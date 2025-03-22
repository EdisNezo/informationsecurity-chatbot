"""Document processing for the RAG chatbot."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

from models.schema import DocumentChunk
from utils import load_json, save_json, read_text_file, generate_id
from ollama_client import OllamaClient
from embedding_service import NomicEmbeddingService
import config

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents for the RAG chatbot."""

    def __init__(self):
        """Initialize the document processor."""
        self.ollama_client = OllamaClient()
        # Use the new embedding service
        self.embedding_service = NomicEmbeddingService()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.index = None
        self.id_to_text = {}
        self.load_or_create_index()

    def load_or_create_index(self) -> None:
        """Load an existing FAISS index or create a new one."""
        # Rest of method remains the same...

    def create_new_index(self) -> None:
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
        self.id_to_text = {}
        logger.info("Created new FAISS index")

    def index_documents(self) -> None:
        """Index all documents in the source docs directory."""
        # Rest of method remains the same...

    def index_document(self, file_path: Path) -> None:
        """Index a single document."""
        logger.info(f"Indexing document: {file_path}")
        
        if file_path.suffix.lower() == ".json":
            document = load_json(file_path)
            text = json.dumps(document, ensure_ascii=False)
        else:
            text = read_text_file(file_path)
        
        chunks = self.text_splitter.split_text(text)
        
        for chunk in chunks:
            chunk_id = generate_id()
            self.id_to_text[chunk_id] = {
                "text": chunk,
                "source": file_path.name,
                "metadata": {
                    "path": str(file_path),
                    "type": file_path.suffix.lower()[1:],
                }
            }
            
            # Use the new embedding service
            embedding = self.embedding_service.get_embedding(chunk)
            if embedding:
                self.index.add(np.array([embedding], dtype=np.float32))
            else:
                logger.warning(f"Could not get embedding for chunk: {chunk[:50]}...")

    def get_chunks_for_query(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Get relevant document chunks for a query."""
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty, no chunks to retrieve")
            return []
        
        # Use the new embedding service
        query_embedding = self.embedding_service.get_embedding(query)
        if not query_embedding:
            logger.error("Could not get embedding for query")
            return []
            
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        
        scores, indices = self.index.search(query_embedding_np, top_k)
        
        chunks = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 indicates no result
                chunk_id = list(self.id_to_text.keys())[idx]
                chunk_data = self.id_to_text[chunk_id]
                chunks.append(DocumentChunk(
                    id=chunk_id,
                    text=chunk_data["text"],
                    source=chunk_data["source"],
                    metadata=chunk_data["metadata"]
                ))
        
        return chunks