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
import config

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents for the RAG chatbot."""

    def __init__(self):
        """Initialize the document processor."""
        self.ollama_client = OllamaClient()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.index = None
        self.id_to_text = {}
        self.load_or_create_index()

    def load_or_create_index(self) -> None:
        """Load an existing FAISS index or create a new one."""
        needs_new_index = False
        
        # First, check if the files exist
        if not os.path.exists(config.FAISS_INDEX_PATH) or not os.path.exists(config.FAISS_MAPPING_PATH):
            logger.info("FAISS index or mapping file not found, creating new index")
            needs_new_index = True
        
        if not needs_new_index:
            try:
                # Try to load the existing index
                self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
                self.id_to_text = load_json(config.FAISS_MAPPING_PATH)
                
                # Check if dimensions match
                if self.index.d != config.EMBEDDING_DIMENSION:
                    logger.warning(f"FAISS index dimension ({self.index.d}) doesn't match "
                                f"current embedding dimension ({config.EMBEDDING_DIMENSION})")
                    logger.warning("Creating new index with correct dimensions")
                    needs_new_index = True
                else:
                    logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                needs_new_index = True
    
    # Create a new index if needed
    if needs_new_index:
        self.create_new_index()

    def create_new_index(self) -> None:
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
        self.id_to_text = {}
        logger.info("Created new FAISS index")

    def index_documents(self) -> None:
        """Index all documents in the source docs directory."""
        for file_path in config.SOURCE_DOCS_DIR.glob("*.*"):
            if file_path.suffix.lower() in [".txt", ".json", ".md"]:
                self.index_document(file_path)
        
        self.save_index()
        logger.info(f"Indexed all documents, total {self.index.ntotal} vectors")

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
            
            embedding = self.ollama_client.get_embedding(chunk)
            if embedding:
                self.index.add(np.array([embedding], dtype=np.float32))
            else:
                logger.warning(f"Could not get embedding for chunk: {chunk[:50]}...")

    def save_index(self) -> None:
        """Save the FAISS index and ID to text mapping."""
        try:
            faiss.write_index(self.index, str(config.FAISS_INDEX_PATH))
            save_json(self.id_to_text, config.FAISS_MAPPING_PATH)
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

    def get_chunks_for_query(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Get relevant document chunks for a query."""
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty, no chunks to retrieve")
            return []
        
        query_embedding = self.ollama_client.get_embedding(query)
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