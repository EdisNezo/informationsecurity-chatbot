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
                                 f"configured dimension ({config.EMBEDDING_DIMENSION})")
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
        logger.info(f"Created new FAISS index with dimension {config.EMBEDDING_DIMENSION}")

    def index_documents(self) -> None:
        """Index all documents in the source docs directory."""
        # Verify dimension consistency before starting
        test_embedding = self.ollama_client.get_embedding("Test embedding")
        if test_embedding and len(test_embedding) != config.EMBEDDING_DIMENSION:
            logger.warning(f"Embedding dimension from model ({len(test_embedding)}) differs from config ({config.EMBEDDING_DIMENSION})")
            logger.warning(f"Updating config to match embedding model")
            config.EMBEDDING_DIMENSION = len(test_embedding)
            self.create_new_index()
        
        # Find and process documents
        document_count = 0
        for file_path in config.SOURCE_DOCS_DIR.glob("*.*"):
            if file_path.suffix.lower() in [".txt", ".json", ".md"]:
                self.index_document(file_path)
                document_count += 1
        
        if document_count > 0:
            self.save_index()
            logger.info(f"Indexed {document_count} documents, total {self.index.ntotal} vectors")
        else:
            logger.warning("No documents found to index")

    def index_document(self, file_path: Path) -> None:
        """Index a single document."""
        logger.info(f"Indexing document: {file_path}")
        
        # Read the document
        if file_path.suffix.lower() == ".json":
            document = load_json(file_path)
            text = json.dumps(document, ensure_ascii=False)
        else:
            text = read_text_file(file_path)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Process each chunk
        chunk_count = 0
        for chunk in chunks:
            # Get embedding
            embedding = self.ollama_client.get_embedding(chunk)
            
            # Verify embedding is valid and has correct dimensions
            if embedding and len(embedding) > 0:
                # Double check dimensions
                if len(embedding) != config.EMBEDDING_DIMENSION:
                    logger.error(f"Embedding dimension mismatch: got {len(embedding)}, expected {config.EMBEDDING_DIMENSION}")
                    continue
                
                # Store the chunk and its embedding
                chunk_id = generate_id()
                self.id_to_text[chunk_id] = {
                    "text": chunk,
                    "source": file_path.name,
                    "metadata": {
                        "path": str(file_path),
                        "type": file_path.suffix.lower()[1:],
                    }
                }
                
                # Add to index
                try:
                    self.index.add(np.array([embedding], dtype=np.float32))
                    chunk_count += 1
                except Exception as e:
                    logger.error(f"Error adding vector to index: {e}")
            else:
                logger.warning(f"Could not get embedding for chunk: {chunk[:50]}...")
        
        logger.info(f"Successfully indexed {chunk_count} chunks from {file_path.name}")

    def save_index(self) -> None:
        """Save the FAISS index and ID to text mapping."""
        try:
            # Only save if there's data in the index
            if self.index.ntotal > 0:
                faiss.write_index(self.index, str(config.FAISS_INDEX_PATH))
                save_json(self.id_to_text, config.FAISS_MAPPING_PATH)
                logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
            else:
                logger.warning("Not saving empty index")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

    def get_chunks_for_query(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Get relevant document chunks for a query."""
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty, no chunks to retrieve")
            return []
        
        # Get embedding for query
        query_embedding = self.ollama_client.get_embedding(query)
        
        if not query_embedding:
            logger.error("Could not get embedding for query")
            return []
        
        # Verify dimensions
        if len(query_embedding) != self.index.d:
            logger.error(f"Query embedding dimension ({len(query_embedding)}) doesn't match index dimension ({self.index.d})")
            return []
            
        # Search for similar chunks
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query_embedding_np, min(top_k, self.index.ntotal))
        
        # Convert results to DocumentChunk objects
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
        
        logger.info(f"Found {len(chunks)} relevant chunks for query")
        return chunks