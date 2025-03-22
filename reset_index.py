"""Reset the FAISS index to use 1024-dimensional embeddings."""

import os
import logging
import sys
import faiss
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def reset_index():
    """Reset the FAISS index with 1024-dimensional vectors."""
    # Set the embedding dimension to 1024 for mxbai-embed-large
    config.EMBEDDING_DIMENSION = 1024
    
    # Create a new empty FAISS index with this dimension
    new_index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
    
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(config.FAISS_INDEX_PATH), exist_ok=True)
    
    # Save the index
    logger.info(f"Creating new FAISS index with dimension {config.EMBEDDING_DIMENSION}")
    faiss.write_index(new_index, str(config.FAISS_INDEX_PATH))
    
    # Delete the mapping file if it exists
    if os.path.exists(config.FAISS_MAPPING_PATH):
        logger.info(f"Removing old mapping file: {config.FAISS_MAPPING_PATH}")
        os.remove(config.FAISS_MAPPING_PATH)
    
    logger.info(f"Index reset complete. Created new empty index with dimension {config.EMBEDDING_DIMENSION}")
    logger.info(f"You can now start your application to begin indexing documents")

if __name__ == "__main__":
    reset_index()