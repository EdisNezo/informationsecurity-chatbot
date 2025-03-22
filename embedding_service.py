"""Embedding service using nomic-embed-text model."""

import logging
import numpy as np
from typing import List, Union
import torch
from transformers import AutoTokenizer, AutoModel

import config

logger = logging.getLogger(__name__)

class NomicEmbeddingService:
    """Service for generating embeddings using nomic-embed-text."""
    
    def __init__(self):
        """Initialize the embedding service."""
        self.model_name = "nomic-ai/nomic-embed-text-v1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_dimension = 768  # nomic-embed-text dimension
        
        try:
            logger.info(f"Loading nomic-embed-text model on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            logger.info("nomic-embed-text model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading nomic-embed-text model: {e}")
            self.tokenizer = None
            self.model = None
            
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using nomic-embed-text."""
        if self.tokenizer is None or self.model is None:
            logger.error("Model not loaded, returning zeros")
            return [0.0] * self.embedding_dimension
            
        try:
            # Tokenize and prepare inputs
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   truncation=True, max_length=512, 
                                   padding=True).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use the [CLS] token embedding as the text embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Convert to list and return
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.embedding_dimension
            
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)