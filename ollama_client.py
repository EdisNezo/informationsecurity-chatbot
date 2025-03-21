"""Client for interacting with Ollama API."""

import logging
import requests
from typing import Dict, List, Any, Optional
import numpy as np

import config

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url=None, model=None):
        """Initialize the Ollama client."""
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.model = model or config.LLM_MODEL
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.embeddings_endpoint = f"{self.base_url}/api/embeddings"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = None, max_tokens: int = None) -> str:
        """Generate text using Ollama."""
        temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(self.generate_endpoint, json=payload)
            response.raise_for_status()
            return response.json().get('response', '')
        except Exception as e:
            logger.error(f"Error generating text with Ollama: {e}")
            return f"Error generating text: {str(e)}"

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for text using Ollama."""
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(self.embeddings_endpoint, json=payload)
            response.raise_for_status()
            return response.json().get('embedding', [])
        except Exception as e:
            logger.error(f"Error getting embeddings with Ollama: {e}")
            return []

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                # If embedding failed, add a zero vector
                embeddings.append([0.0] * config.EMBEDDING_DIMENSION)
        
        return np.array(embeddings, dtype=np.float32)