"""Client for interacting with Ollama API."""

import logging
import requests
from typing import Dict, List, Any, Optional
import numpy as np
import json

import config

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(self, base_url=None, model=None, embedding_model=None):
        """Initialize the Ollama client."""
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.model = model or config.LLM_MODEL
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.embeddings_endpoint = f"{self.base_url}/api/embeddings"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
             temperature: float = None, max_tokens: int = None) -> str:
        """Generate text using Ollama with robust error handling for malformed JSON."""
        temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS

        payload = {
            "model": self.model,
            "prompt": prompt
        }
        
        # Add optional parameters if provided
        if temperature is not None:
            payload["temperature"] = temperature
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # Ensure we're explicitly setting stream to false
        payload["stream"] = False
        
        try:
            # Send the request to Ollama
            response = requests.post(self.generate_endpoint, json=payload)
            
            if response.status_code == 200:
                # Get the raw text response
                raw_response_text = response.text
                
                try:
                    # Try to parse as standard JSON
                    response_data = json.loads(raw_response_text)
                    return response_data.get('response', '')
                except json.JSONDecodeError as json_err:
                    logger.warning(f"JSON parsing error: {json_err}")
                    
                    # Attempt to extract the first valid JSON object from the response
                    import re
                    json_pattern = r'(\{.*?\})'
                    json_matches = re.findall(json_pattern, raw_response_text, re.DOTALL)
                    
                    if json_matches:
                        # Try each potential JSON object until we find a valid one
                        for potential_json in json_matches:
                            try:
                                obj = json.loads(potential_json)
                                if 'response' in obj:
                                    logger.info("Successfully extracted response from malformed JSON")
                                    return obj.get('response', '')
                            except json.JSONDecodeError:
                                continue
                    
                    # If we can't extract JSON, look for response pattern directly
                    response_pattern = r'"response"\s*:\s*"(.*?)(?:"|$)'
                    response_match = re.search(response_pattern, raw_response_text, re.DOTALL)
                    
                    if response_match:
                        # Found response text directly in pattern
                        logger.info("Extracted response using regex pattern")
                        return response_match.group(1)
                    
                    # As a last resort, return the raw response text with a warning
                    logger.warning("Couldn't parse JSON response, returning raw text")
                    return f"Error parsing response. Raw text: {raw_response_text[:100]}..."
            else:
                error_message = f"Error {response.status_code}: {response.text}"
                logger.error(error_message)
                return f"Error generating text: {error_message}"
        except Exception as e:
            logger.error(f"Exception in generate request: {e}")
            return f"Error generating text: {str(e)}"

    def _filter_reasoning_output(self, text: str) -> str:
        """Filter out reasoning/thinking sections from model output."""
        # Pattern 1: Remove <think> tags and content between them
        import re
        
        # Pattern: <think>any content</think>
        pattern1 = r'<think>.*?</think>'
        # In case closing tag is missing
        pattern2 = r'<think>.*?$'
        # Looser pattern that catches variations
        pattern3 = r'<think.*?>.*?(?:</think>|$)'
        
        # Apply all patterns
        text = re.sub(pattern1, '', text, flags=re.DOTALL)
        text = re.sub(pattern2, '', text, flags=re.DOTALL)
        text = re.sub(pattern3, '', text, flags=re.DOTALL)
        
        # Pattern 2: Just the <think> tag without closing (more aggressive)
        pattern4 = r'<think>.*'
        text = re.sub(pattern4, '', text, flags=re.DOTALL)
        
        # Additional patterns for other reasoning formats
        # Handle square bracket thinking: [thinking: ...]
        text = re.sub(r'\[thinking:.*?\]', '', text, flags=re.DOTALL)
        text = re.sub(r'\[thought:.*?\]', '', text, flags=re.DOTALL)
        
        # Clean up any double spaces or newlines that resulted from removing content
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for text using Ollama with mxbai-embed-large."""
        # Construct a proper prompt for passage embeddings
        formatted_text = f"Represent this text for searching relevant passages: {text}"
        
        payload = {
            "model": self.embedding_model,
            "prompt": formatted_text
        }
        
        try:
            logger.debug(f"Getting embedding for text (length {len(text)})")
            response = requests.post(self.embeddings_endpoint, json=payload)
            response.raise_for_status()
            embedding = response.json().get('embedding', [])
            
            if embedding:
                logger.debug(f"Successfully retrieved embedding of dimension {len(embedding)}")
            else:
                logger.warning(f"Received empty embedding from Ollama API")
                
            return embedding
        except Exception as e:
            logger.error(f"Error getting embeddings with Ollama ({self.embedding_model}): {e}")
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