import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedding_dimension():
    """Get the dimension of embeddings from mxbai-embed-large."""
    try:
        # Test with a simple sentence
        payload = {
            "model": "mxbai-embed-large",
            "prompt": "Represent this text for searching relevant passages: This is a test."
        }
        
        response = requests.post("http://localhost:11434/api/embeddings", json=payload)
        response.raise_for_status()
        
        embedding = response.json().get('embedding', [])
        dimension = len(embedding)
        
        logger.info(f"mxbai-embed-large produces embeddings with dimension: {dimension}")
        return dimension
    except Exception as e:
        logger.error(f"Error getting embedding dimension: {e}")
        return None

if __name__ == "__main__":
    dimension = get_embedding_dimension()
    if dimension:
        print(f"\nYou should update your config.py file to set EMBEDDING_DIMENSION = {dimension}")