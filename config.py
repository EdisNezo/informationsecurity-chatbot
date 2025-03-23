"""Configuration settings for the RAG chatbot using Ollama."""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SOURCE_DOCS_DIR = DATA_DIR / "source_docs"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Ensure directories exist
os.makedirs(SOURCE_DOCS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:27b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# Embedding configuration
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
EMBEDDING_DIMENSION = 1024

# FAISS configuration
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss_index.bin"
FAISS_MAPPING_PATH = EMBEDDINGS_DIR / "id_to_text.json"

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Dialog configuration
MAX_QUESTIONS_PER_SECTION = 2
CONTEXT_QUESTIONS = [
    "In welcher medizinischen Einrichtung sollen die Schulungen umgesetzt werden?",
    "Gibt es spezifische Personengruppen, die zu berücksichtigen sind?",
    "Welche Vorkenntnisse haben die Teilnehmer im Bereich Informationssicherheit?",
    "Wie lang darf die Schulung maximal sein?",
    "Welche spezifischen Bedrohungen sind für Ihre Einrichtung besonders relevant?",
    "Welche sensiblen Daten werden in Ihrer Einrichtung verarbeitet?",
    "Gab es in der Vergangenheit bereits Sicherheitsvorfälle und welcher Art waren diese?",
    "Wie sieht der typische Arbeitsalltag der Mitarbeiter in Bezug auf IT-Systeme aus?"
]

# FastAPI configuration
API_PORT = int(os.getenv("API_PORT", "8000"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")