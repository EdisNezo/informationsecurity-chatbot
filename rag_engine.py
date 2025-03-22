"""RAG (Retrieval Augmented Generation) engine for the chatbot."""

import logging
from typing import Dict, List, Any, Optional

from models.schema import DocumentChunk, RetrievalResult
from document_processor import DocumentProcessor
from ollama_client import OllamaClient
import config
from embedding_service import NomicEmbeddingService

logger = logging.getLogger(__name__)


class RAGEngine:
    """Retrieval Augmented Generation engine."""

    def __init__(self):
        """Initialize the RAG engine."""
        self.document_processor = DocumentProcessor()
        self.ollama_client = OllamaClient()
        self.embedding_service = NomicEmbeddingService()

    def retrieve(self, query: str, context: Dict[str, Any] = None, top_k: int = 5) -> RetrievalResult:
        """Retrieve relevant documents for a query."""
        # Enhance query with context if available
        enhanced_query = query
        if context:
            context_str = ", ".join([f"{k}: {v}" for k, v in context.items() if v])
            enhanced_query = f"{query} Context: {context_str}"
        
        chunks = self.document_processor.get_chunks_for_query(enhanced_query, top_k)
        sources = list(set([chunk.source for chunk in chunks]))
        
        return RetrievalResult(chunks=chunks, source_documents=sources)

    def generate(self, query: str, retrieved_result: RetrievalResult, 
                 context: Dict[str, Any] = None) -> str:
        """Generate a response using the query and retrieved documents."""
        # Create a context string from retrieved documents
        retrieved_context = "\n\n".join([chunk.text for chunk in retrieved_result.chunks])
        
        # Create context string from additional context
        context_str = ""
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items() if v])
        
        # Create the prompt
        prompt = f"""
Beantworte die folgende Frage basierend auf dem bereitgestellten Kontext.

Frage: {query}

Kontext:
{retrieved_context}

Zusätzlicher Kontext:
{context_str}
"""

        system_prompt = "Du bist ein hilfreicher Assistent, der Skripte für Informationssicherheitskurse im Gesundheitswesen erstellt."
        
        # Call Ollama
        response = self.ollama_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
        
        return response

    def generate_script_section(self, section_id: int, section_title: str, 
                               section_questions: Dict[str, str], 
                               context_answers: Dict[str, str]) -> str:
        """Generate a section of the script."""
        # Retrieve relevant documents for this section
        section_query = f"Informationssicherheit {section_title} im Gesundheitswesen"
        retrieved_result = self.retrieve(section_query, context=context_answers, top_k=3)
        
        # Create prompt for this section
        questions_answers = "\n".join([f"Frage: {q}\nAntwort: {a}" for q, a in section_questions.items()])
        
        retrieved_context = "\n\n".join([chunk.text for chunk in retrieved_result.chunks])
        
        context_str = "\n".join([f"{k}: {v}" for k, v in context_answers.items() if v])
        
        prompt = f"""
Erstelle einen Abschnitt für ein Schulungsskript zur Informationssicherheit.
Der Abschnitt gehört zu: {section_title}

Die folgenden Fragen und Antworten wurden gesammelt:
{questions_answers}

Kontext aus der Organisation:
{context_str}

Zusätzliche Informationen:
{retrieved_context}

Erstelle einen zusammenhängenden, gut lesbaren Skriptabschnitt, der wie der Beispielabschnitt eines Schulungsvideos klingt.
Der Text sollte didaktisch wertvoll sein und zum Lernen anregen.
Schreibe den Text so, dass er von einem Sprecher vorgelesen werden kann.
"""

        system_prompt = "Du bist ein Experte für Informationssicherheit im Gesundheitswesen und erstellst Schulungsskripte."
        
        response = self.ollama_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS
        )
        
        return response