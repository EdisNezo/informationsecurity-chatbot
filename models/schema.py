"""Data models for the RAG chatbot."""

from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A message in a conversation."""
    role: MessageRole
    content: str


class Conversation(BaseModel):
    """A conversation between a user and the assistant."""
    id: str
    messages: List[Message] = []
    context_answers: Dict[str, str] = {}
    section_answers: Dict[int, Dict[str, str]] = {}
    current_state: str = "greeting"
    current_context_question: int = 0
    current_section: int = 1
    questions_asked_in_section: int = 0


class Section(BaseModel):
    """A section in the information security script template."""
    id: int
    title: str
    description: str
    questions: List[str]
    example: str


class Template(BaseModel):
    """The template for the information security script."""
    template: str
    version: str
    sections: List[Section]
    contextQuestions: List[str]


class DocumentChunk(BaseModel):
    """A chunk of a document."""
    id: str
    text: str
    source: str
    metadata: Dict = {}


class RetrievalResult(BaseModel):
    """Results from the RAG retrieval."""
    chunks: List[DocumentChunk]
    source_documents: List[str]


class ChatResponse(BaseModel):
    """Response to a chat message."""
    message: str
    conversation_id: str
    state: str = "in_progress"
    script: Optional[str] = None
    next_question: Optional[str] = None


class ChatRequest(BaseModel):
    """Request for a chat message."""
    message: str
    conversation_id: Optional[str] = None