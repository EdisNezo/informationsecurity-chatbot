"""FastAPI application for the RAG chatbot."""

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

from models.schema import ChatRequest, ChatResponse
from dialog_manager import DialogManager
from script_generator import ScriptGenerator
from document_processor import DocumentProcessor
import config

logger = logging.getLogger(__name__)

app = FastAPI(title="Healthcare Security Script Generator")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize components
dialog_manager = DialogManager()
script_generator = ScriptGenerator()
document_processor = DocumentProcessor()

# Initialize document index on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the document index on startup."""
    document_processor.index_documents()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Process a chat message."""
    conversation_id = request.conversation_id
    
    # Create a new conversation if needed
    if not conversation_id:
        conversation = dialog_manager.create_conversation()
        conversation_id = conversation.id
        return ChatResponse(
            message=conversation.messages[-1].content,
            conversation_id=conversation_id,
            state=conversation.current_state,
            next_question=conversation.messages[-1].content if conversation.current_state != "finished" else None
        )
    
    # Process the message
    response_message, script, state = dialog_manager.process_message(conversation_id, request.message)
    
    # If we need to generate a script, do it in the background
    if state == "generating_script":
        background_tasks.add_task(
            script_generator.process_script_generation,
            conversation_id, 
            dialog_manager
        )
        return ChatResponse(
            message=response_message,
            conversation_id=conversation_id,
            state="generating_script"
        )
    
    return ChatResponse(
        message=response_message,
        conversation_id=conversation_id,
        state=state,
        script=script
    )


@app.get("/api/script/{conversation_id}")
async def get_script(conversation_id: str):
    """Get the generated script for a conversation."""
    conversation = dialog_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conversation.current_state != "finished":
        return {"status": "generating", "message": "Skript wird noch generiert..."}
    
    # Generate script if not done yet
    script = script_generator.generate_script(conversation)
    
    return {"status": "complete", "script": script}


@app.post("/api/reload-documents")
async def reload_documents():
    """Reload and reindex all documents."""
    try:
        document_processor.index_documents()
        return {"status": "success", "message": "Documents reindexed successfully"}
    except Exception as e:
        logger.error(f"Error reindexing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))