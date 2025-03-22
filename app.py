"""FastAPI application for the RAG chatbot."""

import asyncio
import logging
import os
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from models.schema import ChatRequest, ChatResponse
from dialog_manager import DialogManager
from script_generator import ScriptGenerator
from document_processor import DocumentProcessor
from ollama_client import OllamaClient
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

# Create static directory if it doesn't exist
static_dir = Path("static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    
# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates_dir = Path("templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
templates = Jinja2Templates(directory="templates")

# Initialize components
dialog_manager = DialogManager()
script_generator = ScriptGenerator()
document_processor = DocumentProcessor()

# Store script generation tasks
script_tasks = {}

# Initialize document index and test Ollama on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the document index on startup and test Ollama connection."""
    # Index documents
    try:
        document_processor.index_documents()
        logger.info("Document indexing completed successfully")
    except Exception as e:
        logger.error(f"Error indexing documents: {e}", exc_info=True)
    
    # Test Ollama connection
    try:
        ollama_client = OllamaClient()
        test_response = ollama_client.generate("Test", system_prompt="This is a test.", max_tokens=20)
        logger.info(f"Ollama connection test successful: {test_response[:20]}...")
    except Exception as e:
        logger.error(f"Ollama connection test failed: {e}", exc_info=True)


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
    
    # If we need to generate a script, do it in the background with timeout protection
    if state == "generating_script":
        # Remove any existing task
        if conversation_id in script_tasks:
            script_tasks.pop(conversation_id, None)
        
        # Add the new background task
        task = asyncio.create_task(
            script_generator_with_timeout(conversation_id, dialog_manager, 300)  # 5 minute timeout
        )
        script_tasks[conversation_id] = task
        
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
    
    # Check if generation is still in progress
    if conversation.current_state == "generating_script":
        # Check if the task is still running
        task = script_tasks.get(conversation_id)
        if task and not task.done():
            return {"status": "generating", "message": "Skript wird noch generiert..."}
        else:
            # Task is done or missing, but state wasn't updated - something went wrong
            if task:
                try:
                    # Try to get the result and handle any exceptions
                    await task
                except Exception as e:
                    logger.error(f"Script generation task failed: {e}", exc_info=True)
                    dialog_manager.set_error_state(conversation_id, str(e))
                    return {"status": "error", "message": str(e)}
            
            # Check the state again after potentially handling task exceptions
            if conversation.current_state == "finished":
                return {"status": "complete", "script": dialog_manager.get_generated_script(conversation_id) or ""}
            elif conversation.current_state == "error":
                return {"status": "error", "message": "Fehler bei der Skripterstellung"}
            else:
                # Generate the script now as fallback
                try:
                    script = script_generator.generate_script(conversation)
                    dialog_manager.set_generated_script(conversation_id, script)
                    return {"status": "complete", "script": script}
                except Exception as e:
                    logger.error(f"Error in fallback script generation: {e}", exc_info=True)
                    dialog_manager.set_error_state(conversation_id, str(e))
                    return {"status": "error", "message": str(e)}
    
    # If generation is finished, return the script
    if conversation.current_state == "finished":
        script = dialog_manager.get_generated_script(conversation_id)
        if not script:
            # Generate it if missing (shouldn't happen normally)
            try:
                script = script_generator.generate_script(conversation)
                dialog_manager.set_generated_script(conversation_id, script)
            except Exception as e:
                logger.error(f"Error generating missing script: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}
        
        return {"status": "complete", "script": script}
    
    # If there was an error, report it
    if conversation.current_state == "error":
        return {"status": "error", "message": "Fehler bei der Skripterstellung"}
    
    # Default fallback
    return {"status": "unknown", "message": f"Unerwarteter Zustand: {conversation.current_state}"}


@app.post("/api/reload-documents")
async def reload_documents():
    """Reload and reindex all documents."""
    try:
        document_processor.index_documents()
        return {"status": "success", "message": "Documents reindexed successfully"}
    except Exception as e:
        logger.error(f"Error reindexing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Check system health."""
    try:
        # Test Ollama connection
        ollama_client = OllamaClient()
        test_response = ollama_client.generate("Test", system_prompt="This is a test.", max_tokens=10)
        ollama_status = "ok" if test_response else "error"
        
        # Get memory usage
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            memory_usage = {
                "total_gb": round(memory_info.total / (1024**3), 2),
                "available_gb": round(memory_info.available / (1024**3), 2),
                "percent_used": memory_info.percent
            }
        except ImportError:
            memory_usage = {"error": "psutil not available"}
        
        return {
            "status": "ok",
            "ollama_status": ollama_status,
            "memory": memory_usage,
            "active_conversations": len(dialog_manager.conversations),
            "script_tasks": len(script_tasks)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


async def script_generator_with_timeout(conversation_id: str, dialog_manager, timeout_seconds: int = 300):
    """Run script generation with a timeout."""
    try:
        # Create a task for script generation
        script_future = asyncio.get_event_loop().run_in_executor(
            None, 
            script_generator.process_script_generation, 
            conversation_id, 
            dialog_manager
        )
        
        # Wait for it to complete with a timeout
        script = await asyncio.wait_for(script_future, timeout=timeout_seconds)
        logger.info(f"Script generation completed for conversation {conversation_id}")
        return script
    except asyncio.TimeoutError:
        logger.error(f"Script generation timed out after {timeout_seconds} seconds")
        dialog_manager.set_error_state(
            conversation_id, 
            f"Die Skripterstellung hat das Zeitlimit von {timeout_seconds} Sekunden überschritten."
        )
        return None
    except Exception as e:
        logger.error(f"Error in script generation task: {e}", exc_info=True)
        dialog_manager.set_error_state(conversation_id, str(e))
        return None