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
import faiss

logger = logging.getLogger(__name__)

app = FastAPI(title="Healthcare Security Script Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path("static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    
app.mount("/static", StaticFiles(directory="static"), name="static")
templates_dir = Path("templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
templates = Jinja2Templates(directory="templates")

dialog_manager = DialogManager()
script_generator = ScriptGenerator()
document_processor = DocumentProcessor()

script_tasks = {}

@app.on_event("startup")
async def startup_event():
    # Initialisiert den Dokumentenindex beim Start und testet Verbindungen
    logger.info("Starting application initialization...")
    
    try:
        logger.info("Testing connection to Ollama embedding service...")
        ollama_client = OllamaClient()
        test_embedding = ollama_client.get_embedding("This is a test sentence for embeddings.")
        
        if test_embedding:
            actual_dimension = len(test_embedding)
            logger.info(f"Successfully connected to Ollama embeddings API")
            logger.info(f"Embedding dimension from mxbai-embed-large: {actual_dimension}")
            
            if actual_dimension != config.EMBEDDING_DIMENSION:
                logger.warning(f"⚠️ Dimension mismatch: Config has {config.EMBEDDING_DIMENSION}, but model produces {actual_dimension}")
                logger.warning(f"Updating in-memory EMBEDDING_DIMENSION to {actual_dimension}")
                config.EMBEDDING_DIMENSION = actual_dimension
                
                if os.path.exists(config.FAISS_INDEX_PATH):
                    try:
                        index = faiss.read_index(str(config.FAISS_INDEX_PATH))
                        if index.d != actual_dimension:
                            logger.warning("Existing FAISS index has incompatible dimensions")
                            logger.warning("You should run the reset_index.py script before continuing")
                    except Exception as e:
                        logger.error(f"Error checking existing index: {e}")
        else:
            logger.error("❌ Failed to get test embedding - received empty result")
            logger.error("Check if mxbai-embed-large is available in Ollama")
    except Exception as e:
        logger.error(f"❌ Error testing embedding connection: {e}", exc_info=True)
        logger.error("Application may not function correctly without embeddings")
    
    try:
        logger.info(f"Testing connection to Ollama LLM service with model {config.LLM_MODEL}...")
        test_response = ollama_client.generate(
            "Test", 
            system_prompt="This is a test.", 
            max_tokens=20
        )
        if test_response:
            logger.info(f"Successfully connected to Ollama LLM API")
            logger.info(f"Response preview: {test_response[:30]}...")
        else:
            logger.error(f"❌ Failed to get response from LLM model {config.LLM_MODEL}")
    except Exception as e:
        logger.error(f"❌ Error testing LLM connection: {e}", exc_info=True)
        logger.error("Application may not function correctly without LLM capabilities")
    
    try:
        logger.info("Initializing document processor...")
        
        if os.path.exists(config.FAISS_INDEX_PATH):
            try:
                index = faiss.read_index(str(config.FAISS_INDEX_PATH))
                if index.d != config.EMBEDDING_DIMENSION:
                    logger.warning(f"⚠️ Existing FAISS index has dimension {index.d}, but we need {config.EMBEDDING_DIMENSION}")
                    logger.warning("Deleting old index files to force recreation")
                    os.remove(config.FAISS_INDEX_PATH)
                    if os.path.exists(config.FAISS_MAPPING_PATH):
                        os.remove(config.FAISS_MAPPING_PATH)
            except Exception as e:
                logger.error(f"Error checking existing FAISS index: {e}")
                logger.warning("Will try to create a new index")
        
        logger.info("Indexing documents...")
        document_processor.index_documents()
        logger.info("Document indexing completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during document indexing: {e}", exc_info=True)
        logger.error("Application may have limited retrieval capabilities")
    
    try:
        logger.info("Testing retrieval pipeline...")
        test_query = "Informationssicherheit im Gesundheitswesen"
        chunks = document_processor.get_chunks_for_query(test_query, top_k=1)
        if chunks:
            logger.info(f"Retrieval test successful: found {len(chunks)} relevant chunks")
            logger.info(f"Sample chunk: {chunks[0].text[:50]}...")
        else:
            logger.warning("⚠️ Retrieval test returned no results")
    except Exception as e:
        logger.error(f"❌ Error during retrieval test: {e}", exc_info=True)
    
    logger.info("Application startup complete!")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Rendert die Hauptseite
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    # Verarbeitet eine Chat-Nachricht
    conversation_id = request.conversation_id
    
    if not conversation_id:
        conversation = dialog_manager.create_conversation()
        conversation_id = conversation.id
        return ChatResponse(
            message=conversation.messages[-1].content,
            conversation_id=conversation_id,
            state=conversation.current_state,
            next_question=conversation.messages[-1].content if conversation.current_state != "finished" else None
        )
    
    response_message, script, state = dialog_manager.process_message(conversation_id, request.message)
    
    if state == "generating_script":
        if conversation_id in script_tasks:
            script_tasks.pop(conversation_id, None)
        
        task = asyncio.create_task(
            script_generator_with_timeout(conversation_id, dialog_manager, 3000)
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
    # Holt das generierte Skript für eine Konversation
    conversation = dialog_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    if conversation.current_state == "generating_script":
        task = script_tasks.get(conversation_id)
        if task and not task.done():
            return {"status": "generating", "message": "Skript wird noch generiert..."}
        else:
            if task:
                try:
                    await task
                except Exception as e:
                    logger.error(f"Script generation task failed: {e}", exc_info=True)
                    dialog_manager.set_error_state(conversation_id, str(e))
                    return {"status": "error", "message": str(e)}
            
            if conversation.current_state == "finished":
                return {"status": "complete", "script": dialog_manager.get_generated_script(conversation_id) or ""}
            elif conversation.current_state == "error":
                return {"status": "error", "message": "Fehler bei der Skripterstellung"}
            else:
                try:
                    script = script_generator.generate_script(conversation)
                    dialog_manager.set_generated_script(conversation_id, script)
                    return {"status": "complete", "script": script}
                except Exception as e:
                    logger.error(f"Error in fallback script generation: {e}", exc_info=True)
                    dialog_manager.set_error_state(conversation_id, str(e))
                    return {"status": "error", "message": str(e)}
    
    if conversation.current_state == "finished":
        script = dialog_manager.get_generated_script(conversation_id)
        if not script:
            try:
                script = script_generator.generate_script(conversation)
                dialog_manager.set_generated_script(conversation_id, script)
            except Exception as e:
                logger.error(f"Error generating missing script: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}
        
        return {"status": "complete", "script": script}
    
    if conversation.current_state == "error":
        return {"status": "error", "message": "Fehler bei der Skripterstellung"}
    
    return {"status": "unknown", "message": f"Unerwarteter Zustand: {conversation.current_state}"}


@app.post("/api/reload-documents")
async def reload_documents():
    # Lädt alle Dokumente neu und indexiert sie
    try:
        document_processor.index_documents()
        return {"status": "success", "message": "Documents reindexed successfully"}
    except Exception as e:
        logger.error(f"Error reindexing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    # Überprüft den Systemzustand
    try:
        ollama_client = OllamaClient()
        test_response = ollama_client.generate("Test", system_prompt="This is a test.", max_tokens=10)
        ollama_status = "ok" if test_response else "error"
        
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
    # Führt die Skripterstellung mit einem Timeout aus
    try:
        script_future = asyncio.get_event_loop().run_in_executor(
            None, 
            script_generator.process_script_generation, 
            conversation_id, 
            dialog_manager
        )
        
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