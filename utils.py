"""Utility functions for the RAG chatbot."""

import json
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def load_json(file_path: Path) -> Dict:
    """Load JSON from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}


def save_json(data: Dict, file_path: Path) -> None:
    """Save data as JSON to a file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")


def read_text_file(file_path: Path) -> str:
    """Read text from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        return ""


def save_text_file(text: str, file_path: Path) -> None:
    """Save text to a file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        logger.error(f"Error saving text file {file_path}: {e}")


def group_messages_by_section(messages: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group messages by section."""
    sections = {}
    current_section = None

    for message in messages:
        if message.get("metadata", {}).get("section"):
            current_section = message["metadata"]["section"]
            if current_section not in sections:
                sections[current_section] = []
        
        if current_section is not None:
            sections[current_section].append(message)
    
    return sections


def extract_template_sections(template_data: Dict) -> List[Dict]:
    """Extract sections from the template."""
    return template_data.get("sections", [])


def select_questions_for_section(section: Dict, context_answers: Dict[str, str], max_questions: int = 2) -> List[str]:
    """Select most relevant questions for a section based on context answers."""
    # For now, simply return the first max_questions
    return section["questions"][:max_questions]


def format_script_section(section_id: int, section_title: str, content: str) -> str:
    """Format a section of the script."""
    return f"\n\n{section_title}\n\n{content}"