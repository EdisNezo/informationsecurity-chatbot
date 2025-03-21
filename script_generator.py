"""Script generation for the RAG chatbot."""

import logging
from typing import Dict, List, Any, Optional

from models.schema import Conversation
from rag_engine import RAGEngine
from utils import format_script_section
import config

logger = logging.getLogger(__name__)


class ScriptGenerator:
    """Generate information security scripts."""

    def __init__(self):
        """Initialize the script generator."""
        self.rag_engine = RAGEngine()

    def generate_script(self, conversation: Conversation) -> str:
        """Generate a complete script from the conversation data."""
        if not conversation.section_answers:
            return "Keine ausreichenden Daten für die Skripterstellung vorhanden."
        
        # Generate each section
        sections = []
        
        # Generate introduction
        intro = self._generate_introduction(conversation)
        sections.append(intro)
        
        # Generate each content section
        for section_id, questions_answers in conversation.section_answers.items():
            if not questions_answers:
                continue
            
            # Find the section title
            section_title = f"Sektion {section_id}"
            for section in conversation.section_answers:
                if section.id == section_id:
                    section_title = section.title
                    break
            
            section_content = self.rag_engine.generate_script_section(
                section_id=section_id,
                section_title=section_title,
                section_questions=questions_answers,
                context_answers=conversation.context_answers
            )
            
            formatted_section = format_script_section(section_id, section_title, section_content)
            sections.append(formatted_section)
        
        # Generate conclusion
        conclusion = self._generate_conclusion(conversation)
        sections.append(conclusion)
        
        # Combine all sections
        full_script = "\n\n".join(sections)
        
        return full_script

    def _generate_introduction(self, conversation: Conversation) -> str:
        """Generate the introduction for the script."""
        organization_type = conversation.context_answers.get(
            "In welcher medizinischen Einrichtung sollen die Schulungen umgesetzt werden?", 
            "Ihrer Gesundheitseinrichtung"
        )
        
        prompt = f"""
Erstelle eine einleitende Passage für ein Schulungsskript zur Informationssicherheit im Gesundheitswesen.
Die Schulung ist für {organization_type}.

Die Einleitung sollte:
1. Die Teilnehmer begrüßen
2. Die Wichtigkeit von Informationssicherheit im Gesundheitswesen betonen
3. Die Ziele der Schulung kurz erläutern
4. Die Teilnehmer motivieren, aktiv teilzunehmen

Schreibe die Einleitung so, dass sie von einem Sprecher vorgelesen werden kann.
"""
        
        # Use the RAG engine for generation
        context = {"organization_type": organization_type}
        for question, answer in conversation.context_answers.items():
            # Create a simplified key for the context
            key = question.split("?")[0].split()[-3:]
            key = "_".join(key).lower()
            context[key] = answer
        
        retrieved_result = self.rag_engine.retrieve(prompt, context=context)
        introduction = self.rag_engine.generate(prompt, retrieved_result, context=context)
        
        return format_script_section(0, "Einleitung", introduction)

    def _generate_conclusion(self, conversation: Conversation) -> str:
        """Generate the conclusion for the script."""
        organization_type = conversation.context_answers.get(
            "In welcher medizinischen Einrichtung sollen die Schulungen umgesetzt werden?", 
            "Ihrer Gesundheitseinrichtung"
        )
        
        prompt = f"""
Erstelle eine abschließende Passage für ein Schulungsskript zur Informationssicherheit im Gesundheitswesen.
Die Schulung ist für {organization_type}.

Die Schlussfolgerung sollte:
1. Die wichtigsten Punkte der Schulung zusammenfassen
2. Die Teilnehmer ermutigen, das Gelernte in der Praxis anzuwenden
3. Danke für die Aufmerksamkeit sagen und zum Fragen stellen ermutigen
4. Auf weitere Ressourcen oder Unterstützung hinweisen

Schreibe die Schlussfolgerung so, dass sie von einem Sprecher vorgelesen werden kann.
"""
        
        # Use the RAG engine for generation
        context = {"organization_type": organization_type}
        for question, answer in conversation.context_answers.items():
            # Create a simplified key for the context
            key = question.split("?")[0].split()[-3:]
            key = "_".join(key).lower()
            context[key] = answer
        
        retrieved_result = self.rag_engine.retrieve(prompt, context=context)
        conclusion = self.rag_engine.generate(prompt, retrieved_result, context=context)
        
        return format_script_section(8, "Abschluss", conclusion)

    def process_script_generation(self, conversation_id: str, dialog_manager) -> str:
        """Process script generation for a conversation."""
        conversation = dialog_manager.get_conversation(conversation_id)
        if not conversation:
            return "Konversation nicht gefunden."
        
        try:
            script = self.generate_script(conversation)
            dialog_manager.set_generated_script(conversation_id, script)
            return script
        except Exception as e:
            logger.error(f"Error generating script: {e}")
            return "Es ist ein Fehler bei der Skripterstellung aufgetreten. Bitte versuchen Sie es später erneut."