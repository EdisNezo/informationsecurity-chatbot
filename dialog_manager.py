"""Dialog management for the RAG chatbot."""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from models.schema import Conversation, Message, MessageRole, Template, Section
from utils import generate_id, load_json, select_questions_for_section
import config

logger = logging.getLogger(__name__)


class DialogManager:
    """Manage the conversation flow."""

    def __init__(self):
        """Initialize the dialog manager."""
        self.conversations = {}
        self.template = self.load_template()
        self.sections = self.template.sections if self.template else []

    def load_template(self) -> Optional[Template]:
        """Load the script template."""
        template_path = config.SOURCE_DOCS_DIR / "blank_template.json"
        if not template_path.exists():
            logger.error(f"Template file not found: {template_path}")
            return None
        
        template_data = load_json(template_path)
        if not template_data:
            return None
        
        return Template(**template_data)

    def create_conversation(self) -> Conversation:
        """Create a new conversation."""
        conversation_id = generate_id()
        conversation = Conversation(id=conversation_id)
        
        # Add greeting message
        greeting = """
Willkommen! Ich bin Ihr Assistent für die Erstellung von Informationssicherheits-Schulungsskripten im Gesundheitswesen. 
Ich werde Ihnen einige Fragen stellen, um ein maßgeschneidertes Skript für Ihre Einrichtung zu erstellen.

Zunächst benötige ich einige Informationen zum Kontext Ihrer Einrichtung.
"""
        conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=greeting))
        
        # Add first context question
        if config.CONTEXT_QUESTIONS:
            conversation.current_state = "context_questions"
            conversation.current_context_question = 0
            first_question = config.CONTEXT_QUESTIONS[0]
            conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=first_question))
        
        self.conversations[conversation_id] = conversation
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)

    def process_message(self, conversation_id: str, message: str) -> Tuple[str, Optional[str], str]:
        """Process a user message and determine the next step in the conversation."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            conversation = self.create_conversation()
            return conversation.messages[-1].content, None, conversation.current_state
        
        # Add user message to conversation
        conversation.messages.append(Message(role=MessageRole.USER, content=message))
        
        # Process based on current state
        if conversation.current_state == "context_questions":
            return self._handle_context_question(conversation, message)
        elif conversation.current_state == "section_questions":
            return self._handle_section_question(conversation, message)
        elif conversation.current_state == "finished":
            return "Ihr Skript wurde bereits erstellt. Möchten Sie ein neues Skript erstellen?", None, "finished"
        else:
            return "Es tut mir leid, aber ich verstehe Ihren aktuellen Status nicht.", None, "error"

    def _handle_context_question(self, conversation: Conversation, message: str) -> Tuple[str, Optional[str], str]:
        """Handle responses to context questions."""
        # Save the answer to the current context question
        current_question = config.CONTEXT_QUESTIONS[conversation.current_context_question]
        conversation.context_answers[current_question] = message
        
        # Move to the next context question or to section questions
        conversation.current_context_question += 1
        if conversation.current_context_question < len(config.CONTEXT_QUESTIONS):
            next_question = config.CONTEXT_QUESTIONS[conversation.current_context_question]
            conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=next_question))
            return next_question, None, conversation.current_state
        else:
            # Transition to section questions
            conversation.current_state = "section_questions"
            conversation.current_section = 1
            conversation.questions_asked_in_section = 0
            
            if self.sections:
                section = self.sections[0]  # First section
                selected_questions = select_questions_for_section(
                    section.dict(), 
                    conversation.context_answers,
                    config.MAX_QUESTIONS_PER_SECTION
                )
                
                if selected_questions:
                    transition_message = f"""
Vielen Dank für diese Informationen! Nun werde ich Ihnen einige Fragen zu den verschiedenen Sektionen des Schulungsskripts stellen.

Beginnen wir mit Sektion 1: {section.title} - {section.description}
"""
                    conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=transition_message))
                    
                    next_question = selected_questions[0]
                    conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=next_question))
                    return transition_message + "\n\n" + next_question, None, conversation.current_state
            
            # If no sections or questions, go to script generation
            return self._finalize_conversation(conversation)

    def _handle_section_question(self, conversation: Conversation, message: str) -> Tuple[str, Optional[str], str]:
        """Handle responses to section questions."""
        # Make sure we have sections
        if not self.sections or conversation.current_section > len(self.sections):
            return self._finalize_conversation(conversation)
        
        # Get current section
        section_idx = conversation.current_section - 1
        section = self.sections[section_idx]
        
        # Save the answer
        selected_questions = select_questions_for_section(
            section.dict(), 
            conversation.context_answers,
            config.MAX_QUESTIONS_PER_SECTION
        )
        
        current_question_idx = conversation.questions_asked_in_section
        if current_question_idx < len(selected_questions):
            current_question = selected_questions[current_question_idx]
            
            # Initialize the section answers dict if needed
            if conversation.current_section not in conversation.section_answers:
                conversation.section_answers[conversation.current_section] = {}
            
            # Save the answer
            conversation.section_answers[conversation.current_section][current_question] = message
            
            # Increment the question counter
            conversation.questions_asked_in_section += 1
            
            # Check if we have more questions for this section
            if conversation.questions_asked_in_section < len(selected_questions) and conversation.questions_asked_in_section < config.MAX_QUESTIONS_PER_SECTION:
                next_question = selected_questions[conversation.questions_asked_in_section]
                conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=next_question))
                return next_question, None, conversation.current_state
            else:
                # Move to the next section
                conversation.current_section += 1
                conversation.questions_asked_in_section = 0
                
                # Check if we have more sections
                if conversation.current_section <= len(self.sections):
                    next_section = self.sections[conversation.current_section - 1]
                    selected_questions = select_questions_for_section(
                        next_section.dict(), 
                        conversation.context_answers,
                        config.MAX_QUESTIONS_PER_SECTION
                    )
                    
                    if selected_questions:
                        transition_message = f"""
Vielen Dank für Ihre Antworten zu Sektion {section.id}: {section.title}.

Fahren wir fort mit Sektion {next_section.id}: {next_section.title} - {next_section.description}
"""
                        conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=transition_message))
                        
                        next_question = selected_questions[0]
                        conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=next_question))
                        return transition_message + "\n\n" + next_question, None, conversation.current_state
                
                # If no more sections or questions, finalize
                return self._finalize_conversation(conversation)
        
        # Something went wrong
        return "Es tut mir leid, aber ich konnte Ihre Antwort nicht verarbeiten.", None, "error"

    def _finalize_conversation(self, conversation: Conversation) -> Tuple[str, Optional[str], str]:
        """Finalize the conversation and trigger script generation."""
        final_message = """
Vielen Dank für all Ihre Antworten! Ich habe genügend Informationen gesammelt, um ein maßgeschneidertes Schulungsskript für Informationssicherheit zu erstellen.

Ihr Skript wird jetzt generiert. Dies kann einen Moment dauern...
"""
        conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=final_message))
        conversation.current_state = "generating_script"
        
        return final_message, None, "generating_script"

    def set_generated_script(self, conversation_id: str, script: str) -> None:
        """Set the generated script for a conversation."""
        conversation = self.get_conversation(conversation_id)
        if conversation:
            completion_message = """
Ihr Schulungsskript wurde erfolgreich erstellt! Sie können es nun für Ihre Informationssicherheits-Schulungen verwenden.

Falls Sie Änderungen wünschen oder weitere Fragen haben, stehe ich Ihnen gerne zur Verfügung.
"""
            conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=completion_message))
            conversation.current_state = "finished"