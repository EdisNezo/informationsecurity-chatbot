"""Dialog management for the RAG chatbot."""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from models.schema import Conversation, Message, MessageRole, Template, Section
from utils import generate_id, load_json
import config

logger = logging.getLogger(__name__)


class DialogManager:
    """Manage the conversation flow."""

    def __init__(self):
        """Initialize the dialog manager."""
        self.conversations = {}
        self.template = self.load_template()
        self.sections = self.template.sections if self.template else []
        # Store adapted questions separately to avoid Pydantic validation errors
        self._adapted_questions = {}
        self._generated_scripts = {}

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
        elif conversation.current_state == "error":
            return "Es ist ein Fehler aufgetreten. Möchten Sie von vorne beginnen?", None, "error"
        else:
            return "Es tut mir leid, aber ich verstehe Ihren aktuellen Status nicht.", None, "error"

    def _adapt_question_for_user(self, question: str, context_answers: Dict[str, str], section_title: str) -> str:
        """Adapt a question to be more understandable based on context answers."""
        # Import here to avoid circular imports
        from rag_engine import RAGEngine
        
        organization_type = context_answers.get(
            "In welcher medizinischen Einrichtung sollen die Schulungen umgesetzt werden?", 
            "Ihre Einrichtung"
        )
        
        audience = context_answers.get(
            "Gibt es spezifische Personengruppen, die zu berücksichtigen sind?", 
            "Ihre Mitarbeiter"
        )
        
        knowledge_level = context_answers.get(
            "Welche Vorkenntnisse haben die Teilnehmer im Bereich Informationssicherheit?",
            "grundlegende"
        )
        
        # Create a context-aware prompt for the RAG engine
        context_info = "\n".join([f"{k}: {v}" for k, v in context_answers.items() if v])
        
        # Use the RAG engine to adapt the question
        rag_engine = RAGEngine()
        prompt = f"""
Bitte formuliere die folgende Frage zur Informationssicherheit um, damit sie für Mitarbeiter in {organization_type} 
ohne tiefgreifendes Sicherheitswissen verständlicher wird. Die Frage gehört zum Bereich "{section_title}".

Ursprüngliche Frage: {question}

Kontext zur Organisation und den Teilnehmern:
{context_info}

Die neue Frage sollte:
1. Alltagssprache verwenden statt technischen Fachjargon
2. Konkrete Beispiele aus dem medizinischen Umfeld einbeziehen
3. Direkt auf die Arbeitssituation der Teilnehmer bezogen sein
4. Weniger abstrakt und mehr praxisorientiert sein

Gib nur die umformulierte Frage zurück, ohne Erklärungen oder zusätzlichen Text.
"""
        
        try:
            # Get retrieval results
            retrieval_result = rag_engine.retrieve(prompt, context=context_answers)
            
            # Generate adapted question
            adapted_question = rag_engine.generate(prompt, retrieval_result, context=context_answers)
            
            # If generation succeeded and produced a reasonable result
            if adapted_question and len(adapted_question.strip()) > 10:
                # Clean up any extra text that might have been generated
                adapted_question = adapted_question.strip()
                # Take the first sentence if there are multiple
                if "." in adapted_question:
                    adapted_question = adapted_question.split(".")[0].strip() + "?"
                return adapted_question
                
        except Exception as e:
            logger.warning(f"Error adapting question: {e}")
        
        # Fallback: Use the original question with minimal adaptations
        return question.replace("Bedrohung", f"Bedrohung in {organization_type}").replace(
            "Maßnahmen", f"Maßnahmen für {audience}")

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
                
                # Get the base questions
                raw_questions = section.dict()["questions"][:config.MAX_QUESTIONS_PER_SECTION]
                
                # Initialize adapted questions storage
                if conversation.id not in self._adapted_questions:
                    self._adapted_questions[conversation.id] = {}
                
                # Adapt each question based on context
                selected_questions = []
                for question in raw_questions:
                    adapted_question = self._adapt_question_for_user(
                        question, 
                        conversation.context_answers, 
                        section.title
                    )
                    selected_questions.append(adapted_question)
                
                # Store for later use
                self._adapted_questions[conversation.id][1] = selected_questions
                
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
        
        # Get base questions for the current section
        raw_questions = section.dict()["questions"][:config.MAX_QUESTIONS_PER_SECTION]
        
        # Initialize the adapted questions storage for this conversation if needed
        if conversation.id not in self._adapted_questions:
            self._adapted_questions[conversation.id] = {}
        
        # Get or create adapted questions for this section
        if conversation.current_section not in self._adapted_questions[conversation.id]:
            selected_questions = []
            for question in raw_questions:
                adapted_question = self._adapt_question_for_user(
                    question, 
                    conversation.context_answers, 
                    section.title
                )
                selected_questions.append(adapted_question)
            
            # Store the adapted questions
            self._adapted_questions[conversation.id][conversation.current_section] = selected_questions
        
        # Get the adapted questions for this section
        selected_questions = self._adapted_questions[conversation.id][conversation.current_section]
        
        # Save the answer
        current_question_idx = conversation.questions_asked_in_section
        if current_question_idx < len(selected_questions):
            current_question = selected_questions[current_question_idx]
            original_question = raw_questions[current_question_idx]
            
            # Initialize the section answers dict if needed
            if conversation.current_section not in conversation.section_answers:
                conversation.section_answers[conversation.current_section] = {}
            
            # Save the answer with the original question as the key
            conversation.section_answers[conversation.current_section][original_question] = message
            
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
                    
                    # Get base questions for the next section
                    next_raw_questions = next_section.dict()["questions"][:config.MAX_QUESTIONS_PER_SECTION]
                    
                    # Adapt questions for the next section
                    next_adapted_questions = []
                    for question in next_raw_questions:
                        adapted_question = self._adapt_question_for_user(
                            question, 
                            conversation.context_answers, 
                            next_section.title
                        )
                        next_adapted_questions.append(adapted_question)
                    
                    # Store the adapted questions
                    self._adapted_questions[conversation.id][conversation.current_section] = next_adapted_questions
                    
                    if next_adapted_questions:
                        transition_message = f"""
Vielen Dank für Ihre Antworten zu Sektion {section.id}: {section.title}.

Fahren wir fort mit Sektion {next_section.id}: {next_section.title} - {next_section.description}
"""
                        conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=transition_message))
                        
                        next_question = next_adapted_questions[0]
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
            # Store the script in our dictionary instead of on the conversation
            self._generated_scripts[conversation_id] = script

    def get_generated_script(self, conversation_id: str) -> Optional[str]:
        """Get the generated script for a conversation."""
        return self._generated_scripts.get(conversation_id)

    def set_error_state(self, conversation_id: str, error_message: str) -> None:
        """Set the conversation to an error state."""
        conversation = self.get_conversation(conversation_id)
        if conversation:
            error_notice = f"""
Es ist ein Fehler bei der Erstellung Ihres Skripts aufgetreten: {error_message}

Bitte versuchen Sie es erneut oder kontaktieren Sie den Support.
"""
            conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=error_notice))
            conversation.current_state = "error"