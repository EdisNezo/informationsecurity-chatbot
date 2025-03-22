"""Dialog management for the RAG chatbot."""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from models.schema import Conversation, Message, MessageRole, Template, Section
from utils import generate_id, load_json
import config

# Ensure the logger is properly configured for console output
logger = logging.getLogger(__name__)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG level to capture all logs


class DialogManager:
    """Manage the conversation flow."""

    def __init__(self):
        """Initialize the dialog manager."""
        logger.info("Initializing DialogManager")
        self.conversations = {}
        self.template = self.load_template()
        logger.info(f"Template loaded: {self.template is not None}")
        self.sections = self.template.sections if self.template else []
        logger.info(f"Loaded {len(self.sections)} sections from template")
        # Store adapted questions separately to avoid Pydantic validation errors
        self._adapted_questions = {}
        self._generated_scripts = {}
        logger.debug("DialogManager initialization complete")

    def load_template(self) -> Optional[Template]:
        """Load the script template."""
        template_path = config.SOURCE_DOCS_DIR / "blank_template.json"
        logger.info(f"Attempting to load template from: {template_path}")
        
        if not template_path.exists():
            logger.error(f"Template file not found: {template_path}")
            return None
        
        template_data = load_json(template_path)
        if not template_data:
            logger.error("Failed to load template data - empty or invalid JSON")
            return None
        
        logger.info(f"Template loaded successfully with version: {template_data.get('version', 'unknown')}")
        return Template(**template_data)

    def create_conversation(self) -> Conversation:
        """Create a new conversation."""
        conversation_id = generate_id()
        logger.info(f"Creating new conversation with ID: {conversation_id}")
        conversation = Conversation(id=conversation_id)
        
        # Add greeting message
        greeting = """
        Willkommen! Ich bin Ihr Assistent für die Erstellung von Informationssicherheits-Schulungsskripten im Gesundheitswesen. 
        Ich werde Ihnen einige Fragen stellen, um ein maßgeschneidertes Skript für Ihre Einrichtung zu erstellen.

        Zunächst benötige ich einige Informationen zum Kontext Ihrer Einrichtung.
        """
        logger.debug(f"Adding greeting message to conversation {conversation_id}")
        conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=greeting))
        
        # Add first context question
        if config.CONTEXT_QUESTIONS:
            logger.debug(f"Setting up first context question for conversation {conversation_id}")
            conversation.current_state = "context_questions"
            conversation.current_context_question = 0
            first_question = config.CONTEXT_QUESTIONS[0]
            logger.debug(f"First question: {first_question}")
            conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=first_question))
        else:
            logger.warning("No context questions defined in configuration")
        
        self.conversations[conversation_id] = conversation
        logger.info(f"New conversation {conversation_id} created and initialized")
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        logger.debug(f"Retrieving conversation with ID: {conversation_id}")
        conversation = self.conversations.get(conversation_id)
        if conversation:
            logger.debug(f"Found conversation {conversation_id} in state: {conversation.current_state}")
        else:
            logger.warning(f"Conversation with ID {conversation_id} not found")
        return conversation

    def process_message(self, conversation_id: str, message: str) -> Tuple[str, Optional[str], str]:
        """Process a user message and determine the next step in the conversation."""
        logger.info(f"Processing message for conversation {conversation_id}")
        logger.debug(f"User message: {message[:50]}..." if len(message) > 50 else f"User message: {message}")
        
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            logger.info(f"Conversation {conversation_id} not found, creating new conversation")
            conversation = self.create_conversation()
            return conversation.messages[-1].content, None, conversation.current_state
        
        # Add user message to conversation
        logger.debug(f"Adding user message to conversation {conversation_id}")
        conversation.messages.append(Message(role=MessageRole.USER, content=message))
        
        # Process based on current state
        logger.info(f"Processing message in state: {conversation.current_state}")
        if conversation.current_state == "context_questions":
            logger.debug(f"Handling context question {conversation.current_context_question}")
            response_message, script, state = self._handle_context_question(conversation, message)
        elif conversation.current_state == "section_questions":
            logger.debug(f"Handling section question for section {conversation.current_section}")
            response_message, script, state = self._handle_section_question(conversation, message)
        elif conversation.current_state == "finished":
            logger.debug("Conversation already finished")
            response_message = "Ihr Skript wurde bereits erstellt. Möchten Sie ein neues Skript erstellen?"
            script, state = None, "finished"
        elif conversation.current_state == "error":
            logger.debug("Conversation is in error state")
            response_message = "Es ist ein Fehler aufgetreten. Möchten Sie von vorne beginnen?"
            script, state = None, "error"
        else:
            logger.error(f"Unknown conversation state: {conversation.current_state}")
            response_message = "Es tut mir leid, aber ich verstehe Ihren aktuellen Status nicht."
            script, state = None, "error"
        
        # No cleaning needed for Gemma responses
        logger.debug("Skipping response cleaning since Gemma doesn't use reasoning tags")
        
        # Add the message to the conversation history
        logger.debug("Adding assistant response to conversation history")
        conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=response_message))
        
        logger.info(f"Completed message processing. New state: {state}")
        return response_message, script, state
    
    def _adapt_question_for_user(self, question: str, context_answers: Dict[str, str], section_title: str) -> str:
        """Adapt a question to be more understandable based on context answers."""
        logger.info(f"Adapting question for section '{section_title}'")
        logger.debug(f"Original question: {question}")
        
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
        
        logger.debug(f"Context parameters - Organization: {organization_type}, Audience: {audience}, Knowledge level: {knowledge_level}")
        
        # Create a context-aware prompt for the RAG engine
        context_info = "\n".join([f"{k}: {v}" for k, v in context_answers.items() if v])
        logger.debug(f"Created context info with {len(context_answers)} answers")
        
        # Use the RAG engine to adapt the question
        logger.debug("Initializing RAG engine for question adaptation")
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
            logger.debug("Retrieving context for question adaptation")
            # Get retrieval results
            retrieval_result = rag_engine.retrieve(prompt, context=context_answers)
            logger.debug(f"Retrieved {len(retrieval_result.chunks)} relevant chunks")
            
            logger.debug("Generating adapted question")
            # Generate adapted question
            adapted_question = rag_engine.generate(prompt, retrieval_result, context=context_answers)
            logger.debug(f"Raw adapted question: {adapted_question[:100]}..." if adapted_question and len(adapted_question) > 100 else f"Raw adapted question: {adapted_question}")
            
            return adapted_question
                
                    
        except Exception as e:
            logger.error(f"Error adapting question: {e}", exc_info=True)
        
        # Fallback: Use the original question with minimal adaptations
        logger.warning("Using fallback for question adaptation")
        adapted = question.replace("Bedrohung", f"Bedrohung in {organization_type}").replace(
            "Maßnahmen", f"Maßnahmen für {audience}")
        logger.info(f"Using minimally adapted question: {adapted}")
        return adapted

    def _handle_context_question(self, conversation: Conversation, message: str) -> Tuple[str, Optional[str], str]:
        """Handle responses to context questions."""
        logger.info(f"Handling context question {conversation.current_context_question} for conversation {conversation.id}")
        
        # Save the answer to the current context question
        current_question = config.CONTEXT_QUESTIONS[conversation.current_context_question]
        logger.debug(f"Current question: {current_question}")
        logger.debug(f"Saving answer: {message[:50]}..." if len(message) > 50 else message)
        conversation.context_answers[current_question] = message
        
        # Move to the next context question or to section questions
        conversation.current_context_question += 1
        logger.debug(f"Incremented context question counter to {conversation.current_context_question}")
        
        if conversation.current_context_question < len(config.CONTEXT_QUESTIONS):
            logger.info(f"Moving to next context question {conversation.current_context_question}")
            next_question = config.CONTEXT_QUESTIONS[conversation.current_context_question]
            logger.debug(f"Next question: {next_question}")
            conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=next_question))
            return next_question, None, conversation.current_state
        else:
            # Transition to section questions
            logger.info("All context questions completed, transitioning to section questions")
            conversation.current_state = "section_questions"
            conversation.current_section = 1
            conversation.questions_asked_in_section = 0
            logger.debug(f"Set initial section to 1 and questions_asked_in_section to 0")
            
            if self.sections:
                section = self.sections[0]  # First section
                logger.debug(f"First section: {section.title} (ID: {section.id})")
                
                # Get the base questions
                raw_questions = section.dict()["questions"][:config.MAX_QUESTIONS_PER_SECTION]
                logger.debug(f"Got {len(raw_questions)} raw questions for first section")
                
                # Initialize adapted questions storage
                if conversation.id not in self._adapted_questions:
                    logger.debug(f"Initializing adapted questions storage for conversation {conversation.id}")
                    self._adapted_questions[conversation.id] = {}
                
                # Adapt each question based on context
                logger.info("Adapting questions for first section")
                selected_questions = []
                for idx, question in enumerate(raw_questions):
                    logger.debug(f"Adapting question {idx+1}: {question[:50]}..." if len(question) > 50 else question)
                    adapted_question = self._adapt_question_for_user(
                        question, 
                        conversation.context_answers, 
                        section.title
                    )
                    selected_questions.append(adapted_question)
                
                # Store for later use
                logger.debug(f"Storing {len(selected_questions)} adapted questions for section 1")
                self._adapted_questions[conversation.id][1] = selected_questions
                
                if selected_questions:
                    logger.info("Creating transition to first section question")
                    transition_message = f"""
Vielen Dank für diese Informationen! Nun werde ich Ihnen einige Fragen zu den verschiedenen Sektionen des Schulungsskripts stellen.
"""
                    conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=transition_message))
                    
                    next_question = selected_questions[0]
                    logger.debug(f"First section question: {next_question}")
                    conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=next_question))
                    return transition_message + "\n\n" + next_question, None, conversation.current_state
                else:
                    logger.warning("No adapted questions available for first section")
            else:
                logger.warning("No sections available in template")
            
            # If no sections or questions, go to script generation
            logger.info("No sections or questions available, finalizing conversation")
            return self._finalize_conversation(conversation)

    def _handle_section_question(self, conversation: Conversation, message: str) -> Tuple[str, Optional[str], str]:
        """Handle responses to section questions."""
        logger.info(f"Handling section question for section {conversation.current_section}, question {conversation.questions_asked_in_section}")
        
        # Make sure we have sections
        if not self.sections or conversation.current_section > len(self.sections):
            logger.warning(f"No more sections available (current: {conversation.current_section}, total: {len(self.sections) if self.sections else 0})")
            return self._finalize_conversation(conversation)
        
        # Get current section
        section_idx = conversation.current_section - 1
        section = self.sections[section_idx]
        logger.debug(f"Current section: {section.title} (ID: {section.id})")
        
        # Get base questions for the current section
        raw_questions = section.dict()["questions"][:config.MAX_QUESTIONS_PER_SECTION]
        logger.debug(f"Got {len(raw_questions)} raw questions for current section")
        
        # Initialize the adapted questions storage for this conversation if needed
        if conversation.id not in self._adapted_questions:
            logger.debug(f"Initializing adapted questions storage for conversation {conversation.id}")
            self._adapted_questions[conversation.id] = {}
        
        # Get or create adapted questions for this section
        if conversation.current_section not in self._adapted_questions[conversation.id]:
            logger.info(f"Adapting questions for section {conversation.current_section}")
            selected_questions = []
            for idx, question in enumerate(raw_questions):
                logger.debug(f"Adapting question {idx+1}: {question[:50]}..." if len(question) > 50 else question)
                adapted_question = self._adapt_question_for_user(
                    question, 
                    conversation.context_answers, 
                    section.title
                )
                selected_questions.append(adapted_question)
            
            # Store the adapted questions
            logger.debug(f"Storing {len(selected_questions)} adapted questions for section {conversation.current_section}")
            self._adapted_questions[conversation.id][conversation.current_section] = selected_questions
        
        # Get the adapted questions for this section
        selected_questions = self._adapted_questions[conversation.id][conversation.current_section]
        logger.debug(f"Retrieved {len(selected_questions)} adapted questions for section {conversation.current_section}")
        
        # Save the answer
        current_question_idx = conversation.questions_asked_in_section
        logger.debug(f"Current question index: {current_question_idx}")
        if current_question_idx < len(selected_questions):
            current_question = selected_questions[current_question_idx]
            original_question = raw_questions[current_question_idx]
            logger.debug(f"Saving answer for question: {current_question[:50]}..." if len(current_question) > 50 else current_question)
            
            # Initialize the section answers dict if needed
            if conversation.current_section not in conversation.section_answers:
                logger.debug(f"Initializing section answers for section {conversation.current_section}")
                conversation.section_answers[conversation.current_section] = {}
            
            # Save the answer with the original question as the key
            logger.debug(f"Saving answer: {message[:50]}..." if len(message) > 50 else message)
            conversation.section_answers[conversation.current_section][original_question] = message
            
            # Increment the question counter
            conversation.questions_asked_in_section += 1
            logger.debug(f"Incremented questions_asked_in_section to {conversation.questions_asked_in_section}")
            
            # Check if we have more questions for this section
            if conversation.questions_asked_in_section < len(selected_questions) and conversation.questions_asked_in_section < config.MAX_QUESTIONS_PER_SECTION:
                logger.info(f"Moving to next question in section {conversation.current_section}")
                next_question = selected_questions[conversation.questions_asked_in_section]
                logger.debug(f"Next question: {next_question[:50]}..." if len(next_question) > 50 else next_question)
                conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=next_question))
                return next_question, None, conversation.current_state
            else:
                # Move to the next section
                logger.info(f"Completed all questions for section {conversation.current_section}, moving to next section")
                conversation.current_section += 1
                conversation.questions_asked_in_section = 0
                logger.debug(f"Set current_section to {conversation.current_section} and questions_asked_in_section to 0")
                
                # Check if we have more sections
                if conversation.current_section <= len(self.sections):
                    next_section = self.sections[conversation.current_section - 1]
                    logger.debug(f"Next section: {next_section.title} (ID: {next_section.id})")
                    
                    # Get base questions for the next section
                    next_raw_questions = next_section.dict()["questions"][:config.MAX_QUESTIONS_PER_SECTION]
                    logger.debug(f"Got {len(next_raw_questions)} raw questions for next section")
                    
                    # Adapt questions for the next section
                    logger.info(f"Adapting questions for section {conversation.current_section}")
                    next_adapted_questions = []
                    for idx, question in enumerate(next_raw_questions):
                        logger.debug(f"Adapting question {idx+1}: {question[:50]}..." if len(question) > 50 else question)
                        adapted_question = self._adapt_question_for_user(
                            question, 
                            conversation.context_answers, 
                            next_section.title
                        )
                        next_adapted_questions.append(adapted_question)
                    
                    # Store the adapted questions
                    logger.debug(f"Storing {len(next_adapted_questions)} adapted questions for section {conversation.current_section}")
                    self._adapted_questions[conversation.id][conversation.current_section] = next_adapted_questions
                    
                    if next_adapted_questions:
                        logger.info(f"Creating transition to section {conversation.current_section}")
                        transition_message = f""""""
                        conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=transition_message))
                        
                        next_question = next_adapted_questions[0]
                        logger.debug(f"First question for next section: {next_question[:50]}..." if len(next_question) > 50 else next_question)
                        conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=next_question))
                        return transition_message + "\n\n" + next_question, None, conversation.current_state
                    else:
                        logger.warning(f"No adapted questions available for section {conversation.current_section}")
                else:
                    logger.info("No more sections available")
                
                # If no more sections or questions, finalize
                logger.info("No more sections or questions, finalizing conversation")
                return self._finalize_conversation(conversation)
        else:
            logger.error(f"Current question index {current_question_idx} is out of bounds for selected questions (length: {len(selected_questions)})")
        
        # Something went wrong
        logger.error("Error processing section question")
        return "Es tut mir leid, aber ich konnte Ihre Antwort nicht verarbeiten.", None, "error"

    def _finalize_conversation(self, conversation: Conversation) -> Tuple[str, Optional[str], str]:
        """Finalize the conversation and trigger script generation."""
        logger.info(f"Finalizing conversation {conversation.id} and triggering script generation")
        
        # Log summary of collected data
        logger.info(f"Context answers collected: {len(conversation.context_answers)}")
        logger.info(f"Section answers collected: {len(conversation.section_answers)} sections")
        for section_id, answers in conversation.section_answers.items():
            logger.debug(f"Section {section_id}: {len(answers)} answers")
        
        final_message = """
Vielen Dank für all Ihre Antworten! Ich habe genügend Informationen gesammelt, um ein maßgeschneidertes Schulungsskript für Informationssicherheit zu erstellen.

Ihr Skript wird jetzt generiert. Dies kann einen Moment dauern...
"""
        logger.debug("Adding finalization message to conversation")
        conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=final_message))
        conversation.current_state = "generating_script"
        logger.info(f"Conversation state changed to {conversation.current_state}")
        
        return final_message, None, "generating_script"

    def set_generated_script(self, conversation_id: str, script: str) -> None:
        """Set the generated script for a conversation."""
        logger.info(f"Setting generated script for conversation {conversation_id}")
        logger.debug(f"Script length: {len(script)} characters")
        
        conversation = self.get_conversation(conversation_id)
        if conversation:
            completion_message = """
Ihr Schulungsskript wurde erfolgreich erstellt! Sie können es nun für Ihre Informationssicherheits-Schulungen verwenden.

Falls Sie Änderungen wünschen oder weitere Fragen haben, stehe ich Ihnen gerne zur Verfügung.
"""
            logger.debug("Adding completion message to conversation")
            conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=completion_message))
            conversation.current_state = "finished"
            logger.info(f"Conversation state changed to {conversation.current_state}")
            
            # Store the script in our dictionary instead of on the conversation
            logger.debug("Storing script in _generated_scripts dictionary")
            self._generated_scripts[conversation_id] = script
        else:
            logger.error(f"Could not find conversation {conversation_id} to set generated script")

    def get_generated_script(self, conversation_id: str) -> Optional[str]:
        """Get the generated script for a conversation."""
        logger.debug(f"Retrieving generated script for conversation {conversation_id}")
        script = self._generated_scripts.get(conversation_id)
        if script:
            logger.debug(f"Found script with length: {len(script)} characters")
        else:
            logger.debug(f"No script found for conversation {conversation_id}")
        return script

    def set_error_state(self, conversation_id: str, error_message: str) -> None:
        """Set the conversation to an error state."""
        logger.info(f"Setting error state for conversation {conversation_id}")
        logger.error(f"Error message: {error_message}")
        
        conversation = self.get_conversation(conversation_id)
        if conversation:
            error_notice = f"""
Es ist ein Fehler bei der Erstellung Ihres Skripts aufgetreten: {error_message}

Bitte versuchen Sie es erneut oder kontaktieren Sie den Support.
"""
            logger.debug("Adding error notice to conversation")
            conversation.messages.append(Message(role=MessageRole.ASSISTANT, content=error_notice))
            conversation.current_state = "error"
            logger.info(f"Conversation state changed to {conversation.current_state}")
        else:
            logger.error(f"Could not find conversation {conversation_id} to set error state")
            
    def _clean_llm_response(self, text: str) -> str:
        """Clean LLM response from reasoning artifacts and other unwanted content."""
        if not text:
            logger.debug("Empty text provided to _clean_llm_response")
            return text
            
        logger.debug(f"Cleaning LLM response with length {len(text)}")
        # Remove thinking tags and content
        import re
        
        # List of patterns to clean
        patterns = [
            # Thinking tags
            r'<think>.*?</think>',
            r'<think>.*?$',
            r'<think.*?>.*?(?:</think>|$)',
            r'<think>.*',
            
            # Other reasoning formats
            r'\[thinking:.*?\]',
            r'\[thought:.*?\]',
            r'\(thinking:.*?\)',
            r'\(thinking.*?\)',
            
            # Instruction leakage
            r'<instruction>.*?</instruction>',
            r'<system>.*?</system>',
            
            # Planning patterns
            r'Step \d+:.*',  # Sometimes models outline steps before answering
            r'Let me plan my response:.*',
            r'I need to:.*',
            r'First, I will.*Then, I will',
            
            # Meta-commentary
            r'I should respond with.*',
            r'I need to formulate.*',
        ]
        
        original_length = len(text)
        
        # Apply all patterns
        for pattern in patterns:
            prev_length = len(text)
            text = re.sub(pattern, '', text, flags=re.DOTALL)
            if len(text) != prev_length:
                logger.debug(f"Pattern '{pattern}' removed {prev_length - len(text)} characters")
        
        # Clean up spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        cleaned_text = text.strip()
        logger.debug(f"Removed {original_length - len(cleaned_text)} characters during cleaning")
        
        return cleaned_text