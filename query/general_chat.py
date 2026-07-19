from typing import Dict, Any
from llm.groq_client import GroqLLMClient
from config.settings import settings
from custom_logging.logger import app_logger

class GeneralChatHandler:
    """Handles general chat interactions (greetings, small talk, identity questions)."""
    
    def __init__(self):
        self.groq_client = GroqLLMClient(settings.groq_api_key)
        self.greeting_responses = {
            "hi": "Hello! I am your AI Research Assistant. I can help you understand research papers and answer general AI-related questions.",
            "hello": "Hello! I am your AI Research Assistant. I can help you understand research papers and answer general AI-related questions.",
            "hey": "Hey! I am your AI Research Assistant. How can I help you today?",
            "hey bro": "Hey! I am your AI Research Assistant. I can help you with research papers and general AI questions.",
            "good morning": "Good morning! I am your AI Research Assistant. How can I help you today?",
            "good afternoon": "Good afternoon! I am your AI Research Assistant. How can I help you today?",
            "good evening": "Good evening! I am your AI Research Assistant. How can I help you today?",
        }
        
        self.identity_responses = {
            "who are you": "I am an AI Research Assistant capable of answering both general questions and questions related to the uploaded research papers.",
            "what are you": "I am an AI Research Assistant designed to help you understand research papers and answer general AI/ML questions.",
            "your name": "I am an AI Research Assistant. You can ask me about research papers or general AI/ML topics.",
        }
    
    def handle_greeting(self, query: str) -> str:
        """Handles greeting queries with predefined responses."""
        query_lower = query.lower().strip()
        
        for greeting, response in self.greeting_responses.items():
            if greeting in query_lower:
                return response
        
        # Default greeting response
        return "Hello! I am your AI Research Assistant. I can help you understand research papers and answer general AI-related questions."
    
    def handle_identity(self, query: str) -> str:
        """Handles identity questions (who are you, what are you)."""
        query_lower = query.lower().strip()
        
        for identity, response in self.identity_responses.items():
            if identity in query_lower:
                return response
        
        # Default identity response
        return "I am an AI Research Assistant capable of answering both general questions and questions related to the uploaded research papers."
    
    def handle_small_talk(self, query: str) -> str:
        """Handles small talk and casual conversation."""
        query_lower = query.lower().strip()
        
        if "thank" in query_lower:
            return "You're welcome! Feel free to ask me more questions about research papers or AI/ML topics."
        elif "bye" in query_lower or "goodbye" in query_lower:
            return "Goodbye! Feel free to come back anytime you have questions about research papers or AI."
        elif any(word in query_lower for word in ["ok", "okay", "sure", "alright"]):
            return "Great! Let me know if you have any questions about research papers or AI/ML topics."
        else:
            return "I'm here to help with research papers and AI/ML questions. What would you like to know?"
    
    def handle_general_chat(self, query: str) -> str:
        """
        Routes general chat queries to appropriate handlers.
        
        Args:
            query: The user's query
            
        Returns:
            str: The response
        """
        query_lower = query.lower().strip()
        
        # Check for greetings - use word boundaries to avoid false matches like "hive" containing "hi"
        import re
        greeting_patterns = [
            r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening)',
            r'^(hi|hello|hey) there',
            r'^(hi|hello|hey) bro',
            r'^how are you',
            r'^what\'s up',
        ]
        
        if self._matches_patterns(query_lower, greeting_patterns):
            return self.handle_greeting(query)
        
        # Check for identity questions - use word boundaries
        identity_patterns = [
            r'^(who are you|what are you)',
            r'^(your name|what\'s your name)',
        ]
        
        if self._matches_patterns(query_lower, identity_patterns):
            return self.handle_identity(query)
        
        # Handle other small talk
        return self.handle_small_talk(query)
    
    def _matches_patterns(self, text: str, patterns: list) -> bool:
        """Check if text matches any of the regex patterns."""
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False
