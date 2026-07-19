from typing import Literal
from config.settings import settings
from custom_logging.logger import app_logger
import re

QueryType = Literal[
    "greeting",
    "small_talk",
    "rag_query"
]

class QueryClassifier:
    """Classifies user queries to determine the appropriate response mode.
    
    Conservative approach: Only classify as greeting/small_talk if clearly matched.
    Everything else goes through RAG pipeline first, with fallback to general knowledge.
    """
    
    def __init__(self):
        self.greeting_patterns = [
            r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening)',
            r'^(hi|hello|hey) there',
            r'^(hi|hello|hey) bro',
            r'^how are you',
            r'^what\'s up',
        ]
        
        self.small_talk_patterns = [
            r'^(who are you|what are you)',
            r'^(your name|what\'s your name)',
            r'^(thank you|thanks)',
            r'^(bye|goodbye|see you)',
            r'^(ok|okay|sure|alright)',
        ]
    
    def classify(self, query: str, has_context: bool = False) -> QueryType:
        """
        Classifies the query type based on content and context availability.
        
        Args:
            query: The user's query string
            has_context: Whether documents are available in the corpus
            
        Returns:
            QueryType: The classified query type (greeting, small_talk, or rag_query)
        """
        query_lower = query.lower().strip()
        
        # Check for greetings
        if self._matches_patterns(query_lower, self.greeting_patterns):
            return "greeting"
        
        # Check for small talk
        if self._matches_patterns(query_lower, self.small_talk_patterns):
            return "small_talk"
        
        # Everything else goes through RAG pipeline
        return "rag_query"
    
    def _matches_patterns(self, text: str, patterns: list) -> bool:
        """Check if text matches any of the regex patterns."""
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _contains_greeting_words(self, text: str) -> bool:
        """Check if text contains greeting words (more lenient check)."""
        greeting_words = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        for word in greeting_words:
            if word in text:
                return True
        return False
    
    def get_response_mode(self, query: str, has_context: bool = False) -> str:
        """
        Determines the response mode based on query classification.
        
        Returns:
            str: 'general_chat' or 'rag'
        """
        query_type = self.classify(query, has_context)
        
        if query_type in ["greeting", "small_talk"]:
            return "general_chat"
        else:
            return "rag"  # Everything else goes through RAG pipeline
