from typing import Dict, Any
from llm.groq_client import GroqLLMClient
from config.settings import settings
from custom_logging.logger import app_logger

class GeneralKnowledgeHandler:
    """Handles general knowledge questions about AI, ML, Python, etc."""
    
    def __init__(self):
        self.groq_client = GroqLLMClient(settings.groq_api_key)
    
    def handle_general_knowledge(self, query: str, chat_history: list = None) -> str:
        """
        Handles general knowledge questions using Groq LLM without RAG.
        
        Args:
            query: The user's query
            chat_history: Optional conversation history for context (string format)
            
        Returns:
            str: The LLM response
        """
        system_prompt = """You are an AI Research Assistant. Answer the user's question about AI, Machine Learning, Python, Data Science, or related topics.

Rules:
1. Provide accurate, comprehensive information.
2. Give detailed explanations with examples.
3. Include relevant context and background information.
4. If you're unsure about something, acknowledge it.
5. Provide thorough, well-structured responses."""
        
        # Format chat history if provided (chat_history is a string from BufferMemory.get_history())
        history_context = ""
        if chat_history and isinstance(chat_history, str):
            # Take last few lines from the string history
            history_lines = chat_history.split('\n')
            recent_history = history_lines[-6:] if len(history_lines) > 6 else history_lines
            history_context = "\n".join(recent_history)
        
        user_prompt = f"""Question: {query}"""
        
        if history_context:
            user_prompt = f"""Conversation History:
{history_context}

Question: {query}"""
        
        try:
            response = self.groq_client.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response
        except Exception as e:
            app_logger.error(f"General knowledge generation failed: {str(e)}")
            return "I apologize, but I encountered an error while answering your question. Please try again."
