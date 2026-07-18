from typing import List, Dict, Any, Tuple
from llm.groq_client import GroqLLMClient
from config.settings import settings
from custom_logging.logger import app_logger
from pydantic_models.responses import RetrievedChunk

class HybridQueryHandler:
    """Handles hybrid questions that combine general knowledge with document context."""
    
    def __init__(self):
        self.groq_client = GroqLLMClient(settings.groq_api_key)
    
    def handle_hybrid_query(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        chat_history: list = None
    ) -> str:
        """
        Handles hybrid queries by combining general knowledge with document context.
        
        Args:
            query: The user's hybrid query
            retrieved_chunks: Retrieved document chunks
            chat_history: Optional conversation history
            
        Returns:
            str: The hybrid response combining general knowledge and document context
        """
        # First, get general knowledge about the topic
        general_knowledge = self._get_general_knowledge(query, chat_history)
        
        # Then, extract document context
        document_context = self._extract_document_context(retrieved_chunks)
        
        # Combine both in a structured response
        hybrid_response = self._combine_contexts(query, general_knowledge, document_context)
        
        return hybrid_response
    
    def _get_general_knowledge(self, query: str, chat_history: list = None) -> str:
        """Gets general knowledge about the query topic."""
        system_prompt = """You are an AI Research Assistant. Provide a clear, concise explanation of the concept mentioned in the user's question.

Rules:
1. Focus on the general concept (e.g., Attention Mechanism, RAG, Transformers).
2. Provide a clear definition and explanation.
3. Keep it concise (2-3 sentences).
4. Don't mention documents or papers."""
        
        user_prompt = f"Question: {query}"
        
        if chat_history and isinstance(chat_history, str):
            # Take last few lines from the string history
            history_lines = chat_history.split('\n')
            recent_history = history_lines[-4:] if len(history_lines) > 4 else history_lines
            history_context = "\n".join(recent_history)
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
                max_tokens=300
            )
            return response
        except Exception as e:
            app_logger.error(f"General knowledge generation failed: {str(e)}")
            return ""
    
    def _extract_document_context(self, retrieved_chunks: List[RetrievedChunk]) -> str:
        """Extracts relevant context from retrieved document chunks."""
        if not retrieved_chunks:
            return "No specific information found in the uploaded documents."
        
        context_parts = []
        for chunk in retrieved_chunks[:3]:  # Use top 3 chunks
            context_parts.append(
                f"From '{chunk.title}' (Page {chunk.page}):\n{chunk.content[:200]}..."
            )
        
        return "\n\n".join(context_parts)
    
    def _combine_contexts(self, query: str, general_knowledge: str, document_context: str) -> str:
        """Combines general knowledge with document context into a coherent response."""
        
        if not general_knowledge and not document_context:
            return "I don't have enough information to answer this question."
        
        if not general_knowledge:
            return f"Based on the uploaded documents:\n\n{document_context}"
        
        if not document_context or "No specific information" in document_context:
            return f"General explanation:\n\n{general_knowledge}\n\nNote: No specific information was found in the uploaded documents about this topic."
        
        # Combine both contexts
        combined_response = f"""General explanation:
{general_knowledge}

Additional context from uploaded documents:
{document_context}"""
        
        return combined_response
