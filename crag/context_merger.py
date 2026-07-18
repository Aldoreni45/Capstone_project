from typing import List, Dict, Any, Optional
from pydantic_models.responses import RetrievedChunk
from crag.web_search import WebSearchResult
from custom_logging.logger import app_logger
from langsmith import traceable

class ContextMerger:
    """
    Merges retrieved document context with web search context.
    
    Strategies:
    - Prioritize document chunks (higher reliability)
    - Supplement with web search when documents are insufficient
    - Deduplicate overlapping information
    - Maintain source attribution
    """
    
    def __init__(self, max_web_context_length: int = 1000):
        self.max_web_context_length = max_web_context_length
    
    @traceable(name="Context Merging")
    def merge_contexts(
        self,
        retrieved_chunks: List[RetrievedChunk],
        web_results: List[WebSearchResult] = None,
        strategy: str = "document_priority"
    ) -> str:
        """
        Merges document and web search contexts.
        
        Args:
            retrieved_chunks: Retrieved document chunks
            web_results: Web search results (optional)
            strategy: Merging strategy ('document_priority', 'web_supplement', 'balanced')
            
        Returns:
            str: Merged context string
        """
        if not retrieved_chunks and not web_results:
            return ""
        
        if strategy == "document_priority":
            return self._document_priority_merge(retrieved_chunks, web_results)
        elif strategy == "web_supplement":
            return self._web_supplement_merge(retrieved_chunks, web_results)
        elif strategy == "balanced":
            return self._balanced_merge(retrieved_chunks, web_results)
        else:
            return self._document_priority_merge(retrieved_chunks, web_results)
    
    def _document_priority_merge(
        self,
        retrieved_chunks: List[RetrievedChunk],
        web_results: List[WebSearchResult] = None
    ) -> str:
        """
        Prioritizes document chunks, adds web context only if documents are limited.
        
        Strategy:
        - Use all document chunks first
        - Add web results only if < 3 document chunks
        - Limit web context to avoid overwhelming document context
        """
        context_parts = []
        
        # Add document context
        if retrieved_chunks:
            context_parts.append("=== Document Context ===")
            for idx, chunk in enumerate(retrieved_chunks, 1):
                context_parts.append(
                    f"[Document {idx}] {chunk.title} (Page {chunk.page}):\n{chunk.content}"
                )
        
        # Add web context only if documents are limited
        if web_results and len(retrieved_chunks) < 3:
            context_parts.append("\n=== Additional Web Context ===")
            web_context = self._format_web_context(web_results)
            context_parts.append(web_context)
        
        merged = "\n\n".join(context_parts)
        app_logger.info(f"Document priority merge: {len(retrieved_chunks)} docs + {len(web_results) if web_results else 0} web results")
        return merged
    
    def _web_supplement_merge(
        self,
        retrieved_chunks: List[RetrievedChunk],
        web_results: List[WebSearchResult] = None
    ) -> str:
        """
        Uses web search to supplement document context.
        
        Strategy:
        - Always include web context
        - Useful when documents are partial or outdated
        """
        context_parts = []
        
        # Add document context
        if retrieved_chunks:
            context_parts.append("=== Document Context ===")
            for idx, chunk in enumerate(retrieved_chunks, 1):
                context_parts.append(
                    f"[Document {idx}] {chunk.title} (Page {chunk.page}):\n{chunk.content}"
                )
        
        # Add web context
        if web_results:
            context_parts.append("\n=== Supplementary Web Context ===")
            web_context = self._format_web_context(web_results)
            context_parts.append(web_context)
        
        merged = "\n\n".join(context_parts)
        app_logger.info(f"Web supplement merge: {len(retrieved_chunks)} docs + {len(web_results) if web_results else 0} web results")
        return merged
    
    def _balanced_merge(
        self,
        retrieved_chunks: List[RetrievedChunk],
        web_results: List[WebSearchResult] = None
    ) -> str:
        """
        Balanced approach with equal weight to both sources.
        
        Strategy:
        - Interleave document and web context
        - Useful when both sources are equally important
        """
        context_parts = []
        
        # Add document context
        if retrieved_chunks:
            context_parts.append("=== Document Context ===")
            for idx, chunk in enumerate(retrieved_chunks, 1):
                context_parts.append(
                    f"[Document {idx}] {chunk.title} (Page {chunk.page}):\n{chunk.content}"
                )
        
        # Add web context
        if web_results:
            context_parts.append("\n=== Web Context ===")
            web_context = self._format_web_context(web_results)
            context_parts.append(web_context)
        
        merged = "\n\n".join(context_parts)
        app_logger.info(f"Balanced merge: {len(retrieved_chunks)} docs + {len(web_results) if web_results else 0} web results")
        return merged
    
    def _format_web_context(self, web_results: List[WebSearchResult]) -> str:
        """
        Formats web search results into context string.
        
        Args:
            web_results: List of web search results
            
        Returns:
            str: Formatted web context
        """
        if not web_results:
            return ""
        
        context_parts = []
        total_length = 0
        
        for idx, result in enumerate(web_results, 1):
            # Truncate snippet if needed to stay within limits
            snippet = result.snippet
            if total_length + len(snippet) > self.max_web_context_length:
                remaining = self.max_web_context_length - total_length
                if remaining > 50:  # Only add if we have space for meaningful content
                    snippet = snippet[:remaining] + "..."
                    context_parts.append(
                        f"[Web Source {idx}] {result.title}:\n{snippet}"
                    )
                break
            
            context_parts.append(
                f"[Web Source {idx}] {result.title}:\n{snippet}"
            )
            total_length += len(snippet)
        
        return "\n\n".join(context_parts)
    
    def deduplicate_context(
        self,
        context: str,
        similarity_threshold: float = 0.8
    ) -> str:
        """
        Removes duplicate or highly similar content from merged context.
        
        Args:
            context: Merged context string
            similarity_threshold: Threshold for considering content as duplicate
            
        Returns:
            str: Deduplicated context
        """
        # Simple deduplication by removing exact duplicate lines
        lines = context.split('\n')
        seen_lines = set()
        deduplicated_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            if stripped_line and stripped_line not in seen_lines:
                seen_lines.add(stripped_line)
                deduplicated_lines.append(line)
            elif not stripped_line:
                deduplicated_lines.append(line)  # Keep empty lines for structure
        
        deduplicated = '\n'.join(deduplicated_lines)
        
        removed = len(lines) - len(deduplicated_lines)
        if removed > 0:
            app_logger.info(f"Removed {removed} duplicate lines from context")
        
        return deduplicated
