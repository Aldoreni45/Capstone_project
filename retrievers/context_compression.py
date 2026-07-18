from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from pydantic_models.responses import RetrievedChunk
from retrievers.base import BasePaperRetriever
from custom_logging.logger import app_logger
from langsmith import traceable


class RuleBasedContextCompressor:
    """
    Rule-based context compression that filters and compresses retrieved chunks
    without using LLM calls, providing fast and efficient context optimization.
    """

    def __init__(
        self,
        min_relevance_score: float = 0.5,
        max_context_length: int = 3000,
        remove_duplicates: bool = True,
        dedup_threshold: float = 0.8,
        use_score_filtering: bool = False  # Disabled by default to avoid filtering good chunks
    ):
        self.min_relevance_score = min_relevance_score
        self.max_context_length = max_context_length
        self.remove_duplicates = remove_duplicates
        self.dedup_threshold = dedup_threshold
        self.use_score_filtering = use_score_filtering  # Only enable if using Cross Encoder scores

    @traceable(name="Context Compression")
    def compress_context(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        target_length: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """
        Compresses context by filtering, deduplicating, and length optimization.
        
        IMPORTANT: Score filtering is disabled by default to avoid incorrectly
        filtering good chunks when using RRF scores. Only enable score filtering
        when using Cross Encoder scores (0.0-1.0 range).
        
        Args:
            query: The original query
            chunks: Retrieved chunks to compress
            target_length: Target total length in characters (overrides max_context_length)
            
        Returns:
            Compressed list of chunks
        """
        target_length = target_length or self.max_context_length
        
        app_logger.info(
            f"Compressing {len(chunks)} chunks with target length {target_length}, "
            f"score_filtering={self.use_score_filtering}"
        )

        # Step 1: Filter by relevance score (only if enabled and using Cross Encoder scores)
        if self.use_score_filtering:
            filtered_chunks = self._filter_by_relevance(chunks)
            app_logger.info(f"Relevance filtering: {len(chunks)} -> {len(filtered_chunks)}")
        else:
            filtered_chunks = chunks
            app_logger.info("Score filtering disabled - using all chunks")

        # Step 2: Remove duplicates
        if self.remove_duplicates:
            deduped_chunks = self._remove_duplicates(filtered_chunks)
            app_logger.info(f"Deduplication: {len(filtered_chunks)} -> {len(deduped_chunks)}")
        else:
            deduped_chunks = filtered_chunks

        # Step 3: Optimize length by selecting best chunks
        length_optimized_chunks = self._optimize_length(
            deduped_chunks,
            target_length
        )
        app_logger.info(
            f"Length optimization: {len(deduped_chunks)} -> {len(length_optimized_chunks)} chunks"
        )

        return length_optimized_chunks

    def _filter_by_relevance(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Filters chunks by minimum relevance score.
        
        IMPORTANT: Only use this when chunks have Cross Encoder scores (0.0-1.0 range).
        Do NOT use with RRF scores (0.01-0.02 range) as it will filter all good chunks.
        """
        return [
            chunk for chunk in chunks
            if chunk.score is not None and chunk.score >= self.min_relevance_score
        ]

    def _remove_duplicates(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Removes duplicate chunks using content similarity.
        Uses Jaccard similarity for efficient comparison.
        """
        if not chunks:
            return []

        unique_chunks = []
        seen_signatures = []

        for chunk in chunks:
            # Create a content signature (set of words)
            content_words = set(chunk.content.lower().split())
            
            # Check for duplicates
            is_duplicate = False
            for signature in seen_signatures:
                # Calculate Jaccard similarity
                if content_words and signature:
                    intersection = len(content_words & signature)
                    union = len(content_words | signature)
                    jaccard_sim = intersection / union if union > 0 else 0.0
                    
                    if jaccard_sim >= self.dedup_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                # Store as frozenset in list to avoid hashability issues
                seen_signatures.append(frozenset(content_words))

        return unique_chunks

    def _optimize_length(
        self,
        chunks: List[RetrievedChunk],
        target_length: int
    ) -> List[RetrievedChunk]:
        """
        Selects chunks to fit within target length while maximizing relevance.
        Uses a greedy approach to select highest-scoring chunks first.
        """
        if not chunks:
            return []

        # Sort chunks by relevance score (descending)
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)

        selected_chunks = []
        current_length = 0

        for chunk in sorted_chunks:
            chunk_length = len(chunk.content)
            
            # Check if adding this chunk would exceed target length
            if current_length + chunk_length <= target_length:
                selected_chunks.append(chunk)
                current_length += chunk_length
            else:
                # Try to add a truncated version of the chunk
                remaining_length = target_length - current_length
                if remaining_length > 100:  # Only add if we have meaningful space
                    # Create a truncated chunk
                    truncated_content = chunk.content[:remaining_length]
                    truncated_content = truncated_content.rsplit(' ', 1)[0] + "..."
                    
                    truncated_chunk = RetrievedChunk(
                        content=truncated_content,
                        score=chunk.score * 0.8,  # Slightly lower score for truncated
                        title=chunk.title,
                        page=chunk.page,
                        chunk_id=chunk.chunk_id,
                        source=chunk.source,
                        author=chunk.author
                    )
                    selected_chunks.append(truncated_chunk)
                    current_length += len(truncated_content)
                break

        # Re-sort selected chunks by original order (optional, keeps document flow)
        # selected_chunks.sort(key=lambda x: chunks.index(x) if x in chunks else len(chunks))

        return selected_chunks

    def extract_relevant_sentences(
        self,
        query: str,
        chunk_content: str,
        max_sentences: int = 3
    ) -> str:
        """
        Extracts the most relevant sentences from a chunk based on query terms.
        Uses keyword matching for sentence selection.
        """
        query_terms = list(query.lower().split())
        sentences = chunk_content.split('. ')
        
        # Score sentences by query term overlap
        scored_sentences = []
        for sentence in sentences:
            sentence_words = list(sentence.lower().split())
            overlap = len(set(query_terms) & set(sentence_words))
            scored_sentences.append((sentence, overlap))
        
        # Sort by overlap score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences
        top_sentences = [s[0] for s in scored_sentences[:max_sentences]]
        
        return '. '.join(top_sentences)


class ContextualCompressionRetriever(BasePaperRetriever):
    """
    Retriever wrapper that applies context compression to retrieved chunks.
    Uses rule-based compression for efficiency.
    """

    def __init__(
        self,
        base_retriever: BasePaperRetriever,
        compressor: Optional[RuleBasedContextCompressor] = None
    ):
        self.base_retriever = base_retriever
        self.compressor = compressor or RuleBasedContextCompressor()

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieves chunks and applies context compression.
        """
        # Retrieve chunks using base retriever
        raw_chunks = self.base_retriever.retrieve(
            query=query,
            top_k=top_k,
            namespace=namespace,
            filter_dict=filter_dict
        )

        # Apply context compression
        compressed_chunks = self.compressor.compress_context(query, raw_chunks)

        app_logger.info(
            f"Contextual compression: {len(raw_chunks)} -> {len(compressed_chunks)} chunks"
        )

        return compressed_chunks
