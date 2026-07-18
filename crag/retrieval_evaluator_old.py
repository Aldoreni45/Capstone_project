from typing import List, Dict, Any, Literal
from pydantic import BaseModel
from pydantic_models.responses import RetrievedChunk
from custom_logging.logger import app_logger
from config.settings import settings
from langsmith import traceable
from embeddings import get_embeddings
from query.query_type_classifier import QueryTypeClassifier

RetrievalQuality = Literal["GOOD", "PARTIAL", "BAD"]

class RetrievalEvaluationResult(BaseModel):
    """Result of retrieval quality evaluation."""
    quality: RetrievalQuality
    reason: str
    avg_score: float
    chunk_count: int
    confidence: float
    answerable: bool  # Whether the query can be answered from retrieved context

class RetrievalEvaluator:
    """
    Production-grade retrieval quality evaluator that determines whether retrieved
    chunks can ACTUALLY answer the user's question.
    
    Evaluation factors (weights):
    1. Query Coverage (30%) - How much of the query is covered by context
    2. Context Relevance (25%) - Are chunks actually relevant to the question
    3. Top Chunk Relevance (15%) - Query vs top chunk similarity and coverage
    4. Context Sufficiency (15%) - Is context sufficient to answer the question
    5. Semantic Similarity (10%) - Partial contribution only
    6. Metadata Match (5%) - Minimal contribution
    
    SPECIAL RULES:
    - If Query Coverage < 25% → CANNOT be GOOD
    - If Context Sufficiency = False → CANNOT be GOOD
    - If No Named Entity Match → CANNOT be GOOD
    
    IMPORTANT: Semantic similarity alone should NEVER make a retrieval GOOD.
    The evaluator determines whether context can answer the question, not just
    whether it's semantically related.
    """
    
    def __init__(
        self,
        embedding_model: str = "huggingface",
        min_chunks_for_good: int = 2,
        min_chunks_for_partial: int = 1,
        query_coverage_weight: float = 0.30,
        context_relevance_weight: float = 0.25,
        top_chunk_weight: float = 0.15,
        context_sufficiency_weight: float = 0.15,
        semantic_weight: float = 0.10,
        metadata_weight: float = 0.05
    ):
        self.embedding_model = embedding_model
        self.min_chunks_for_good = min_chunks_for_good
        self.min_chunks_for_partial = min_chunks_for_partial
        self.query_coverage_weight = query_coverage_weight
        self.context_relevance_weight = context_relevance_weight
        self.top_chunk_weight = top_chunk_weight
        self.context_sufficiency_weight = context_sufficiency_weight
        self.semantic_weight = semantic_weight
        self.metadata_weight = metadata_weight
        self.embedding_client = None
        self.query_type_classifier = QueryTypeClassifier()
    
    @traceable(name="Retrieval Evaluation")
    def evaluate(
        self,
        retrieved_chunks: List[RetrievedChunk],
        query: str = None
    ) -> RetrievalEvaluationResult:
        """
        Evaluates retrieval quality using production-grade multi-factor analysis.
        
        Factors:
        1. Query Coverage (30%) - How much of the query is covered by context
        2. Context Relevance (25%) - Are chunks actually relevant to the question
        3. Top Chunk Relevance (15%) - Query vs top chunk similarity and coverage
        4. Context Sufficiency (15%) - Is context sufficient to answer the question
        5. Semantic Similarity (10%) - Partial contribution only
        6. Metadata Match (5%) - Minimal contribution
        
        Args:
            retrieved_chunks: List of retrieved chunks
            query: Original query for semantic analysis
            
        Returns:
            RetrievalEvaluationResult: Quality classification and metadata
        """
        if not retrieved_chunks:
            return RetrievalEvaluationResult(
                quality="BAD",
                reason="No chunks retrieved",
                avg_score=0.0,
                chunk_count=0,
                confidence=0.0
            )
        
        # Initialize embedding client if needed
        if self.embedding_client is None and query:
            try:
                self.embedding_client = get_embeddings(self.embedding_model)
            except Exception as e:
                app_logger.warning(f"Could not initialize embedding client: {e}")
        
        # Classify query type to determine evaluation strategy
        query_type = self.query_type_classifier.classify(query) if query else "general_research"
        use_entity_matching = self.query_type_classifier.should_use_entity_matching(query_type)
        
        app_logger.info(f"Query type: {query_type}, Entity matching: {use_entity_matching}")
        
        # Calculate multi-factor scores
        query_coverage_score = self._evaluate_query_coverage(retrieved_chunks, query) if query else 0.5
        context_relevance_score = self._evaluate_context_relevance(retrieved_chunks, query) if query else 0.5
        
        # Only use entity matching for named entity queries
        if use_entity_matching:
            named_entity_match = self._evaluate_named_entity_match(retrieved_chunks, query) if query else 0.5
        else:
            named_entity_match = 1.0  # Neutral score for non-named-entity queries
        
        context_sufficiency_score = self._evaluate_context_sufficiency(retrieved_chunks, query) if query else 0.5
        top_chunk_score = self._evaluate_top_chunk_relevance(retrieved_chunks, query) if query else 0.5
        semantic_score = self._evaluate_semantic_similarity(retrieved_chunks, query) if query else 0.5
        metadata_score = self._evaluate_metadata_match(retrieved_chunks, query)
        
        # Calculate weighted confidence score
        confidence = (
            query_coverage_score * self.query_coverage_weight +
            context_relevance_score * self.context_relevance_weight +
            top_chunk_score * self.top_chunk_weight +
            context_sufficiency_score * self.context_sufficiency_weight +
            semantic_score * self.semantic_weight +
            metadata_score * self.metadata_weight
        )
        
        # Apply special rules for classification
        quality, reason = self._classify_quality_with_special_rules(
            confidence,
            query_coverage_score,
            context_relevance_score,
            context_sufficiency_score,
            named_entity_match,
            len(retrieved_chunks),
            use_entity_matching
        )
        
        # Calculate average Cross Encoder score (for logging only, not for classification)
        avg_score = sum(chunk.score for chunk in retrieved_chunks if chunk.score is not None) / len(retrieved_chunks) if retrieved_chunks else 0.0
        
        app_logger.info(
            f"Retrieval Evaluation: {quality} | "
            f"Confidence: {confidence:.3f} | "
            f"QueryCoverage: {query_coverage_score:.3f} | "
            f"ContextRelevance: {context_relevance_score:.3f} | "
            f"EntityMatch: {named_entity_match:.3f} | "
            f"ContextSufficiency: {context_sufficiency_score:.3f} | "
            f"TopChunk: {top_chunk_score:.3f} | "
            f"Semantic: {semantic_score:.3f} | "
            f"Metadata: {metadata_score:.3f} | "
            f"Chunks: {len(retrieved_chunks)} | "
            f"AvgCrossEncoder: {avg_score:.3f} | "
            f"Reason: {reason}"
        )
        
        return RetrievalEvaluationResult(
            quality=quality,
            reason=reason,
            avg_score=avg_score,
            chunk_count=len(retrieved_chunks),
            confidence=confidence
        )
    
    def _evaluate_query_coverage(self, chunks: List[RetrievedChunk], query: str) -> float:
        """
        Evaluates how much of the user's query is covered by retrieved chunks.
        This is the PRIMARY metric for determining if the question can be answered.
        """
        if not chunks or not query:
            return 0.0
        
        # Extract query terms (remove common stop words)
        query_terms = set(query.lower().split())
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                     'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
                     'about', 'tell', 'me', 'what', 'how', 'why', 'when', 'where', 'who', 'explain'}
        
        query_terms = query_terms - stop_words
        
        if not query_terms:
            return 0.5  # Neutral if no meaningful terms
        
        # Check coverage in chunk contents
        covered_terms = set()
        for chunk in chunks:
            chunk_content = chunk.content.lower()
            for term in query_terms:
                if term in chunk_content:
                    covered_terms.add(term)
        
        coverage = len(covered_terms) / len(query_terms)
        return coverage
    
    def _evaluate_context_relevance(self, chunks: List[RetrievedChunk], query: str) -> float:
        """
        Determines whether retrieved chunks are ACTUALLY relevant to the question.
        This checks if the context contains the specific information requested.
        """
        if not chunks or not query:
            return 0.0
        
        query_lower = query.lower()
        relevant_chunks = 0
        
        for chunk in chunks:
            chunk_content = chunk.content.lower()
            
            # Check for exact query terms in the chunk
            query_terms = query_lower.split()
            term_matches = sum(1 for term in query_terms if term in chunk_content)
            
            # If more than 50% of query terms appear, consider it relevant
            if term_matches / len(query_terms) >= 0.5:
                relevant_chunks += 1
            # Also check for semantic phrases (3+ consecutive characters)
            elif len(query_lower) > 3:
                for i in range(len(query_lower) - 2):
                    substring = query_lower[i:i+3]
                    if substring in chunk_content:
                        relevant_chunks += 1
                        break
        
        return relevant_chunks / len(chunks) if chunks else 0.0
    
    def _evaluate_named_entity_match(self, chunks: List[RetrievedChunk], query: str) -> float:
        """
        Evaluates named entity matching between query and retrieved chunks.
        This is critical for document-specific queries.
        """
        if not chunks or not query:
            return 0.0
        
        # Extract potential named entities from query (capitalized words, proper nouns)
        query_words = query.split()
        named_entities = [word for word in query_words if word[0].isupper() or len(word) > 3]
        
        if not named_entities:
            # If no clear named entities, use all meaningful terms
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                         'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
                         'about', 'tell', 'me', 'what', 'how', 'why', 'when', 'where', 'who', 'explain'}
            named_entities = [word for word in query_words if word.lower() not in stop_words]
        
        if not named_entities:
            return 0.5  # Neutral if no entities to match
        
        # Check for entity matches in chunks
        entity_matches = 0
        for chunk in chunks:
            chunk_content = chunk.content.lower()
            for entity in named_entities:
                if entity.lower() in chunk_content:
                    entity_matches += 1
                    break  # Count each chunk only once
        
        match_ratio = entity_matches / len(chunks)
        return match_ratio
    
    def _evaluate_context_sufficiency(self, chunks: List[RetrievedChunk], query: str) -> float:
        """
        Determines whether the retrieved context is sufficient to answer the question.
        This checks if the context contains explicit information that can answer the query.
        """
        if not chunks or not query:
            return 0.0
        
        query_lower = query.lower()
        sufficient_chunks = 0
        
        for chunk in chunks:
            chunk_content = chunk.content.lower()
            
            # Check if chunk contains answer-like patterns
            # Look for definitions, explanations, or direct answers
            answer_patterns = ['is', 'are', 'was', 'were', 'refers to', 'means', 'defined as', 
                            'describes', 'explains', 'provides', 'contains', 'includes']
            
            has_answer_pattern = any(pattern in chunk_content for pattern in answer_patterns)
            
            # Check if chunk covers query terms
            query_terms = query_lower.split()
            term_coverage = sum(1 for term in query_terms if term in chunk_content) / len(query_terms)
            
            # Chunk is sufficient if it has answer patterns AND covers query terms
            if has_answer_pattern and term_coverage >= 0.3:
                sufficient_chunks += 1
            # Or if it has very high term coverage
            elif term_coverage >= 0.7:
                sufficient_chunks += 1
        
        return sufficient_chunks / len(chunks) if chunks else 0.0
    
    def _evaluate_semantic_similarity(self, chunks: List[RetrievedChunk], query: str) -> float:
        """Evaluates semantic similarity between query and chunks."""
        if not self.embedding_client or not chunks or not query:
            return 0.5  # Neutral score if embeddings unavailable
        
        try:
            query_embedding = self.embedding_client.embed_query(query)
            
            # Calculate similarity with top chunk
            top_chunk = chunks[0]
            chunk_embedding = self.embedding_client.embed_query(top_chunk.content)
            
            # Simple cosine similarity (dot product for normalized embeddings)
            similarity = sum(q * c for q, c in zip(query_embedding, chunk_embedding))
            
            # Normalize to 0-1 range (assuming embeddings are normalized)
            similarity = max(0.0, min(1.0, (similarity + 1) / 2))
            
            return similarity
        except Exception as e:
            app_logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.5
    
    def _evaluate_query_coverage(self, chunks: List[RetrievedChunk], query: str) -> float:
        """Evaluates how much of the query is covered by retrieved context."""
        if not chunks or not query:
            return 0.5
        
        # Extract query terms (remove common stop words)
        query_terms = set(query.lower().split())
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                     'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
                     'about', 'tell', 'me', 'what', 'how', 'why', 'when', 'where', 'who'}
        
        query_terms = query_terms - stop_words
        
        if not query_terms:
            return 0.5
        
        # Check coverage in chunk contents
        covered_terms = set()
        for chunk in chunks:
            chunk_content = chunk.content.lower()
            for term in query_terms:
                if term in chunk_content:
                    covered_terms.add(term)
        
        coverage = len(covered_terms) / len(query_terms)
        return coverage
    
    def _evaluate_top_chunk_relevance(self, chunks: List[RetrievedChunk], query: str) -> float:
        """
        Evaluates relevance of the top retrieved chunk with enhanced analysis.
        The top chunk should have strong influence on retrieval quality.
        """
        if not chunks or not query:
            return 0.5
        
        top_chunk = chunks[0]
        query_lower = query.lower()
        chunk_content = top_chunk.content.lower()
        
        # Check for exact query terms
        query_terms = query_lower.split()
        term_matches = sum(1 for term in query_terms if term in chunk_content)
        
        # Calculate term match ratio
        term_ratio = term_matches / len(query_terms) if query_terms else 0
        
        # Check for semantic relevance (longer matches)
        if len(query_lower) > 3:
            # Check if any significant portion of query appears in chunk
            for i in range(len(query_lower) - 2):
                substring = query_lower[i:i+3]
                if substring in chunk_content:
                    term_ratio = max(term_ratio, 0.7)  # Boost for semantic match
                    break
        
        # Check for answer patterns in top chunk
        answer_patterns = ['is', 'are', 'was', 'were', 'refers to', 'means', 'defined as', 
                        'describes', 'explains', 'provides']
        has_answer = any(pattern in chunk_content for pattern in answer_patterns)
        
        if has_answer and term_ratio >= 0.5:
            term_ratio = min(1.0, term_ratio + 0.2)  # Boost for answer-containing chunk
        
        return term_ratio
    
    def _evaluate_metadata_match(self, chunks: List[RetrievedChunk], query: str = None) -> float:
        """Evaluates metadata matching (document title, author, source)."""
        if not chunks:
            return 0.0
        
        # Check if chunks have metadata fields
        metadata_matches = 0
        total_chunks = len(chunks)
        
        for chunk in chunks:
            # Check for document title, author, source using direct field access
            if chunk.title or chunk.author or chunk.source:
                metadata_matches += 1
        
        # If query is provided, check for query terms in metadata fields
        if query:
            query_terms = set(query.lower().split())
            for chunk in chunks:
                # Check title
                if chunk.title and any(term in chunk.title.lower() for term in query_terms):
                    metadata_matches += 0.5  # Partial credit for term match
                # Check author
                elif chunk.author and any(term in chunk.author.lower() for term in query_terms):
                    metadata_matches += 0.5
                # Check source
                elif chunk.source and any(term in chunk.source.lower() for term in query_terms):
                    metadata_matches += 0.5
        
        return min(1.0, metadata_matches / total_chunks)
    
    def _classify_quality_with_special_rules(
        self,
        confidence: float,
        query_coverage: float,
        context_relevance: float,
        context_sufficiency: float,
        named_entity_match: float,
        chunk_count: int,
        use_entity_matching: bool
    ) -> tuple:
        """
        Classifies quality with special rules to prevent incorrect GOOD classifications.
        
        SPECIAL RULES:
        - If Query Coverage < 25% → CANNOT be GOOD
        - If Context Sufficiency < 30% → CANNOT be GOOD
        - If No Named Entity Match AND entity matching is enabled → CANNOT be GOOD
        """
        
        # Apply special blocking rules for GOOD classification
        can_be_good = True
        blocking_reasons = []
        
        if query_coverage < 0.25:
            can_be_good = False
            blocking_reasons.append(f"low query coverage ({query_coverage:.1%})")
        
        if context_sufficiency < 0.3:
            can_be_good = False
            blocking_reasons.append(f"insufficient context ({context_sufficiency:.1%})")
        
        # Only apply entity matching rule if entity matching is enabled
        if use_entity_matching and named_entity_match < 0.3:
            can_be_good = False
            blocking_reasons.append(f"no named entity match ({named_entity_match:.1%})")
        
        # GOOD: High confidence with no blocking rules
        if can_be_good and confidence >= 0.65 and chunk_count >= self.min_chunks_for_good:
            reasons = []
            if query_coverage > 0.7:
                reasons.append("excellent query coverage")
            if context_relevance > 0.7:
                reasons.append("high context relevance")
            if context_sufficiency > 0.7:
                reasons.append("sufficient context")
            
            reason = f"High confidence retrieval ({', '.join(reasons) if reasons else 'multiple strong factors'})"
            return "GOOD", reason
        
        # PARTIAL: Moderate confidence or blocked from GOOD
        elif confidence >= 0.4 and chunk_count >= self.min_chunks_for_partial:
            if not can_be_good:
                reason = f"Moderate retrieval but blocked from GOOD: {', '.join(blocking_reasons)}"
            else:
                reasons = []
                if query_coverage > 0.5:
                    reasons.append("moderate query coverage")
                if context_relevance > 0.5:
                    reasons.append("moderate context relevance")
                reason = f"Moderate confidence retrieval ({', '.join(reasons) if reasons else 'some relevant factors'})"
            return "PARTIAL", reason
        
        # BAD: Low confidence or poor metrics
        else:
            reasons = []
            if query_coverage < 0.3:
                reasons.append("poor query coverage")
            if context_relevance < 0.3:
                reasons.append("low context relevance")
            if context_sufficiency < 0.3:
                reasons.append("insufficient context")
            if use_entity_matching and named_entity_match < 0.3:
                reasons.append("no entity match")
            
            reason = f"Low confidence retrieval ({', '.join(reasons) if reasons else 'insufficient relevance'})"
            return "BAD", reason
    
    def validate_partial_quality(
        self,
        evaluation: RetrievalEvaluationResult,
        query_coverage: float,
        context_sufficiency: float
    ) -> tuple[bool, str]:
        """
        Validates whether PARTIAL quality is sufficient to proceed with pipeline.
        
        PARTIAL is only valid if:
        - Query Coverage >= 50%
        - Context Sufficiency >= 50%
        - At least one supporting chunk exists
        
        Returns:
            tuple: (is_valid, reason)
        """
        if evaluation.quality != "PARTIAL":
            return True, "Not a PARTIAL quality evaluation"
        
        validation_failures = []
        
        if query_coverage < 0.5:
            validation_failures.append(f"query coverage below 50% ({query_coverage:.1%})")
        
        if context_sufficiency < 0.5:
            validation_failures.append(f"context sufficiency below 50% ({context_sufficiency:.1%})")
        
        if evaluation.chunk_count < 1:
            validation_failures.append("no supporting chunks")
        
        if validation_failures:
            return False, f"PARTIAL validation failed: {', '.join(validation_failures)}"
        
        return True, "PARTIAL validation passed"
