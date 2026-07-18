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
    Production-grade retrieval quality evaluator using Answerability Check + Weighted Confidence Scoring.
    
    Architecture:
    1. Answerability Check: Can the retrieved chunks answer the user's question?
       - If NO → BAD (stop pipeline)
       - If YES → Continue to confidence scoring
    
    2. Weighted Confidence Scoring:
       - Semantic Similarity (30%)
       - Context Relevance (25%)
       - Query Coverage (20%)
       - Context Sufficiency (15%)
       - Top Chunk Score (10%)
    
    3. Classification:
       - GOOD: Context sufficiently answers the query
       - PARTIAL: Context partially answers the query (pipeline continues)
       - BAD: Context cannot answer the query (pipeline stops)
    
    Key Principles:
    - NO hard-coded threshold-based if-else rules
    - Semantic similarity alone is NOT sufficient for answerability
    - PARTIAL retrieval is VALID and should continue
    - Context incompleteness does NOT automatically mean BAD
    """
    
    def __init__(
        self,
        embedding_model: str = "huggingface",
        semantic_weight: float = 0.30,
        relevance_weight: float = 0.25,
        coverage_weight: float = 0.20,
        sufficiency_weight: float = 0.15,
        top_chunk_weight: float = 0.10
    ):
        self.embedding_model = embedding_model
        self.semantic_weight = semantic_weight
        self.relevance_weight = relevance_weight
        self.coverage_weight = coverage_weight
        self.sufficiency_weight = sufficiency_weight
        self.top_chunk_weight = top_chunk_weight
        self.embedding_client = None
        self.query_type_classifier = QueryTypeClassifier()
        
        # Concept coverage tracking (for concept queries)
        self._concept_coverage = 0.0
        self._found_concepts = []
        self._missing_concepts = []
    
    @traceable(name="Retrieval Evaluation")
    def evaluate(
        self,
        retrieved_chunks: List[RetrievedChunk],
        query: str = None
    ) -> RetrievalEvaluationResult:
        """
        Evaluates retrieval quality using Answerability Check + Weighted Confidence Scoring.
        
        Architecture:
        1. Answerability Check: Can retrieved chunks answer the question?
        2. Weighted Confidence Scoring: Compute confidence score
        3. Classification: GOOD/PARTIAL/BAD based on answerability and confidence
        
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
                confidence=0.0,
                answerable=False
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
        
        # STEP 1: Answerability Check - Can the retrieved chunks answer the question?
        answerable, answerability_reason = self._check_answerability(retrieved_chunks, query, query_type)
        
        if not answerable:
            # BAD: Context cannot answer the query at all
            app_logger.info(f"Answerability Check FAILED: {answerability_reason}")
            return RetrievalEvaluationResult(
                quality="BAD",
                reason=f"Cannot answer query: {answerability_reason}",
                avg_score=0.0,
                chunk_count=len(retrieved_chunks),
                confidence=0.0,
                answerable=False
            )
        
        # STEP 2: Calculate weighted confidence scores
        semantic_score = self._evaluate_semantic_similarity(retrieved_chunks, query) if query else 0.5
        relevance_score = self._evaluate_context_relevance(retrieved_chunks, query) if query else 0.5
        coverage_score = self._evaluate_query_coverage(retrieved_chunks, query) if query else 0.5
        sufficiency_score = self._evaluate_context_sufficiency(retrieved_chunks, query) if query else 0.5
        top_chunk_score = self._evaluate_top_chunk_relevance(retrieved_chunks, query) if query else 0.5
        
        # Calculate weighted confidence score
        confidence = (
            semantic_score * self.semantic_weight +
            relevance_score * self.relevance_weight +
            coverage_score * self.coverage_weight +
            sufficiency_score * self.sufficiency_weight +
            top_chunk_score * self.top_chunk_weight
        )
        
        # STEP 3: Classification based on answerability and confidence
        quality, reason = self._classify_by_answerability_and_confidence(
            confidence,
            semantic_score,
            relevance_score,
            coverage_score,
            sufficiency_score,
            len(retrieved_chunks),
            query_type
        )
        
        # Calculate average Cross Encoder score (for logging only)
        avg_score = sum(chunk.score for chunk in retrieved_chunks if chunk.score is not None) / len(retrieved_chunks) if retrieved_chunks else 0.0
        
        app_logger.info(
            f"Retrieval Evaluation: {quality} | "
            f"Answerable: True | "
            f"Confidence: {confidence:.3f} | "
            f"Semantic: {semantic_score:.3f} | "
            f"Relevance: {relevance_score:.3f} | "
            f"Coverage: {coverage_score:.3f} | "
            f"Sufficiency: {sufficiency_score:.3f} | "
            f"TopChunk: {top_chunk_score:.3f} | "
            f"Chunks: {len(retrieved_chunks)} | "
            f"AvgCrossEncoder: {avg_score:.3f} | "
            f"Reason: {reason}"
        )
        
        return RetrievalEvaluationResult(
            quality=quality,
            reason=reason,
            avg_score=avg_score,
            chunk_count=len(retrieved_chunks),
            confidence=confidence,
            answerable=True
        )
    
    def _check_answerability(
        self,
        chunks: List[RetrievedChunk],
        query: str,
        query_type: str
    ) -> tuple[bool, str]:
        """
        PRIMARY EVALUATION STEP: Can the retrieved chunks answer the user's question?
        
        This is the most critical check. If the answer is NO, the pipeline should stop immediately.
        
        Answerability Criteria:
        1. For named entity queries: Entity must be present in chunks
        2. For concept queries: Concept must be present in chunks
        3. For general queries: Query terms must be covered in chunks
        4. Semantic similarity alone is NOT sufficient
        
        Returns:
            tuple: (is_answerable, reason)
        """
        if not chunks or not query:
            return False, "No chunks or query provided"
        
        query_lower = query.lower()
        
        # Check 1: Named Entity Queries - Entity must be present
        if query_type == "named_entity":
            # Extract named entities from query (capitalized words)
            query_words = query.split()
            named_entities = [word for word in query_words if word[0].isupper() and len(word) > 1]
            
            if not named_entities:
                # Fallback to meaningful terms if no capitalized words
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                             'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                             'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
                             'about', 'tell', 'me', 'what', 'how', 'why', 'when', 'where', 'who', 'explain'}
                named_entities = [word for word in query_words if word.lower() not in stop_words]
            
            # Check if entities are present in chunks
            for entity in named_entities:
                entity_lower = entity.lower()
                entity_found = False
                for chunk in chunks:
                    if entity_lower in chunk.content.lower():
                        entity_found = True
                        break
                
                if not entity_found:
                    return False, f"Named entity '{entity}' not found in retrieved context"
        
        # Check 2: Concept Queries - Multi-concept extraction and coverage calculation
        elif query_type == "concept":
            # Extract ALL concepts from the query
            concepts = self._extract_concepts_from_query(query_lower)
            
            if not concepts:
                # Fallback to general query coverage if no concepts extracted
                pass
            else:
                # Check concept coverage using multiple matching strategies
                concept_coverage, found_concepts, missing_concepts = self._check_concept_coverage(
                    concepts, chunks, query_lower
                )
                
                app_logger.info(
                    f"Concept Coverage: {concept_coverage:.1%} | "
                    f"Found: {found_concepts} | "
                    f"Missing: {missing_concepts}"
                )
                
                # Concept Coverage = 0% → BAD
                if concept_coverage == 0.0:
                    return False, f"No concepts found in retrieved context. Missing: {', '.join(missing_concepts)}"
                
                # Concept Coverage > 0% → Answerable (PARTIAL or GOOD will be determined later)
                # Store concept coverage for classification
                self._concept_coverage = concept_coverage
                self._found_concepts = found_concepts
                self._missing_concepts = missing_concepts
        
        # Check 3: General Query Coverage - Query terms must be covered
        # Extract meaningful query terms
        query_terms = set(query_lower.split())
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                     'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
                     'about', 'tell', 'me', 'what', 'how', 'why', 'when', 'where', 'who', 'explain'}
        
        query_terms = query_terms - stop_words
        
        if query_terms:
            # Check if query terms are covered in chunks
            covered_terms = set()
            for chunk in chunks:
                chunk_content = chunk.content.lower()
                for term in query_terms:
                    if term in chunk_content:
                        covered_terms.add(term)
            
            coverage = len(covered_terms) / len(query_terms)
            
            # If less than 20% of query terms are covered, not answerable
            if coverage < 0.2:
                return False, f"Query terms not sufficiently covered in context ({coverage:.1%})"
        
        # Check 4: Context must contain actual information (not just noise)
        # Check if chunks contain answer-like patterns
        answer_patterns = ['is', 'are', 'was', 'were', 'refers to', 'means', 'defined as', 
                        'describes', 'explains', 'provides', 'contains', 'includes']
        
        has_answer_content = False
        for chunk in chunks:
            chunk_content = chunk.content.lower()
            if any(pattern in chunk_content for pattern in answer_patterns):
                has_answer_content = True
                break
        
        if not has_answer_content:
            return False, "Retrieved context does not contain answer-like content"
        
        # All checks passed - query is answerable
        return True, "Query is answerable from retrieved context"
    
    def _classify_by_answerability_and_confidence(
        self,
        confidence: float,
        semantic_score: float,
        relevance_score: float,
        coverage_score: float,
        sufficiency_score: float,
        chunk_count: int,
        query_type: str = None
    ) -> tuple:
        """
        Classifies retrieval quality based on answerability (already confirmed) and confidence.
        
        Classification Logic:
        - For Concept Queries: Based on concept coverage
          * Concept Coverage = 100% → GOOD
          * Concept Coverage > 0% → PARTIAL
          * Concept Coverage = 0% → BAD (already handled in answerability check)
        
        - For Other Queries: Based on confidence scores
          * GOOD: Context sufficiently answers the query (high confidence)
          * PARTIAL: Context partially answers the query (moderate confidence)
          * BAD: Context cannot answer the query (already handled in answerability check)
        
        Note: This method is only called when answerable=True, so BAD is only returned
        for edge cases with extremely low confidence.
        """
        # For concept queries, use concept coverage for classification
        if query_type == "concept" and hasattr(self, '_concept_coverage'):
            concept_coverage = self._concept_coverage
            found_concepts = self._found_concepts
            missing_concepts = self._missing_concepts
            
            # Concept Coverage = 100% → GOOD
            if concept_coverage == 1.0:
                reason = f"All concepts found in retrieved context: {', '.join(found_concepts)}"
                return "GOOD", reason
            
            # Concept Coverage > 0% → PARTIAL
            elif concept_coverage > 0.0:
                reason = (
                    f"Partial concept coverage ({concept_coverage:.1%}). "
                    f"Found: {', '.join(found_concepts)}. "
                    f"Missing: {', '.join(missing_concepts)}."
                )
                return "PARTIAL", reason
            
            # Concept Coverage = 0% → BAD (should not reach here due to answerability check)
            else:
                reason = f"No concepts found in retrieved context. Missing: {', '.join(missing_concepts)}"
                return "BAD", reason
        
        # For non-concept queries, use confidence-based classification
        # GOOD: High confidence across multiple metrics
        if confidence >= 0.7:
            reasons = []
            if semantic_score >= 0.7:
                reasons.append("strong semantic match")
            if relevance_score >= 0.7:
                reasons.append("high context relevance")
            if coverage_score >= 0.7:
                reasons.append("excellent query coverage")
            if sufficiency_score >= 0.7:
                reasons.append("sufficient context")
            
            reason = f"Context sufficiently answers query ({', '.join(reasons) if reasons else 'high overall confidence'})"
            return "GOOD", reason
        
        # PARTIAL: Moderate confidence - context partially answers the query
        elif confidence >= 0.4:
            reasons = []
            if semantic_score >= 0.5:
                reasons.append("moderate semantic match")
            if relevance_score >= 0.5:
                reasons.append("moderate context relevance")
            if coverage_score >= 0.5:
                reasons.append("partial query coverage")
            if sufficiency_score >= 0.5:
                reasons.append("some context sufficiency")
            
            reason = f"Context partially answers query ({', '.join(reasons) if reasons else 'moderate overall confidence'})"
            return "PARTIAL", reason
        
        # Edge case: Very low confidence even though answerable
        else:
            reason = "Context has answerable content but low confidence scores"
            return "PARTIAL", reason  # PARTIAL is still valid, pipeline should continue
    
    def _evaluate_semantic_similarity(self, chunks: List[RetrievedChunk], query: str) -> float:
        """Evaluates semantic similarity between query and chunks."""
        if not query:
            return 0.5
        
        # If embedding client is unavailable, use term-based similarity as fallback
        if not self.embedding_client:
            return self._term_based_similarity(chunks, query)
        
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
            return self._term_based_similarity(chunks, query)
    
    def _term_based_similarity(self, chunks: List[RetrievedChunk], query: str) -> float:
        """Fallback term-based similarity when embeddings are unavailable."""
        if not chunks or not query:
            return 0.5
        
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        if not query_terms:
            return 0.5
        
        # Calculate Jaccard similarity with top chunk
        top_chunk = chunks[0]
        chunk_terms = set(top_chunk.content.lower().split())
        
        intersection = len(query_terms & chunk_terms)
        union = len(query_terms | chunk_terms)
        
        if union == 0:
            return 0.5
        
        similarity = intersection / union
        return similarity
    
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
    
    def _evaluate_query_coverage(self, chunks: List[RetrievedChunk], query: str) -> float:
        """
        Evaluates how much of and user's query is covered by retrieved chunks.
        This measures the extent to which query terms appear in the context.
        """
        if not chunks or not query:
            return 0.5
        
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
    
    def _evaluate_context_sufficiency(self, chunks: List[RetrievedChunk], query: str) -> float:
        """
        Determines whether the retrieved context is sufficient to answer the question.
        This checks if the context contains explicit information that can answer the query.
        
        IMPORTANT: Context incompleteness does NOT automatically mean BAD.
        Partial information is still sufficient for PARTIAL classification.
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
    
    def _evaluate_top_chunk_relevance(self, chunks: List[RetrievedChunk], query: str) -> float:
        """
        Evaluates relevance of the top retrieved chunk.
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
    
    def _extract_concepts_from_query(self, query_lower: str) -> List[str]:
        """
        Extract ALL concepts from a concept query.
        
        Examples:
        - "what is rnn and attention?" → ["rnn", "attention"]
        - "explain transformer, encoder and decoder" → ["transformer", "encoder", "decoder"]
        - "what is hive?" → ["hive"]
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            List of extracted concepts
        """
        # Remove common question patterns
        query_clean = query_lower
        
        # Remove question words and patterns
        question_patterns = [
            "what is", "what are", "define", "explain", "describe",
            "tell me about", "what does", "what do"
        ]
        
        for pattern in question_patterns:
            if query_clean.startswith(pattern):
                query_clean = query_clean[len(pattern):].strip()
        
        # Remove question mark
        query_clean = query_clean.replace("?", "").strip()
        
        # Split by common separators (and, comma, &)
        separators = [" and ", ", ", " & ", " and the ", " and a ", " and an "]
        concepts = []
        
        # Split by separators
        for sep in separators:
            if sep in query_clean:
                parts = query_clean.split(sep)
                concepts.extend([part.strip() for part in parts if part.strip()])
                break
        else:
            # No separator found, treat entire query as single concept
            if query_clean:
                concepts.append(query_clean)
        
        # Remove stop words from concepts
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                     'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
                     'about', 'tell', 'me', 'what', 'how', 'why', 'when', 'where', 'who', 'explain',
                     'define', 'describe', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from'}
        
        concepts = [c for c in concepts if c.lower() not in stop_words and len(c) > 1]
        
        return concepts
    
    def _check_concept_coverage(
        self,
        concepts: List[str],
        chunks: List[RetrievedChunk],
        query_lower: str
    ) -> tuple[float, List[str], List[str]]:
        """
        Check concept coverage using multiple matching strategies.
        
        Matching Strategies:
        1. Exact Match - Concept appears exactly in chunk
        2. Keyword Match - Concept keywords appear in chunk
        3. Semantic Match - Concept appears in different form (e.g., "attention" vs "self-attention")
        4. Metadata Match - Concept appears in chunk metadata
        
        Args:
            concepts: List of concepts to check
            chunks: Retrieved chunks
            query_lower: Original query for context
            
        Returns:
            tuple: (coverage_ratio, found_concepts, missing_concepts)
        """
        found_concepts = []
        missing_concepts = []
        
        for concept in concepts:
            concept_found = False
            
            # Strategy 1: Exact Match
            for chunk in chunks:
                if concept in chunk.content.lower():
                    concept_found = True
                    found_concepts.append(concept)
                    break
            
            if concept_found:
                continue
            
            # Strategy 2: Keyword Match (split concept into keywords)
            concept_keywords = concept.split()
            keyword_matches = 0
            
            for chunk in chunks:
                chunk_content = chunk.content.lower()
                for keyword in concept_keywords:
                    if keyword in chunk_content:
                        keyword_matches += 1
                
                # If 50%+ of keywords match, consider it found
                if keyword_matches / len(concept_keywords) >= 0.5:
                    concept_found = True
                    found_concepts.append(concept)
                    break
            
            if concept_found:
                continue
            
            # Strategy 3: Semantic Match (check for variations)
            # Common variations for technical terms
            concept_variations = self._get_concept_variations(concept)
            
            for variation in concept_variations:
                for chunk in chunks:
                    if variation in chunk.content.lower():
                        concept_found = True
                        found_concepts.append(concept)
                        break
                
                if concept_found:
                    break
            
            if concept_found:
                continue
            
            # Strategy 4: Metadata Match
            for chunk in chunks:
                # Check title
                if chunk.title and concept in chunk.title.lower():
                    concept_found = True
                    found_concepts.append(concept)
                    break
                # Check author
                elif chunk.author and concept in chunk.author.lower():
                    concept_found = True
                    found_concepts.append(concept)
                    break
                # Check source
                elif chunk.source and concept in chunk.source.lower():
                    concept_found = True
                    found_concepts.append(concept)
                    break
            
            if not concept_found:
                missing_concepts.append(concept)
        
        # Calculate coverage ratio
        coverage_ratio = len(found_concepts) / len(concepts) if concepts else 0.0
        
        return coverage_ratio, found_concepts, missing_concepts
    
    def _get_concept_variations(self, concept: str) -> List[str]:
        """
        Get common variations for technical concepts.
        
        Examples:
        - "attention" → ["self-attention", "multi-head attention", "attention mechanism"]
        - "transformer" → ["transformer architecture", "transformer model"]
        - "encoder" → ["encoder layer", "encoder block"]
        
        Args:
            concept: Base concept
            
        Returns:
            List of concept variations
        """
        variations = [concept]
        concept_lower = concept.lower()
        
        # Common technical term variations
        variations_map = {
            "attention": ["self-attention", "multi-head attention", "attention mechanism", "attention layer"],
            "transformer": ["transformer architecture", "transformer model", "transformer network"],
            "encoder": ["encoder layer", "encoder block", "encoder component"],
            "decoder": ["decoder layer", "decoder block", "decoder component"],
            "rnn": ["recurrent neural network", "rnn layer", "rnn cell"],
            "lstm": ["long short-term memory", "lstm cell", "lstm layer"],
            "cnn": ["convolutional neural network", "cnn layer", "cnn architecture"],
            "bert": ["bert model", "bert architecture", "bert transformer"],
            "gpt": ["gpt model", "gpt architecture", "gpt transformer"],
            "hive": ["apache hive", "hive database", "hive data warehouse"],
            "kafka": ["apache kafka", "kafka messaging", "kafka stream"],
            "spark": ["apache spark", "spark framework", "spark processing"],
        }
        
        if concept_lower in variations_map:
            variations.extend(variations_map[concept_lower])
        
        # Add common prefixes/suffixes
        variations.append(f"{concept} architecture")
        variations.append(f"{concept} model")
        variations.append(f"{concept} layer")
        variations.append(f"{concept} mechanism")
        
        return variations
