# -----------------------------------------------------------------------------
# Production Optimized Prompt Templates
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """Answer comprehensively using the provided context. Provide detailed explanations with examples when relevant. If information is unavailable, clearly state what you don't know."""


RAG_PROMPT_TEMPLATE = """Context:
{context_text}

Question: {question}

Provide a comprehensive answer using the context. Include relevant details, explanations, and examples when appropriate. If the context is insufficient, clearly state what information is missing."""


CITATION_PROMPT_TEMPLATE = """Format: Paper Title, Page, Chunk ID"""


FOLLOWUP_PROMPT_TEMPLATE = """Generate 3 relevant follow-up questions. Output only questions."""


EVALUATION_PROMPT_TEMPLATE = """Rate (0-1): faithfulness, relevancy, precision. JSON: {{"faithfulness":0,"answer_relevancy":0,"context_precision":0,"verdict":""}}"""