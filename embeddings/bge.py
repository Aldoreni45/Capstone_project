from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
from custom_logging.logger import app_logger
from custom_logging.error_handler import EmbeddingModelError
import os

# Singleton instance cache
_bge_embeddings_instance = None

def get_bge_embeddings() -> HuggingFaceEmbeddings:
    """
    Instantiates and returns the HuggingFace BGE Small model using Inference API.
    Uses HF_TOKEN from environment for authentication.
    No local model download - all embedding generation happens via API calls.
    Implements singleton pattern to prevent multiple initializations.
    """
    global _bge_embeddings_instance
    
    if _bge_embeddings_instance is not None:
        app_logger.debug("Returning cached BGE Inference API embeddings instance")
        return _bge_embeddings_instance
    
    model_name = settings.get("embeddings", "bge", "model_name", default="BAAI/bge-small-en-v1.5")
    hf_token = os.getenv("HF_TOKEN", "")

    if not hf_token:
        app_logger.warning("HF_TOKEN not found in environment. Using free tier with rate limits.")
        app_logger.warning("Set HF_TOKEN in .env file for higher rate limits and faster inference.")
    else:
        # Set HUGGINGFACE_HUB_TOKEN for authentication
        if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            app_logger.info("Using HF_TOKEN for Hugging Face Hub authentication")

    app_logger.info(f"Initializing BGE Inference API embeddings ({model_name})...")

    try:
        # HuggingFaceEmbeddings from langchain_huggingface uses environment variables for authentication
        # No direct API token parameter is supported
        bge_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # Not used for Inference API but kept for compatibility
            encode_kwargs={"normalize_embeddings": True}
        )
        app_logger.info("BGE Inference API embeddings initialized successfully.")
        _bge_embeddings_instance = bge_model
        return bge_model
    except Exception as e:
        app_logger.error(f"Failed to initialize BGE Inference API embedding model: {str(e)}")
        raise EmbeddingModelError(f"BGE Inference API embedding error: {str(e)}")
