from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
from custom_logging.logger import app_logger
from custom_logging.error_handler import EmbeddingModelError
import os

# Singleton instance cache
_huggingface_embeddings_instance = None

def get_huggingface_embeddings() -> HuggingFaceEmbeddings:
    """
    Instantiates and returns a HuggingFace embedding model using the Inference API.
    Uses HF_TOKEN from environment for authentication.
    Defaults to BAAI/bge-large-en-v1.5 as configured in config.yaml.
    No local model download - all embedding generation happens via API calls.
    Implements singleton pattern to prevent multiple initializations.
    """
    global _huggingface_embeddings_instance
    
    if _huggingface_embeddings_instance is not None:
        app_logger.debug("Returning cached HuggingFace Inference API embeddings instance")
        return _huggingface_embeddings_instance
    
    model_name = settings.get("embeddings", "huggingface", "model_name", default="BAAI/bge-large-en-v1.5")
    hf_token = os.getenv("HF_TOKEN", "")

    if not hf_token:
        app_logger.warning("HF_TOKEN not found in environment. Using free tier with rate limits.")
        app_logger.warning("Set HF_TOKEN in .env file for higher rate limits and faster inference.")
    else:
        # Set HUGGINGFACE_HUB_TOKEN for authentication
        if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            app_logger.info("Using HF_TOKEN for Hugging Face Hub authentication")

    app_logger.info(f"Initializing HuggingFace Inference API embeddings ({model_name})...")

    try:
        # HuggingFaceEmbeddings from langchain_huggingface uses environment variables for authentication
        # No direct API token parameter is supported
        hf_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},  # Not used for Inference API but kept for compatibility
            encode_kwargs={"normalize_embeddings": True}
        )
        app_logger.info(f"HuggingFace Inference API embeddings ({model_name}) initialized successfully.")
        _huggingface_embeddings_instance = hf_model
        return hf_model
    except Exception as e:
        app_logger.error(f"Failed to initialize HuggingFace Inference API embedding model '{model_name}': {str(e)}")
        raise EmbeddingModelError(f"HuggingFace Inference API embedding error: {str(e)}")
