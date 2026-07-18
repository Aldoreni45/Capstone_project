from embeddings.base import EmbeddingBenchmarker
from embeddings.bge import get_bge_embeddings
from embeddings.huggingface import get_huggingface_embeddings
from langchain_core.embeddings import Embeddings
from config.settings import settings
from custom_logging.logger import app_logger

def get_embeddings(model_type: str = None) -> Embeddings:
    """
    Factory to retrieve an embeddings model based on name ('bge' or 'huggingface').
    Defaults to config settings if model_type is not provided.
    """
    model_type = model_type or settings.get("embeddings", "default_model", default="huggingface")
    model_type = model_type.lower().strip()

    if model_type == "bge":
        return get_bge_embeddings()
    elif model_type in ("huggingface", "hf"):
        return get_huggingface_embeddings()
    else:
        app_logger.warning(f"Unknown embedding model type '{model_type}'. Defaulting to HuggingFace BGE.")
        return get_huggingface_embeddings()
