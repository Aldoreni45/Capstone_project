from langchain_openai import OpenAIEmbeddings
from config.settings import settings
from custom_logging.logger import app_logger
from custom_logging.error_handler import EmbeddingModelError

def get_openai_embeddings() -> OpenAIEmbeddings:
    """Instantiates and returns the OpenAI Embeddings client."""
    api_key = settings.openai_api_key
    model_name = settings.get("embeddings", "openai", "model_name", default="text-embedding-3-small")
    
    if not api_key:
        app_logger.warning("OPENAI_API_KEY environment variable is missing. OpenAI Embeddings calls will fail.")
        
    app_logger.info(f"Initializing OpenAI Embeddings ({model_name})...")
    
    try:
        openai_model = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key
        )
        app_logger.info("OpenAI embeddings client initialized successfully.")
        return openai_model
    except Exception as e:
        app_logger.error(f"Failed to instantiate OpenAI embeddings client: {str(e)}")
        raise EmbeddingModelError(f"OpenAI embedding error: {str(e)}")
