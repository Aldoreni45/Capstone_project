import os
from config.settings import settings
from custom_logging.logger import app_logger

class LangSmithTracker:
    """Manages LangSmith tracing environment and verifies connection settings."""

    @staticmethod
    def init_tracing():
        """Applies tracing environment variables dynamically."""
        if settings.langchain_tracing_v2 and settings.langsmith_api_key:
            app_logger.info("Initializing LangSmith Tracing...")
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            app_logger.info(f"LangSmith tracking activated for project '{settings.langchain_project}'.")
        else:
            app_logger.info("LangSmith tracing is disabled (missing LANGCHAIN_TRACING_V2=true or API Key).")
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            
    @staticmethod
    def is_active() -> bool:
        return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
