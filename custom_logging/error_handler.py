import time
import functools
from typing import Any, Callable, Tuple, Type
from custom_logging.logger import app_logger

class AppBaseException(Exception):
    """Base exception for the application."""
    def __init__(self, message: str, details: Any = None):
        super().__init__(message)
        self.message = message
        self.details = details

class ConfigurationError(AppBaseException):
    """Raised when there is an issue with application configuration or env vars."""

class PDFParsingError(AppBaseException):
    """Raised when a PDF fails to load or parse."""

class EmbeddingModelError(AppBaseException):
    """Raised when the embedding generation fails."""

class WeaviateAPIError(AppBaseException):
    """Raised when Weaviate database operations fail."""

class LLMGenerationError(AppBaseException):
    """Raised when Groq or fallback LLM inference fails."""

class RerankingError(AppBaseException):
    """Raised when Cross-Encoder reranking fails."""

class TimeoutError(AppBaseException):
    """Raised when an operation times out."""


def retry_on_exception(
    exceptions: Tuple[Type[BaseException], ...],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    Decorator that retries a function with exponential backoff on specified exceptions.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    app_logger.warning(
                        f"Attempt {attempt}/{max_retries} for function '{func.__name__}' "
                        f"failed with error: {str(e)}. Retrying in {delay:.2f} seconds..."
                    )
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay *= backoff_factor
            
            app_logger.error(
                f"Function '{func.__name__}' failed after {max_retries} attempts. "
                f"Last error: {str(last_exception)}"
            )
            raise last_exception
        return wrapper
    return decorator
