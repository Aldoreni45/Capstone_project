from typing import List, Dict, Any, Optional
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_models.responses import StructuredAnswer
from config.settings import settings
from custom_logging.logger import app_logger
from custom_logging.error_handler import LLMGenerationError
from prompts.templates import SYSTEM_PROMPT

class PydanticAIEngine:
    """Orchestrates LLM runs using the PydanticAI Agent framework with type guarantees."""

    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or settings.groq_api_key
        self.model_name = model_name or settings.get("llm", "groq", "default_model", default="llama-3.3-70b-versatile")
        
        if not self.api_key:
            app_logger.warning("GROQ_API_KEY is not defined. PydanticAI Agent calls will fail.")

        # Groq is OpenAI compatible, so we configure OpenAIModel with Groq endpoints
        try:
            self.model = OpenAIModel(
                model_name=self.model_name,
                base_url="https://api.groq.com/openai/v1",
                api_key=self.api_key
            )
            
            # Setup PydanticAI Agent
            self.agent = Agent(
                model=self.model,
                result_type=StructuredAnswer,
                system_prompt=SYSTEM_PROMPT
            )
            app_logger.info(f"PydanticAI agent initialized successfully with model '{self.model_name}'.")
        except Exception as e:
            app_logger.error(f"Failed to initialize PydanticAI agent: {str(e)}")
            raise LLMGenerationError(f"PydanticAI initialization error: {str(e)}")

    def run_agent(self, prompt: str) -> StructuredAnswer:
        """Runs the PydanticAI agent synchronously and returns a validated StructuredAnswer model."""
        app_logger.info("Executing PydanticAI agent structured query...")
        try:
            result = self.agent.run_sync(prompt)
            # The result.data will already be parsed and validated as a StructuredAnswer instance
            app_logger.info("PydanticAI agent executed and validated output successfully.")
            return result.data
        except Exception as e:
            app_logger.error(f"PydanticAI agent execution failed: {str(e)}")
            raise LLMGenerationError(f"PydanticAI agent run failed: {str(e)}")
