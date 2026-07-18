import json
import re
from typing import Generator, List, Dict, Any, Type, TypeVar, Union
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langsmith import traceable
from config.settings import settings
from custom_logging.logger import app_logger
from custom_logging.error_handler import LLMGenerationError, retry_on_exception
from utils.metrics import MetricsRegistry

T = TypeVar("T", bound=BaseModel)

class GroqLLMClient:
    """Production-ready client wrapper for Groq inference engines with LangSmith tracing."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.groq_api_key
        if not self.api_key:
            app_logger.warning("GROQ_API_KEY is not defined. LLM steps will fail.")
            
        self.model = model or settings.get("llm", "groq", "default_model", default="llama-3.3-70b-versatile")
        self.temperature = settings.get("llm", "groq", "temperature", default=0.0)
        self.max_tokens = settings.get("llm", "groq", "max_tokens", default=2048)
        self.timeout = settings.get("llm", "groq", "timeout", default=30.0)
        
        # Use LangChain ChatGroq for automatic LangSmith tracing
        self.llm = ChatGroq(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )

    @traceable(name="Groq Generation")
    @retry_on_exception(exceptions=(Exception,), max_retries=3)
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """Generates conversational responses from Groq models with streaming support."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        app_logger.debug(f"Calling Groq completion (model={self.model}, temp={temp}, stream={stream})")
        
        try:
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            # Update temperature if provided
            if temperature is not None:
                self.llm.temperature = temperature
            if max_tokens is not None:
                self.llm.max_tokens = max_tokens
            
            # Use LangChain ChatGroq for automatic LangSmith tracing
            if stream:
                response = self.llm.stream(langchain_messages)
                def gen():
                    for chunk in response:
                        content = chunk.content
                        if content:
                            yield content
                return gen()
            else:
                response = self.llm.invoke(langchain_messages)
                usage = response.response_metadata.get("token_usage", {})
                if usage:
                    MetricsRegistry.record_tokens(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
                return response.content
                
        except Exception as e:
            app_logger.error(f"Groq API generation failure: {str(e)}")
            raise LLMGenerationError(f"Groq completion failed: {str(e)}")

    @traceable(name="Groq Structured Generation")
    @retry_on_exception(exceptions=(Exception,), max_retries=3)
    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.0,
        max_tokens: int = None
    ) -> T:
        """Requests structured JSON output from Groq and parses it into the target Pydantic model."""
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        app_logger.debug(f"Calling Groq structured generation (model={self.model}, response_model={response_model.__name__})")
        
        # Inject JSON instruction to system message
        sys_instructions = (
            f" You MUST respond strictly with a valid JSON matching this schema: {json.dumps(response_model.model_json_schema())}. "
            "Do not output markdown code blocks or reasoning text outside the JSON structure."
        )
        
        modified_messages = []
        for msg in messages:
            if msg["role"] == "system":
                modified_messages.append({"role": "system", "content": msg["content"] + sys_instructions})
            else:
                modified_messages.append(msg)
                
        if not any(msg["role"] == "system" for msg in modified_messages):
            modified_messages.insert(0, {"role": "system", "content": sys_instructions})

        try:
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in modified_messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            # Update temperature if provided
            if temperature is not None:
                self.llm.temperature = temperature
            if max_tokens is not None:
                self.llm.max_tokens = max_tokens
            
            # Use LangChain ChatGroq with JSON mode
            self.llm.model_kwargs = {"response_format": {"type": "json_object"}}
            response = self.llm.invoke(langchain_messages)
            self.llm.model_kwargs = {}  # Reset model_kwargs
            
            usage = response.response_metadata.get("token_usage", {})
            if usage:
                MetricsRegistry.record_tokens(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

            raw_content = response.content.strip()
            
            # Remove any trailing/leading backticks
            clean_content = re.sub(r"^```json\s*", "", raw_content, flags=re.IGNORECASE)
            clean_content = re.sub(r"\s*```$", "", clean_content, flags=re.IGNORECASE)
            
            parsed_json = json.loads(clean_content)
            return response_model.model_validate(parsed_json)
            
        except Exception as e:
            app_logger.error(f"Groq structured generation failed: {str(e)}")
            raise LLMGenerationError(f"Failed to generate structured answer: {str(e)}")
