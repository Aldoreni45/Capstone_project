import os
import yaml
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AppSettings(BaseModel):
    # API Keys loaded from environment variables
    weaviate_url: str = Field(default_factory=lambda: os.getenv("WEAVIATE_URL", ""))
    weaviate_api_key: str = Field(default_factory=lambda: os.getenv("WEAVIATE_API_KEY", ""))
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    langsmith_api_key: str = Field(default_factory=lambda: os.getenv("LANGCHAIN_API_KEY", ""))
    langchain_tracing_v2: bool = Field(default_factory=lambda: os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true")
    langchain_project: str = Field(default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "research-paper-answer-bot"))
    
    # Configuration options loaded from config.yaml
    config_data: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def load(cls) -> "AppSettings":
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "config" / "config.yaml"
        
        config_data = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        
        return cls(config_data=config_data)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Helper to get a nested key from config.yaml."""
        val = self.config_data
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

# Global settings instance
settings = AppSettings.load()
