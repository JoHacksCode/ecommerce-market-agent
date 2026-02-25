"""Application settings loaded from environment variables / .env file."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM — OpenRouter is OpenAI-compatible, swap for any provider
    openrouter_api_key: str = Field(default="sk-or-placeholder", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = Field(default="deepseek/deepseek-chat", alias="MODEL_NAME")

    # Server
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    # Agent tuning
    agent_recursion_limit: int = 25

    model_config = {"env_file": ".env", "populate_by_name": True}


settings = Settings()
