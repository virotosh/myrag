"""
Core configuration settings for the Chat RAG application.
"""
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Database
    database_url: str = "postgresql://admin:admin@localhost:5432/chatrag_db"
    
    # OpenAI
    openai_api_key: str
    # LLM configuration
    llm_model: str = "gpt-5.1"
    llm_temperature: float = 1
    llm_max_tokens: int = 1000
    
    # Vector Database
    chroma_persist_directory: str = "./chroma_db"
    embedding_model: str = "text-embedding-ada-002"
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # File Storage
    upload_dir: str = "./uploads"
    max_file_size: int = 1048576  # 1MB (1 * 1024 * 1024)
    
    # CORS
    allowed_origins: List[str]
    
    # Development
    debug: bool = True
    log_level: str = "INFO"
    
    # Error Logging
    error_logging: int = 1  # 1 = local file, 2 = Sentry
    error_log_dir: str = "./logs"
    sentry_dsn: str = ""
    
    # API
    api_v1_str: str = ""
    project_name: str = "RAG agent"
    port: int = 7000
    
    @field_validator("allowed_origins", mode="before")
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            # Handle both comma-separated and JSON array formats
            if v.startswith('[') and v.endswith(']'):
                # Handle JSON array format
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            # Handle comma-separated string
            return [i.strip() for i in v.split(",")]
        return v
    
    @field_validator("upload_dir")
    def create_upload_dir(cls, v):
        """Create upload directory if it doesn't exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("chroma_persist_directory")
    def create_chroma_dir(cls, v):
        """Create Chroma directory if it doesn't exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator("error_log_dir")
    def create_error_log_dir(cls, v):
        """Create error log directory if it doesn't exist."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    # Pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=".env",
    )


# Global settings instance
settings = Settings()
