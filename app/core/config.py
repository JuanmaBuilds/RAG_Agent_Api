import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # MongoDB Atlas Configuration
    mongodb_atlas_uri: str
    mongodb_database_name: str = "rag_database"
    mongodb_collection_name: str = "documents"
    
    # OpenAI Configuration
    openai_api_key: str
    
    # LangSmith Configuration
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "rag-agent-api"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_tracing_enabled: bool = True
    
    # Application Configuration
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = True
    
    # Vector Database Configuration
    vector_dimension: int = 1536  # text-embedding-3-small uses 1536 dimensions
    embedding_model: str = "text-embedding-3-small"
    
    # Document Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_separators: list = ["\n\n", "\n", " ", ""]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings() 