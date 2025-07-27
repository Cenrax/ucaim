"""CAIM framework configuration."""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path


class CAIMConfig(BaseModel):
    """Configuration for CAIM framework."""
    
    max_memory_size: int = Field(default=10000, description="Maximum number of memories to store")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model for vector representations")
    vector_db_type: str = Field(default="chromadb", description="Vector database type")
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    
    memory_decay_factor: float = Field(default=0.95, description="Factor for memory importance decay")
    consolidation_threshold: float = Field(default=0.8, description="Threshold for memory consolidation")
    retrieval_top_k: int = Field(default=5, description="Number of top memories to retrieve")
    
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    huggingface_api_token: Optional[str] = Field(default=None, description="HuggingFace API token")
    
    log_level: str = Field(default="INFO", description="Logging level")
    cache_dir: str = Field(default="./cache", description="Cache directory")
    
    model_class_name: str = Field(default="CAIMAgent", description="Default agent class")
    
    @classmethod
    def from_env(cls) -> "CAIMConfig":
        """Create configuration from environment variables."""
        return cls(
            max_memory_size=int(os.getenv("CAIM_MAX_MEMORY_SIZE", 10000)),
            embedding_model=os.getenv("CAIM_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            vector_db_type=os.getenv("CAIM_VECTOR_DB_TYPE", "chromadb"),
            database_url=os.getenv("DATABASE_URL"),
            redis_url=os.getenv("REDIS_URL"),
            memory_decay_factor=float(os.getenv("CAIM_MEMORY_DECAY_FACTOR", 0.95)),
            consolidation_threshold=float(os.getenv("CAIM_CONSOLIDATION_THRESHOLD", 0.8)),
            retrieval_top_k=int(os.getenv("CAIM_RETRIEVAL_TOP_K", 5)),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            huggingface_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
            log_level=os.getenv("CAIM_LOG_LEVEL", "INFO"),
            cache_dir=os.getenv("CAIM_CACHE_DIR", "./cache"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
    
    @classmethod
    def load(cls, path: str) -> "CAIMConfig":
        """Load configuration from file."""
        with open(path, "r") as f:
            data = f.read()
        return cls.model_validate_json(data)