from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    app_name: str = "PolicyLens"
    environment: str = "development"
    log_level: str = "INFO"

    # Legal & Safety
    disclaimer: str = (
        "Informational only, not insurance advice. Refer to the official policy document."
    )

    # LLM Settings
    llm_provider: Literal["gemini", "mock", "openai", "anthropic"] = "gemini"
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "gemini-3.5-flash-lite"
    llm_temperature: float = 0.0

    # Embeddings & Reranker
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_device: str = "cpu"
    reranker_enabled: bool = False
    reranker_candidate_pool: int = 30

    # Qdrant Vector DB
    qdrant_mode: Literal["server", "local"] = "local"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_local_path: str = "./qdrant_storage"
    qdrant_collection_name: str = "policy_chunks"

    # SQLite Database
    database_url: str = "sqlite+aiosqlite:///policylens.db"

    # Retrieval & Ranking Parameters
    top_k_dense: int = 15
    top_k_bm25: int = 15
    top_k_rerank: int = 5
    rrf_k: int = 60
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    abstention_threshold: float = 0.35

    # Observability
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "data"
    eval_dir: Path = base_dir / "eval"


settings = Settings()
