"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Application
    APP_NAME: str = "AegisAI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://aegisai:aegisai@localhost:5432/aegisai"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Qdrant Vector Database
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "aegisai_documents"

    # Ollama LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    # Embeddings
    EMBEDDING_DIMENSION: int = 768
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5

    # Hybrid RAG — all backward compatible, new pipeline is opt-in
    HYBRID_RAG_ENABLED: bool = False
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    VECTOR_TOP_K: int = 20
    BM25_TOP_K: int = 20
    FUSION_TOP_K: int = 12
    RRF_K: int = 60
    RERANK_TOP_K: int = 5
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MAX_CONTEXT_CHARS: int = 12000
    CITATION_VALIDATION_ENABLED: bool = True

    # Phase 2 — Advanced Multi-Document RAG (all opt-in, backward compatible)
    PHASE2_ENABLED: bool = False
    PHASE2_MULTI_QUERY_ENABLED: bool = False
    PHASE2_QUERY_REWRITE_ENABLED: bool = False
    PHASE2_QUERY_EXPANSION_ENABLED: bool = False
    PHASE2_DECOMPOSITION_ENABLED: bool = False
    PHASE2_RERANK_VISIBILITY_ENABLED: bool = True
    PHASE2_DIVERSITY_ENABLED: bool = False
    PHASE2_DIVERSITY_MAX_PER_DOC: int = 3
    PHASE2_ADJACENT_EXPANSION_ENABLED: bool = False
    PHASE2_ADJACENT_THRESHOLD: float = 0.7
    PHASE2_VERSION_HANDLING_ENABLED: bool = False
    PHASE2_CONFLICT_DETECTION_ENABLED: bool = True
    PHASE2_COVERAGE_ENABLED: bool = True
    PHASE2_PROMPT_DEFENSE_ENABLED: bool = True
    PHASE2_TABLE_HANDLING_ENABLED: bool = True

    # File Upload
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt", ".md"]
    UPLOAD_DIR: str = "./uploads"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_SENSITIVE_DATA: bool = False


settings = Settings()