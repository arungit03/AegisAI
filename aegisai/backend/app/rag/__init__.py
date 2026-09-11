"""RAG pipeline for AegisAI - local document processing and retrieval."""
from .ingestion import ingest_document, IngestionResult
from .chunking import chunk_text, ChunkResult, ChunkMetadata
from .embeddings import get_embedding_model, embed_text, embed_texts, EmbeddingProvider, set_embedding_provider
from .qdrant import QdrantManager, PointStruct
from .types import RAGQuery, RAGResult, SearchHit
from .service import RAGService

__all__ = [
    "ingest_document",
    "IngestionResult",
    "chunk_text",
    "ChunkResult",
    "ChunkMetadata",
    "EmbeddingProvider",
    "set_embedding_provider",
    "get_embedding_model",
    "embed_text",
    "embed_texts",
    "QdrantManager",
    "SearchHit",
    "PointStruct",
    "RAGQuery",
    "RAGResult",
    "RAGSearchHit",
    "RAGService",
]