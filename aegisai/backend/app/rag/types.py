"""Type definitions for the RAG pipeline."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from uuid import UUID
from dataclasses import dataclass, field


@dataclass
class SearchHit:
    """A search result from Qdrant with relevance score."""
    id: str
    score: float
    payload: Dict[str, Any]
    document_id: UUID
    filename: str
    page_number: Optional[int]
    chunk_text: str
    classification: str
    department: Optional[str]
    # Enhanced citation metadata
    section_title: Optional[str] = None
    chunk_index: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    document_title: Optional[str] = None
    author: Optional[str] = None


@dataclass
class RAGQuery:
    """Query object for RAG pipeline."""
    question: str
    conversation_id: Optional[UUID] = None
    user_role: Optional[str] = None
    user_id: Optional[UUID] = None
    department: Optional[str] = None
    request_id: Optional[str] = None
    intent: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    memory_summary: Optional[str] = None
    # Hybrid RAG extensions (opt-in, backward compatible)
    enable_hybrid: Optional[bool] = None  # None = use global HYBRID_RAG_ENABLED
    hybrid_top_k: Optional[int] = None


@dataclass
class RAGResult:
    """Result from RAG pipeline execution."""
    answer: str
    sources: List[SearchHit]
    model_used: str
    processing_time_ms: int
    tokens_used: Optional[int] = None
    question: str = ""
    # Hybrid RAG extensions
    confidence: Optional[float] = None
    pipeline: str = "legacy"  # "legacy" | "hybrid"
    retrieval_debug: Optional[Dict[str, Any]] = None

    def has_sources(self) -> bool:
        """Check if the result has source citations."""
        return len(self.sources) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "sources": [
                {
                    "document_id": str(s.document_id),
                    "filename": s.filename,
                    "page_number": s.page_number,
                    "chunk_text": s.chunk_text[:200] + "..." if len(s.chunk_text) > 200 else s.chunk_text,
                    "classification": s.classification,
                }
                for s in self.sources
            ],
            "model_used": self.model_used,
            "processing_time_ms": self.processing_time_ms,
        }
