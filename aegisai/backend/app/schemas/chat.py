"""Chat/Conversation schemas."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID

from app.models.chat import MessageRole


class SourceCitation(BaseModel):
    """Source citation for RAG responses."""
    document_id: UUID
    document_filename: str
    document_title: Optional[str] = None
    page_number: Optional[int] = None
    chunk_text: str
    similarity_score: float
    # Enhanced citation metadata
    section_title: Optional[str] = None
    chunk_index: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    author: Optional[str] = None


class ConversationBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)


class ConversationCreate(ConversationBase):
    pass


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    is_archived: Optional[bool] = None


class ConversationResponse(ConversationBase):
    id: UUID
    user_id: UUID
    is_archived: bool
    message_count: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse]
    total: int
    page: int
    page_size: int


class MessageBase(BaseModel):
    role: MessageRole
    content: str


class MessageCreate(MessageBase):
    conversation_id: UUID


class MessageResponse(MessageBase):
    id: UUID
    conversation_id: UUID
    sources: Optional[List[Dict[str, Any]]] = None
    token_count: Optional[int] = None
    model_used: Optional[str] = None
    processing_time_ms: Optional[int] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[UUID] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=8192)
    stream: bool = False
    # Opt-in hybrid retrieval; None = use server default (HYBRID_RAG_ENABLED)
    enable_hybrid: Optional[bool] = None


class ChatResponse(BaseModel):
    message: str
    conversation_id: UUID
    sources: List[SourceCitation] = []
    token_count: Optional[int] = None
    processing_time_ms: Optional[int] = None
    # Hybrid observability (optional, backward compatible)
    confidence: Optional[float] = None
    pipeline: Optional[str] = None  # "legacy" | "hybrid"