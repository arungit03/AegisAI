"""Document schemas."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID

from app.models.document import DocumentStatus
from app.models.user import DocumentClassification


class DocumentPermissionBase(BaseModel):
    role_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    can_read: bool = True
    can_write: bool = False
    can_delete: bool = False


class DocumentPermissionCreate(DocumentPermissionBase):
    document_id: UUID


class DocumentPermissionUpdate(BaseModel):
    can_read: Optional[bool] = None
    can_write: Optional[bool] = None
    can_delete: Optional[bool] = None


class DocumentPermissionResponse(DocumentPermissionBase):
    id: UUID
    document_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentBase(BaseModel):
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    classification: DocumentClassification = DocumentClassification.PUBLIC_INTERNAL
    department: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = None


class DocumentCreate(DocumentBase):
    pass


class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    classification: Optional[DocumentClassification] = None
    department: Optional[str] = Field(None, max_length=100)
    tags: Optional[List[str]] = None


class DocumentResponse(DocumentBase):
    id: UUID
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    extension: str
    version: int
    status: DocumentStatus
    page_count: Optional[int]
    chunk_count: Optional[int]
    error_message: Optional[str]
    processed_at: Optional[datetime]
    uploaded_by_id: UUID
    department_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime
    permissions: List[DocumentPermissionResponse] = []

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class DocumentReindexRequest(BaseModel):
    document_ids: Optional[List[UUID]] = None
    force: bool = False