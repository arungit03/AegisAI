"""Audit log schemas."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


class AuditLogResponse(BaseModel):
    id: UUID
    action: str
    user_id: Optional[UUID]
    ip_address: Optional[str]
    user_agent: Optional[str]
    request_id: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[UUID]
    details: Optional[Dict[str, Any]]
    success: bool
    error_message: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AuditLogListResponse(BaseModel):
    logs: List[AuditLogResponse]
    total: int
    page: int
    page_size: int