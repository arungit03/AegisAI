"""Pydantic schemas package."""
from app.schemas.user import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserListResponse,
    RoleCreate,
    RoleUpdate,
    RoleResponse,
    DepartmentCreate,
    DepartmentUpdate,
    DepartmentResponse,
    Token,
    TokenData,
    LoginRequest,
    RefreshTokenRequest,
)
from app.schemas.document import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    DocumentListResponse,
    DocumentPermissionCreate,
    DocumentPermissionUpdate,
    DocumentPermissionResponse,
    DocumentReindexRequest,
)
from app.schemas.chat import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationListResponse,
    MessageCreate,
    MessageResponse,
    ChatRequest,
    ChatResponse,
    SourceCitation,
)
from app.schemas.audit import (
    AuditLogResponse,
    AuditLogListResponse,
)
from app.schemas.health import (
    HealthResponse,
    SystemStatusResponse,
)

__all__ = [
    # User
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserListResponse",
    "RoleCreate",
    "RoleUpdate",
    "RoleResponse",
    "DepartmentCreate",
    "DepartmentUpdate",
    "DepartmentResponse",
    "Token",
    "TokenData",
    "LoginRequest",
    "RefreshTokenRequest",
    # Document
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentResponse",
    "DocumentListResponse",
    "DocumentPermissionCreate",
    "DocumentPermissionUpdate",
    "DocumentPermissionResponse",
    "DocumentReindexRequest",
    # Chat
    "ConversationCreate",
    "ConversationUpdate",
    "ConversationResponse",
    "ConversationListResponse",
    "MessageCreate",
    "MessageResponse",
    "ChatRequest",
    "ChatResponse",
    "SourceCitation",
    # Audit
    "AuditLogResponse",
    "AuditLogListResponse",
    # Health
    "HealthResponse",
    "SystemStatusResponse",
]