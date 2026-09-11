"""Audit logging service."""
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import UUID
from uuid import UUID as UUIDType

from app.models.audit import AuditLog
from app.models.user import User


async def audit_log(
    db: AsyncSession,
    request: Optional[object] = None,
    action: str = "",
    user_id: Optional[UUIDType] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[UUIDType] = None,
    details: Optional[Dict[str, Any]] = None,
    success: bool = True,
    error_message: Optional[str] = None,
) -> AuditLog:
    """Create an audit log entry."""
    # Extract request info if available
    ip_address = None
    user_agent = None
    request_id = None

    if request and hasattr(request, "client") and request.client:
        ip_address = request.client.host
    if request and hasattr(request, "headers"):
        user_agent = request.headers.get("user-agent")
    if request and hasattr(request, "state") and hasattr(request.state, "request_id"):
        request_id = request.state.request_id

    audit_entry = AuditLog(
        action=action,
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        request_id=request_id,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        success=success,
        error_message=error_message,
    )

    db.add(audit_entry)
    await db.flush()
    return audit_entry