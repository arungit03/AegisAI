"""Structured logging configuration for AegisAI.

Provides consistent logging across all modules with:
- JSON format for production
- Human-readable format for development
- Request ID tracking for tracing
- Audit event logging for security events
- Sensitive data protection

Security:
- Full user queries, AI responses, and document contents are NOT logged
- Only metadata, IDs, counts, and timing are logged by default
- Sensitive data logging requires explicit LOG_SENSITIVE_DATA=true in config
- Filenames are sanitized to avoid leaking sensitive information
"""
import logging
import structlog
import hashlib
from typing import Any, Dict, Optional

from app.core.config import settings


# Maximum query preview length when sensitive data logging is enabled
SENSITIVE_PREVIEW_LENGTH = 20


def _should_log_sensitive() -> bool:
    """Check if sensitive data should be logged (debug/dev only)."""
    return settings.DEBUG and settings.LOG_SENSITIVE_DATA


def _sanitize_query(query: Optional[str]) -> Optional[str]:
    """Sanitize query text for logging.

    When sensitive logging is disabled (default), returns a fingerprint
    (hash) instead of the actual query text.

    Args:
        query: The user query text.

    Returns:
        Query fingerprint when sensitive logging is off,
        brief preview when sensitive logging is enabled.
    """
    if not query:
        return None

    if _should_log_sensitive():
        # Only show a very short preview in sensitive mode
        return query[:SENSITIVE_PREVIEW_LENGTH] + "..." if len(query) > SENSITIVE_PREVIEW_LENGTH else query
    else:
        # Return a content fingerprint instead of the actual query
        return hashlib.sha256(query.encode()).hexdigest()[:16]


def _sanitize_filename(filename: Optional[str]) -> Optional[str]:
    """Sanitize filename for logging to avoid leaking sensitive names.

    Args:
        filename: The original filename.

    Returns:
        Sanitized filename (extension preserved, name hashed if sensitive logging disabled).
    """
    if not filename:
        return None

    if _should_log_sensitive():
        return filename

    # Hash the filename but preserve the extension for context
    import os
    name, ext = os.path.splitext(filename)
    name_hash = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"{name_hash}{ext}"


def _sanitize_error(error: Optional[str]) -> Optional[str]:
    """Sanitize error messages to avoid leaking internal details.

    Args:
        error: The raw error message.

    Returns:
        Sanitized error message.
    """
    if not error:
        return None

    if _should_log_sensitive():
        return error

    import re

    sanitized = error
    # Redact connection strings with credentials (must be done before URL redaction)
    # Pattern: protocol://user:password@host:port/path
    sanitized = re.sub(
        r'[a-zA-Z][a-zA-Z0-9+.-]*://[^:@\s]+:[^:@\s]+@[^\s]+',
        '[REDACTED_CONNECTION_STRING]',
        sanitized,
    )
    # Redact Bearer tokens and JWTs (JWT has 3 base64 parts separated by dots)
    sanitized = re.sub(
        r'\b[Bb]earer\s+[A-Za-z0-9_\-.=]+',
        '[REDACTED_BEARER_TOKEN]',
        sanitized,
    )
    # Redact standalone JWT tokens (3 base64-encoded parts separated by dots)
    sanitized = re.sub(
        r'\b[A-Za-z0-9_\-]+\.(?:[A-Za-z0-9_\-]+\.)+[A-Za-z0-9_\-]+',
        '[REDACTED_JWT]',
        sanitized,
    )
    # Redact URLs (protocol://host:port/path) - must be before path redaction
    sanitized = re.sub(r'https?://[^\s]+', '[REDACTED_URL]', sanitized)
    sanitized = re.sub(r'[a-zA-Z]+://[^\s]+', '[REDACTED_URL]', sanitized)
    # Redact file paths (Unix and Windows)
    sanitized = re.sub(r'/[a-zA-Z0-9_./-]+', '[REDACTED_PATH]', sanitized)
    sanitized = re.sub(r'[A-Za-z]:\\[a-zA-Z0-9_\\\\.-]+', '[REDACTED_PATH]', sanitized)
    # Redact any port numbers that might remain (:8443, :3306, etc.)
    sanitized = re.sub(r':\d{4,5}\b', '[REDACTED_PORT]', sanitized)
    # Redact API keys, tokens, passwords in key=value or "key: value" patterns
    sanitized = re.sub(
        r'(api_key|token|password|secret|key)[\s:=]+[^\s&\s]+',
        r'\1=[REDACTED]',
        sanitized,
        flags=re.IGNORECASE,
    )

    return sanitized


def _sanitize_details(details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitize details dict to remove sensitive values.

    Args:
        details: Dictionary of detail key-value pairs.

    Returns:
        Sanitized dictionary.
    """
    if not details:
        return {}

    # Keys that should never be logged (exact match or contains pattern)
    SENSITIVE_KEY_PATTERNS = (
        "password", "token", "api_key", "secret", "authorization",
        "query", "answer", "response", "prompt", "text", "content",
        "embeddings", "vector", "document_text", "chunk_text",
        "access_token", "refresh_token", "jwt", "salary", "ssn",
        "resource", "filename", "file_path", "path",
    )

    sanitized = {}
    for key, value in details.items():
        key_lower = key.lower()
        # Check if key contains any sensitive pattern
        if any(pattern in key_lower for pattern in SENSITIVE_KEY_PATTERNS):
            continue  # Skip sensitive keys entirely
        if isinstance(value, str) and len(value) > 200:
            sanitized[key] = value[:200] + "..."
        else:
            sanitized[key] = value

    return sanitized


def configure_logging() -> structlog.BoundLogger:
    """Configure structlog with appropriate processors based on environment.

    Uses JSONRenderer in production for machine-parseable logs,
    ConsoleRenderer in development for readability.

    Security:
        - Debug logging is disabled in production (ENVIRONMENT=production)
        - Sensitive data is never logged unless LOG_SENSITIVE_DATA=true AND DEBUG=true
    """

    # Determine effective log level
    # Force INFO minimum in production regardless of config
    if settings.ENVIRONMENT == "production":
        effective_level = logging.INFO
    else:
        effective_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Use JSON renderer in production, console in development
    if settings.LOG_FORMAT == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set the root logging level
    logging.basicConfig(
        level=effective_level,
        format="%(message)s" if settings.LOG_FORMAT == "json" else "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = structlog.get_logger()

    logger.info(
        "logging_configured",
        level=logging.getLevelName(effective_level),
        format=settings.LOG_FORMAT,
        environment=settings.ENVIRONMENT,
        sensitive_data_logging=_should_log_sensitive(),
    )

    return logger


def get_logger(name: str = "aegisai") -> structlog.BoundLogger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically module path).

    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)


def log_audit_event(
    logger: structlog.BoundLogger,
    event_type: str,
    user_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a security audit event.

    Audit logs record security-relevant events for compliance and investigation.
    Sensitive data (queries, document contents, passwords, tokens) must NEVER
    be included in audit events.

    Safe to log:
        - user_id, resource_type, resource_id, request_id
        - Counts, timestamps, status codes
        - Non-sensitive metadata (classification level, department)

    Args:
        logger: The logger instance.
        event_type: The type of event (e.g., "document_accessed", "chat_query").
        user_id: ID of the user performing the action.
        resource_type: Type of resource accessed (e.g., "document", "conversation").
        resource_id: ID of the resource.
        request_id: Request ID for tracing.
        details: Additional context details (sanitized before logging).
    """
    sanitized_details = _sanitize_details(details)
    logger.info(
        "AUDIT_EVENT",
        event_type=event_type,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        request_id=request_id,
        details=sanitized_details,
    )


def log_rag_event(
    logger: structlog.BoundLogger,
    event_type: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    user_role: Optional[str] = None,
    department: Optional[str] = None,
    query: Optional[str] = None,
    sources_count: Optional[int] = None,
    answer_length: Optional[int] = None,
    processing_time_ms: Optional[int] = None,
    error: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a RAG pipeline event.

    Security:
        - The full user query is NEVER logged. Only a fingerprint (hash) is stored.
        - AI responses are never logged. Only their length is recorded.
        - Document contents and embeddings are never logged.

    Safe metadata includes: user_id, role, department, request_id,
    sources_count, answer_length, processing_time_ms, error category.

    Args:
        logger: The logger instance.
        event_type: The type of event (e.g., "query", "ingestion", "error").
        user_id: ID of the user who made the query.
        request_id: Correlation ID for tracing the request.
        user_role: Role of the user making the query.
        department: User's department.
        query: The user query text (will be hashed/sanitized).
        sources_count: Number of source documents retrieved.
        answer_length: Length of the answer in characters.
        processing_time_ms: Processing time in milliseconds.
        error: Error category/message if any (sanitized).
        details: Additional context (sanitized).
    """
    sanitized_details = _sanitize_details(details)
    query_fingerprint = _sanitize_query(query)

    logger.info(
        "RAG_EVENT",
        event_type=event_type,
        user_id=user_id,
        request_id=request_id,
        user_role=user_role,
        department=department,
        query_fingerprint=query_fingerprint,
        sources_count=sources_count,
        answer_length=answer_length,
        processing_time_ms=processing_time_ms,
        error=_sanitize_error(error),
        details=sanitized_details,
    )


def log_document_ingestion_event(
    logger: structlog.BoundLogger,
    document_id: Optional[str] = None,
    filename: Optional[str] = None,
    file_type: Optional[str] = None,
    file_size: Optional[int] = None,
    page_count: Optional[int] = None,
    chunk_count: Optional[int] = None,
    classification: Optional[str] = None,
    department: Optional[str] = None,
    status: str = "started",
    processing_time_ms: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Log a document ingestion event.

    Security:
        - Document contents and chunk texts are NEVER logged.
        - Filenames are sanitized (hashed) to avoid leaking sensitive names.
        - Only metadata: IDs, type, size, page count, chunk count, timing, status.

    Args:
        logger: The logger instance.
        document_id: UUID of the document.
        filename: Name of the file being processed (sanitized).
        file_type: File type/extension.
        file_size: Size of the file in bytes.
        page_count: Number of pages extracted.
        chunk_count: Number of chunks produced.
        classification: Document classification level.
        department: Document department.
        status: Processing status (e.g., "started", "processed", "failed").
        processing_time_ms: Processing duration in milliseconds.
        error: Error category if any (sanitized).
    """
    logger.info(
        "DOCUMENT_INGESTION_EVENT",
        document_id=document_id,
        filename=_sanitize_filename(filename),
        file_type=file_type,
        file_size=file_size,
        page_count=page_count,
        chunk_count=chunk_count,
        classification=classification,
        department=department,
        status=status,
        processing_time_ms=processing_time_ms,
        error=_sanitize_error(error),
    )


def log_security_event(
    logger: structlog.BoundLogger,
    event_type: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    success: bool = True,
    failure_reason: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a security-relevant event.

    For events like login attempts, permission denials, role changes, etc.

    Security:
        - Passwords, tokens, and secrets are NEVER logged.
        - Error messages are sanitized to avoid leaking internals.

    Args:
        logger: The logger instance.
        event_type: The type of security event (e.g., "login_success", "permission_denied").
        user_id: ID of the user involved.
        request_id: Correlation ID for tracing.
        ip_address: Client IP address.
        user_agent: Client user agent string (truncated).
        resource_type: Type of resource (e.g., "document", "conversation").
        resource_id: ID of the resource.
        success: Whether the operation succeeded.
        failure_reason: Reason for failure if applicable (sanitized).
        details: Additional context (sanitized).
    """
    sanitized_details = _sanitize_details(details)
    truncated_ua = (user_agent[:100] + "...") if user_agent and len(user_agent) > 100 else user_agent

    logger.info(
        "SECURITY_EVENT",
        event_type=event_type,
        user_id=user_id,
        request_id=request_id,
        ip_address=ip_address,
        user_agent=truncated_ua,
        resource_type=resource_type,
        resource_id=resource_id,
        success=success,
        failure_reason=_sanitize_error(failure_reason),
        details=sanitized_details,
    )
