"""API v1 routes."""
from app.api.v1 import health, auth, users, documents, chat, audit

__all__ = ["health", "auth", "users", "documents", "chat", "audit"]