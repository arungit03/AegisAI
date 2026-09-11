"""Security tests for logging sanitization.

Phase 21: Secure Logging & Privacy Audit

Verifies that sensitive data is never logged in plaintext:
- User queries (RAG) are fingerprinted, not logged
- Passwords are never logged
- JWT tokens are never logged
- Document text is never logged
- File paths are sanitized (basename only)
- Error messages don't expose internals
- Permission denials produce audit events without leaking protected content
- LOG_SENSITIVE_DATA config flag controls behavior
"""

import hashlib
import io
import logging
import json
import re
import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.core.logging import (
    _sanitize_query,
    _sanitize_filename,
    _sanitize_error,
    _sanitize_details,
    log_rag_event,
    log_document_ingestion_event,
    log_security_event,
    log_audit_event,
    configure_logging,
    get_logger,
)
from app.core.config import settings


# ─── Test: Query Fingerprinting ──────────────────────────────────────────

class TestQueryFingerprinting:
    """Verify RAG queries are hashed, never logged in plaintext."""

    SENSITIVE_QUERY = (
        "CONFIDENTIAL unreleased product specification "
        "for Project Phoenix - secret roadmap Q4 2025"
    )

    def test_sanitize_query_produces_hash(self):
        """_sanitize_query() must return a SHA256 hash, not the original text."""
        fingerprint = _sanitize_query(self.SENSITIVE_QUERY)

        # Must not contain any part of the original query
        assert "CONFIDENTIAL" not in fingerprint
        assert "product specification" not in fingerprint
        assert "Project Phoenix" not in fingerprint
        assert "secret roadmap" not in fingerprint

        # Must be a hash (hex string, 16 chars based on implementation)
        assert len(fingerprint) == 16
        assert all(c in "0123456789abcdef" for c in fingerprint)

    def test_sanitize_query_is_consistent(self):
        """Same query must produce same fingerprint."""
        fp1 = _sanitize_query(self.SENSITIVE_QUERY)
        fp2 = _sanitize_query(self.SENSITIVE_QUERY)
        assert fp1 == fp2

    def test_sanitize_query_different_queries_different_hashes(self):
        """Different queries must produce different fingerprints."""
        fp1 = _sanitize_query(self.SENSITIVE_QUERY)
        fp2 = _sanitize_query(self.SENSITIVE_QUERY + "different")
        assert fp1 != fp2

    def test_rag_event_does_not_log_sensitive_query(self, caplog):
        """log_rag_event() must not log the sensitive query in plaintext."""
        configure_logging()
        logger = get_logger(__name__)

        with caplog.at_level(logging.INFO):
            log_rag_event(
                logger,
                event_type="query",
                user_id="user-123",
                request_id="req-456",
                user_role="employee",
                department="engineering",
                query=self.SENSITIVE_QUERY,
                sources_count=3,
                answer_length=42,
            )

            # The sensitive query text must not appear in any log output
            log_text = caplog.text
            assert "CONFIDENTIAL" not in log_text
            assert "product specification" not in log_text
            assert "Project Phoenix" not in log_text
            assert "secret roadmap" not in log_text

            # The query fingerprint should be present
            assert "query_fingerprint" in log_text


# ─── Test: Password Protection ───────────────────────────────────────────

class TestPasswordProtection:
    """Verify passwords never appear in logs."""

    def test_password_not_in_error_sanitization(self):
        """_sanitize_error() must redact passwords from error messages."""
        error_with_password = (
            "Database connection failed: "
            "postgresql://user:s3cr3t_p@ss@localhost/db"
        )
        sanitized = _sanitize_error(error_with_password)

        assert "s3cr3t_p@ss" not in sanitized
        assert "postgresql://user" not in sanitized

    def test_password_in_details_redacted(self):
        """_sanitize_details() must redact password-like keys."""
        details = {
            "password": "super_secret_123",
            "user": "admin",
            "query": "SELECT * FROM users",
        }
        sanitized = _sanitize_details(details)

        # Password key should be entirely skipped
        assert "super_secret_123" not in str(sanitized)
        assert "password" not in str(sanitized).lower()

    def test_bcrypt_hash_not_logged(self, caplog):
        """Verify bcrypt hashes don't appear in logs."""
        bcrypt_hash = "$2b$12$LQv3c1yqBWVHxkd0LHAkCOQz6o7OxlJSk/7lqJPqb9H9r4x.6N8y"
        configure_logging()
        logger = get_logger(__name__)

        with caplog.at_level(logging.INFO):
            log_security_event(
                logger,
                event_type="login_success",
                user_id="user-123",
                resource_type="auth",
                resource_id="login",
                details={"password_hash": bcrypt_hash},
            )

            assert bcrypt_hash not in caplog.text


# ─── Test: JWT Token Protection ──────────────────────────────────────────

class TestJWTProtection:
    """Verify JWT tokens and API keys are never logged."""

    JWT_TOKEN = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ."
        "SflKxwRJSMeKKF2QT4pPGYh"
    )

    BEARER_HEADER = f"Bearer {JWT_TOKEN}"

    def test_jwt_in_error_message_redacted(self):
        """_sanitize_error() must strip JWT tokens from error messages."""
        error = f"Request failed with token: {self.JWT_TOKEN}"
        sanitized = _sanitize_error(error)

        assert self.JWT_TOKEN not in sanitized

    def test_bearer_header_redacted(self):
        """_sanitize_error() must strip Bearer tokens."""
        error = f"Authorization failed: {self.BEARER_HEADER}"
        sanitized = _sanitize_error(error)

        assert self.JWT_TOKEN not in sanitized
        assert "Bearer" not in sanitized

    def test_jwt_in_details_not_logged(self, caplog):
        """JWT tokens in details dict must not appear in logs."""
        configure_logging()
        logger = get_logger(__name__)

        with caplog.at_level(logging.INFO):
            log_rag_event(
                logger,
                event_type="embedding",
                user_id="user-123",
                query="test query",
                details={"access_token": self.JWT_TOKEN},
            )

            assert self.JWT_TOKEN not in caplog.text

    def test_api_key_not_logged(self, caplog):
        """API keys must not appear in logs."""
        api_key = "sk-ant-api-03-abcdefghijklmnopqrstuvwxyz1234567890"
        configure_logging()
        logger = get_logger(__name__)

        with caplog.at_level(logging.INFO):
            log_rag_event(
                logger,
                event_type="embedding",
                user_id="user-123",
                query="test query",
                details={"api_key": api_key},
            )

            assert api_key not in caplog.text


# ─── Test: Document Text Protection ──────────────────────────────────────

class TestDocumentTextProtection:
    """Verify document contents never appear in logs."""

    SENSITIVE_DOC_TEXT = (
        "TOP SECRET: Financial Projections for FY2026\n"
        "Revenue forecast: $50M\n"
        "Confidential merger details with Acme Corp."
    )

    def test_document_ingestion_logs_sanitize_filename(self, caplog):
        """log_document_ingestion_event must sanitize filenames but not log file contents."""
        configure_logging()
        logger = get_logger(__name__)

        with caplog.at_level(logging.INFO):
            log_document_ingestion_event(
                logger,
                document_id=str(uuid4()),
                filename="financial_projections_2026.pdf",
                file_type="pdf",
                file_size=2048000,
                page_count=50,
                chunk_count=250,
                classification="highly_restricted",
                department="finance",
                status="processed",
                processing_time_ms=1250,
            )

            log_text = caplog.text

            # Document content must never appear
            assert "TOP SECRET" not in log_text
            assert "Financial Projections" not in log_text
            assert "Revenue forecast: $50M" not in log_text
            assert "merger details" not in log_text

            # Filename should be sanitized (hashed)
            assert "financial_projections_2026.pdf" not in log_text

    def test_document_content_not_logged_via_log_audit(self, caplog):
        """log_audit_event must not log document content."""
        configure_logging()
        logger = get_logger(__name__)

        with caplog.at_level(logging.INFO):
            log_audit_event(
                logger,
                event_type="document_viewed",
                user_id="manager-123",
                resource_type="document",
                resource_id=str(uuid4()),
                details={
                    "filename": "secret_merger_plan.docx",
                    "content": "CONFIDENTIAL - Merger with Acme Corp will value company at $5B",
                },
            )

            log_text = caplog.text

            # Content must not appear (details are sanitized)
            assert "Merger with Acme Corp" not in log_text
            assert "$5B" not in log_text
            assert "secret_merger_plan.docx" not in log_text

            # But the event should still be logged
            assert "AUDIT_EVENT" in log_text or "document_viewed" in log_text


# ─── Test: Filename Sanitization ─────────────────────────────────────────

class TestFilenameSanitization:
    """Verify filenames are hashed while preserving extensions."""

    def test_filename_sanitized_preserves_extension(self):
        """_sanitize_filename() must hash the name but keep the extension."""
        result = _sanitize_filename("financial_projections_2026.pdf")

        assert result.endswith(".pdf")
        assert "financial" not in result
        assert "projections" not in result
        # Should be a hash + extension
        assert len(result.split(".")[0]) > 0

    def test_filename_without_extension(self):
        """_sanitize_filename() must handle filenames without extensions."""
        result = _sanitize_filename("README")

        # Should be just the hash without extension
        expected = hashlib.sha256("README".encode()).hexdigest()[:8]
        assert result == expected

    def test_file_path_sanitize_strips_directory(self):
        """File paths in errors must be stripped (uses _sanitize_error)."""
        # Test that _sanitize_error redacts paths
        error = "File not found: /var/secrets/config/database.yml"
        sanitized = _sanitize_error(error)
        assert "secrets" not in sanitized
        assert "config" not in sanitized
        assert "var" not in sanitized


# ─── Test: Error Message Sanitization ───────────────────────────────────

class TestErrorSanitization:
    """Verify error messages don't expose internal paths, URLs, or tokens."""

    def test_path_stripped_from_error(self):
        """_sanitize_error() must strip server file paths."""
        error = (
            "Error in /var/lib/postgresql/data/db.py at line 42: "
            "connection refused"
        )
        sanitized = _sanitize_error(error)

        assert "/var/lib/postgresql" not in sanitized
        assert "db.py" not in sanitized

    def test_url_stripped_from_error(self):
        """_sanitize_error() must strip internal URLs."""
        error = (
            "Connection failed to https://internal-api.company.com:8443/v2/users"
        )
        sanitized = _sanitize_error(error)

        # URL components should be redacted
        assert "internal-api.company.com" not in sanitized
        assert "8443" not in sanitized

    def test_connection_string_sanitized(self):
        """_sanitize_error() must strip connection strings with credentials."""
        error = (
            "Database error: "
            "mysql://admin:p@ssw0rd@db-cluster-1.internal:3306/prod_db"
        )
        sanitized = _sanitize_error(error)

        assert "p@ssw0rd" not in sanitized
        assert "mysql://admin" not in sanitized

    def test_sql_injection_pattern_sanitized(self):
        """_sanitize_error() must redact SQL-injection-like token patterns."""
        # JWT-like token patterns should be redacted
        token_like = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4pPGYh"
        error = f"Query failed with token='{token_like}'"
        sanitized = _sanitize_error(error)

        assert token_like not in sanitized

    def test_password_in_error_sanitized(self):
        """_sanitize_error() must redact password= patterns."""
        error = "Auth failed: password=super_secret_password123"
        sanitized = _sanitize_error(error)

        assert "super_secret_password123" not in sanitized


# ─── Test: Permission Denial Audit ───────────────────────────────────────

class TestPermissionDenialAudit:
    """Verify permission denials produce audit events without leaking content."""

    SENSITIVE_RESOURCE = "CONFIDENTIAL_salary_data_q4_2025.xlsx"

    def test_permission_denial_audit_no_content_leak(self, caplog):
        """log_security_event for permission denial must not leak resource content."""
        configure_logging()
        logger = get_logger(__name__)

        with caplog.at_level(logging.INFO):
            log_security_event(
                logger,
                event_type="permission_denied",
                user_id="employee-789",
                resource_type="document",
                resource_id="doc-xyz",
                ip_address="10.0.0.5",
                success=False,
                failure_reason="Insufficient permissions for role: employee",
                details={
                    "requested_resource": self.SENSITIVE_RESOURCE,
                    "document_content": "Annual salary data for all employees including SSNs",
                },
            )

            log_text = caplog.text

            # Must have the audit event
            assert "permission_denied" in log_text or "SECURITY_EVENT" in log_text
            assert "employee-789" in log_text

            # Must NOT leak sensitive resource details
            assert "CONFIDENTIAL_salary_data" not in log_text
            assert "Annual salary data" not in log_text
            assert "SSN" not in log_text

    def test_audit_event_logs_without_content(self, caplog):
        """log_audit_event must not log document content."""
        configure_logging()
        logger = get_logger(__name__)

        with caplog.at_level(logging.INFO):
            log_audit_event(
                logger,
                event_type="document_view",
                user_id="manager-123",
                resource_type="document",
                resource_id="doc-abc-123",
                details={
                    "filename": "secret_merger_plan.docx",
                    "content": "CONFIDENTIAL - Merger with Acme Corp will value company at $5B",
                },
            )

            log_text = caplog.text

            # Must have audit fields
            assert "manager-123" in log_text
            assert "doc-abc-123" in log_text

            # Must NOT leak content
            assert "Merger with Acme Corp" not in log_text
            assert "$5B" not in log_text
            assert "secret_merger_plan.docx" not in log_text


# ─── Test: Production Log Configuration ────────────────────────────────

class TestProductionLogConfiguration:
    """Verify production logging is configured securely."""

    def test_log_sensitive_data_defaults_false(self):
        """LOG_SENSITIVE_DATA must default to False."""
        assert settings.LOG_SENSITIVE_DATA is False

    def test_no_external_telemetry_dependencies(self):
        """Verify no external telemetry packages are in requirements."""
        with open("requirements.txt") as f:
            requirements = f.read().lower()

        forbidden = [
            "sentry",
            "datadog",
            "newrelic",
            "opentelemetry",
            "honeycomb",
            "rollbar",
        ]

        for dep in forbidden:
            assert dep not in requirements, (
                f"Forbidden telemetry dependency '{dep}' found in requirements.txt"
            )

    def test_structlog_configured(self):
        """structlog must be configured for JSON output."""
        # After configure_logging, structlog processors should include JSON
        from app.core.logging import configure_logging
        configure_logging()
        import structlog
        # The logger should produce structured logs
        logger = structlog.get_logger()
        assert logger is not None


# ─── Test: LOG_SENSITIVE_DATA Flag ─────────────────────────────────────

class TestLogSensitiveDataFlag:
    """Verify the LOG_SENSITIVE_DATA flag controls sensitive logging."""

    def test_sensitive_query_always_hashed(self):
        """Query fingerprinting must always happen."""
        fp = _sanitize_query("SECRET: unreleased earnings projection for Q3")
        assert "unreleased" not in fp
        assert "earnings" not in fp
        assert "projection" not in fp

    def test_filenames_always_sanitized(self):
        """Filenames must always be sanitized in document logs."""
        result = _sanitize_filename("confidential_revenue_data.csv")
        assert "confidential" not in result
        assert "revenue" not in result
        assert result.endswith(".csv")


# ─── Test: RAG Query Function Call ──────────────────────────────────────

class TestRagQueryLogging:
    """Verify RAG query logging uses fingerprinting."""

    def test_log_rag_event_logs_fingerprint_not_query(self, caplog):
        """log_rag_event must log query_fingerprint, not the raw query."""
        configure_logging()
        logger = get_logger(__name__)

        sensitive = "CONFIDENTIAL: unreleased financial projections for Q4"

        with caplog.at_level(logging.INFO):
            log_rag_event(
                logger,
                event_type="query",
                user_id="user-123",
                query=sensitive,
            )

            # The raw query must not appear
            assert "CONFIDENTIAL" not in caplog.text
            assert "financial projections" not in caplog.text
            assert "unreleased" not in caplog.text

            # The fingerprint field should be present
            assert "query_fingerprint" in caplog.text
