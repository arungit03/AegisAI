"""Authorized document-record queries used by chat document-awareness answers."""

import re
from typing import Iterable, List

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentPermission, DocumentStatus
from app.models.user import User


def build_authorized_document_query(user: User, owner_only: bool = False):
    """Build the same permission-scoped document query used by the API."""
    query = select(Document).where(Document.status != DocumentStatus.DELETED)

    if owner_only:
        return query.where(Document.uploaded_by_id == user.id)

    permission_scope = select(DocumentPermission.document_id).where(
        DocumentPermission.can_read == True,
        or_(
            DocumentPermission.user_id == user.id,
            and_(
                DocumentPermission.role_id == user.role_id,
                user.role_id is not None,
            ),
        ),
    )
    return query.where(Document.id.in_(permission_scope))


async def get_authorized_documents(
    db: AsyncSession,
    user: User,
    owner_only: bool = False,
) -> List[Document]:
    """Load non-deleted documents within the caller's authorized scope."""
    result = await db.execute(
        build_authorized_document_query(user, owner_only=owner_only).order_by(
            Document.created_at.desc()
        )
    )
    return list(result.scalars().all())


def _metadata_text(document: Document) -> str:
    tags = " ".join(document.tags or [])
    return " ".join(
        value or ""
        for value in (
            document.original_filename,
            document.filename,
            document.title,
            document.description,
            tags,
        )
    ).lower()


def _is_sih_document(document: Document) -> bool:
    """Identify SIH using stored metadata, filenames, titles, or descriptions."""
    return bool(re.search(r"(?<![a-z])sih(?:\d+)?(?![a-z])", _metadata_text(document)))


def _asks_yes_no_only(question: str) -> bool:
    return bool(re.search(r"\bsay only (?:yes|no)\b", question.lower()))


def answer_document_awareness(question: str, documents: Iterable[Document]) -> str:
    """Answer document existence/list questions without invoking Qdrant or Ollama."""
    documents = list(documents)
    normalized = question.lower()
    sih_requested = bool(re.search(r"\bsih(?:\d+)?\b", normalized))
    matching_documents = [
        document for document in documents
        if not sih_requested or _is_sih_document(document)
    ]

    if _asks_yes_no_only(question):
        return "YES" if matching_documents else "NO"

    if re.search(r"\blist\b|\bwhat documents?\b|\bwhich documents?\b", normalized):
        if not matching_documents:
            return "I couldn't find any accessible uploaded documents."
        lines = ["Your accessible uploaded documents:"]
        for document in matching_documents:
            status = getattr(document.status, "value", str(document.status))
            lines.append(f"- {document.original_filename} ({status})")
        return "\n".join(lines)

    if matching_documents:
        return f"Yes, I can see {len(matching_documents)} accessible uploaded document(s)."
    return "No, I can't see any accessible uploaded documents."
