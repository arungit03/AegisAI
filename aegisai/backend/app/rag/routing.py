"""Deterministic routing for conversational and company-knowledge queries."""
from enum import Enum
import re


class QueryIntent(str, Enum):
    """The supported chat execution paths."""

    GENERAL = "general"
    MEMORY = "memory"
    DOCUMENT_AWARE = "document_aware"
    COMPANY = "company"


_MEMORY_PATTERNS = (
    r"\bwhat(?:'s| is) my\b",
    r"\bdo you remember\b",
    r"\bwhat did i (?:say|tell)\b",
    r"\bwhat have i (?:said|told)\b",
    r"\b(?:earlier|previously|before)\b",
    r"\bwe (?:discussed|talked about)\b",
)

_COMPANY_TERMS = (
    "company",
    "organization",
    "company document",
    "company documents",
    "knowledge base",
    "internal",
    "policy",
    "policies",
    "procedure",
    "procedures",
    "sop",
    "employee handbook",
    "hr policy",
    "safety protocol",
    "maintenance schedule",
    "working hours",
    "office hours",
    "office working",
    "remote work",
    "engineering department",
    "maternity leave",
    "cto",
    "sih",
    "problem statement",
    "technology required",
    "technology is required",
    "technical requirements",
    "department",
    "benefits",
    "reimbursement",
    "leave policy",
    "our documents",
    "our policy",
)

_DOCUMENT_AWARE_PATTERNS = (
    r"\bdo i have (?:any )?(?:documents?|files?)\b",
    r"\b(?:can|do) you (?:see|access) (?:my|the) (?:documents?|files?)\b",
    r"\bdid i (?:upload|add)\b",
    r"\b(?:are|is) my (?:[a-z0-9_-]+ )?(?:documents?|files?) (?:uploaded|available|there)\b",
    r"\b(?:what|which) documents? have i uploaded\b",
    r"\blist (?:my|the documents? i)\b",
    r"\blist\b.*\b(?:documents?|files?)\b.*\b(?:uploaded|added)\b",
    r"\blist\b.*\b(?:uploaded|added)\b.*\b(?:documents?|files?)\b",
    r"\bdocuments? (?:that )?i (?:added|uploaded)\b",
)

_STRUCTURED_LOOKUP_PATTERNS = (
    r"\b(?:ps|problem statement)\s+(?:number|id)\b",
    r"\bwhat\s+is\s+the\s+(?:problem statement\s+)?title\b",
    r"\bwhat\s+title\s+(?:belongs|is associated)\b",
    r"\blist\b.*\b(?:sih|problem statements?)\b",
)


def is_document_awareness_query(question: str) -> bool:
    """Return whether a question asks about document records, not contents."""
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    return any(re.search(pattern, normalized) for pattern in _DOCUMENT_AWARE_PATTERNS)


def classify_query(question: str) -> QueryIntent:
    """Classify a user message without sending it to an external classifier."""
    normalized = re.sub(r"\s+", " ", question.strip().lower())

    if any(re.search(pattern, normalized) for pattern in _MEMORY_PATTERNS):
        return QueryIntent.MEMORY

    if is_document_awareness_query(normalized):
        return QueryIntent.DOCUMENT_AWARE

    # Structured document lookups must use the deterministic SIH record path,
    # even when the question contains a title but not the literal word SIH.
    if any(re.search(pattern, normalized) for pattern in _STRUCTURED_LOOKUP_PATTERNS):
        return QueryIntent.COMPANY

    if any(term in normalized for term in _COMPANY_TERMS):
        return QueryIntent.COMPANY

    return QueryIntent.GENERAL
