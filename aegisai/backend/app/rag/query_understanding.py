"""Query understanding — 8 query types A-H, deterministic, local-only.

No LLM calls. Purely rule/regex based. Follows style of routing.py.

Query types:
  A SIMPLE_FACTUAL    — single fact lookup e.g. "What is X?"
  B MULTI_PART        — multiple sub-questions joined by and/also/lists/? ?
  C COMPARISON        — compare / versus / difference between / contrast
  D AGGREGATION       — how many / count / list all / total / aggregate
  E FOLLOW_UP         — pronouns this/that/it needing context resolution
  F EXACT_IDENTIFIER  — SIH + 5 digits pattern
  G DOCUMENT_SPECIFIC — mentions a specific document name/file
  H MULTI_DOCUMENT    — asks across multiple sources/documents
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional


class QueryType(str, Enum):
    SIMPLE_FACTUAL = "simple_factual"        # A
    MULTI_PART = "multi_part"                # B
    COMPARISON = "comparison"                # C
    AGGREGATION = "aggregation"              # D
    FOLLOW_UP = "follow_up"                  # E
    EXACT_IDENTIFIER = "exact_identifier"    # F
    DOCUMENT_SPECIFIC = "document_specific"  # G
    MULTI_DOCUMENT = "multi_document"        # H


# --- Compiled patterns ---

_IDENTIFIER_RE = re.compile(r"\bsih[\s-]*\d{5}\b", re.IGNORECASE)

_COMPARISON_PATTERNS = (
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bdifference between\b",
    r"\bdifferences? between\b",
    r"\bcontrast\b",
)

_AGGREGATION_PATTERNS = (
    r"\bhow many\b",
    r"\bcount\b",
    r"\btotal\b",
    r"\blist all\b",
    r"\ball of\b",
    r"\baggregate\b",
    r"\bsum\b",
    r"\baverage\b",
    r"\bstatistics?\b",
)

_FOLLOW_UP_PRONOUN_RE = re.compile(
    r"\b(?:this|that|it|they|these|those|the (?:same|above|previous))\b",
    re.IGNORECASE,
)

# G — document-specific: mentions a specific document name or file reference.
# Requires an explicit document/file indicator, not just a generic topic word.
# Matches e.g. "in the employee handbook", "according to the safety manual",
# "in PS SIH 26 -1.pdf", 'document "HR Policy"', filenames with extensions.
_DOCUMENT_SPECIFIC_PATTERNS = (
    r"\b[\w-]+\.(?:pdf|docx?|xlsx?|txt|md|csv)\b",
    r"\b(?:document|file|report|manual|handbook|contract|agreement)\b\s*[\"'][\w\s.\-()]+[\"']",
    r"\b(?:in|from|according to)\s+(?:the\s+)?[\w\s.\-()]*?(?:document|file|report|manual|handbook|contract|agreement)\b",
    r"\bemployee handbook\b",
    r"\bsafety (?:manual|protocol)\b",
    r"\bhr policy\b",
    r"\bmaintenance schedule\b",
    r"\bproblem statement document\b",
    r"\bPS\s*SIH\b",
)

_MULTI_DOC_PATTERNS = (
    r"\bacross (?:all |multiple )?documents?\b",
    r"\bmultiple (?:sources?|documents?)\b",
    r"\bboth documents?\b",
    r"\beach document\b",
    r"\bcompare documents?\b",
    r"\b(?:multiple|various|several|two|2) documents?\b",
    r"\bacross (?:all |multiple )?sources?\b",
    r"\bfrom (?:multiple|various|several) sources?\b",
    r"\bacross (?:multiple|various|several) sources?\b",
)


@dataclass
class QueryAnalysis:
    """Result of query understanding."""

    query_types: List[QueryType] = field(default_factory=list)
    is_simple: bool = False
    is_multi_part: bool = False
    is_comparison: bool = False
    is_aggregation: bool = False
    has_identifier: bool = False
    is_follow_up: bool = False
    is_multi_document: bool = False
    # Extra convenience fields (not in base spec, kept for backward compat)
    is_document_specific: bool = False
    sub_questions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_query: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_identifier(text: str) -> bool:
    return bool(_IDENTIFIER_RE.search(text))


def _identifier_count(text: str) -> int:
    return len(_IDENTIFIER_RE.findall(text))


def _is_comparison(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in _COMPARISON_PATTERNS)


def _is_aggregation(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in _AGGREGATION_PATTERNS)


def _is_document_specific(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in _DOCUMENT_SPECIFIC_PATTERNS)


def _is_multi_document(text: str, sub_questions: Optional[List[str]] = None) -> bool:
    low = text.lower()
    if any(re.search(p, low) for p in _MULTI_DOC_PATTERNS):
        return True
    # Multiple distinct SIH identifiers imply multi-document
    if _identifier_count(text) >= 2:
        return True
    # Multiple sub-questions that each look like distinct info needs
    # (only when explicitly multi-doc phrasing is present above; otherwise
    # multi-part alone does not imply multi-document)
    return False


def _is_follow_up(
    text: str,
    conversation_history: Optional[List[Dict]] = None,
) -> bool:
    """Detect pronouns/references that need context resolution.

    Returns True when:
    - conversation_history is non-empty and question contains a pronoun/demonstrative
      (needs previous context to resolve), OR
    - question is a short fragment (< ~12 words) consisting mainly of
      pronoun reference such as "What about that?" / "Tell me more about it."
    """
    if not _FOLLOW_UP_PRONOUN_RE.search(text):
        return False

    # With conversation context, any pronoun indicates follow-up
    if conversation_history and len(conversation_history) > 0:
        return True

    # Without history: short queries where pronoun is central are follow-ups.
    # Longer self-contained questions that happen to contain "that"/"this"
    # (e.g. "What is that policy about?") are NOT follow-ups.
    words = text.strip().split()
    if len(words) < 10:
        # Check that pronoun is not inside a concrete noun phrase
        # like "that policy" — still ambiguous when no history, so flag it.
        return True

    # Longer query: only flag if it looks like a dependent fragment,
    # e.g. starts with pronoun or is mostly pronouns.
    low = text.strip().lower()
    if re.match(r"^(?:and\s+)?(?:what|how|why|tell|give|show)\b.*\b(?:this|that|it)\s*\??\s*$", low):
        return True

    return False


def _split_multi_part(text: str) -> List[str]:
    """Split a multi-part question into atomic sub-questions.

    Boundaries (in priority order):
    1. Multiple question marks
    2. Numbered list  e.g.  1. ...  2. ...
    3. Semicolons joining clauses
    4. Conjunctions  "and" / "also"  joining clauses with wh-words
    """
    q = text.strip()
    if not q:
        return [q]

    # 1 — multiple '?'  (each segment is a sub-question)
    if q.count("?") > 1:
        segs = [s.strip().lstrip(";").strip() for s in q.split("?") if s.strip().lstrip(";").strip()]
        return [s + "?" for s in segs]

    # 2 — numbered list  e.g. "1. What is X? 2. Who is responsible?"
    if re.search(r"\b\d+[\.\)]\s+\w", q):
        raw = re.split(r"\s*\d+[\.\)]\s*", q)
        parts = [p.strip().rstrip(",;") for p in raw if p.strip()]
        if len(parts) > 1:
            # Ensure each ends with ?
            return [p if p.endswith("?") else p + "?" for p in parts]

    # 3 — semicolons (strip leading/trailing punctuation leftovers)
    if ";" in q:
        raw = [p.strip().lstrip(";").strip() for p in q.split(";") if p.strip()]
        raw = [p for p in raw if p]
        if len(raw) > 1:
            return [p if p.endswith("?") else p + "?" for p in raw]

    # 4 — conjunctions "and" / "also"  joining distinct information needs.
    # Require at least 2 wh-words or 2 distinct question intents to avoid
    # splitting simple compounds like "What is X and Y?".
    has_and = bool(re.search(r"\band\b", q, re.IGNORECASE))
    has_also = bool(re.search(r"\balso\b", q, re.IGNORECASE))
    wh_count = len(re.findall(r"\b(?:what|who|when|where|why|how|which)\b", q, re.IGNORECASE))

    # Also-split: "What is X? Also, who is responsible?"
    if has_also and wh_count >= 2:
        raw = re.split(r"\s*,?\s*also\s*,?\s*", q, flags=re.IGNORECASE)
        raw = [p.strip().rstrip(",") for p in raw if p.strip()]
        if len(raw) > 1:
            return [p if p.endswith("?") else p + "?" for p in raw]

    if has_and and wh_count >= 2:
        raw = re.split(r"\s+and\s+", q, flags=re.IGNORECASE)
        raw = [p.strip().rstrip(",") for p in raw if p.strip()]
        if len(raw) > 1:
            return [p if p.endswith("?") else p + "?" for p in raw]

    # Comma joining wh-clauses: "What equipment requires inspection, who is responsible..."
    if wh_count >= 2 and "," in q and not has_and and not has_also:
        raw = [p.strip() for p in q.split(",") if p.strip()]
        # Only treat as multi-part if each segment looks like a question fragment
        wh_per_seg = [
            len(re.findall(r"\b(?:what|who|when|where|why|how|which)\b", seg, re.IGNORECASE))
            for seg in raw
        ]
        if sum(1 for c in wh_per_seg if c >= 1) >= 2:
            return [p if p.endswith("?") else p + "?" for p in raw]

    return [q]


def _is_multi_part(text: str) -> bool:
    return len(_split_multi_part(text)) > 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_query_types(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
) -> List[QueryType]:
    """Classify *question* into one or more :class:`QueryType` values.

    Purely rule/regex based — no LLM calls.

    Args:
        question: Raw user question.
        conversation_history: Optional conversation turns (used only for
            follow-up detection). Kept optional so callers matching the
            bare spec signature ``classify_query_types(question)`` still work.
    """
    if not question or not question.strip():
        return [QueryType.SIMPLE_FACTUAL]

    q = question.strip()
    types: List[QueryType] = []

    has_id = _has_identifier(q)
    is_comparison = _is_comparison(q)
    is_aggregation = _is_aggregation(q)
    is_follow_up = _is_follow_up(q, conversation_history)
    is_doc_specific = _is_document_specific(q)

    # Pre-compute multi-part split once for both B and H checks
    sub = _split_multi_part(q)
    is_multi_part = len(sub) > 1
    is_multi_doc = _is_multi_document(q, sub)

    if has_id:
        types.append(QueryType.EXACT_IDENTIFIER)
    if is_comparison:
        types.append(QueryType.COMPARISON)
    if is_aggregation:
        types.append(QueryType.AGGREGATION)
    if is_follow_up:
        types.append(QueryType.FOLLOW_UP)
    if is_doc_specific:
        types.append(QueryType.DOCUMENT_SPECIFIC)
    if is_multi_doc:
        types.append(QueryType.MULTI_DOCUMENT)
    if is_multi_part:
        types.append(QueryType.MULTI_PART)

    # Default: if nothing else matched, this is a simple factual query
    if not types:
        types.append(QueryType.SIMPLE_FACTUAL)

    return types


def analyze_query(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
) -> QueryAnalysis:
    """Full query analysis with convenience booleans and sub-question split.

    For simple factual queries (only type == SIMPLE_FACTUAL) no decomposition
    is performed and ``sub_questions`` is left empty — per spec requirement
    "Do NOT over-process simple queries".

    Args:
        question: Raw user question.
        conversation_history: Optional prior turns for follow-up detection.
    """
    qtypes = classify_query_types(question, conversation_history)

    is_simple = qtypes == [QueryType.SIMPLE_FACTUAL]
    is_multi = QueryType.MULTI_PART in qtypes
    is_comp = QueryType.COMPARISON in qtypes
    is_agg = QueryType.AGGREGATION in qtypes
    has_id = QueryType.EXACT_IDENTIFIER in qtypes
    is_follow = QueryType.FOLLOW_UP in qtypes
    is_mdoc = QueryType.MULTI_DOCUMENT in qtypes
    is_dspec = QueryType.DOCUMENT_SPECIFIC in qtypes

    # Only decompose non-simple multi-part queries
    sub_qs: List[str] = []
    if is_multi and not is_simple:
        sub_qs = _split_multi_part(question)

    # Confidence: higher when classification is unambiguous / single-type
    if is_simple:
        confidence = 0.92
    elif len(qtypes) == 1:
        confidence = 0.85
    elif len(qtypes) == 2:
        confidence = 0.68
    elif len(qtypes) == 3:
        confidence = 0.52
    else:
        confidence = 0.40

    # Clamp well inside (0,1)
    confidence = round(max(0.05, min(0.99, confidence)), 3)

    return QueryAnalysis(
        query_types=qtypes,
        is_simple=is_simple,
        is_multi_part=is_multi,
        is_comparison=is_comp,
        is_aggregation=is_agg,
        has_identifier=has_id,
        is_follow_up=is_follow,
        is_multi_document=is_mdoc,
        is_document_specific=is_dspec,
        sub_questions=sub_qs,
        confidence=confidence,
        raw_query=question,
    )
