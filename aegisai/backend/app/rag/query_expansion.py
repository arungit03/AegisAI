"""P3 Query Expansion — controlled, careful, local-only.

No LLM. Uses a small curated domain synonym map. Expands only when
appropriate: not every query, not unrelated terms, and never exact
identifiers. Preserves the original query as the first element and
caps alternative phrasings at three.

Example:
    "machine inspection frequency"
        -> ["machine inspection frequency",
            "equipment inspection frequency",
            "machine audit frequency",
            "machine inspection interval"]
"""

from __future__ import annotations

import re
from typing import List

from app.core.logging import get_logger

logger = get_logger(__name__)

# Exact-identifier guard — must not be modified or expanded.
_IDENTIFIER_RE = re.compile(r"\bSIH[\s-]*\d{5}\b", re.IGNORECASE)

# Domain synonym groups — each group is fully interchangeable. Derived
# map is bidirectional so "machine -> equipment" and "equipment -> machine"
# both trigger expansion. Groups are intentionally small and curated;
# generic terms are not included.
_SYNONYM_GROUPS: List[List[str]] = [
    ["machine", "equipment"],
    ["frequency", "interval"],
    ["policy", "procedure", "guideline"],
    ["inspection", "audit", "check", "examination"],
    ["maintenance", "upkeep", "servicing"],
]

_SYNONYM_MAP: dict[str, List[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _term in _group:
        _SYNONYM_MAP[_term] = [t for t in _group if t != _term]

_MAX_EXPANSIONS = 3


def _is_multi_query(question: str) -> bool:
    """Return True when the question already contains multiple queries.

    Detects already multi-query contexts so expansion does not blindly
    multiply a decomposed question (e.g. "What is X? What is Y?" or
    "Tell me about X and tell me about Y" with multiple WH-clauses).
    Mirrors the heuristics in query_understanding._split_multi_part.
    """
    q = question.strip()
    if not q:
        return False
    # Multiple question marks -> multiple queries
    if q.count("?") > 1:
        return True
    # Numbered list like "1. ... 2. ..."
    if re.search(r"\b\d+[\.\)]\s+\w", q):
        return True
    # Multiple WH-words joined by conjunction -> multi-part
    wh = re.findall(r"\b(?:what|who|when|where|why|how|which)\b", q, flags=re.IGNORECASE)
    if len(wh) > 1 and re.search(r"\band\b", q, flags=re.IGNORECASE):
        return True
    # Explicit semicolon-separated clauses
    if ";" in q and len([p for p in q.split(";") if p.strip()]) > 1:
        return True
    # Multiple non-empty lines with a question
    lines = [line.strip() for line in q.splitlines() if line.strip()]
    if len(lines) > 1 and sum(1 for line in lines if "?" in line) >= 1 and len(lines) > 1:
        # If there are multiple lines and at least one is a question,
        # treat as already decomposed context.
        # Require at least 2 substantive lines to avoid false positives
        # on single-line queries that merely contain a newline.
        if len(lines) >= 2 and any(len(line.split()) >= 3 for line in lines):
            # Only flag when lines look like separate questions/clauses
            question_lines = [line for line in lines if "?" in line or len(line.split()) >= 3]
            if len(question_lines) > 1:
                return True
    return False


def _contains_synonym_term(question: str) -> bool:
    """Check whether any curated synonym term occurs as a whole word."""
    for term in _SYNONYM_MAP:
        if re.search(rf"\b{re.escape(term)}\b", question, flags=re.IGNORECASE):
            return True
    return False


def should_expand(question: str) -> bool:
    """Return whether *question* is a candidate for expansion.

    Controlled and careful — returns ``False`` for:
    * empty / whitespace-only queries
    * exact identifier queries (``SIH`` followed by 5 digits)
    * very short queries (< 3 words)
    * already multi-query contexts
    * queries with no curated synonym term present
    """
    if not question or not question.strip():
        return False
    if _IDENTIFIER_RE.search(question):
        return False
    if len(question.strip().split()) < 3:
        return False
    if _is_multi_query(question):
        return False
    if not _contains_synonym_term(question):
        return False
    return True


def expand_query(question: str) -> List[str]:
    """Generate 1-3 alternative phrasings via curated synonym substitution.

    * Preserves the original query as the first element.
    * Each alternative replaces a single whole-word term with one of its
      domain synonyms (word-boundary, case-insensitive).
    * Caps at ``_MAX_EXPANSIONS`` (3) alternatives — total list length
      is at most 4.
    * Returns ``[question]`` unchanged when :func:`should_expand` is
      ``False`` or no synonym is found.
    * Local only — no LLM, no external service.

    Example:
        >>> expand_query("machine inspection frequency")
        ['machine inspection frequency',
         'equipment inspection frequency',
         'machine audit frequency',
         'machine inspection interval']
    """
    if not should_expand(question):
        return [question]

    # Collect synonym-bearing keys sorted by first occurrence in the query
    # so expansions follow query order rather than map insertion order.
    found: List[tuple[int, str]] = []
    for term in _SYNONYM_MAP:
        m = re.search(rf"\b{re.escape(term)}\b", question, flags=re.IGNORECASE)
        if m:
            found.append((m.start(), term))
    found.sort(key=lambda x: x[0])

    if not found:
        return [question]

    expansions: List[str] = [question]
    seen = {question.lower()}

    # Pre-collect terms present in original to avoid degenerate
    # expansions like "policy procedure" -> "procedure procedure"
    # when both terms belong to the same synonym group.
    original_terms = {
        t for t in _SYNONYM_MAP if re.search(rf"\b{re.escape(t)}\b", question, flags=re.IGNORECASE)
    }

    # Round-robin across keys so we cover distinct terms first
    # (machine->equipment, inspection->audit, frequency->interval)
    # before using second synonyms of the same term (inspection->check).
    max_rank = max(len(_SYNONYM_MAP[term]) for _, term in found)
    for rank in range(max_rank):
        for _, term in found:
            syns = _SYNONYM_MAP[term]
            if rank >= len(syns):
                continue
            syn = syns[rank]
            # Skip if synonym already occurs in the original query as a
            # whole word — would create duplicate or circular expansion
            # (e.g. "policy procedure" with policy->procedure).
            if syn.lower() in original_terms:
                continue
            # Whole-word, single-occurrence substitution
            variant = re.sub(
                rf"\b{re.escape(term)}\b", syn, question, count=1, flags=re.IGNORECASE
            )
            v_low = variant.lower()
            if v_low in seen:
                continue
            # Guard against accidental duplicate words (e.g. "procedure procedure")
            if re.search(r"\b(\w+)\s+\1\b", variant, flags=re.IGNORECASE):
                continue
            expansions.append(variant)
            seen.add(v_low)
            if len(expansions) >= _MAX_EXPANSIONS + 1:
                logger.info(
                    "query_expansion_generated",
                    original_len=len(question),
                    expansions=len(expansions) - 1,
                )
                return expansions
        if len(expansions) >= _MAX_EXPANSIONS + 1:
            break

    logger.info(
        "query_expansion_generated",
        original_len=len(question),
        expansions=len(expansions) - 1,
    )
    return expansions[: _MAX_EXPANSIONS + 1]
