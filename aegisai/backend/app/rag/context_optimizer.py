"""Context optimization — relevance density under budget.

P11 Context Optimization
-----------------------
Improve MAX_CONTEXT_CHARS usage:

* Highest-value first — sort by rerank_score > rrf_score > score descending
* Remove redundant near-duplicate chunk_text (Jaccard > 0.85)
* Preserve neighboring context / page / section metadata in source header
* Incremental budget build; avoid cutting important evidence mid-block
* Try truncated hit before stopping; never exceed max_chars
* More context is NOT automatically better — optimize for relevance density

Reference: app.rag.fusion.build_context
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ContextStats:
    """Observability stats for context optimization."""

    total_chars: int
    included_count: int
    truncated_count: int
    avg_relevance: float
    utilization: float


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count ~1 token per 4 chars."""
    return max(1, len(text) // 4)


def truncate_to_budget(text: str, max_chars: int) -> str:
    """Truncate text to max_chars with ellipsis if needed."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _relevance_score(hit: Dict[str, Any]) -> float:
    """Relevance: rerank_score > rrf_score > score."""
    for key in ("rerank_score", "rrf_score", "score"):
        if key in hit and hit[key] is not None:
            try:
                return float(hit[key])
            except Exception:
                continue
    return 0.0


def _jaccard(a: str, b: str) -> float:
    """Token Jaccard on [a-z0-9]{2,} tokens, case-insensitive."""
    ta = set(re.findall(r"[a-z0-9]{2,}", a.lower()))
    tb = set(re.findall(r"[a-z0-9]{2,}", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0


def _format_header(payload: Dict[str, Any]) -> str:
    """Format: [Source: filename p.X | section] preserving metadata."""
    filename = payload.get("filename", "unknown") or "unknown"
    page = payload.get("page_number")
    section = payload.get("section_title")
    header = f"[Source: {filename}"
    if page is not None:
        header += f" p.{page}"
    if section:
        header += f" | {section}"
    header += "]"
    return header


# ---------------------------------------------------------------------------
# public API — matches spec exports exactly
# ---------------------------------------------------------------------------

def optimize_context(
    hits: List[Dict[str, Any]],
    max_chars: int = 12000,
    query: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build context string under max_chars, highest-value first.

    Steps (do NOT blindly concatenate):

    1. Sort hits by relevance (rerank_score > rrf_score > score) descending.
    2. Remove near-duplicates: Jaccard(chunk_text) > 0.85.
    3. Incrementally build context tracking char budget (including \\n\\n).
    4. Each hit formatted as ``[Source: filename p.X | section]\\nchunk_text``.
    5. If next hit would exceed budget, try a truncated version; else stop.

    Args:
        hits: Candidate hits (each with payload.chunk_text etc.).
        max_chars: Hard cap for the returned context string.
        query: Original query (reserved for future relevance weighting).

    Returns:
        (context_string, included_hits) — 2-tuple per spec. Stats are logged.

    Note:
        For callers needing stats without parsing logs, use
        :func:`optimize_context_with_stats`.
    """
    context, included, _stats = optimize_context_with_stats(hits, max_chars=max_chars, query=query)
    return context, included


def optimize_context_with_stats(
    hits: List[Dict[str, Any]],
    max_chars: int = 12000,
    query: str = "",
) -> Tuple[str, List[Dict[str, Any]], ContextStats]:
    """Same as :func:`optimize_context` but also returns :class:`ContextStats`.

    Used by ``hybrid_service`` phase2 path to populate retrieval_debug.
    """

    _ = query  # reserved for future query-aware weighting

    if not hits:
        stats = ContextStats(0, 0, 0, 0.0, 0.0)
        logger.info(
            "context_optimized",
            total_chars=0,
            included=0,
            truncated=0,
            utilization=0.0,
            avg_relevance=0.0,
            max_chars=max_chars,
            input_hits=0,
        )
        return "", [], stats

    if max_chars <= 0:
        stats = ContextStats(0, 0, len(hits), 0.0, 0.0)
        logger.info(
            "context_optimized",
            total_chars=0,
            included=0,
            truncated=len(hits),
            utilization=0.0,
            avg_relevance=0.0,
            max_chars=max_chars,
            input_hits=len(hits),
        )
        return "", [], stats

    # 1) Highest-value first
    ordered = sorted(hits, key=_relevance_score, reverse=True)

    # 2) Remove redundant near-duplicates (Jaccard > 0.85)
    deduped: List[Dict[str, Any]] = []
    for h in ordered:
        txt = str((h.get("payload") or {}).get("chunk_text", "") or "")
        is_dup = False
        for d in deduped:
            other = str((d.get("payload") or {}).get("chunk_text", "") or "")
            if _jaccard(txt, other) > 0.85:
                is_dup = True
                break
        if not is_dup:
            deduped.append(h)

    # 3-5) Incremental build with budget
    parts: List[str] = []
    included: List[Dict[str, Any]] = []
    scores: List[float] = []
    total = 0  # len("\n\n".join(parts))

    for hit in deduped:
        payload = hit.get("payload") or {}
        header = _format_header(payload)
        chunk = str(payload.get("chunk_text", "") or "")
        block = f"{header}\n{chunk}"
        sep = 2 if parts else 0
        needed = len(block) + sep

        if total + needed > max_chars:
            # Try truncated version to still use budget
            remaining = max_chars - total - sep
            if remaining > len(header) + 20:
                budget = remaining - len(header) - 1 - 3  # newline + "..."
                if budget < 0:
                    budget = 0
                truncated_chunk = chunk[:budget] if budget > 0 else ""
                block_trunc = f"{header}\n{truncated_chunk}..." if truncated_chunk else f"{header}\n..."
                parts.append(block_trunc)
                included.append(hit)
                scores.append(_relevance_score(hit))
                total += len(block_trunc) + sep
            break

        parts.append(block)
        included.append(hit)
        scores.append(_relevance_score(hit))
        total += needed

    context = "\n\n".join(parts)
    if len(context) > max_chars:
        context = truncate_to_budget(context, max_chars)

    avg_rel = sum(scores) / len(scores) if scores else 0.0
    utilization = len(context) / max_chars if max_chars else 0.0
    stats = ContextStats(
        total_chars=len(context),
        included_count=len(included),
        truncated_count=len(deduped) - len(included),
        avg_relevance=round(avg_rel, 4),
        utilization=round(utilization, 3),
    )

    logger.info(
        "context_optimized",
        total_chars=stats.total_chars,
        included=stats.included_count,
        truncated=stats.truncated_count,
        utilization=stats.utilization,
        avg_relevance=stats.avg_relevance,
        max_chars=max_chars,
        input_hits=len(hits),
        deduped=len(deduped),
        estimated_tokens=estimate_tokens(context),
    )
    return context, included, stats
