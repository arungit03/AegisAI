"""Document diversity — relevance-aware, no forced diversity.

P8 Document Diversity: avoid selecting only Document A with 20 similar chunks
vs B with 5 relevant. Relevance-aware; do NOT force multiple docs if only
one contains the answer.

Logic:
1. Group hits by document_id
2. If only 1 document in candidates, return top_k from that doc (no forced diversity)
3. If multiple docs, ensure no single doc monopolizes results: cap per-doc at
   max_per_doc while preserving relevance order
4. Relevance-aware: only diversify if candidate docs have hits within
   relevance_threshold of top hit
5. If secondary docs have much lower scores (< threshold below top), prefer
   the dominant doc
6. Return hits ordered by rerank/rrf/score still, but with diversity guarantee

Do NOT over-diversify: a query answered by 1 document should cite that
document, not force in irrelevant docs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DiversityInfo:
    """Observability for document diversity outcome."""

    total_docs: int
    per_doc_counts: Dict[str, int]
    max_per_doc: int
    was_diversified: bool


def document_distribution(hits: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count hits per document_id.

    Args:
        hits: List of hit dicts with payload.document_id.

    Returns:
        Mapping document_id -> count.
    """
    counts: Counter = Counter()
    for hit in hits:
        payload = hit.get("payload") or {}
        doc_id = payload.get("document_id", "unknown")
        # Normalize None -> "unknown" and coerce to str
        if doc_id is None:
            doc_id = "unknown"
        counts[str(doc_id)] += 1
    return dict(counts)


def _score(hit: Dict[str, Any]) -> float:
    """Best available relevance score for a hit (rerank > rrf > score)."""
    for key in ("rerank_score", "rrf_score", "score"):
        if key in hit and hit[key] is not None:
            try:
                return float(hit[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _doc_id(hit: Dict[str, Any]) -> str:
    payload = hit.get("payload") or {}
    doc_id = payload.get("document_id", "unknown")
    if doc_id is None:
        doc_id = "unknown"
    return str(doc_id)


def diversify_by_document(
    hits: List[Dict[str, Any]],
    top_k: int = 5,
    max_per_doc: int = 3,
    relevance_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Relevance-aware per-document capping.

    Preserves input relevance order (assumed sorted by rerank/rrf/score desc)
    while ensuring no single document monopolizes the top_k. Secondary
    documents are only preferred when they have hits within
    relevance_threshold (absolute score difference) of the top hit; otherwise
    the dominant document is preferred.

    Args:
        hits: Candidate hits ordered by relevance (highest first).
        top_k: Number of hits to return.
        max_per_doc: Maximum hits per document in the diversified set.
        relevance_threshold: Absolute score delta. A secondary-doc hit whose
            score is more than this below the top hit is considered much
            less relevant and is deferred (only used to fill if needed).

    Returns:
        Diversified list of at most top_k hits, still in relevance order
        except for per-doc capping adjustments.
    """
    if not hits or top_k <= 0:
        return []
    if max_per_doc <= 0:
        max_per_doc = top_k  # no effective cap

    # Group / distribution check - no forced diversity
    dist = document_distribution(hits)
    if len(dist) <= 1:
        # Only one document contains the answer -> return its top_k
        return hits[:top_k]

    # If the pool is already at or below top_k we still attempt to
    # diversify via reordering (cap + deferred fill) rather than early-return,
    # so that a monopolized small pool (e.g. 4x A + 1x B with cap 3) can be
    # rebalanced to 3x A + 1x B + 1x deferred A while preserving order.
    # For Truly small pools where all hits already satisfy cap, the logic
    # below returns them unchanged.

    top_score = _score(hits[0])
    top_doc = _doc_id(hits[0])

    selected: List[Dict[str, Any]] = []
    per_doc: Counter = Counter()
    deferred: List[Dict[str, Any]] = []

    for hit in hits:
        doc = _doc_id(hit)
        score = _score(hit)
        is_top_doc = doc == top_doc

        # Relevance-aware gate: secondary docs far below top are deferred.
        # Absolute difference: if secondary hit is more than threshold below
        # top, prefer the dominant doc.
        if not is_top_doc and (top_score - score) > relevance_threshold:
            deferred.append(hit)
            continue

        # Per-doc cap (soft - deferred hits can still fill remaining slots)
        if per_doc[doc] >= max_per_doc:
            deferred.append(hit)
            continue

        selected.append(hit)
        per_doc[doc] += 1
        if len(selected) >= top_k:
            break

    # Fill remaining slots from deferred in original relevance order.
    # This prefers the highest-scoring remaining hits regardless of cap,
    # which means if diversity is not achievable (or secondary docs are
    # much less relevant) we fall back to the dominant document.
    for hit in deferred:
        if len(selected) >= top_k:
            break
        selected.append(hit)

    return selected[:top_k]
