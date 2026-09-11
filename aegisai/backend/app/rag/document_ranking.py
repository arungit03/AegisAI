"""P10 Document-Level Ranking — aggregate chunk evidence per document.

Additional signal, not replacement. Used for multi-document selection,
source ordering, comparison, and aggregation tasks. Combines per-document
max relevance with evidence breadth (multiple supporting chunks).

Two equivalent combined-score formulations are documented; the log-boost
variant is used as the default (simpler, rewards docs with multiple
supporting chunks) while the weighted blend is shown as an alternative:

  weighted: combined = 0.5*max_score + 0.3*avg_score + 0.2*normalized_total
  log-boost: combined = max_score * (1 + 0.1 * log(chunk_count))

Scores per hit are resolved as rerank_score > rrf_score > score,
consistent with fusion.confidence_score.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class DocumentScore:
    """Aggregated relevance for a single document across its retrieved chunks."""

    document_id: str
    filename: str
    chunk_count: int
    max_score: float
    avg_score: float
    total_score: float
    combined_score: float
    title: Optional[str] = None


def _score(hit: Dict[str, Any]) -> float:
    """Resolve hit score with priority rerank_score > rrf_score > score."""
    for k in ("rerank_score", "rrf_score", "score"):
        if k in hit and hit[k] is not None:
            return float(hit[k])
    return 0.0


def aggregate_document_scores(hits: List[Dict[str, Any]]) -> Dict[str, DocumentScore]:
    """Group hits by document_id and compute per-document aggregates.

    For each document computes max_score, avg_score, total_score and
    chunk_count, then derives combined_score as:

        combined = max_score * (1 + 0.1 * log(chunk_count))

    This rewards documents with multiple supporting chunks without
    overwhelming peak relevance. Weighted alternative:

        combined = 0.5*max + 0.3*avg + 0.2*(total/max_total)

    is left commented for reference.

    Args:
        hits: Fused/reranked hits each with payload.document_id etc.

    Returns:
        Mapping document_id -> DocumentScore.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for h in hits:
        doc_id = str((h.get("payload") or {}).get("document_id", "unknown"))
        groups[doc_id].append(h)

    if not groups:
        return {}

    # First pass: totals and global max_total for normalized_total
    totals: Dict[str, float] = {}
    max_total = 0.0
    for doc_id, lst in groups.items():
        t = sum(_score(h) for h in lst)
        totals[doc_id] = t
        if t > max_total:
            max_total = t

    out: Dict[str, DocumentScore] = {}
    for doc_id, lst in groups.items():
        scores = [_score(h) for h in lst]
        mx = max(scores) if scores else 0.0
        avg = sum(scores) / len(scores) if scores else 0.0
        tot = totals[doc_id]
        # normalized_total for weighted-blend alternative
        norm_tot = (tot / max_total) if max_total > 0 else 0.0
        # Primary: log-boost formulation (rewards multi-evidence docs)
        combined = mx * (1 + 0.1 * math.log(max(1, len(lst))))
        # Alternative weighted blend (kept for reference, not active):
        # combined = 0.5 * mx + 0.3 * avg + 0.2 * norm_tot
        _ = norm_tot  # suppress unused-variable warning; kept for alt formula
        payload0 = (lst[0].get("payload") or {})
        out[doc_id] = DocumentScore(
            document_id=doc_id,
            filename=str(payload0.get("filename", "unknown")),
            chunk_count=len(lst),
            max_score=round(mx, 4),
            avg_score=round(avg, 4),
            total_score=round(tot, 4),
            combined_score=round(combined, 4),
            title=payload0.get("document_title") or payload0.get("title"),
        )
    return out


def rank_documents(hits: List[Dict[str, Any]], top_docs: int = 5) -> List[DocumentScore]:
    """Rank documents by combined_score descending.

    Convenience wrapper over aggregate_document_scores; sorts the
    aggregated DocumentScore objects by combined_score and returns the
    top N. Intended as an additional signal for ordering sources and
    for multi-doc comparison / aggregation.

    Args:
        hits: Retrieved hits to aggregate.
        top_docs: Maximum documents to return (default 5).

    Returns:
        Ranked list of DocumentScore, highest combined_score first.
    """
    if top_docs <= 0:
        return []
    agg = aggregate_document_scores(hits)
    ranked = sorted(agg.values(), key=lambda d: d.combined_score, reverse=True)
    return ranked[:top_docs]
