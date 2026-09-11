"""Reranking visibility — wraps fusion reranking with Phase 2 observability.

Provides a typed RerankingResult dataclass and a visibility wrapper
around the fusion reranker. Thin wrapper — fusion.py owns the
CrossEncoder logic and is the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class RerankingResult:
    """Result of reranking with full visibility info."""

    hits: List[Dict[str, Any]]
    reranked: bool
    model_loaded: bool
    model_name: str
    fallback_reason: Optional[str] = None


def is_reranking_available(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> bool:
    """Return True if the cross-encoder model is importable.

    Tries to import CrossEncoder locally. No model load attempted.

    Args:
        model_name: Model id (kept for signature consistency; not used
            to decide availability beyond import check).

    Returns:
        True if sentence_transformers.CrossEncoder is importable.
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def rerank_with_visibility(
    query: str,
    hits: List[Dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
) -> RerankingResult:
    """Rerank with explicit visibility info.

    Delegates to fusion.rerank_with_visibility (which itself wraps
    fusion.rerank_cross_encoder) so all CrossEncoder logic lives in
    one place. Falls back gracefully when fusion helpers are not
    importable.

    Args:
        query: User query.
        hits: Candidate hits with payload.chunk_text.
        model_name: Cross-encoder model id.
        top_k: Return top-K after reranking.

    Returns:
        RerankingResult with hits, reranked flag, model_loaded,
        model_name, and fallback_reason if not reranked.
    """
    if not hits:
        # Report model_loaded from import probe when there are no candidates
        model_loaded = is_reranking_available(model_name)
        return RerankingResult(
            hits=[],
            reranked=False,
            model_loaded=model_loaded,
            model_name=model_name,
            fallback_reason="no_candidates",
        )

    # Prefer the typed visibility wrapper in fusion (has candidate/returned counts)
    try:
        from app.rag.fusion import rerank_with_visibility as _fusion_vis  # type: ignore

        fusion_hits, info = _fusion_vis(query, hits, model_name=model_name, top_k=top_k)
        return RerankingResult(
            hits=fusion_hits,
            reranked=bool(info.get("reranked", False)),
            model_loaded=bool(info.get("model_loaded", False)),
            model_name=str(info.get("model_name", model_name)),
            fallback_reason=info.get("fallback_reason"),
        )
    except Exception:
        pass

    # Fallback: use rerank_cross_encoder + import probe
    try:
        from app.rag.fusion import rerank_cross_encoder  # type: ignore

        reranked = rerank_cross_encoder(query, hits, model_name=model_name, top_k=top_k)
        has_scores = any("rerank_score" in h for h in reranked)
        if has_scores:
            return RerankingResult(
                hits=reranked,
                reranked=True,
                model_loaded=True,
                model_name=model_name,
            )
        # No scores — infer whether import failed
        model_loaded = is_reranking_available(model_name)
        if model_loaded:
            fallback = "rerank_failed_kept_order"
        else:
            fallback = "import_failed_or_rerank_failed"
        return RerankingResult(
            hits=reranked,
            reranked=False,
            model_loaded=model_loaded,
            model_name=model_name,
            fallback_reason=fallback,
        )
    except Exception as e:
        model_loaded = is_reranking_available(model_name)
        return RerankingResult(
            hits=hits[:top_k],
            reranked=False,
            model_loaded=model_loaded,
            model_name=model_name,
            fallback_reason=f"error:{type(e).__name__}",
        )
