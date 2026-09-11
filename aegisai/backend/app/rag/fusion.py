"""Hybrid fusion: Reciprocal Rank Fusion (RRF), dedup, diversity, reranking, context.

All local. No external API.
"""

from __future__ import annotations

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)


def rrf_fuse(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    Args:
        ranked_lists: List of ranked lists, each item has {id, score, payload}.
        k: RRF constant (higher = flatter rank discount).
        top_k: Final fused list size.

    Returns:
        Fused list sorted by RRF score desc, each item is best payload for that id.
    """
    rrf_scores: Dict[str, float] = {}
    best_item: Dict[str, Dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = item.get("id", "")
            if not doc_id:
                continue
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            # keep highest original score payload if duplicate
            if doc_id not in best_item or item.get("score", 0) > best_item[doc_id].get("score", 0):
                best_item[doc_id] = item
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results: List[Dict[str, Any]] = []
    for doc_id, rrf_score in fused:
        item = dict(best_item[doc_id])
        item["rrf_score"] = rrf_score
        results.append(item)
    logger.info("rrf_fused", inputs=len(ranked_lists), candidates=len(rrf_scores), returned=len(results))
    return results


def dedup_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate chunk_text hits (normalized), keep first (highest fused)."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for h in hits:
        payload = h.get("payload") or {}
        text = str(payload.get("chunk_text", "") or "").strip().lower()
        key = re.sub(r"\s+", " ", text)
        # also consider point id uniqueness for exact duplicates
        id_key = h.get("id", "")
        composite = f"{id_key}:{hashlib.sha256(key.encode()).hexdigest()[:16]}" if key else id_key
        if composite in seen or (key and key in seen):
            continue
        seen.add(composite)
        if key:
            seen.add(key)
        out.append(h)
    return out


def mmr_select(
    hits: List[Dict[str, Any]],
    top_k: int = 5,
    lambda_mult: float = 0.6,
) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance diversity selection using token Jaccard.

    Local, no embeddings needed. Balances relevance vs diversity.
    Relevance is rrf_score or score. Diversity is Jaccard distance to selected set.

    Args:
        hits: Fused ranked list (ordered by relevance).
        top_k: Number to select.
        lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity.

    Returns:
        Diverse subset preserving relevance order bias.
    """
    if not hits or top_k <= 0:
        return []
    if len(hits) <= top_k:
        return hits

    def token_set(payload: Dict[str, Any]):
        text = str(payload.get("chunk_text", "") or "").lower()
        return set(re.findall(r"[a-z0-9]{2,}", text))

    selected: List[Dict[str, Any]] = [hits[0]]
    selected_tokens = [token_set(hits[0].get("payload") or {})]
    remaining = hits[1:]

    while len(selected) < top_k and remaining:
        best_idx = -1
        best_score = float("-inf")
        for idx, hit in enumerate(remaining):
            rel = float(hit.get("rrf_score", hit.get("score", 0)))
            # Normalize relevance to ~[0,1] via min-max over remaining+selected
            cand_tokens = token_set(hit.get("payload") or {})
            max_sim = 0.0
            for st in selected_tokens:
                if not cand_tokens or not st:
                    continue
                inter = len(cand_tokens & st)
                union = len(cand_tokens | st)
                sim = inter / union if union else 0.0
                if sim > max_sim:
                    max_sim = sim
            mmr = lambda_mult * rel - (1 - lambda_mult) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        if best_idx >= 0:
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            selected_tokens.append(token_set(chosen.get("payload") or {}))
        else:
            break
    return selected


def rerank_cross_encoder(
    query: str,
    hits: List[Dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Local cross-encoder reranking. Gracefully falls back to input order.

    Tries to load sentence-transformers CrossEncoder locally. If unavailable
    or fails, returns hits[:top_k] unchanged. No external API.

    Args:
        query: User query.
        hits: Candidate hits with payload.chunk_text.
        model_name: Local cross-encoder model id.
        top_k: Return top-K after reranking.

    Returns:
        Reranked hits[:top_k] with added rerank_score field when successful.
    """
    if not hits:
        return []
    # Lazy import so missing optional dep doesn't break the service
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as e:
        logger.warning("rerank_cross_encoder_unavailable", error_type=type(e).__name__)
        return hits[:top_k]

    try:
        model = CrossEncoder(model_name, max_length=512)
        pairs = [(query, str((h.get("payload") or {}).get("chunk_text", "")[:2000])) for h in hits]
        scores = model.predict(pairs)
        # scores may be numpy array
        scored = list(zip(hits, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = []
        for hit, s in scored[:top_k]:
            h = dict(hit)
            h["rerank_score"] = s
            reranked.append(h)
        logger.info("rerank_completed", model=model_name, candidates=len(hits), returned=len(reranked))
        return reranked
    except Exception as e:
        logger.warning("rerank_failed", error_type=type(e).__name__)
        return hits[:top_k]


def get_rerank_status(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Dict[str, Any]:
    """Report whether the cross-encoder model is available.

    Tries to import CrossEncoder locally. No model load is attempted.

    Args:
        model_name: Cross-encoder model id to report.

    Returns:
        Dict with keys:
            available (bool), model_loaded (bool, alias of available),
            model_name (str), fallback_reason (str or None).
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore  # noqa: F401

        return {
            "available": True,
            "model_loaded": True,
            "model_name": model_name,
            "fallback_reason": None,
        }
    except Exception as e:
        return {
            "available": False,
            "model_loaded": False,
            "model_name": model_name,
            "fallback_reason": f"import_failed:{type(e).__name__}",
        }


def rerank_with_info(
    query: str,
    hits: List[Dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rerank with full visibility info.

    Preferred backward-compatible visibility wrapper: keeps
    rerank_cross_encoder returning List[Dict] but provides a tuple
    variant that also returns a rerank_info dict.

    Args:
        query: User query.
        hits: Candidate hits with payload.chunk_text.
        model_name: Cross-encoder model id.
        top_k: Return top-K after reranking.

    Returns:
        (reranked_hits, rerank_info) where rerank_info has keys:
            reranked (bool), model_loaded (bool), model_name (str),
            fallback_reason (str or None), candidate_count (int),
            returned_count (int).
    """
    candidate_count = len(hits) if hits else 0
    if not hits:
        info: Dict[str, Any] = {
            "reranked": False,
            "model_loaded": False,
            "model_name": model_name,
            "fallback_reason": "no_candidates",
            "candidate_count": 0,
            "returned_count": 0,
        }
        # Even with no candidates, report whether import would succeed
        try:
            from sentence_transformers import CrossEncoder  # type: ignore  # noqa: F401

            info["model_loaded"] = True
        except Exception:
            pass
        return [], info

    # Check CrossEncoder import availability
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception as e:
        info = {
            "reranked": False,
            "model_loaded": False,
            "model_name": model_name,
            "fallback_reason": f"import_failed:{type(e).__name__}",
            "candidate_count": candidate_count,
            "returned_count": min(candidate_count, top_k),
        }
        logger.warning("rerank_with_info_unavailable", error_type=type(e).__name__)
        return hits[:top_k], info

    try:
        model = CrossEncoder(model_name, max_length=512)
        pairs = [(query, str((h.get("payload") or {}).get("chunk_text", "")[:2000])) for h in hits]
        scores = model.predict(pairs)
        scored = list(zip(hits, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        reranked: List[Dict[str, Any]] = []
        for hit, s in scored[:top_k]:
            h = dict(hit)
            h["rerank_score"] = s
            reranked.append(h)
        logger.info("rerank_with_info_completed", model=model_name, candidates=candidate_count, returned=len(reranked))
        info = {
            "reranked": True,
            "model_loaded": True,
            "model_name": model_name,
            "fallback_reason": None,
            "candidate_count": candidate_count,
            "returned_count": len(reranked),
        }
        return reranked, info
    except Exception as e:
        logger.warning("rerank_with_info_failed", error_type=type(e).__name__)
        info = {
            "reranked": False,
            "model_loaded": True,
            "model_name": model_name,
            "fallback_reason": f"rerank_failed:{type(e).__name__}",
            "candidate_count": candidate_count,
            "returned_count": min(candidate_count, top_k),
        }
        return hits[:top_k], info


def rerank_with_visibility(
    query: str,
    hits: List[Dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Wrap rerank_cross_encoder and return visibility info.

    Keeps rerank_cross_encoder backward compatible (returns List[Dict])
    while this wrapper returns (hits, visibility_dict) where
    visibility_dict has keys: reranked, model_loaded, model_name,
    fallback_reason, candidate_count, returned_count.

    Unlike rerank_with_info, this function delegates to
    rerank_cross_encoder and infers visibility from its output
    (presence of rerank_score), ensuring both wrappers stay
    consistent with the single source of reranking logic.

    Args:
        query: User query.
        hits: Candidate hits.
        model_name: Cross-encoder model id.
        top_k: Return top-K after reranking.

    Returns:
        (reranked_hits, visibility_dict)
    """
    candidate_count = len(hits) if hits else 0
    if not hits:
        # Use get_rerank_status to inform model_loaded even when no candidates
        status = get_rerank_status(model_name)
        info: Dict[str, Any] = {
            "reranked": False,
            "model_loaded": bool(status.get("model_loaded", False)),
            "model_name": model_name,
            "fallback_reason": "no_candidates",
            "candidate_count": 0,
            "returned_count": 0,
        }
        return [], info

    # Delegate to the canonical reranker (backward-compatible source of truth)
    reranked = rerank_cross_encoder(query, hits, model_name=model_name, top_k=top_k)
    has_scores = any("rerank_score" in h for h in reranked)
    status = get_rerank_status(model_name)
    # If reranking succeeded, model must have been loaded; otherwise rely on status
    model_loaded = True if has_scores else bool(status.get("model_loaded", False))
    # Derive fallback_reason
    if has_scores:
        fallback_reason = None
    else:
        # Prefer status fallback if import failed, else generic kept-order reason
        fallback_reason = status.get("fallback_reason")
        if not fallback_reason or status.get("available"):
            # Available but still no scores => rerank failed but kept order
            fallback_reason = "rerank_failed_kept_order"
        # If status says unavailable, keep its import_failed reason

    info = {
        "reranked": bool(has_scores),
        "model_loaded": bool(model_loaded),
        "model_name": model_name,
        "fallback_reason": fallback_reason,
        "candidate_count": candidate_count,
        "returned_count": len(reranked),
    }
    return reranked, info


def build_context(
    hits: List[Dict[str, Any]],
    max_chars: int = 12000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build LLM context string under max_chars, preserving order.

    Args:
        hits: Ordered hits (already reranked/diverse).
        max_chars: Hard cap on context string length.

    Returns:
        (context_string, included_hits) — hits actually included.
    """
    parts: List[str] = []
    included: List[Dict[str, Any]] = []
    total = 0
    for hit in hits:
        payload = hit.get("payload") or {}
        filename = payload.get("filename", "unknown")
        chunk = payload.get("chunk_text", "") or ""
        page = payload.get("page_number")
        header = f"[Source: {filename}" + (f" p.{page}" if page else "") + "]"
        block = f"{header}\n{chunk}"
        # +2 for separator
        needed = len(block) + (2 if parts else 0)
        if total + needed > max_chars:
            # Try to truncate this block to fit
            remaining = max_chars - total - (2 if parts else 0)
            if remaining > len(header) + 20:
                chunk_budget = remaining - len(header) - 1
                block = f"{header}\n{chunk[:chunk_budget]}..."
                parts.append(block)
                included.append(hit)
            break
        parts.append(block)
        included.append(hit)
        total += needed
    context = "\n\n".join(parts)
    return context, included


def validate_citations(
    answer: str,
    hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Lightweight citation validation: every hit should have non-empty chunk_text grounded.

    Returns:
        {valid: bool, issues: [...], cited_ids: [...]}
    """
    issues: List[str] = []
    cited_ids: List[str] = []
    for hit in hits:
        payload = hit.get("payload") or {}
        chunk = str(payload.get("chunk_text", "") or "").strip()
        if not chunk:
            issues.append(f"hit {hit.get('id','?')} has empty chunk_text")
        else:
            cited_ids.append(hit.get("id", ""))
        if not payload.get("document_id"):
            issues.append(f"hit {hit.get('id','?')} missing document_id")
    valid = len(issues) == 0
    return {"valid": valid, "issues": issues, "cited_ids": cited_ids}


def confidence_score(hits: List[Dict[str, Any]]) -> float:
    """Heuristic confidence in [0,1] from rerank/rrf/score distribution.

    Uses top score magnitude and score gap. Not a calibrated probability —
    for display and thresholding only.
    """
    if not hits:
        return 0.0
    # Prefer rerank_score, then rrf_score, then raw score
    def hit_score(h):
        for k in ("rerank_score", "rrf_score", "score"):
            if k in h and h[k] is not None:
                return float(h[k])
        return 0.0
    scores = [hit_score(h) for h in hits]
    top = scores[0]
    # Normalize: rerank scores can be negative/large; rrf in (0,1); cosine in [-1,1]
    # Use sigmoid-ish clamp
    import math as _math
    # Map top through tanh for bounded confidence
    # Scale factor 2 works for both rerank (~5) and cosine (~0.8)
    conf = _math.tanh(top * 1.2) if top > 0 else 0.0
    # Boost if there's a gap between 1st and 2nd
    if len(scores) > 1:
        gap = max(0.0, scores[0] - scores[1])
        conf = min(1.0, conf + gap * 0.15)
    # Penalize single-source answers slightly
    if len(hits) == 1:
        conf *= 0.92
    return round(float(max(0.0, min(1.0, conf))), 3)
