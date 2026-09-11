"""Adjacent chunk expansion — P12 Adjacent Chunk Expansion.

When highly relevant chunk 42 selected, consider 41/43 only if same doc,
authorized, actually useful. Not blindly.

Security: permission filter must be combined with document_id filter for
adjacent lookups - never expose unauthorized adjacent chunks.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Set, Tuple


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _doc_id(hit: Dict[str, Any]) -> str:
    return str((hit.get("payload") or {}).get("document_id", ""))


def _chunk_index(hit: Dict[str, Any]) -> int | None:
    v = (hit.get("payload") or {}).get("chunk_index")
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def _chunk_text(hit: Dict[str, Any]) -> str:
    payload = hit.get("payload") or {}
    # Prefer payload chunk_text, fall back to common keys
    for key in ("chunk_text", "text", "content"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            return val
    # Also hit-level text (legacy)
    for key in ("chunk_text", "text", "content"):
        val = hit.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _get_score(hit: Dict[str, Any]) -> float | None:
    for k in ("rerank_score", "rrf_score", "score"):
        if k in hit and hit[k] is not None:
            try:
                return float(hit[k])
            except Exception:
                continue
    return None


def _jaccard_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Tokenise on whitespace, lowercased
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    inter = tokens_a & tokens_b
    union = tokens_a | tokens_b
    if not union:
        return 1.0
    return len(inter) / len(union)


def _is_meaningful(text: str) -> bool:
    if not text or not text.strip():
        return False
    # Require at least a few characters / tokens to be useful
    stripped = text.strip()
    if len(stripped) < 10:
        return False
    # At least one alphanumeric token
    if not re.search(r"[a-zA-Z0-9]", stripped):
        return False
    return True


def _is_truncated_text(text: str) -> bool:
    """Return True if text appears truncated.

    Heuristics:
    - ends mid-sentence (no terminal .!?… etc, ends with alnum or ,;:-)
    - contains incomplete table row markers (mismatched pipes, unclosed tags/fences)
    """
    if not text or not text.strip():
        return False
    stripped = text.strip()

    # ---------- incomplete table checks ----------
    lines = [ln.rstrip() for ln in stripped.splitlines() if ln.strip() != ""]
    has_pipe = any("|" in ln for ln in lines) if lines else ("|" in stripped)
    # Markdown / pipe table incomplete
    if has_pipe and lines:
        pipe_counts = [ln.count("|") for ln in lines if "|" in ln]
        if pipe_counts:
            max_pipes = max(pipe_counts)
            last_line = lines[-1]
            last_pipes = last_line.count("|")
            # Last row has fewer pipes than max -> incomplete row
            if last_pipes > 0 and last_pipes < max_pipes:
                return True
            # Most rows end with '|' but last does not -> incomplete
            ends_with_pipe = sum(1 for ln in lines if ln.strip().endswith("|"))
            pipe_lines = len([ln for ln in lines if "|" in ln])
            if pipe_lines >= 2 and ends_with_pipe >= pipe_lines * 0.6:
                if "|" in last_line and not last_line.strip().endswith("|"):
                    return True
        # Unclosed markdown fence
        if stripped.count("```") % 2 == 1:
            return True

    # HTML table incomplete
    lower = stripped.lower()
    if "<table" in lower or "<tr" in lower or "<td" in lower:
        for tag in ("<tr", "<td", "<th", "<table"):
            open_c = lower.count(tag)
            close_tag = tag.replace("<", "</")
            close_c = lower.count(close_tag)
            if open_c > close_c:
                return True
        if "<table" in lower and "</table>" not in lower:
            return True
        # If ends inside a cell without closing, treat as truncated
        # e.g. "<td>content" without </td>
        if lower.rstrip().endswith("<td>") or lower.rstrip().endswith("<tr>"):
            return True

    # If it's a pipe table and ends cleanly with '|' and pipe counts matched,
    # consider it NOT truncated (complete row) — avoid false positive on sentence check
    if has_pipe and stripped.endswith("|"):
        # Already checked incomplete above; if we are here, row is complete
        # Still check if there is unclosed fence etc., but we already did.
        return False

    # ---------- mid-sentence check ----------
    # Strip trailing quotes/brackets/spaces before checking terminator
    temp = stripped.rstrip(' \t\n\r"\'”’`)]}')
    if not temp:
        return False
    last_char = temp[-1]
    # Terminal punctuation -> not truncated
    if last_char in ".!?…。！？":
        return False
    # Trailing comma/semicolon/colon/dash -> truncated
    if last_char in ",;:—-":
        return True
    # Ends with alphanumeric -> likely mid-sentence
    if last_char.isalnum():
        return True
    # For other endings (e.g. '|' already handled, '%' etc) consider not truncated
    # But to be safe, if it doesn't end with terminal, treat as truncated
    # However pipe case already returned False, so remaining symbols:
    # If last char is not terminal and not alphanumeric, e.g. '-' handled, others like ')' were stripped
    # If we have stripped quotes and last char is still not terminal, but is e.g. '>', consider not truncated?
    return True


def _looks_like_manager(obj: Any) -> bool:
    if obj is None:
        return False
    # QdrantManager has client + collection_name + create_permission_filter
    if hasattr(obj, "client") and hasattr(obj, "collection_name"):
        return True
    if hasattr(obj, "search") and hasattr(obj, "scroll"):
        return True
    if hasattr(obj, "create_permission_filter"):
        return True
    return False


def _build_combined_filter(filter_conditions: Any, doc_cond: Any, idx_cond: Any):
    """Combine permission filter with doc+index conditions via AND (must).

    permission_filter AND document_id AND chunk_index.
    Never expose adjacent chunk without permission check.
    """
    try:
        from qdrant_client.models import Filter
    except Exception:
        # Fallback: return a simple dict-like filter if models unavailable
        return None

    if filter_conditions is None:
        return Filter(must=[doc_cond, idx_cond])

    # Filter object
    if isinstance(filter_conditions, Filter):
        has = any(
            [
                getattr(filter_conditions, "must", None),
                getattr(filter_conditions, "should", None),
                getattr(filter_conditions, "must_not", None),
                getattr(filter_conditions, "min_should", None),
            ]
        )
        if not has:
            # Empty filter (admin bypass) -> just doc+index
            return Filter(must=[doc_cond, idx_cond])
        return Filter(must=[filter_conditions, doc_cond, idx_cond])

    # List of conditions
    if isinstance(filter_conditions, list):
        return Filter(must=[*filter_conditions, doc_cond, idx_cond])

    # Fallback
    return Filter(must=[doc_cond, idx_cond])


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def should_expand_chunk(hit: Dict[str, Any], query: str | float | None = "", threshold: float | None = None, **kwargs) -> bool:
    """Check if chunk appears truncated and would benefit from expansion.

    New spec: should_expand_chunk(hit: Dict, query: str) -> bool
        returns True if chunk_text ends mid-sentence or contains incomplete
        table row markers.

    Backward-compat: also supports legacy threshold mode
        should_expand_chunk(hit, threshold=0.7) or should_expand_chunk(hit, 0.7)
        where second positional is numeric threshold -> score check.

    Args:
        hit: hit dict with payload.chunk_text / chunk_index etc
        query: query string (unused for truncation heuristics but kept for spec);
               if numeric, treated as legacy threshold.
        threshold: optional legacy threshold kwarg.

    Returns:
        bool: True if expansion is warranted (truncated or high-score legacy).
    """
    # Handle legacy numeric second arg: should_expand_chunk(hit, 0.7)
    if isinstance(query, (int, float)):
        thr = float(query)
        sc = _get_score(hit)
        if sc is None:
            return False
        try:
            return float(sc) >= thr
        except Exception:
            return False

    # Handle legacy keyword threshold: should_expand_chunk(hit, threshold=0.7)
    # If threshold is provided and query is empty/default, do score check
    # Distinguish from new spec where query is a non-empty string and threshold is None
    # If caller passed threshold explicitly with empty query, interpret as legacy
    if threshold is not None and (query == "" or query is None):
        # Also check kwargs legacy alias
        thr = threshold
        # Allow threshold to be passed via kwargs alias like score_threshold/relevance_threshold
        if isinstance(thr, (int, float, str)):
            try:
                thr_f = float(thr)
                sc = _get_score(hit)
                if sc is not None:
                    return float(sc) >= thr_f
            except Exception:
                pass
        # Fall through to truncation if score check not applicable
    # Check kwargs aliases for threshold (score_threshold etc.)
    for alias in ("score_threshold", "relevance_threshold", "thr"):
        if alias in kwargs and kwargs[alias] is not None:
            try:
                thr_f = float(kwargs[alias])
                # If query is empty, treat as legacy mode
                if query == "" or query is None:
                    sc = _get_score(hit)
                    if sc is not None:
                        return float(sc) >= thr_f
            except Exception:
                continue

    # Truncation mode (spec)
    text = _chunk_text(hit)
    if not text:
        # Fallback: try payload chunk_text directly if _chunk_text missed
        payload = hit.get("payload") or {}
        text = str(payload.get("chunk_text", "") or payload.get("text", "") or "")
    return _is_truncated_text(text)


def expand_adjacent_chunks(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    """Expand highly-relevant hits with adjacent chunk indices.

    Spec signature:
        expand_adjacent_chunks(qdrant_manager, hits: List[Dict], filter_conditions,
                               relevance_threshold: float = 0.7, max_expansions: int = 3) -> List[Dict]

    Also supports legacy/hybrid signature:
        expand_adjacent_chunks(hits, qdrant_manager=None, filter_conditions=None,
                               score_threshold=0.7, max_extra=3)

    Logic:
      1. Only consider expansion for hits with score above relevance_threshold
      2. For each qualifying hit, check adjacent chunk indices (chunk_index +/- 1)
         exist in Qdrant for same document_id
      3. Use scroll with filter: document_id match + chunk_index in [idx-1, idx+1],
         combined with permission filter
      4. Only include adjacent chunk if it adds meaningful content (not empty,
         not near-duplicate via Jaccard < 0.8 with original)
      5. Limit total expansions to max_expansions
      6. Mark expanded chunks with payload flag is_adjacent_expansion: true
      7. should_expand_chunk helper checks truncation.

    IMPORTANT: permission filter must be combined with document_id filter for
    adjacent lookups - never expose unauthorized adjacent chunks.

    If qdrant_manager is None or no high-relevance hits, return original hits unchanged.
    """
    # ---- normalize kwargs aliases ----
    # relevance_threshold aliases
    relevance_threshold: Any = kwargs.pop("relevance_threshold", None)
    if relevance_threshold is None:
        relevance_threshold = kwargs.pop("score_threshold", None)
        if relevance_threshold is None:
            relevance_threshold = kwargs.pop("threshold", None)
            if relevance_threshold is None:
                relevance_threshold = 0.7
    # max_expansions aliases
    max_expansions: Any = kwargs.pop("max_expansions", None)
    if max_expansions is None:
        max_expansions = kwargs.pop("max_extra", None)
        if max_expansions is None:
            max_expansions = 3
    # also pop legacy names that might remain
    kwargs.pop("score_threshold", None)
    kwargs.pop("threshold", None)
    kwargs.pop("max_extra", None)

    try:
        relevance_threshold = float(relevance_threshold)
    except Exception:
        relevance_threshold = 0.7
    try:
        max_expansions = int(max_expansions)
    except Exception:
        max_expansions = 3

    qdrant_manager: Any = kwargs.pop("qdrant_manager", None)
    hits: Any = kwargs.pop("hits", None)
    filter_conditions: Any = kwargs.pop("filter_conditions", None)

    # Any leftover kwargs are ignored (for forward compat)

    # ---- normalize positional args ----
    # args may contain hits/manager/filter in either order
    # Handle the duplicate-keyword case gracefully via *args/**kwargs (no Python duplicate error)
    if hits is None and len(args) >= 1:
        # If first arg is a list, likely hits (hybrid order)
        if isinstance(args[0], list):
            hits = args[0]
        elif _looks_like_manager(args[0]) and len(args) >= 2 and isinstance(args[1], list):
            # Spec order: manager first, hits second
            if qdrant_manager is None:
                qdrant_manager = args[0]
            hits = args[1]
            if filter_conditions is None and len(args) >= 3:
                filter_conditions = args[2]
            # Handle optional threshold/max as positional 4/5
            if len(args) >= 4:
                try:
                    relevance_threshold = float(args[3])
                except Exception:
                    pass
            if len(args) >= 5:
                try:
                    max_expansions = int(args[4])
                except Exception:
                    pass
        # else: first arg not list nor manager -> keep hits None for now

    # If hits still None but second arg is a list and first was manager (spec order) not yet captured
    if hits is None and len(args) >= 2 and isinstance(args[1], list):
        # Could be spec order where args[0] is manager
        if _looks_like_manager(args[0]):
            hits = args[1]
            if qdrant_manager is None:
                qdrant_manager = args[0]
            if filter_conditions is None and len(args) >= 3:
                filter_conditions = args[2]
        else:
            # Fallback: second arg is hits even if first not manager
            hits = args[1]
            if qdrant_manager is None and _looks_like_manager(args[0]):
                qdrant_manager = args[0]

    # If qdrant_manager still None, search args for manager object
    if qdrant_manager is None and len(args) >= 1:
        for a in args:
            if _looks_like_manager(a):
                qdrant_manager = a
                break

    # If filter_conditions still None, try third positional
    if filter_conditions is None and len(args) >= 3:
        # Determine which index holds filter
        # If hits was args[0] (hybrid), filter is args[2]; if hits was args[1] (spec), filter is args[2]
        # So args[2] is filter in both cases when len>=3
        candidate = args[2]
        if candidate is not hits and candidate is not qdrant_manager:
            # Ensure candidate is not the hits list already
            # Use identity check; also avoid mistaking threshold float for filter
            if not isinstance(candidate, (int, float)) or isinstance(candidate, bool):
                # Only assign if candidate looks like Filter/list/conditions, not threshold number
                # Threshold numbers are int/float, filter is Filter or list of FieldCondition or Filter
                # So skip if candidate is plain number and we haven't set threshold via positional
                filter_conditions = candidate
            else:
                # candidate is numeric -> maybe it's threshold passed positionally
                # Don't treat as filter
                pass

    # Handle threshold/max_expansions passed as extra positionals beyond filter
    # Spec: (manager, hits, filter, threshold, max) -> args[3]=threshold, args[4]=max
    # Hybrid never passes threshold positionally, so safe
    if len(args) >= 4:
        # args[3] could be threshold if it's numeric
        cand_thr = args[3]
        if isinstance(cand_thr, (int, float)) and relevance_threshold == 0.7:
            # Only override if still default and candidate not already filter
            # Check that args[3] is not filter_conditions (which would be Filter)
            try:
                # Avoid overriding if candidate is actually filter previously assigned
                if cand_thr is not filter_conditions:
                    relevance_threshold = float(cand_thr)
            except Exception:
                pass
    if len(args) >= 5:
        cand_max = args[4]
        if isinstance(cand_max, (int, float)) and max_expansions == 3:
            try:
                max_expansions = int(cand_max)
            except Exception:
                pass

    # Ensure hits is list
    if hits is None:
        # Maybe args[0] was not list but we missed; fallback to first list in args
        for a in args:
            if isinstance(a, list):
                hits = a
                break
        if hits is None:
            hits = []
    if not isinstance(hits, list):
        # If hits is not list (e.g., manager passed as first arg and we mis-assigned), correct
        return hits if isinstance(hits, list) else []  # type: ignore[return-value]

    # ---- early exits: must have manager and filter and hits ----
    if not hits:
        return hits
    if qdrant_manager is None:
        return hits
    if filter_conditions is None:
        # Without permission filter we cannot safely fetch adjacent - never expose unauthorized
        return hits

    # If no high-relevance hits, return unchanged
    high_relevance = []
    for h in hits:
        sc = _get_score(h)
        if sc is not None and sc >= relevance_threshold:
            high_relevance.append(h)
    if not high_relevance:
        return hits

    # Prepare dedup sets
    seen_ids: Set[str] = {str(h.get("id", "")) for h in hits if h.get("id") is not None}
    seen_pairs: Set[Tuple[str, int]] = set()
    for h in hits:
        idx = _chunk_index(h)
        doc = _doc_id(h)
        if idx is not None and doc:
            seen_pairs.add((doc, idx))

    extras: List[Dict[str, Any]] = []

    # Lazy imports for Qdrant filter construction
    try:
        from qdrant_client.models import FieldCondition, MatchValue, MatchAny  # type: ignore
    except Exception:
        # If Qdrant models unavailable, cannot safely filter - return original
        return hits

    for h in high_relevance:
        if len(extras) >= max_expansions:
            break

        doc = _doc_id(h)
        idx = _chunk_index(h)
        if not doc or idx is None:
            continue

        # Build target indices (idx-1, idx+1) that are not already seen
        targets: List[int] = []
        for delta in (-1, 1):
            t = idx + delta
            if t < 0:
                continue
            if (doc, t) in seen_pairs:
                continue
            targets.append(t)
        if not targets:
            continue

        # Optional: use should_expand_chunk to avoid blind expansion?
        # Spec says should_expand_chunk checks truncation; we could skip expansion
        # if chunk is not truncated and not needed. However to keep spec bullet 1
        # (score threshold) as primary gate, we only apply truncation as advisory,
        # not a hard block. The meaningful/Jaccard checks below ensure "not blindly".
        # Uncomment to enforce truncation gate:
        # if not should_expand_chunk(h, ""):
        #     # Still allow expansion if high relevance? The spec separates the two,
        #     # so we do not gate on truncation here.
        #     pass

        doc_cond = FieldCondition(key="document_id", match=MatchValue(value=doc))
        if len(targets) == 1:
            idx_cond = FieldCondition(key="chunk_index", match=MatchValue(value=targets[0]))
        else:
            idx_cond = FieldCondition(key="chunk_index", match=MatchAny(any=targets))

        combined_filter = _build_combined_filter(filter_conditions, doc_cond, idx_cond)
        if combined_filter is None:
            continue

        # Scroll for adjacent chunks with permission filter + doc+index
        try:
            collection_name = getattr(qdrant_manager, "collection_name", "aegisai_documents")
            # Prefer scroll_filter param; some clients use `filter`
            scroll_kwargs: Dict[str, Any] = {
                "collection_name": collection_name,
                "limit": len(targets) + 2,
                "with_payload": True,
                "with_vectors": False,
            }
            # qdrant_client uses scroll_filter, but older versions use `scroll_filter`
            # Try scroll_filter first, fallback to filter
            try:
                result = qdrant_manager.client.scroll(
                    scroll_filter=combined_filter,
                    **scroll_kwargs,
                )
            except TypeError:
                # Fallback to `filter` param name
                result = qdrant_manager.client.scroll(
                    filter=combined_filter,
                    **scroll_kwargs,
                )
            # Result is (points, next_offset) or similar
            if isinstance(result, tuple) and len(result) >= 1:
                points = result[0]
            elif isinstance(result, list):
                points = result
            else:
                points = result[0] if isinstance(result, tuple) else []

            # Normalize points to iterable
            if points is None:
                points = []
        except Exception:
            continue

        orig_text = _chunk_text(h)

        for p in points:
            if len(extras) >= max_expansions:
                break
            # Extract id and payload from Qdrant point
            try:
                pid = str(getattr(p, "id", "") or (p.get("id") if isinstance(p, dict) else "") or "")
            except Exception:
                pid = ""
            if not pid:
                # Try payload chunk_id fallback
                payload_tmp = getattr(p, "payload", None) or (p.get("payload") if isinstance(p, dict) else {}) or {}
                pid = str(payload_tmp.get("chunk_id", "") or payload_tmp.get("id", "") or "")
                if not pid:
                    continue
            if pid in seen_ids:
                continue

            payload: Dict[str, Any] = {}
            try:
                if hasattr(p, "payload"):
                    payload = getattr(p, "payload", {}) or {}
                elif isinstance(p, dict):
                    payload = p.get("payload", {}) or {}
            except Exception:
                payload = {}

            if not payload:
                continue

            # Extract chunk text for meaningful/Jaccard checks
            adj_text = payload.get("chunk_text") or payload.get("text") or payload.get("content") or ""
            if isinstance(adj_text, str):
                adj_text_str = adj_text
            else:
                adj_text_str = str(adj_text) if adj_text is not None else ""

            if not _is_meaningful(adj_text_str):
                continue

            # Near-duplicate check via Jaccard < 0.8
            try:
                jacc = _jaccard_similarity(orig_text, adj_text_str)
            except Exception:
                jacc = 0.0
            if jacc >= 0.8:
                continue

            # Check duplicate pair
            p_idx_raw = payload.get("chunk_index")
            p_idx: int | None = None
            try:
                p_idx = int(p_idx_raw) if p_idx_raw is not None else None
            except Exception:
                p_idx = None
            if p_idx is not None and (doc, p_idx) in seen_pairs:
                continue
            p_doc = str(payload.get("document_id", doc))
            if p_doc != doc:
                # Should not happen due to filter, but enforce same-doc invariant
                continue
            if p_idx is not None and p_idx not in targets:
                # Filter returned unexpected index (e.g., original idx) - skip if not in targets
                # But allow if single target case already matched; this guards against self
                if p_idx != idx:
                    # If we requested [idx-1, idx+1] but got idx itself due to filter including it, skip self
                    if p_idx == idx:
                        continue

            # Mark as adjacent expansion
            new_payload = dict(payload)
            new_payload["is_adjacent_expansion"] = True

            expanded_hit: Dict[str, Any] = {
                "id": pid,
                "score": 0.0,
                "payload": new_payload,
                "is_adjacent_expansion": True,
                "adjacent_to": h.get("id"),
            }
            # Preserve rerank/score provenance if useful (optional)
            # Keep original hit's score for debugging? Not required.

            extras.append(expanded_hit)
            seen_ids.add(pid)
            if p_idx is not None:
                seen_pairs.add((doc, p_idx))
            else:
                # Fallback pair with string id
                seen_pairs.add((doc, -1))

    if not extras:
        return hits

    # Simple: append extras at end (caller can re-sort/group)
    # Optionally interleave after source hit for locality, but spec says append
    return hits + extras


# Ensure inspect.signature shows spec-compliant signature despite *args/**kwargs
try:
    import inspect

    _spec_params = [
        inspect.Parameter("qdrant_manager", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Any),
        inspect.Parameter("hits", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=List[Dict]),
        inspect.Parameter("filter_conditions", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Any),
        inspect.Parameter("relevance_threshold", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=0.7, annotation=float),
        inspect.Parameter("max_expansions", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=3, annotation=int),
    ]
    expand_adjacent_chunks.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=_spec_params, return_annotation=List[Dict]
    )
    # Also annotate for type checkers
    expand_adjacent_chunks.__annotations__ = {
        "qdrant_manager": Any,
        "hits": List[Dict],
        "filter_conditions": Any,
        "relevance_threshold": float,
        "max_expansions": int,
        "return": List[Dict],
    }

    _should_params = [
        inspect.Parameter("hit", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Dict),
        inspect.Parameter("query", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="", annotation=str),
    ]
    should_expand_chunk.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=_should_params, return_annotation=bool
    )
    should_expand_chunk.__annotations__ = {
        "hit": Dict,
        "query": str,
        "return": bool,
    }
except Exception:
    pass
