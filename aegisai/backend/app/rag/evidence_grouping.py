"""P9 Multi-Document Evidence Grouping — Phase 2.

Group selected evidence by source document/pages; use during context
construction while maintaining source attribution. Complements
``app.rag.fusion.build_context`` and ``app.rag.types.SearchHit``.

Every chunk remains traceable to its document/page via the original hit dict
(payload.document_id / payload.page_number) and via grouped context markers.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class EvidenceGroup:
    """Evidence chunks grouped by source document.

    Attributes:
        document_id: Source document identifier (str of payload.document_id).
        filename: Source filename for display / headers.
        document_title: Optional human title (payload.document_title or title).
        chunks: Hits belonging to this document, sorted by page then chunk_index.
        page_range: (min_page, max_page) or (None, None) when pages absent.
        total_chars: Sum of chunk_text lengths in group (for budgeting).
    """

    document_id: str
    filename: str
    document_title: Optional[str]
    chunks: List[Dict]
    page_range: Tuple[Optional[int], Optional[int]]
    total_chars: int


def _doc_id(hit: Dict) -> str:
    payload = hit.get("payload") or {}
    return str(payload.get("document_id", "unknown"))


def _score(hit: Dict) -> float:
    for k in ("rerank_score", "rrf_score", "score"):
        if k in hit and hit[k] is not None:
            try:
                return float(hit[k])
            except Exception:
                continue
    return 0.0


def _page_number(hit: Dict) -> Optional[int]:
    v = (hit.get("payload") or {}).get("page_number")
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _chunk_index(hit: Dict) -> Optional[int]:
    v = (hit.get("payload") or {}).get("chunk_index")
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def group_evidence_by_document(hits: List[Dict]) -> List[EvidenceGroup]:
    """Group hits by document_id, sorting chunks within each group.

    Sort key per group: (page_number then chunk_index). Missing values sort
    first (0 sentinel) to keep undated chunks at the top without breaking
    attribution — original page_number is preserved in the hit.

    Each group records its page range and total_chars.

    Args:
        hits: Ranked hits as produced by fusion/reranking. Each hit is a
            dict with at least ``payload`` containing ``document_id``,
            ``filename``, ``chunk_text``, optional ``page_number``,
            ``chunk_index``, ``document_title``/``title``.

    Returns:
        List of EvidenceGroup, one per distinct document_id (order = first
        appearance in ``hits``; use ``sort_groups_by_relevance`` to order).
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for h in hits:
        groups[_doc_id(h)].append(h)

    out: List[EvidenceGroup] = []
    for doc_id, lst in groups.items():
        # Sort within group by page_number then chunk_index for reading order.
        # Use large sentinel for None so unpaged chunks sort before paged? Spec
        # says sort by page_number then chunk_index; we use 0 sentinel to keep
        # missing first (consistent with typical chunk_index ordering).
        def _sort_key(h: Dict):
            pg = _page_number(h)
            ci = _chunk_index(h)
            return (pg if pg is not None else 0, ci if ci is not None else 0)

        lst.sort(key=_sort_key)

        pages = [_page_number(h) for h in lst]
        pages = [p for p in pages if p is not None]
        page_range: Tuple[Optional[int], Optional[int]] = (
            (min(pages), max(pages)) if pages else (None, None)
        )

        total_chars = 0
        for h in lst:
            payload = h.get("payload") or {}
            total_chars += len(str(payload.get("chunk_text", "") or ""))

        payload0 = lst[0].get("payload") or {}
        filename = str(payload0.get("filename", "unknown"))
        document_title = payload0.get("document_title")
        if document_title is None:
            document_title = payload0.get("title")

        out.append(
            EvidenceGroup(
                document_id=doc_id,
                filename=filename,
                document_title=document_title,
                chunks=lst,
                page_range=page_range,
                total_chars=total_chars,
            )
        )
    return out


def sort_groups_by_relevance(groups: List[EvidenceGroup]) -> List[EvidenceGroup]:
    """Sort groups by their best chunk score descending.

    Best score per group is max over chunks of rerank_score > rrf_score > score.
    Maintains source attribution (groups are reordered, chunks untouched).

    Args:
        groups: Evidence groups.

    Returns:
        New list sorted descending by best chunk score.
    """

    def _best(g: EvidenceGroup) -> float:
        return max((_score(c) for c in g.chunks), default=0.0)

    return sorted(groups, key=_best, reverse=True)


def build_grouped_context(
    groups: List[EvidenceGroup], max_chars: int = 12000
) -> Tuple[str, List[Dict]]:
    """Build LLM context string grouped by document, respecting max_chars.

    Format per group:
        === Document: filename (pages X-Y) ===
        [p.X] chunk_text
        [p.Y] chunk_text
        ...

    - Document header example matches spec: ``=== Document: filename (pages X-Y) ===``.
      Single-page groups use ``(pages X)``. Groups without page info omit the
      pages suffix. Title appended as `` — title`` when present.
    - Chunks carry page markers ``[p.X]`` for source attribution.
    - Respects ``max_chars``: groups are considered in relevance order
      (best chunk score descending) so lowest-relevance groups are truncated
      first. If a group's header would exceed budget, remaining groups are
      skipped. If a chunk would exceed budget, the chunk is truncated with
      ``...`` when enough budget remains, then building stops.

    Args:
        groups: Evidence groups (any order; will be sorted by relevance).
        max_chars: Hard cap on returned context string length.

    Returns:
        (context_string, included_hits) — hits actually materialized in order.
    """
    # Lowest-relevance truncation: sort descending so least relevant at end.
    groups = sort_groups_by_relevance(groups)

    parts: List[str] = []
    included: List[Dict] = []
    total = 0  # chars accounted for including separators

    for g in groups:
        # Build header: "=== Document: filename (pages X-Y) ==="
        header = f"=== Document: {g.filename}"
        if g.document_title:
            header += f" — {g.document_title}"
        if g.page_range[0] is not None:
            if g.page_range[0] == g.page_range[1]:
                header += f" (pages {g.page_range[0]})"
            else:
                header += f" (pages {g.page_range[0]}-{g.page_range[1]})"
        header += " ==="

        # +2 for "\n\n" separator before this header (except first)
        header_need = len(header) + (2 if parts else 0)
        if total + header_need > max_chars:
            break
        parts.append(header)
        total += header_need

        for ch in g.chunks:
            payload = ch.get("payload") or {}
            page = payload.get("page_number")
            # Page marker "[p.X]" for attribution; empty when no page.
            tag = f"[p.{page}]" if page is not None else ""
            chunk_text = str(payload.get("chunk_text", "") or "")
            block = f"{tag} {chunk_text}".strip() if tag else chunk_text

            need = len(block) + 2  # +2 for separator
            if total + need > max_chars:
                remaining = max_chars - total - (2 if parts else 0)
                # Only emit a truncated block if enough budget to be useful.
                if remaining > 40:
                    truncated = block[:remaining] + "..."
                    parts.append(truncated)
                    included.append(ch)
                break
            parts.append(block)
            included.append(ch)
            total += need

    context = "\n\n".join(parts)
    return context, included
