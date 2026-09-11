"""Document version handling — use version/date metadata intelligently.

P14 Document Version Handling:
- Current policy prefers latest authoritative version per document_id
  (highest version number or most recent created_at).
- Historical query allows historical versions — never silently discard historical.
- If authority / version metadata is missing/unknown, include all and annotate
  as "version unknown"; do not invent authority.
- If filtering versions, log which versions were excluded.

Version info comes from DocumentVersion model, but Qdrant payload may carry
a ``version`` / ``created_at`` / ``updated_at`` field directly.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

try:
    from app.core.logging import get_logger

    logger = get_logger(__name__)
except Exception:  # fallback when app context not available (tests, scripts)
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data type
# ---------------------------------------------------------------------------


@dataclass
class VersionInfo:
    """Structured version metadata for a single hit / document."""

    document_id: str
    version: int
    created_at: Optional[str]
    is_latest: bool


# ---------------------------------------------------------------------------
# Historical-query detection (P14)
# ---------------------------------------------------------------------------

# Requirement examples that MUST be detected:
#   "previous version", "what was before", "history of", "older version"
# Broader historical signals are also recognised.
_HISTORICAL_RE = re.compile(
    r"("
    r"previous\s+version"
    r"|older\s+version"
    r"|history\s+of"
    r"|what\s+was\s+before"
    r"|what\s+was\b"  # catches "what was X before"
    r"|previous\b"
    r"|older\b"
    r"|earlier\b"
    r"|historical\b"
    r"|\bhistory\b"
    r"|\bprior\b"
    r"|\bformer\b"
    r"|\bold\s+version\b"
    r"|\bversion\s+history\b"
    r"|\bversion\s*\d+\b"
    r"|\bas\s+of\b"
    r"|\bbefore\s+\d{4}\b"
    r")",
    re.IGNORECASE,
)

_LATEST_RE = re.compile(
    r"\b(?:latest|current|newest|most\s+recent|up[\s-]?to[\s-]?date)\b",
    re.IGNORECASE,
)


def _effective_query(query: str = "", **kwargs: Any) -> str:
    """Resolve effective query text from flexible call signatures.

    Supports both the spec signature ``query`` and the legacy ``question``
    keyword used by ``hybrid_service.py`` so existing call-sites remain
    compatible without breaking the spec signature.
    """
    if query:
        return query
    # Legacy aliases — hybrid_service passes question=, tests may pass q=/text=
    for key in ("question", "query", "q", "text"):
        val = kwargs.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def is_historical_query(query: str = "", **kwargs: Any) -> bool:
    """Return True when the user is asking for historical information.

    Detects queries such as "previous version", "what was before",
    "history of", "older version" and related historical signals.
    Matching is case-insensitive and tolerant of extra whitespace.

    ``question`` is accepted as an alias for ``query`` for backward
    compatibility with callers that use ``question=`` (e.g. hybrid_service).
    """
    q = _effective_query(query, **kwargs)
    if not q:
        return False
    return bool(_HISTORICAL_RE.search(q))


def is_latest_query(query: str = "", **kwargs: Any) -> bool:
    """Return True when the user explicitly asks for the latest/current version."""
    q = _effective_query(query, **kwargs)
    if not q:
        return False
    return bool(_LATEST_RE.search(q))


# ---------------------------------------------------------------------------
# Helpers — version / date parsing and payload access
# ---------------------------------------------------------------------------


def _get_payload(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Return the Qdrant payload dict for a hit (always a dict)."""
    p = hit.get("payload")
    if isinstance(p, dict):
        return p
    return {}


def _extract_document_id(hit: Dict[str, Any]) -> str:
    """Extract a stable grouping key for a hit.

    Primary: ``payload.document_id``
    Fallback: ``payload.filename`` / ``payload.document_title`` / ``hit.document_id``
    Last resort: ``"unknown"`` — callers will treat this as its own group and
    include all hits (do not invent authority).
    """
    payload = _get_payload(hit)
    for key in ("document_id", "document_title", "filename"):
        val = payload.get(key)
        if val:
            return str(val)
    # also check top-level hit keys (some callers flatten)
    for key in ("document_id", "filename"):
        val = hit.get(key)
        if val:
            return str(val)
    return "unknown"


def _get_created_at(payload: Dict[str, Any]) -> Optional[Any]:
    """Return the most relevant timestamp field from a payload."""
    for key in ("created_at", "updated_at", "date", "createdAt", "updatedAt", "timestamp"):
        val = payload.get(key)
        if val:
            return val
    return None


def _parse_version(v: Any) -> tuple:
    """Parse a version value into a comparable tuple of ints.

    Examples: 3 -> (3,), "v2.1" -> (2, 1), "version 2" -> (2,), None -> (0,)
    Non-numeric or missing values yield (0,) so they sort lowest without
    inventing authority.
    """
    if v is None:
        return (0,)
    s = str(v).strip()
    if not s:
        return (0,)
    nums = re.findall(r"\d+", s)
    if not nums:
        return (0,)
    try:
        return tuple(int(n) for n in nums)
    except Exception:
        return (0,)


def _parse_date(d: Any) -> Optional[datetime]:
    """Parse a date value into a datetime or None if unparseable."""
    if not d:
        return None
    if isinstance(d, datetime):
        return d
    s = str(d).strip()
    if not s:
        return None
    # Try common formats (prefix match so "2024-01-15T..." still works via slicing)
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(s[: len(fmt)], fmt)
        except Exception:
            continue
    # ISO 8601 with timezone
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _coerce_version_int(v: Any) -> int:
    """Coerce a version value to int for VersionInfo (0 when unknown)."""
    if v is None:
        return 0
    # Prefer tuple leading component
    tup = _parse_version(v)
    if tup and tup != (0,):
        return tup[0]
    try:
        return int(float(str(v).strip()))
    except Exception:
        return 0


def _annotate_hit(
    hit: Dict[str, Any],
    *,
    document_id: Optional[str] = None,
    version_unknown: bool = False,
    is_latest: bool = False,
) -> None:
    """Annotate a single hit in-place with version metadata.

    - Ensures ``hit["payload"]`` exists.
    - Adds ``payload["version_info"]`` / ``payload["version_status"]`` for
      downstream visibility.
    - Adds top-level ``hit["version_info"]`` (VersionInfo) and
      ``hit["version_display"]`` for convenience.
    - When ``version_unknown`` is True, payload annotation is the literal
      string "version unknown" per spec; no authority is invented.
    """
    payload = _get_payload(hit)
    # Ensure hit carries a payload dict
    hit["payload"] = payload

    doc_id = document_id or _extract_document_id(hit)

    if version_unknown:
        payload["version_info"] = "version unknown"
        payload["version_status"] = "version unknown"
        hit["version_display"] = "version unknown"
        hit["version_info"] = VersionInfo(
            document_id=str(doc_id),
            version=0,
            created_at=None,
            is_latest=False,
        )
        # Also expose a serialisable form for JSON logging
        payload["version_info_struct"] = {
            "document_id": str(doc_id),
            "version": 0,
            "created_at": None,
            "is_latest": False,
            "status": "version unknown",
        }
        return

    # Known version path
    v_raw = payload.get("version")
    d_raw = _get_created_at(payload)
    d_str: Optional[str] = str(d_raw) if d_raw is not None else None
    v_int = _coerce_version_int(v_raw)

    vi = VersionInfo(
        document_id=str(doc_id),
        version=v_int,
        created_at=d_str,
        is_latest=bool(is_latest),
    )
    hit["version_info"] = vi
    payload["is_latest"] = bool(is_latest)
    payload["version_info_struct"] = {
        "document_id": vi.document_id,
        "version": vi.version,
        "created_at": vi.created_at,
        "is_latest": vi.is_latest,
    }

    # Human-readable display e.g. "v3 (2024-01-15)"
    if v_raw is not None or d_raw is not None:
        disp = f"v{v_raw}" if v_raw is not None else ""
        if d_raw is not None:
            disp = (disp + f" ({d_raw})").strip()
        hit["version_display"] = disp
    else:
        hit["version_display"] = ""


def _sort_key_for_hit(hit: Dict[str, Any]) -> tuple:
    """Comparable key: (version_tuple, date). Highest wins as latest."""
    payload = _get_payload(hit)
    v = _parse_version(payload.get("version"))
    d = _parse_date(_get_created_at(payload))
    # datetime.min is smaller than any real date, so undated items sort last
    return (v, d or datetime.min)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_document_versions(
    hits: List[Dict[str, Any]],
    query: str = "",
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Resolve document versions according to query intent.

    Args:
        hits: Raw Qdrant-style hits (each with ``id`` / ``payload`` / ``score``).
        query: User query text. For backward compatibility the query may also be
            passed as ``question=`` keyword or as a positional string.

    Returns:
        Filtered/annotated list of hits:
        - If the query is historical (see :func:`is_historical_query`), all
          versions are returned grouped by document_id with version metadata
          annotated. No hits are silently discarded.
        - Otherwise (current policy, the default) only the latest version's
          chunks per document_id are returned. Hits with unknown version
          metadata are included and annotated as "version unknown".
        - Whenever versions are filtered, the excluded versions are logged.
    """
    if not hits:
        return hits

    effective_query = _effective_query(query, **kwargs)
    # Allow legacy prefer_latest override (hybrid_service does not use it,
    # but existing tests or callers might)
    prefer_latest = kwargs.get("prefer_latest", True)

    # Historical query → keep all versions, just annotate + log
    if is_historical_query(effective_query):
        # Annotate every hit with is_latest flag relative to its group so
        # downstream consumers can distinguish latest vs historical without
        # losing any historical chunks.
        try:
            groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for h in hits:
                groups[_extract_document_id(h)].append(h)

            annotated: List[Dict[str, Any]] = []
            for doc_id, lst in groups.items():
                has_meta = any(
                    (_get_payload(x).get("version") is not None)
                    or (_get_created_at(_get_payload(x)) is not None)
                    for x in lst
                )
                if not has_meta:
                    for h in lst:
                        _annotate_hit(h, document_id=doc_id, version_unknown=True)
                        annotated.append(h)
                    continue
                # Determine latest key within group for is_latest flag
                try:
                    best_key = max(_sort_key_for_hit(x) for x in lst)
                except Exception:
                    best_key = None
                for h in lst:
                    is_latest = (_sort_key_for_hit(h) == best_key) if best_key is not None else False
                    _annotate_hit(h, document_id=doc_id, version_unknown=False, is_latest=is_latest)
                    annotated.append(h)

            logger.info(
                "version_handling_historical_query",
                effective_query_preview=effective_query[:80] if isinstance(effective_query, str) else "",
                total_hits=len(hits),
                retained=len(annotated),
                groups=len(groups),
                strategy="historical_keep_all",
            )
            return annotated
        except Exception as exc:
            logger.warning(
                "version_handling_historical_annotate_failed",
                error_type=type(exc).__name__,
                total_hits=len(hits),
            )
            # Fallback: return hits annotated as unknown where needed
            for h in hits:
                payload = _get_payload(h)
                if payload.get("version") is None and _get_created_at(payload) is None:
                    _annotate_hit(h, version_unknown=True)
                else:
                    _annotate_hit(h, is_latest=False)
            return hits

    # Current / default policy: prefer latest version per document_id
    if not prefer_latest:
        # Caller explicitly opted out of prefer_latest but query is not
        # historical → include all with annotation (never silently discard)
        for h in hits:
            payload = _get_payload(h)
            has_meta = payload.get("version") is not None or _get_created_at(payload) is not None
            if not has_meta:
                _annotate_hit(h, version_unknown=True)
            else:
                _annotate_hit(h, is_latest=False)
        logger.info(
            "version_handling_prefer_latest_disabled",
            total_hits=len(hits),
            strategy="include_all",
        )
        return hits

    # Delegate to prefer_latest_version which implements the per-doc filtering
    # and mandatory exclusion logging.
    return prefer_latest_version(hits)


def prefer_latest_version(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate hits by document_id, keeping only the latest version's chunks.

    - "Latest" is the highest version number; ties broken by most recent
      created_at / updated_at. This mirrors DocumentVersion.version ordering.
    - Hits with missing/unknown version metadata are never silently discarded;
      they are included and their payload is annotated as "version unknown".
    - Each retained hit's payload is enriched with version info (VersionInfo
      and is_latest flag) so downstream context building can surface it.
    - If any versions are excluded, they are logged (never silently discarded).

    Args:
        hits: Raw Qdrant-style hits.

    Returns:
        Filtered list containing only latest-version chunks (plus all
        version-unknown hits), with version metadata annotated in-place.
    """
    if not hits:
        return hits

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for h in hits:
        groups[_extract_document_id(h)].append(h)

    result: List[Dict[str, Any]] = []

    for doc_id, lst in groups.items():
        has_version = any(_get_payload(x).get("version") is not None for x in lst)
        has_date = any(_get_created_at(_get_payload(x)) is not None for x in lst)
        has_meta = has_version or has_date

        # Unknown authority → include all, annotate as "version unknown"
        if not has_meta:
            for h in lst:
                _annotate_hit(h, document_id=doc_id, version_unknown=True)
                result.append(h)
            logger.info(
                "version_handling_version_unknown",
                document_id=str(doc_id),
                count=len(lst),
                annotation="version unknown",
                strategy="include_all_no_authority",
            )
            continue

        # Find the latest sort key within this document group
        try:
            best = max(lst, key=_sort_key_for_hit)
            best_key = _sort_key_for_hit(best)
        except Exception as exc:
            logger.warning(
                "version_handling_sort_failed",
                document_id=str(doc_id),
                error_type=type(exc).__name__,
                strategy="include_all_fallback",
            )
            for h in lst:
                _annotate_hit(h, document_id=doc_id, version_unknown=False, is_latest=False)
                result.append(h)
            continue

        best_payload = _get_payload(best)
        best_version = best_payload.get("version")
        best_date = _get_created_at(best_payload)

        kept: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []
        for h in lst:
            try:
                if _sort_key_for_hit(h) == best_key:
                    kept.append(h)
                else:
                    excluded.append(h)
            except Exception:
                # On comparison failure, keep the hit (do not invent discard)
                kept.append(h)

        # Safety: if comparison yielded no kept hits (should not happen), keep all
        if not kept:
            logger.warning(
                "version_handling_no_kept_fallback",
                document_id=str(doc_id),
                total=len(lst),
                strategy="include_all_fallback",
            )
            for h in lst:
                _annotate_hit(h, document_id=doc_id, version_unknown=False, is_latest=False)
                result.extend(lst)
            continue

        # Annotate kept hits as latest
        for h in kept:
            _annotate_hit(h, document_id=doc_id, version_unknown=False, is_latest=True)
        result.extend(kept)

        # Never silently discard — log excluded versions with details
        if excluded:
            excluded_versions = []
            for e in excluded:
                ep = _get_payload(e)
                excluded_versions.append(
                    {
                        "id": str(e.get("id", ""))[:40],
                        "version": ep.get("version"),
                        "created_at": str(_get_created_at(ep)) if _get_created_at(ep) is not None else None,
                    }
                )
            logger.info(
                "version_handling_filtered",
                document_id=str(doc_id),
                kept_version=best_version,
                kept_created_at=str(best_date) if best_date is not None else None,
                kept_count=len(kept),
                excluded_count=len(excluded),
                excluded_versions=excluded_versions,
                strategy="prefer_latest",
            )
        else:
            # No filtering was needed for this document
            logger.debug(
                "version_handling_no_filter_needed",
                document_id=str(doc_id),
                count=len(lst),
                kept_version=best_version,
            )

    return result


def annotate_version_info(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add human-readable version_display and structured VersionInfo to hits.

    Kept for backward compatibility with callers that imported this helper from
    the earlier implementation. Delegates to :func:`_annotate_hit` for each hit.
    """
    for h in hits:
        payload = _get_payload(h)
        has_meta = payload.get("version") is not None or _get_created_at(payload) is not None
        if not has_meta:
            _annotate_hit(h, version_unknown=True)
        else:
            # Preserve is_latest if already set, otherwise False (non-filtering annotate)
            is_latest = bool(payload.get("is_latest", False))
            _annotate_hit(h, version_unknown=False, is_latest=is_latest)
    return hits


__all__ = [
    "VersionInfo",
    "is_historical_query",
    "is_latest_query",
    "resolve_document_versions",
    "prefer_latest_version",
    "annotate_version_info",
]
