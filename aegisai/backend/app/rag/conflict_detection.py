"""Conflict detection — P15: never merge contradictions.

Detects contradictory quantitative/date facts across documents (e.g. 30 days vs
60 days for the same concept). Reports both sides with citations. Only
auto-resolves when explicit metadata establishes authority (newer version,
more recent date, higher classification, explicit supersedes marker, active
status). Otherwise resolution is None and both values are presented.

Detection is conservative and pattern-based (no heavy NLP): numeric + unit
and date patterns grouped by normalized unit. Same-document contradictions are
ignored unless cross-document. Context-keyword overlap or query relevance is
required to avoid flagging unrelated quantities that share a unit.

Example:
    Note: Sources conflict on 'response time' - Handbook.pdf (p.5) states
    "30 days" while Policy_v2.pdf (p.12) states "60 days"
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public dataclass — spec shape + backward-compatible aliases
# ---------------------------------------------------------------------------

@dataclass
class Conflict:
    """A detected conflict between evidence chunks.

    Spec fields:
        field: concept/topic that conflicts (e.g. "response time", "day")
        values: distinct conflicting values with source label
                each tuple is (display_value, source_label) e.g. ("30 days", "DocA.pdf (p.5)")
        conflicting_hits: all hits involved
        resolution: winning value display string if authoritative, else None
        explanation: human-readable explanation (why conflict, why resolved or not)

    Extra (backward-compatible, not in spec but required by existing callers):
        severity: low | medium | high  (high when query directly asks about field)
        topic / supporting_hits / description are aliases for field / conflicting_hits / explanation
    """

    field: str
    values: List[Tuple[str, str]]
    conflicting_hits: List[Dict[str, Any]]
    resolution: Optional[str]
    explanation: str
    severity: str = "medium"

    # --- backward-compatible aliases (hybrid_service, confidence) ------------
    @property
    def topic(self) -> str:
        return self.field

    @property
    def supporting_hits(self) -> List[Dict[str, Any]]:
        return self.conflicting_hits

    @property
    def description(self) -> str:
        return self.explanation


# ---------------------------------------------------------------------------
# Patterns — conservative, numeric/date primarily
# ---------------------------------------------------------------------------

# e.g. "30 days", "60 days", "5 hours", "12 months", "3 weeks", "50 percent", "50%"
_QUANTITY_RE = re.compile(
    r"(\b\d+(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|hours?|minutes?|percent|%)\b)",
    re.IGNORECASE,
)

_MONEY_RE = re.compile(
    r"\$\s*\d[\d,\.]*|\b\d[\d,\.]*\s*(?:USD|EUR|dollars?)\b",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
)

# explicit authority markers in chunk_text / payload
_SUPERSEDES_RE = re.compile(r"supersede", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Helpers: normalization, parsing, metadata
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "which",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "with",
        "about",
        "from",
        "that",
        "this",
        "these",
        "those",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "it",
        "its",
        "as",
        "at",
        "by",
        "if",
        "so",
        "than",
        "then",
        "there",
        "their",
        "they",
        "them",
        "we",
        "you",
        "your",
        "our",
        "are",
        "was",
    }
)

CLASS_RANK: Dict[str, int] = {
    "public_internal": 1,
    "public": 1,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
    "highly_restricted": 4,
}


def _normalize_quantity(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _parse_version(v: Any) -> Tuple[int, ...]:
    if v is None:
        return (0,)
    s = str(v).strip()
    nums = re.findall(r"\d+", s)
    if not nums:
        return (0,)
    try:
        return tuple(int(n) for n in nums)
    except Exception:
        return (0,)


def _parse_date(d: Any) -> Optional[datetime]:
    if not d:
        return None
    if isinstance(d, datetime):
        return d
    s = str(d).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[: len(fmt)], fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _doc_id_of(hit: Dict[str, Any]) -> str:
    return str((hit.get("payload") or {}).get("document_id", "unknown"))


def _payload(hit: Dict[str, Any]) -> Dict[str, Any]:
    return hit.get("payload") or {}


def _hit_source_label(hit: Dict[str, Any]) -> str:
    p = _payload(hit)
    fn = str(p.get("filename") or p.get("document_title") or p.get("document_id") or "unknown")
    page = p.get("page_number")
    if page is not None:
        try:
            return f"{fn} (p.{int(page)})"
        except Exception:
            return f"{fn} (p.{page})"
    return fn


def _has_supersedes(hit: Dict[str, Any]) -> bool:
    p = _payload(hit)
    # payload-level explicit marker
    for key in ("supersedes", "superseded_by", "supersede", "replaces", "replaced_by"):
        if p.get(key):
            # superseded_by indicates this hit is the old one, not authoritative
            if key == "superseded_by":
                return False
            return True
        # also check key name contains supersede
        for k in list(p.keys()):
            if "supersede" in k.lower() and p.get(k):
                return True
    text = str(p.get("chunk_text", "")).lower()
    # "supersede" covers supersedes / superseded / superseding
    # But "superseded by" means this doc is superseded -> not authoritative
    if "superseded by" in text:
        return False
    if _SUPERSEDES_RE.search(text):
        return True
    return False


def _authority_key(hit: Dict[str, Any]) -> Tuple[int, Tuple[int, int, int], float, int, int]:
    """Comparable authority tuple: (supersedes, version(3), timestamp, class_rank, active)."""
    p = _payload(hit)
    supersedes = 1 if _has_supersedes(hit) else 0

    # version: try several keys
    raw_ver = p.get("version")
    if raw_ver is None:
        raw_ver = p.get("document_version") or p.get("ver") or p.get("doc_version")
    ver_tuple = _parse_version(raw_ver)
    # pad to 3 for stable comparison
    ver_padded = tuple(list(ver_tuple) + [0] * (3 - len(ver_tuple)))[:3]

    date_str = p.get("updated_at") or p.get("created_at") or p.get("date") or p.get("timestamp")
    dt = _parse_date(date_str)
    ts = dt.timestamp() if dt else 0.0

    class_rank = CLASS_RANK.get(str(p.get("classification", "")).lower().strip(), 0)

    active = 0
    status = str(p.get("status", "")).lower().strip()
    if status == "active" or p.get("is_active") or p.get("active") or p.get("is_current"):
        active = 1
    # also treat explicit "active": true string
    if str(p.get("active", "")).lower() == "true":
        active = 1

    return (supersedes, ver_padded, ts, class_rank, active)


def _extract_concept(text: str, start: int, end: int) -> str:
    """Infer concept hint from up to 80 chars before the match.

    Takes last up to 2 non-stopword tokens before the quantity.
    Falls back to empty string if no hint.
    """
    window = text[max(0, start - 80): start]
    toks = re.findall(r"[a-zA-Z0-9]{2,}", window.lower())
    filtered = [t for t in toks if t not in _STOPWORDS]
    if filtered:
        # last 2 tokens -> "response time" etc
        hint = " ".join(filtered[-2:])
        # avoid returning the unit itself as concept
        if hint in ("day", "days", "hour", "hours", "week", "weeks", "month", "months", "year", "years", "percent"):
            return ""
        return hint
    return ""


def _context_keywords(context: str, exclude_unit: str = "") -> set[str]:
    toks = re.findall(r"[a-z0-9]{3,}", context.lower())
    out = {t for t in toks if t not in _STOPWORDS}
    # exclude unit tokens to avoid trivial overlap on "days"
    if exclude_unit:
        base = exclude_unit.lower().rstrip("s")
        out.discard(base)
        out.discard(base + "s")
        out.discard("percent")
        out.discard("%")
    return out


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

def _extract_claims_from_hit(hit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract quantity/money/date claims from a single hit."""
    payload = _payload(hit)
    text = str(payload.get("chunk_text") or "")
    if not text.strip():
        return []
    claims: List[Dict[str, Any]] = []

    # Quantity: "30 days", "5 hours" etc
    for m in _QUANTITY_RE.finditer(text):
        raw = m.group(1) if m.lastindex and m.group(1) else m.group(0)
        raw = raw.strip()
        if not raw:
            continue
        norm = _normalize_quantity(raw)
        # numeric
        num_m = re.search(r"\d+(?:\.\d+)?", raw)
        try:
            num = float(num_m.group()) if num_m else None
        except Exception:
            num = None
        # unit
        unit_m = re.search(r"(days?|weeks?|months?|years?|hours?|minutes?|percent|%)", raw, re.IGNORECASE)
        unit_raw = unit_m.group(1) if unit_m else ""
        unit_norm = unit_raw.lower().rstrip("s") if unit_raw.lower() != "%" else "percent"
        if unit_norm == "%":
            unit_norm = "percent"
        # concept + context
        concept = _extract_concept(text, m.start(), m.end())
        ctx_start = max(0, m.start() - 80)
        ctx_end = min(len(text), m.end() + 80)
        context = text[ctx_start:ctx_end]
        claims.append(
            {
                "raw": raw,
                "norm": norm,
                "num": num,
                "unit_raw": unit_raw,
                "unit_norm": unit_norm or "__quantity__",
                "concept": concept,
                "context": context,
                "start": m.start(),
                "end": m.end(),
                "kind": "quantity",
            }
        )

    # Money
    for m in _MONEY_RE.finditer(text):
        raw = m.group(0).strip()
        if not raw:
            continue
        norm = _normalize_quantity(raw)
        num_m = re.search(r"\d[\d,\.]*", raw)
        num_str = num_m.group().replace(",", "") if num_m else ""
        try:
            num = float(num_str) if num_str else None
        except Exception:
            num = None
        concept = _extract_concept(text, m.start(), m.end())
        ctx_start = max(0, m.start() - 80)
        ctx_end = min(len(text), m.end() + 80)
        context = text[ctx_start:ctx_end]
        claims.append(
            {
                "raw": raw,
                "norm": norm,
                "num": num,
                "unit_raw": "money",
                "unit_norm": "__money__",
                "concept": concept,
                "context": context,
                "start": m.start(),
                "end": m.end(),
                "kind": "money",
            }
        )

    # Date
    for m in _DATE_RE.finditer(text):
        raw = m.group(0).strip()
        if not raw:
            continue
        norm = raw.strip().lower()
        concept = _extract_concept(text, m.start(), m.end())
        ctx_start = max(0, m.start() - 80)
        ctx_end = min(len(text), m.end() + 80)
        context = text[ctx_start:ctx_end]
        claims.append(
            {
                "raw": raw,
                "norm": norm,
                "num": None,
                "unit_raw": "date",
                "unit_norm": "__date__",
                "concept": concept,
                "context": context,
                "start": m.start(),
                "end": m.end(),
                "kind": "date",
            }
        )

    return claims


def _claim_display_key(claim: Dict[str, Any]) -> str:
    """Key for grouping distinct values.

    For quantity: numeric + unit (so '30 days' and '30.0 days' collapse).
    For money/date: normalized raw string.
    """
    if claim["kind"] == "quantity" and claim["num"] is not None:
        # Use :g to normalize 30.0 -> 30
        try:
            num_str = ("%g" % claim["num"])
        except Exception:
            num_str = str(claim["num"])
        return f"{num_str} {claim['unit_norm']}"
    return claim["norm"]


def _infer_field_for_unit(
    unit_norm: str,
    claims_in_group: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    query: str,
) -> str:
    """Derive human field name for a unit group.

    Prefers most common concept hint; falls back to unit name.
    For special units uses descriptive defaults.
    """
    concepts: List[str] = []
    for _hit, claim in claims_in_group:
        c = (claim.get("concept") or "").strip().lower()
        if c:
            concepts.append(c)
    # Most common concept
    if concepts:
        counter = Counter(concepts)
        most_common, _ = counter.most_common(1)[0]
        # Use concept if it adds information beyond unit
        if most_common and len(most_common) >= 2:
            # If query mentions the concept, prefer it; otherwise still use it
            # as it describes the field better than bare unit.
            # e.g. concept="response time" + unit "day" -> field "response time"
            return most_common

    # No concept -> fallback to descriptive unit
    if unit_norm == "__money__":
        return "monetary amount"
    if unit_norm == "__date__":
        return "date"
    # For quantity, unit_norm is singular like "day", "hour", "percent"
    # Keep singular for field, but values retain original plural/raw
    return unit_norm


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_conflicts(
    hits: List[Dict[str, Any]],
    query: str = "",
    question: Optional[str] = None,
    **kwargs: Any,
) -> List[Conflict]:
    """Detect conflicting quantitative/date claims across different documents.

    Only flags when:
    - Same normalized unit (e.g. both 'days') but different numeric/date values
    - Values come from different documents (cross-doc)
    - Context overlap or query relevance indicates same concept (conservative)

    Never merges values (e.g. 30 days + 60 days -> 45 days). Both are reported.
    Auto-resolves only when explicit metadata (supersedes / version / date /
    classification / active) unambiguously prefers one side.

    Args:
        hits: List of hit dicts with {id, score, payload:{chunk_text, document_id,
              filename, page_number, version, updated_at, classification, ...}}
        query: User query (preferred name). For backward compat, `question`
               kwarg is also accepted.

    Returns:
        List of Conflict objects (empty if none).
    """
    # Backward compat: hybrid_service calls with question=...
    effective_query = query
    if question is not None:
        effective_query = question if not effective_query else effective_query
    # also accept **kwargs question/query
    if not effective_query and "question" in kwargs:
        effective_query = str(kwargs.get("question") or "")
    if not effective_query and "query" in kwargs:
        effective_query = str(kwargs.get("query") or "")
    effective_query = str(effective_query or "")
    query_lower = effective_query.lower()

    if not hits or len(hits) < 2:
        return []

    # Step 1: extract claims per hit
    # hit_claims: list of (hit, claim_dict)
    all_claims: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        claims = _extract_claims_from_hit(h)
        for c in claims:
            all_claims.append((h, c))

    if len(all_claims) < 2:
        return []

    # Step 2: group by unit_norm
    unit_groups: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for hit, claim in all_claims:
        unit = claim.get("unit_norm") or "__unknown__"
        unit_groups[unit].append((hit, claim))

    conflicts: List[Conflict] = []

    for unit_norm, items in unit_groups.items():
        if len(items) < 2:
            continue

        # Group by display key (distinct values)
        val_to_entries: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
        # Also keep raw display per key (most common raw)
        val_raw_examples: Dict[str, str] = {}
        for hit, claim in items:
            key = _claim_display_key(claim)
            val_to_entries[key].append((hit, claim))
            # track example raw for display
            if key not in val_raw_examples:
                val_raw_examples[key] = claim.get("raw", key)
            else:
                # keep first; could count frequency but first is fine
                pass

        if len(val_to_entries) < 2:
            continue  # no distinct values

        # Cross-doc check: must have >=2 distinct document_ids across keys
        docs_per_val: Dict[str, set[str]] = {}
        for k, entries in val_to_entries.items():
            docs_per_val[k] = {_doc_id_of(h) for h, _ in entries}
        all_docs: set[str] = set().union(*docs_per_val.values()) if docs_per_val else set()
        if len(all_docs) < 2:
            continue  # same document, not a cross-doc conflict

        # Conservative filter: require shared context keywords OR query relevance
        # Build keyword sets per distinct value (union of contexts)
        keywords_per_val: Dict[str, set[str]] = {}
        for k, entries in val_to_entries.items():
            kw_union: set[str] = set()
            for _hit, claim in entries:
                ctx = claim.get("context", "")
                kw = _context_keywords(ctx, exclude_unit=unit_norm)
                kw_union |= kw
                # also add concept words
                concept = claim.get("concept", "")
                if concept:
                    for tok in re.findall(r"[a-z0-9]{3,}", concept.lower()):
                        if tok not in _STOPWORDS:
                            kw_union.add(tok)
            keywords_per_val[k] = kw_union

        # Check pairwise overlap
        has_overlap = False
        keys = list(val_to_entries.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if keywords_per_val[keys[i]] & keywords_per_val[keys[j]]:
                    has_overlap = True
                    break
            if has_overlap:
                break

        # Query relevance: does query mention unit or any keyword?
        query_relevant = False
        if query_lower:
            # unit mention (handle plural)
            if unit_norm not in ("__money__", "__date__", "__unknown__"):
                if unit_norm in query_lower or (unit_norm + "s") in query_lower or (unit_norm.rstrip("s") in query_lower):
                    query_relevant = True
            # money/date query hints
            if unit_norm == "__money__" and any(w in query_lower for w in ("price", "cost", "amount", "dollar", "usd", "payment", "fee")):
                query_relevant = True
            if unit_norm == "__date__" and any(w in query_lower for w in ("date", "deadline", "when", "until", "effective")):
                query_relevant = True
            # keyword overlap with query
            if not query_relevant:
                qtoks = set(re.findall(r"[a-z0-9]{3,}", query_lower))
                qtoks = {t for t in qtoks if t not in _STOPWORDS}
                for k in keys:
                    if qtoks & keywords_per_val[k]:
                        query_relevant = True
                        break
                # also concept in query
                if not query_relevant:
                    for k in keys:
                        for _hit, claim in val_to_entries[k]:
                            concept = (claim.get("concept") or "").lower()
                            if concept and concept in query_lower:
                                query_relevant = True
                                break

        # If neither overlap nor query relevance, skip (conservative)
        if not has_overlap and not query_relevant:
            # Allow exception: if units are rare (money/date) and we have strong cross-doc
            # still report? For now skip to stay conservative.
            logger.debug("conflict_skipped_no_overlap", unit=unit_norm, keys=keys)
            continue

        # We have a genuine conflict for this unit
        # Build values: List[Tuple[str,str]]
        values: List[Tuple[str, str]] = []
        # For deduplication of conflicting_hits, track by hit id
        seen_hit_ids: set[str] = set()
        conflicting_hits: List[Dict[str, Any]] = []

        # Determine field name
        inferred_field = _infer_field_for_unit(unit_norm, items, effective_query)
        pretty_unit = unit_norm if not unit_norm.startswith("__") else unit_norm.strip("_")
        field_name = inferred_field or pretty_unit

        # For explanation we want source labels per value
        for key in sorted(val_to_entries.keys()):
            entries = val_to_entries[key]
            display_val = val_raw_examples.get(key, key)
            # Collect unique source labels for this value
            src_labels_set: set[str] = set()
            src_labels_ordered: List[str] = []
            for hit, _claim in entries:
                label = _hit_source_label(hit)
                if label not in src_labels_set:
                    src_labels_set.add(label)
                    src_labels_ordered.append(label)
                # collect hit for conflicting_hits (dedup)
                hid = str(hit.get("id", "")) or str(_doc_id_of(hit)) + ":" + display_val
                # Use hit id if present, else object id
                unique_key = str(hit.get("id", "")) if hit.get("id") else str(id(hit))
                if unique_key not in seen_hit_ids:
                    seen_hit_ids.add(unique_key)
                    conflicting_hits.append(hit)
            src_label_str = ", ".join(src_labels_ordered) if src_labels_ordered else "unknown"
            values.append((display_val, src_label_str))

        # Sort values for determinism (by display value)
        # But keep conflicting_hits order as encountered
        # values already sorted by key; for quantity, numeric order may be better
        # Try numeric sort for quantity
        if all(v[0] and re.search(r"\d", v[0]) for v in values):
            try:
                def _sort_key(t: Tuple[str, str]) -> float:
                    m = re.search(r"\d+(?:\.\d+)?", t[0])
                    return float(m.group()) if m else 0.0

                values = sorted(values, key=_sort_key)
            except Exception:
                pass

        # Severity
        severity = "medium"
        if query_lower and (field_name.lower() in query_lower or pretty_unit.lower() in query_lower):
            severity = "high"

        # Resolution: check authority
        resolution: Optional[str] = None
        explanation = ""
        # Build per-value best authority
        best_key_per_val: Dict[str, Tuple[int, Tuple[int, int, int], float, int, int]] = {}
        best_hit_per_val: Dict[str, Dict[str, Any]] = {}
        for key, entries in val_to_entries.items():
            # best hit for this value
            best_hit = max((h for h, _ in entries), key=_authority_key)
            best_key_per_val[key] = _authority_key(best_hit)
            best_hit_per_val[key] = best_hit

        # Find overall max
        # Use max key
        max_key = max(best_key_per_val.values()) if best_key_per_val else None
        winners = [k for k, ak in best_key_per_val.items() if ak == max_key]
        # Only resolve if unique winner and max_key is not all zeros (i.e. has explicit metadata)
        has_explicit = False
        if max_key is not None:
            # check if any dimension beyond default indicates explicit
            # supersede=0, version=(0,0,0), ts=0.0, class=0, active=0 => no explicit
            if max_key != (0, (0, 0, 0), 0.0, 0, 0):
                has_explicit = True

        if len(winners) == 1 and has_explicit:
            winner_key = winners[0]
            winner_display = val_raw_examples.get(winner_key, winner_key)
            # Ensure winner is not just default due to missing metadata on all?
            # Verify winner's authority strictly greater than runner-up in at least one dimension
            runner_keys = [ak for k, ak in best_key_per_val.items() if k != winner_key]
            if runner_keys:
                runner_max = max(runner_keys)
                if max_key > runner_max:
                    resolution = winner_display
                    # Determine reason (first differing dimension)
                    reason_labels = ["explicit supersedes marker", "newer version", "more recent date", "higher classification authority", "active status"]
                    winner_hit = best_hit_per_val[winner_key]
                    # Find first index where winner > runner
                    reason = "authoritative metadata"
                    for idx, label in enumerate(reason_labels):
                        # Compare element at idx
                        # version is tuple, need lexicographic compare
                        w_elem = max_key[idx]
                        r_elem = runner_max[idx]
                        if w_elem != r_elem and w_elem > r_elem:
                            reason = label
                            break
                    # Build explanation with resolution
                    winner_src = _hit_source_label(winner_hit)
                    # Collect loser sources for explanation
                    loser_parts = []
                    for k in val_to_entries.keys():
                        if k == winner_key:
                            continue
                        loser_srcs = ", ".join(sorted({_hit_source_label(h) for h, _ in val_to_entries[k]}))
                        loser_parts.append(f'"{val_raw_examples.get(k, k)}" from {loser_srcs}')
                    losers_str = " vs ".join(loser_parts) if loser_parts else ""
                    explanation = (
                        f"Sources conflict on '{field_name}': "
                        + " vs ".join(f'"{v}" from {src}' for v, src in values)
                        + f". Resolved to \"{resolution}\" from {winner_src} based on {reason}."
                    )
        if resolution is None:
            # No authoritative winner -> explain conflict without preference, never merge
            distinct_str = " vs ".join(f'"{v}" from {src}' for v, src in values)
            explanation = (
                f"Sources conflict on '{field_name}': {distinct_str}. "
                "No newer version, higher classification authority, or explicit supersedes marker "
                "establishes precedence; both values are presented without merging or averaging."
            )

        conflicts.append(
            Conflict(
                field=field_name,
                values=values,
                conflicting_hits=conflicting_hits,
                resolution=resolution,
                explanation=explanation,
                severity=severity,
            )
        )
        logger.info(
            "conflict_detected",
            field=field_name,
            unit=unit_norm,
            values=[v for v, _ in values],
            resolution=resolution,
            severity=severity,
            sources=len(conflicting_hits),
        )

    return conflicts


def format_conflict_for_answer(conflicts: List[Conflict]) -> str:
    """Produce human-readable conflict note for inclusion in answer.

    Format per conflict:
        Note: Sources conflict on 'X' - Document A (p.5) states "30 days"
        while Document B (p.12) states "60 days" [optional resolution note]

    Multiple conflicts are joined with newlines. Empty input returns "".

    This never averages or merges values.
    """
    if not conflicts:
        return ""
    lines: List[str] = []
    for c in conflicts:
        field_name = c.field
        # Build per-value phrases: 'Source states "value"'
        parts: List[str] = []
        for val, src in c.values:
            # src already contains filename + page if available
            parts.append(f'{src} states "{val}"')

        if len(parts) == 2:
            body = f"{parts[0]} while {parts[1]}"
        elif len(parts) > 2:
            # Join with commas, last with "while"
            body = ", ".join(parts[:-1]) + f" while {parts[-1]}"
        else:
            body = parts[0] if parts else ""

        line = f'Note: Sources conflict on \'{field_name}\' - {body}'
        if c.resolution:
            line += f' (resolved to "{c.resolution}" based on authoritative metadata)'
        lines.append(line)
        # Optionally add explanation on next line for full transparency?
        # Keep concise for answer; explanation is available in Conflict.explanation
        # and in retrieval_debug. Do not add extra line to avoid verbosity.
    return "\n".join(lines)


def has_high_severity_conflict(conflicts: List[Conflict]) -> bool:
    """Backward-compatible helper for confidence.py / hybrid_service."""
    return any(getattr(c, "severity", "") == "high" for c in conflicts)

