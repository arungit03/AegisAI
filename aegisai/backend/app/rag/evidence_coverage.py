"""P16 Evidence Coverage — per sub-question evidence gating.

Measures whether each decomposed sub-question has supporting evidence in the
retrieved hit set. If only 3 of 4 components are covered, the report must
identify the missing component rather than pretending full coverage. Low
coverage feeds confidence scoring and answer generation:

    "I could not find information about X."

Deterministic, no LLM. Uses keyword overlap + retrieval score.

Public API required by spec
----------------------------
- dataclass CoverageReport(total_subquestions, covered, uncovered,
      coverage_ratio, per_question_coverage, summary)
- assess_coverage(subqueries, hits, score_threshold=0.3) -> CoverageReport
- identify_gaps(query, hits) -> List[str]

Backward compat
---------------
Older call sites (hybrid_service.py) call assess_coverage with
(min_overlap, min_score) as positional/keyword args and read
.total_sub_questions / .covered_count / .coverage_ratio / .per_sub_question
/ .gaps. Both old and new names are exposed as aliases so nothing breaks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = [
    "CoverageReport",
    "SubQuestionCoverage",
    "assess_coverage",
    "identify_gaps",
    "coverage_summary",
]

# ---------------------------------------------------------------------------
# Stopwords stripped before keyword extraction
# ---------------------------------------------------------------------------
_STOP = frozenset(
    {
        "what", "who", "when", "where", "why", "how", "which", "whose", "whom",
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "will", "would", "can", "could", "should", "shall",
        "has", "have", "had", "may", "might", "must",
        "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with",
        "about", "from", "this", "that", "these", "those", "it", "its",
        "as", "at", "by", "if", "so", "but", "not", "no", "yes",
        "requires", "require", "required", "needs", "need", "needed",
    }
)

# Very small synonym/alias map so that e.g. "frequency" matches
# "how often" / "how frequently" in hit text without needing embeddings.
_ALIAS_GROUPS: List[frozenset[str]] = [
    frozenset({"frequency", "frequent", "frequently", "often"}),
    frozenset({"inspection", "inspections", "inspect", "inspected"}),
    frozenset({"responsible", "responsibility", "owner", "owns"}),
]


def _aliases(word: str) -> set[str]:
    out = {word}
    for grp in _ALIAS_GROUPS:
        if word in grp:
            out |= set(grp)
    return out


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SubQuestionCoverage:
    """Per-sub-question coverage detail (internal + legacy alias)."""

    sub_question: str
    covered: bool
    hit_count: int
    best_score: float
    supporting_hit_ids: List[str] = field(default_factory=list)

    # Keyword overlap of the best supporting hit (useful for debugging).
    best_overlap: float = 0.0


@dataclass
class CoverageReport:
    """Phase 2 coverage report.

    Spec fields (required):
        total_subquestions, covered, uncovered, coverage_ratio,
        per_question_coverage, summary

    Legacy aliases (kept for backward compat with existing callers):
        total_sub_questions  -> total_subquestions
        covered_count        -> covered
        gaps                 -> uncovered
        per_sub_question     -> per_question_coverage
        overall_sufficient   -> derived from coverage_ratio
    """

    # Spec names (primary)
    total_subquestions: int = 0
    covered: int = 0
    uncovered: List[str] = field(default_factory=list)
    coverage_ratio: float = 0.0
    per_question_coverage: List[Tuple[str, bool, List[Dict[str, Any]]]] = field(default_factory=list)
    summary: str = ""

    # Legacy/detail fields (kept, populated alongside spec fields)
    per_sub_question: List[SubQuestionCoverage] = field(default_factory=list)
    overall_sufficient: bool = False
    gaps: List[str] = field(default_factory=list)

    # Compatibility aliases as properties — dataclass fields above are canonical,
    # but callers that read .total_sub_questions / .covered_count still work
    # because we mirror values into both sets of fields in assess_coverage.
    @property
    def total_sub_questions(self) -> int:  # noqa: D102  (compat)
        return self.total_subquestions

    @property
    def covered_count(self) -> int:  # noqa: D102  (compat)
        return self.covered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keywords(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    return {t for t in toks if t not in _STOP}


def _hit_score(hit: Dict[str, Any]) -> float:
    for k in ("rerank_score", "rrf_score", "score"):
        if k in hit and hit[k] is not None:
            try:
                return float(hit[k])
            except Exception:
                continue
    return 0.0


def _hit_text(hit: Dict[str, Any]) -> str:
    # Hits may be raw dicts (payload.chunk_text) or SearchHit dataclasses
    if isinstance(hit, dict):
        payload = hit.get("payload") or {}
        txt = payload.get("chunk_text", hit.get("chunk_text", hit.get("text", "")))
        return str(txt or "").lower()
    # Dataclass / object with attributes
    for attr in ("chunk_text", "text", "content"):
        if hasattr(hit, attr):
            try:
                return str(getattr(hit, attr) or "").lower()
            except Exception:
                pass
    return ""


def _hit_id(hit: Dict[str, Any]) -> str:
    if isinstance(hit, dict):
        return str(hit.get("id", (hit.get("payload") or {}).get("id", "")))
    return str(getattr(hit, "id", ""))


def _keyword_overlap_ratio(query_keywords: set[str], hit_text_lower: str) -> float:
    """Fraction of query keywords (or their aliases) present in hit text."""
    if not query_keywords:
        return 0.0
    hit_tokens = set(re.findall(r"[a-z0-9]{3,}", hit_text_lower))
    matched = 0
    for kw in query_keywords:
        variants = _aliases(kw)
        if any(v in hit_text_lower for v in variants) or (variants & hit_tokens):
            matched += 1
        else:
            if any(any(v[:5] == ht[:5] for ht in hit_tokens) for v in variants if len(v) >= 5):
                matched += 1
    return matched / len(query_keywords)


# Heuristic generic terms that appear across many sub-questions in a
# multi-part query (shared context like "inspection", "policy", "equipment").
# A hit that only matches one of these should not count as coverage.
_GENERIC_SHARED = frozenset({"inspection", "inspections", "policy", "equipment", "report", "document", "procedure"})


def _distinctive_overlap(query_keywords: set[str], hit_text_lower: str) -> float:
    """Fraction of *distinctive* (non-generic) query keywords present in hit text.

    Returns 0.0 when the query has no distinctive keywords (all generic), or
    when none of the distinctive ones are found. Used to prevent spurious
    coverage where a hit only shares the shared-context word.
    """
    distinctive = {k for k in query_keywords if k.lower() not in _GENERIC_SHARED}
    if not distinctive:
        return 0.0
    hit_tokens = set(re.findall(r"[a-z0-9]{3,}", hit_text_lower))
    matched = 0
    for kw in distinctive:
        variants = _aliases(kw)
        if any(v in hit_text_lower for v in variants) or (variants & hit_tokens):
            matched += 1
        else:
            if any(any(v[:5] == ht[:5] for ht in hit_tokens) for v in variants if len(v) >= 5):
                matched += 1
    return matched / len(distinctive) if distinctive else 0.0


# ---------------------------------------------------------------------------
# Core: assess_coverage
# ---------------------------------------------------------------------------

def assess_coverage(
    subqueries: List[str],
    hits: List[Dict[str, Any]],
    score_threshold: float = 0.3,
    # Legacy kwarg aliases — accepted so existing positional/keyword call sites
    # keep working after the spec-driven signature change.
    min_score: float | None = None,
    min_overlap: int | None = None,  # legacy: integer threshold (converted)
    keyword_threshold: float | None = None,
) -> CoverageReport:
    """Assess per-sub-question evidence coverage.

    A sub-question is **covered** iff there exists at least one hit where:

        hit_score > score_threshold  OR  keyword_overlap > keyword_threshold

    Per spec: keyword overlap threshold is 0.3 and score threshold is 0.3.
    For multi-keyword sub-questions that share a generic term (e.g.
    "inspection"), a hit must match at least one *distinctive* keyword — a hit
    sharing only the generic term does not count, so the missing component is
    correctly surfaced (P16).

    Args:
        subqueries: Decomposed sub-questions (from query_decomposition).
        hits: Retrieved hits (raw dicts with .payload/.score or SearchHit objects).
        score_threshold: Minimum retrieval/reranker score to count as covered.
        min_score: Legacy alias for score_threshold.
        min_overlap: Legacy alias — if provided, keyword_threshold is derived
            as min_overlap / avg_keywords and keyword_threshold kwarg wins.
        keyword_threshold: Minimum keyword overlap ratio (default 0.3).

    Returns:
        CoverageReport with spec fields populated.
    """
    # Resolve legacy aliases
    if min_score is not None:
        score_threshold = float(min_score)
    if keyword_threshold is None:
        if min_overlap is not None:
            keyword_threshold = 0.3 if int(min_overlap) <= 1 else 0.45
        else:
            keyword_threshold = 0.3
    keyword_threshold = float(keyword_threshold)

    hits = hits or []
    subqueries = [s for s in (subqueries or []) if s and s.strip()]

    # No subqueries (single-question case): report a single synthetic entry.
    if not subqueries:
        covered = False
        best_score_overall = max((_hit_score(h) for h in hits), default=0.0)
        for h in hits:
            if _hit_score(h) > score_threshold:
                covered = True
                break
        report = CoverageReport(
            total_subquestions=1,
            covered=1 if covered else 0,
            uncovered=[] if covered else ["No evidence found for the question"],
            coverage_ratio=1.0 if covered else 0.0,
            per_question_coverage=[
                ("(overall)", covered, [h for h in hits if _hit_score(h) > score_threshold])
            ],
            summary=(
                "Evidence covers the question."
                if covered else
                "Evidence does not cover the question. Missing: No evidence found for the question."
            ),
            per_sub_question=[
                SubQuestionCoverage(
                    sub_question="(overall)",
                    covered=covered,
                    hit_count=len([h for h in hits if _hit_score(h) > score_threshold]),
                    best_score=round(best_score_overall, 4),
                    supporting_hit_ids=[_hit_id(h) for h in hits if _hit_score(h) > score_threshold][:5],
                    best_overlap=0.0,
                )
            ],
            overall_sufficient=covered,
            gaps=[] if covered else ["No evidence found for the question"],
        )
        return report

    per_detail: List[SubQuestionCoverage] = []
    per_spec: List[Tuple[str, bool, List[Dict[str, Any]]]] = []
    uncovered: List[str] = []

    for sq in subqueries:
        kws = _keywords(sq)
        if not kws:
            kws = set(re.findall(r"[a-z0-9]{2,}", sq.lower())) - _STOP
            if not kws:
                kws = set(re.findall(r"[a-z0-9]{2,}", sq.lower()))

        supporting: List[Dict[str, Any]] = []
        best_score = 0.0
        best_overlap = 0.0

        for h in hits:
            txt = _hit_text(h)
            sc = _hit_score(h)
            overlap = _keyword_overlap_ratio(kws, txt) if kws and txt else 0.0
            best_overlap = max(best_overlap, overlap)
            best_score = max(best_score, sc)

            # Distinctive vs generic: hits that only share a highly generic
            # shared term (e.g. "inspection" in every inspection sub-question)
            # must also contain at least one distinctive keyword.
            distinctive_overlap = _distinctive_overlap(kws, txt) if kws and txt else 0.0
            has_distinctive = distinctive_overlap > 0.0
            # All-generic sub-questions (e.g. "What is the inspection?"):
            # fall back to any keyword match so they remain coverable.
            if not any(k.lower() not in _GENERIC_SHARED for k in kws):
                has_distinctive = overlap > 0.0

            # Effective keyword threshold — multi-keyword sub-questions that
            # share a generic term need a higher bar so that the generic alone
            # (e.g. 1/3 = 0.33) does not pass. Single-distinctive queries keep
            # the spec 0.3 bar.
            effective_kw_thresh = keyword_threshold
            if len(kws) >= 3:
                effective_kw_thresh = max(keyword_threshold, 0.5 if len(kws) >= 4 else 0.6)
            elif len(kws) == 2 and any(k.lower() in _GENERIC_SHARED for k in kws):
                # e.g. {"inspection","tools"} — "inspection" alone must not pass
                if not has_distinctive:
                    effective_kw_thresh = 1.1  # impossible — blocks generic-only

            # Keyword path: overall overlap exceeds effective threshold AND at
            # least one distinctive keyword is present (unless all-generic).
            if len(kws) <= 2 and not any(k.lower() in _GENERIC_SHARED for k in kws):
                # Short, fully distinctive queries (no generic) — spec rule
                # applies verbatim: overlap > 0.3 is enough.
                keyword_pass = overlap > keyword_threshold
            else:
                keyword_pass = (overlap > effective_kw_thresh and has_distinctive)
                # Lenient fallback when distinctive present but overall overlap
                # is modest — distinctive is the real signal.
                if not keyword_pass and has_distinctive and overlap > keyword_threshold:
                    if len(kws) >= 3 and effective_kw_thresh > keyword_threshold:
                        keyword_pass = True

            # Score path: high retrieval score counts only when it agrees with
            # keyword evidence — otherwise every hit's high score would spuriously
            # cover every sub-question. Require a distinctive keyword.
            score_pass = (sc > score_threshold and has_distinctive)
            if keyword_pass or score_pass:
                supporting.append(h)

        is_covered = len(supporting) > 0
        supporting_ids = [_hit_id(h) for h in supporting[:5]]

        per_detail.append(
            SubQuestionCoverage(
                sub_question=sq,
                covered=is_covered,
                hit_count=len(supporting),
                best_score=round(best_score, 4),
                supporting_hit_ids=supporting_ids,
                best_overlap=round(best_overlap, 3),
            )
        )
        per_spec.append((sq, is_covered, supporting))
        if not is_covered:
            uncovered.append(sq)

    total = len(subqueries)
    covered_count = sum(1 for d in per_detail if d.covered)
    ratio = (covered_count / total) if total else 0.0

    # Summary — exactly the form required by the spec.
    if total == 0:
        summary = "No sub-questions to assess."
    elif not uncovered:
        summary = f"Evidence covers {covered_count} of {total} sub-questions."
    elif len(uncovered) == 1:
        summary = (
            f"Evidence covers {covered_count} of {total} sub-questions. "
            f"Missing: {uncovered[0]}"
        )
    else:
        missing_str = "; ".join(uncovered)
        summary = (
            f"Evidence covers {covered_count} of {total} sub-questions. "
            f"Missing: {missing_str}"
        )

    # Special sentence for full-failure — still starts with required prefix.
    if covered_count == 0 and total > 0:
        summary = (
            f"Evidence covers 0 of {total} sub-questions. "
            f"Missing: {'; '.join(uncovered)}"
        )

    report = CoverageReport(
        total_subquestions=total,
        covered=covered_count,
        uncovered=list(uncovered),
        coverage_ratio=round(ratio, 3),
        per_question_coverage=per_spec,
        summary=summary,
        per_sub_question=per_detail,
        overall_sufficient=ratio >= 0.75,
        gaps=list(uncovered),
    )
    return report


# ---------------------------------------------------------------------------
# Single-query gap identification
# ---------------------------------------------------------------------------

def _split_aspects(query: str) -> List[str]:
    """Split a single query into aspects (clauses) for gap detection."""
    q = (query or "").strip()
    if not q:
        return []
    parts = re.split(r"\s*;\s*|\s*,\s*and\s+|\s+and\s+|\s*,\s*|\?", q)
    aspects = [p.strip().rstrip(".,;:") for p in parts if p and p.strip()]
    filtered: List[str] = []
    for a in aspects:
        toks = [t for t in re.findall(r"[A-Za-z0-9]+", a) if t.lower() not in _STOP]
        if toks:
            filtered.append(a)
    return filtered if len(filtered) > 1 else [q]


def identify_gaps(
    query: str,
    hits: List[Dict[str, Any]],
    score_threshold: float = 0.3,
    keyword_threshold: float = 0.3,
) -> List[str]:
    """Identify aspects of a single query that lack supporting evidence.

    Uses low-scoring retrieval as the signal: an aspect is a gap when no hit
    matches it above *either* the score threshold or the keyword overlap
    threshold.

    This function keeps the simple ``(query, hits) -> List[str]`` signature
    required by the spec. Internally it reuses :func:`assess_coverage` so the
    two paths cannot drift.

    For callers that already have a :class:`CoverageReport` (e.g.
    ``hybrid_service``), ``identify_gaps(report)``-style calls are also
    supported via duck-typing: if the first argument looks like a
    CoverageReport (has ``.gaps`` / ``.uncovered``), its gaps are returned
    directly.
    """
    # Duck-type: caller passed a CoverageReport instead of a string query.
    if not isinstance(query, str):
        for attr in ("gaps", "uncovered"):
            if hasattr(query, attr):
                val = getattr(query, attr)
                if isinstance(val, list):
                    return list(val)
        return []

    hits = hits or []
    q = (query or "").strip()
    if not q:
        return []

    # No hits at all — the whole query is the gap.
    if not hits:
        return [q]

    # Decompose into aspects and assess each.
    aspects = _split_aspects(q)

    # Single-aspect query: check if *any* hit adequately supports it.
    # If max score is very low AND keyword overlap is weak, it's a gap.
    if len(aspects) <= 1:
        kws = _keywords(q)
        best_overlap = 0.0
        best_score = 0.0
        for h in hits:
            txt = _hit_text(h)
            best_score = max(best_score, _hit_score(h))
            if kws and txt:
                best_overlap = max(best_overlap, _keyword_overlap_ratio(kws, txt))
        if best_score > score_threshold or best_overlap > keyword_threshold:
            return []
        meaningful = " ".join(sorted(kws)) if kws else q
        return [meaningful or q]

    # Multi-aspect: reuse assess_coverage per aspect.
    report = assess_coverage(
        aspects,
        hits,
        score_threshold=score_threshold,
        keyword_threshold=keyword_threshold,
    )
    return list(report.uncovered)


def coverage_summary(report: CoverageReport) -> str:
    """Pretty multi-line summary for logs/debugging (not the spec .summary)."""
    lines = [
        f"Coverage: {report.covered}/{report.total_subquestions} "
        f"({report.coverage_ratio:.0%})"
    ]
    details = report.per_sub_question or []
    if details:
        for d in details:
            mark = "OK" if d.covered else "MISS"
            lines.append(
                f"  {mark} {d.sub_question} -- hits:{d.hit_count} "
                f"best:{d.best_score} overlap:{getattr(d, 'best_overlap', 0):.2f}"
            )
    else:
        for sq, is_covered, supporting in report.per_question_coverage:
            mark = "OK" if is_covered else "MISS"
            lines.append(f"  {mark} {sq} -- hits:{len(supporting)}")
    if report.uncovered:
        lines.append(f"Gaps: {'; '.join(report.uncovered)}")
    if report.summary:
        lines.append(report.summary)
    return "\n".join(lines)
