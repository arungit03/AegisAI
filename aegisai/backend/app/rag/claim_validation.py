"""P19 Claim-to-Evidence Validation — lightweight local groundedness check.

Pipeline:
    Answer -> extract claims -> map to evidence -> validate support ->
    remove / qualify / mark unsupported; never fabricate.

All local — keyword / entity overlap (Jaccard), no NLI model or external API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Union

# ---------------------------------------------------------------------------
# Public dataclass — spec required
# ---------------------------------------------------------------------------

@dataclass
class ClaimValidation:
    """Per-claim validation result (P19 spec)."""

    claim: str
    supported: bool
    supporting_hits: List[Dict[str, Any]]
    confidence: float
    action: str  # "keep" | "qualify" | "remove"


# ---------------------------------------------------------------------------
# Backward-compat dataclasses (used by hybrid_service legacy caller)
# ---------------------------------------------------------------------------

@dataclass
class ClaimResult:
    claim: str
    grounded: bool
    supporting_hit_id: Optional[str] = None
    overlap: int = 0
    reason: str = ""


@dataclass
class ClaimValidationReport:
    total_claims: int
    grounded_count: int
    grounded_ratio: float
    claims: List[ClaimResult] = field(default_factory=list)
    ungrounded_claims: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Regex / constants
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[(?:Source:[^\]]+|p\.\d+)\]", re.IGNORECASE)

# Sentence boundary: split after . ! ? followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Tokens for Jaccard: alphanumeric 3+ chars, lower-cased.
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

# Entities: Capitalised words, all-caps acronyms, numbers / percentages.
# Lightweight heuristic — not an NER model.
_ENTITY_RE = re.compile(
    r"\b[A-Z][a-zA-Z0-9]{1,}\b"  # Capitalised token e.g. Paris, Qdrant
    r"|\b[A-Z]{2,}\b"             # Acronym e.g. AI, RAG
    r"|\b\d+(?:[.,]\d+)?%?\b"     # Number / percent
)

# Hedges / boilerplate that should not be treated as verifiable claims.
_HEDGE_PHRASES = (
    "i could not find",
    "i couldn't find",
    "i couldnt find",
    "i was unable to find",
    "i am unable to find",
    "i don't have",
    "i do not have",
    "i couldn't locate",
    "i could not locate",
    "no information",
    "no relevant information",
    "unable to find",
    "cannot find",
    "cannot answer",
    "insufficient information",
    "not enough information",
    "no evidence",
    "not found",
    "no sources found",
    "don't have enough",
    "do not have enough",
    "as an ai",
    "i am an ai",
    "i don't know",
    "i do not know",
    "there is no information",
    "there are no",
    "no relevant",
    "not available",
    "cannot provide",
)

# Thresholds
_JACCARD_THRESHOLD = 0.20
_QUALIFY_THRESHOLD = 0.08  # below this -> remove, between -> qualify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hit_text(hit: Dict[str, Any]) -> str:
    """Extract chunk_text from a hit dict (supports both flat and payload form)."""
    if not isinstance(hit, dict):
        return ""
    payload = hit.get("payload")
    if isinstance(payload, dict) and "chunk_text" in payload:
        return str(payload.get("chunk_text") or "")
    # fallback: top-level chunk_text
    if "chunk_text" in hit:
        return str(hit.get("chunk_text") or "")
    # fallback: string representation
    return ""


def _token_set(text: str) -> Set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _entity_set(text: str) -> Set[str]:
    # lower-case entities for case-insensitive comparison, but keep original
    # shape detection via regex which is case-sensitive
    return set(m.group(0).lower() for m in _ENTITY_RE.finditer(text))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _is_hedge(sentence: str) -> bool:
    low = sentence.lower()
    for phrase in _HEDGE_PHRASES:
        if phrase in low:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API — spec required
# ---------------------------------------------------------------------------

def extract_claims(answer: str) -> List[str]:
    """Split answer into individual verifiable claims (sentences/facts).

    - Strips citation markers like ``[Source: ...]``.
    - Splits on sentence boundaries (``. ! ?`` + whitespace).
    - Filters out hedges / boilerplate such as ``"I could not find"``
      and very short fragments.

    Args:
        answer: Raw LLM answer string.

    Returns:
        List of claim strings (each retains its trailing punctuation).
    """
    if not answer or not answer.strip():
        return []

    # Remove citation markers for clean claim extraction
    stripped = _CITATION_RE.sub("", answer)

    # Normalise whitespace (preserve sentence boundaries)
    stripped = stripped.strip()
    if not stripped:
        return []

    raw = [s.strip() for s in _SENTENCE_SPLIT_RE.split(stripped) if s.strip()]

    claims: List[str] = []
    for s in raw:
        # Normalise internal whitespace
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) < 12:
            continue
        if _is_hedge(s):
            continue
        # Also filter boilerplate prefixes (legacy behaviour)
        low = s.lower()
        if low.startswith(("i couldn't find", "i don't have", "as an ai", "i am an ai")):
            continue
        claims.append(s)

    return claims


def validate_claims(
    answer: Union[str, List[str]],
    hits: List[Dict[str, Any]],
) -> List[ClaimValidation]:
    """Validate each claim against retrieved evidence.

    For each claim, checks whether any hit supports it via keyword / entity
    overlap.  A claim is *supported* when ``Jaccard > 0.2`` or there is an
    entity match (e.g. same capitalised term / number appears in both).

    Never fabricates supporting evidence — if no hit meets the threshold
    the claim is marked ``supported=False`` with empty ``supporting_hits``.

    Args:
        answer: Either the raw answer string (spec) or, for backward
            compatibility, a pre-split ``List[str]`` of claims as passed by
            ``hybrid_service``.
        hits: Retrieved evidence hits (each has ``payload.chunk_text`` or
            ``chunk_text``).

    Returns:
        List of :class:`ClaimValidation` — one per claim — with fields
        ``supported``, ``supporting_hits``, ``confidence`` (Jaccard score),
        and ``action`` (``"keep"`` | ``"qualify"`` | ``"remove"``).
    """
    # --- backward-compat: caller may pass List[str] claims directly ---
    if isinstance(answer, list):
        claims: List[str] = [str(c).strip() for c in answer if str(c).strip()]
    else:
        if not answer or not answer.strip():
            return []
        claims = extract_claims(answer)
        if not claims:
            return []

    # No evidence at all -> every claim unsupported, no fabrication
    if not hits:
        return [
            ClaimValidation(
                claim=c,
                supported=False,
                supporting_hits=[],
                confidence=0.0,
                action="remove",
            )
            for c in claims
        ]

    # Pre-compute hit token / entity sets
    hit_data: List[tuple[Dict[str, Any], Set[str], Set[str]]] = []
    for h in hits:
        txt = _hit_text(h)
        hit_data.append((h, _token_set(txt), _entity_set(txt)))

    results: List[ClaimValidation] = []
    for claim in claims:
        ctoks = _token_set(claim)
        cents = _entity_set(claim)

        if not ctoks:
            results.append(
                ClaimValidation(
                    claim=claim,
                    supported=False,
                    supporting_hits=[],
                    confidence=0.0,
                    action="remove",
                )
            )
            continue

        supporting: List[Dict[str, Any]] = []
        best_conf = 0.0

        for h, htoks, hents in hit_data:
            j = _jaccard(ctoks, htoks)
            # Entity match: any shared capitalised term / number
            entity_match = bool(cents & hents) if cents and hents else False

            # Track best confidence regardless of support (for qualify decision)
            if j > best_conf:
                best_conf = j
            # If entity matches but Jaccard is 0, give a modest confidence
            # so the claim can be qualified rather than silently removed.
            if entity_match and j == 0.0 and best_conf < 0.15:
                best_conf = max(best_conf, 0.15)

            if j > _JACCARD_THRESHOLD or entity_match:
                supporting.append(h)
                # ensure best_conf reflects supporting threshold
                if j > _JACCARD_THRESHOLD and j > best_conf:
                    best_conf = j

        supported = len(supporting) > 0
        confidence = round(float(best_conf), 3)

        if supported:
            action = "keep"
        else:
            # Borderline overlap -> qualify (keep with hedge), else remove
            if confidence >= _QUALIFY_THRESHOLD:
                action = "qualify"
            else:
                action = "remove"

        results.append(
            ClaimValidation(
                claim=claim,
                supported=supported,
                supporting_hits=supporting,
                confidence=confidence,
                action=action,
            )
        )

    return results


def filter_unsupported_claims(
    answer: str,
    validations: Union[List[ClaimValidation], ClaimValidationReport],
) -> str:
    """Filter an answer based on claim validation results.

    - ``action == "remove"`` and ``supported == False`` -> sentence removed.
    - ``action == "qualify"`` -> sentence kept but prepended with
      ``"Based on limited evidence, ..."``.
    - ``action == "keep"`` (or supported) -> sentence kept verbatim.

    Never fabricates evidence — unsupported claims are either removed or
    explicitly hedged.

    Args:
        answer: Original answer string (used as fallback when validations
            is empty).
        validations: List of :class:`ClaimValidation` (spec) or, for
            backward compatibility, a :class:`ClaimValidationReport`.

    Returns:
        Filtered answer string.
    """
    # --- backward-compat: ClaimValidationReport ---
    if isinstance(validations, ClaimValidationReport):
        # Delegate to legacy behaviour: keep only grounded claims
        report: ClaimValidationReport = validations
        if report.grounded_ratio >= 1.0 or not report.ungrounded_claims:
            return answer
        grounded = [r.claim for r in report.claims if r.grounded]
        if not grounded:
            return answer  # don't empty the answer; let caller decide
        return " ".join(grounded)

    # Spec path: List[ClaimValidation]
    if not validations:
        return answer

    parts: List[str] = []
    for v in validations:
        # Defensive: ensure action is one of expected values
        action = v.action if v.action in ("keep", "qualify", "remove") else (
            "keep" if v.supported else "remove"
        )

        if action == "remove" and not v.supported:
            continue

        if action == "qualify":
            # Don't double-prefix if already hedged
            claim_text = v.claim.strip()
            low = claim_text.lower()
            if low.startswith("based on limited evidence"):
                parts.append(claim_text)
            else:
                # Lower-case first character after prefix for grammar
                if claim_text:
                    qualified = f"Based on limited evidence, {claim_text[0].lower() + claim_text[1:]}"
                else:
                    qualified = claim_text
                parts.append(qualified)
            continue

        # keep
        parts.append(v.claim)

    if not parts:
        # All claims removed — return empty string to signal no grounded
        # content remains.  Callers that prefer to keep the original answer
        # (e.g. to show an insufficient response) can check for "".
        return ""

    # Join with single space; claims already contain their own punctuation
    filtered = " ".join(parts)
    # Normalise whitespace without collapsing sentence punctuation
    filtered = re.sub(r"\s+", " ", filtered).strip()
    # Clean up stray space before punctuation introduced by joins
    filtered = re.sub(r"\s+([.,!?;:])", r"\1", filtered)
    return filtered


# ---------------------------------------------------------------------------
# Optional legacy adapter: validate_claims overload that matches old
# hybrid_service call site  validate_claims(claims: List[str], hits)
# is handled above via Union check.  Expose a helper for typed callers
# that still import ClaimValidationReport.
# ---------------------------------------------------------------------------

def _legacy_validate_claims_report(
    claims: List[str],
    hits: List[Dict[str, Any]],
    min_overlap: int = 2,
    min_overlap_ratio: float = 0.35,
) -> ClaimValidationReport:
    """Legacy report-style validation (kept for internal compat)."""
    if not claims:
        return ClaimValidationReport(0, 0, 1.0, [], [])
    if not hits:
        results = [ClaimResult(claim=c, grounded=False, reason="no_evidence") for c in claims]
        return ClaimValidationReport(len(claims), 0, 0.0, results, claims.copy())

    hit_kw = [(h, _token_set(_hit_text(h))) for h in hits]
    results2: List[ClaimResult] = []
    for claim in claims:
        ctoks = _token_set(claim)
        if not ctoks:
            results2.append(ClaimResult(claim=claim, grounded=False, reason="no_keywords"))
            continue
        best_overlap = 0
        best_id: Optional[str] = None
        for h, hkw in hit_kw:
            overlap = len(ctoks & hkw)
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = str(h.get("id", ""))
        ratio = best_overlap / max(1, len(ctoks))
        threshold = max(min_overlap, int(len(ctoks) * min_overlap_ratio))
        grounded = best_overlap >= threshold or ratio >= min_overlap_ratio
        results2.append(ClaimResult(
            claim=claim,
            grounded=grounded,
            supporting_hit_id=best_id if grounded else None,
            overlap=best_overlap,
            reason="grounded" if grounded else f"overlap {best_overlap}/{len(ctoks)} < {threshold}",
        ))
    grounded_count = sum(1 for r in results2 if r.grounded)
    ratio2 = grounded_count / len(claims) if claims else 1.0
    return ClaimValidationReport(
        total_claims=len(claims),
        grounded_count=grounded_count,
        grounded_ratio=round(ratio2, 3),
        claims=results2,
        ungrounded_claims=[r.claim for r in results2 if not r.grounded],
    )
