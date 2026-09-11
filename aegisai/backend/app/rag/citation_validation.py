"""P18 Citation Quality — strict validation (Phase 2).

Every citation marker in the answer must map to an actual retrieved
chunk/doc/page/section. Prevents hallucinated filenames, pages, IDs,
citations to unused/unauthorized documents, and ensures multi-doc answers
cite the correct source per claim.

Security: validation never exposes unauthorized document info in issues
or returned payloads — generic placeholders are used instead.

Exports required by spec:
  - CitationValidationResult
  - validate_citations_strict(answer, hits, allowed_doc_ids=None)
  - extract_citations_from_answer(answer)
  - map_claims_to_citations(answer, hits)

Backward compatible aliases kept for hybrid_service:
  - CitationIssue, CitationReport
  - extract_citations, strict_validate_citations, citation_coverage_ratio
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CitationValidationResult:
    """Result of strict citation validation (spec)."""
    valid: bool
    issues: List[str] = field(default_factory=list)
    valid_citations: List[Dict[str, Any]] = field(default_factory=list)
    invalid_citations: List[Dict[str, Any]] = field(default_factory=list)
    coverage: float = 0.0


# Backward compat — used by hybrid_service and older callers
@dataclass
class CitationIssue:
    type: str  # missing_source | fake_page | fake_section | text_mismatch | ...
    detail: str
    citation_text: str = ""


@dataclass
class CitationReport:
    valid: bool
    issues: List[CitationIssue] = field(default_factory=list)
    mapped_citations: int = 0
    total_citations: int = 0
    coverage: float = 0.0


# ---------------------------------------------------------------------------
# Regex — citation markers we emit in build_context / build_grouped_context
# ---------------------------------------------------------------------------

# [Source: filename p.12 | Section]  — whole marker
_SOURCE_RE = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)
# [p.12] or [p.12 extra]
_PAGE_ONLY_RE = re.compile(r"\[p\.\s*(\d+)[^\]]*\]", re.IGNORECASE)
# numeric footnote style: [1] [1,2] [1-3] [2, 3, 4]
_NUMERIC_RE = re.compile(r"\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\]", re.IGNORECASE)
# UUID (document_id)
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)

# Master pattern preserving order — alternation left-to-right, Source first
_CITATION_MASTER_RE = re.compile(
    r"\[Source:[^\]]+\]|\[p\.\s*\d+[^\]]*\]|\[\s*\d+(?:\s*[,\-–]\s*\d+)*\s*\]",
    re.IGNORECASE,
)

# Legacy patterns kept for strict_validate_citations compat
_CITATION_RE = re.compile(r"\[(?:Source:[^\]]+|p\.\d+)\]", re.IGNORECASE)
_LEGACY_SOURCE_RE = re.compile(r"\[Source:\s*([^\]|]+)(?:\s*p\.(\d+))?[^\]]*\]", re.IGNORECASE)
_LEGACY_PAGE_RE = re.compile(r"\[p\.(\d+)\]", re.IGNORECASE)

# Boilerplate prefixes that should not be treated as factual claims
_BOILERPLATE_PREFIXES = (
    "i couldn't find",
    "i don't have",
    "as an ai",
    "i am an ai",
    "i am a language",
    "insufficient",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_payload(hit: Any) -> Dict[str, Any]:
    if isinstance(hit, dict):
        return hit.get("payload") or {}
    # SearchHit or similar object
    try:
        p = getattr(hit, "payload", None)
        if isinstance(p, dict):
            return p
    except Exception:
        pass
    return {}


def _get_id(hit: Any) -> str:
    if isinstance(hit, dict):
        return str(hit.get("id", ""))
    try:
        return str(getattr(hit, "id", ""))
    except Exception:
        return ""


def _get_page(payload: Dict[str, Any]) -> Optional[int]:
    v = payload.get("page_number")
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _token_set(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _sanitize_citation(cit: str) -> str:
    """Return citation text for issues — no doc content leakage.

    Citation text itself comes from the answer (already visible to user) so
    echoing it is safe. We truncate very long markers.
    """
    cit = cit.strip()
    if len(cit) > 120:
        return cit[:120] + "...]"
    return cit


# ---------------------------------------------------------------------------
# extract_citations_from_answer  (spec)
# ---------------------------------------------------------------------------

def extract_citations_from_answer(answer: str) -> List[str]:
    """Find citation markers in answer.

    Supported forms (examples):
      [Source: report.pdf p.12]
      [Source: report.pdf | Introduction]
      [Source: report.pdf p.3 | Section 2]
      [p.12]
      [1]  [1, 2]  [1-3]

    Returns markers in order of appearance.
    """
    if not answer:
        return []
    return [m.group(0) for m in _CITATION_MASTER_RE.finditer(answer)]


# Backward compat alias
def extract_citations(answer: str) -> List[str]:
    return extract_citations_from_answer(answer)


# ---------------------------------------------------------------------------
# map_claims_to_citations  (spec: Dict[str, List[Dict]])
# ---------------------------------------------------------------------------

def map_claims_to_citations(answer: str, hits: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Split answer into claims (sentences) and map each to supporting hits.

    Keyword overlap (case-insensitive, tokens [a-z0-9]{3,}) decides support.
    A hit supports a claim when overlap >= max(2, 35% of claim tokens) or
    overlap ratio >= 0.35.

    Security: only hits provided (already permission-filtered) are considered;
    no unauthorized info is introduced.

    Returns:
        Dict mapping cleaned claim sentence -> list of supporting hit dicts
        (sorted by overlap descending). Empty list means no support.
    """
    if not answer or not hits:
        return {}

    # Build hit keyword cache
    hit_keywords: List[Tuple[Any, Set[str]]] = []
    for h in hits:
        payload = _get_payload(h)
        txt = str(payload.get("chunk_text", "") or "")
        toks = _token_set(txt)
        hit_keywords.append((h, toks))

    # Strip citations for clean claim text, but split on original sentence boundaries
    # to keep alignment; we strip citations then split.
    stripped = _CITATION_MASTER_RE.sub("", answer)
    # Also strip any leftover citation remnants like extra spaces
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", stripped.strip()) if s.strip()]

    # Also handle case where answer has no sentence-ending punctuation
    if not sentences and stripped.strip():
        sentences = [stripped.strip()]

    result: Dict[str, List[Dict[str, Any]]] = {}
    for sent in sentences:
        # sent is already without citations; keep as key (clean claim)
        key = sent.strip()
        if not key:
            continue
        # Very short claims — still map but unlikely to be supported
        low = key.lower()
        # Skip boilerplate as unsupported (empty list) but still include key
        is_boilerplate = any(low.startswith(p) for p in _BOILERPLATE_PREFIXES)
        if is_boilerplate:
            result[key] = []
            continue
        stoks = _token_set(key)
        if not stoks:
            result[key] = []
            continue
        # Allow short claims with len < 12 to still be evaluated (threshold handles)
        supporting: List[Tuple[Any, int]] = []
        for h, htoks in hit_keywords:
            if not htoks:
                continue
            overlap = len(stoks & htoks)
            if overlap == 0:
                continue
            threshold = max(2, int(len(stoks) * 0.35))
            ratio = overlap / len(stoks) if stoks else 0
            if overlap >= threshold or ratio >= 0.35:
                supporting.append((h, overlap))
        supporting.sort(key=lambda x: x[1], reverse=True)
        result[key] = [h for h, _ in supporting]

    return result


# Legacy variant kept for internal use (returns List[Tuple[str, Optional[Dict]]])
def _legacy_map_claims_to_citations(
    answer: str,
    hits: List[Dict[str, Any]],
) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    mapping = map_claims_to_citations(answer, hits)
    out: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    for claim, supporting in mapping.items():
        out.append((claim, supporting[0] if supporting else None))
    return out


# ---------------------------------------------------------------------------
# validate_citations_strict  (spec)
# ---------------------------------------------------------------------------

def validate_citations_strict(
    answer: str,
    hits: List[Dict[str, Any]],
    allowed_doc_ids: List[str] | None = None,
) -> CitationValidationResult:
    """Strict citation validation — every marker must map to retrieved evidence.

    Checks:
      * cited filename / document_id exists in hits (no hallucinated names/IDs)
      * cited page number matches a hit payload page_number
      * cited section (if present) matches a hit section_title when hits have sections
      * numeric citations [N] map to 1-indexed hits within range
      * cited hit has non-empty chunk_text
      * cited document_id is in allowed_doc_ids when provided (unauthorized)
      * claims with supporting evidence but no citation → missing citation issue
      * citations to documents not in hits → fake / unused chunk

    Security: unauthorized document details are never echoed. Issues use
    generic placeholders ("***") for filenames/doc_ids/pages from unauthorized
    hits, and valid_citations only expose authorized hit data.

    Args:
        answer: Generated answer text.
        hits: Retrieved hits (List[Dict] with payload.document_id/filename etc.).
        allowed_doc_ids: Optional allowlist from permission filter.

    Returns:
        CitationValidationResult with valid, issues, valid/invalid lists, coverage.
    """
    if hits is None:
        hits = []
    if answer is None:
        answer = ""

    allowed_set: Optional[Set[str]] = None
    if allowed_doc_ids is not None:
        try:
            allowed_set = {str(x).strip().lower() for x in allowed_doc_ids if str(x).strip()}
        except Exception:
            allowed_set = None

    # Partition hits into authorized / unauthorized when allowlist present
    authorized_hits: List[Any] = []
    unauthorized_hits: List[Any] = []
    for h in hits:
        payload = _get_payload(h)
        doc_id = str(payload.get("document_id", "") or "").strip().lower()
        # If allowlist exists and hit has a doc_id not in allowlist → unauthorized
        if allowed_set is not None and doc_id and doc_id not in allowed_set:
            unauthorized_hits.append(h)
            continue
        authorized_hits.append(h)

    # Build lookups from authorized hits only (prevents leaking unauthorized info)
    filenames: Set[str] = set()
    pages_by_file: Dict[str, Set[int]] = {}
    doc_ids: Set[str] = set()
    all_pages: Set[int] = set()
    hit_ids: Set[str] = set()

    for h in authorized_hits:
        payload = _get_payload(h)
        fn = str(payload.get("filename", "") or "").strip()
        if fn:
            fn_low = fn.lower()
            filenames.add(fn_low)
        doc = str(payload.get("document_id", "") or "").strip()
        if doc:
            doc_ids.add(doc.lower())
        hid = _get_id(h).strip().lower()
        if hid:
            hit_ids.add(hid)
        pg = _get_page(payload)
        if fn and pg is not None:
            fn_low = fn.lower()
            pages_by_file.setdefault(fn_low, set()).add(pg)
            all_pages.add(pg)
        elif pg is not None:
            # page without filename — still track globally
            all_pages.add(pg)

    issues: List[str] = []
    valid_citations: List[Dict[str, Any]] = []
    invalid_citations: List[Dict[str, Any]] = []

    # Pre-check hits themselves: empty chunk_text / missing document_id
    for h in authorized_hits:
        payload = _get_payload(h)
        chunk = str(payload.get("chunk_text", "") or "").strip()
        hid = _get_id(h) or "?"
        if not chunk:
            issues.append(f"hit {hid} has empty chunk_text")
        if not str(payload.get("document_id", "") or "").strip():
            issues.append(f"hit {hid} missing document_id")

    if unauthorized_hits:
        # Generic, no doc ids exposed
        issues.append(f"retrieved set contains {len(unauthorized_hits)} unauthorized document(s) — not used for validation")

    citations = extract_citations_from_answer(answer)
    total_citations = len(citations)

    # Validate each citation marker
    for cit in citations:
        cit_s = cit.strip()
        cit_low = cit_s.lower()

        # ---- Source citation: [Source: ...] ----
        if cit_low.startswith("[source:"):
            m = re.match(r"\[Source:\s*([^\]]+)\]", cit_s, re.IGNORECASE)
            if not m:
                issues.append(f"malformed citation {_sanitize_citation(cit_s)}")
                invalid_citations.append({"citation": cit_s, "reason": "malformed_source"})
                continue
            inner = m.group(1).strip()
            # Extract page number if present
            page_match = re.search(r"p\.\s*(\d+)", inner, re.IGNORECASE)
            page_num: Optional[int] = int(page_match.group(1)) if page_match else None

            # Extract section after '|' if present
            section: Optional[str] = None
            filename_part: str
            if "|" in inner:
                # Split only on first '|'
                before, after = inner.split("|", 1)
                section = after.strip()
                filename_part = before
            else:
                filename_part = inner

            # Remove page token from filename_part to isolate filename/doc_id candidate
            filename_candidate = re.sub(r"p\.\s*\d+", "", filename_part, flags=re.IGNORECASE).strip()
            filename_candidate = filename_candidate.strip().strip(",;")

            # Detect UUID inside filename_candidate
            uuid_match = _UUID_RE.search(filename_candidate)
            doc_id_candidate: Optional[str] = uuid_match.group(0).lower() if uuid_match else None

            # If UUID found, remove it from filename_candidate for filename extraction (if both present)
            if doc_id_candidate:
                # If candidate is purely UUID, filename_candidate becomes empty after removal
                # Keep doc_id_candidate for doc lookup
                tmp = _UUID_RE.sub("", filename_candidate).strip().strip(",;")
                # If remaining is non-empty, treat as filename as well
                if tmp:
                    filename_candidate = tmp
                else:
                    filename_candidate = ""

            filename_low = filename_candidate.lower().strip()

            # Empty source (e.g., [Source: p.2]) — treat as fake if no filename/doc_id
            if not filename_candidate and not doc_id_candidate and page_num is None and not section:
                issues.append(f"citation {_sanitize_citation(cit_s)} has empty source")
                invalid_citations.append({"citation": cit_s, "reason": "empty_source"})
                continue

            # If we have a doc_id candidate, validate it first
            matched = False
            matched_hit: Optional[Any] = None

            if doc_id_candidate:
                if doc_id_candidate in doc_ids:
                    matched = True
                    # Find a hit with this doc_id (and page if specified)
                    for h in authorized_hits:
                        payload = _get_payload(h)
                        if str(payload.get("document_id", "") or "").lower() == doc_id_candidate:
                            if page_num is None:
                                matched_hit = h
                                break
                            pg = _get_page(payload)
                            if pg is not None and pg == page_num:
                                matched_hit = h
                                break
                    # If page specified but no hit with that page, matched stays True but page validation will fail below
                else:
                    # Doc id not in authorized hits
                    if allowed_set is not None and doc_id_candidate not in allowed_set:
                        issues.append(f"citation {_sanitize_citation(cit_s)} references unauthorized document")
                        invalid_citations.append({"citation": cit_s, "reason": "unauthorized_document", "document_id": "***"})
                    else:
                        issues.append(f"citation {_sanitize_citation(cit_s)} references unknown document")
                        invalid_citations.append({"citation": cit_s, "reason": "fake_document_id", "document_id": "***"})
                    continue

            # If not matched via doc_id, try filename candidate
            if not matched:
                if not filename_candidate:
                    # Could be page-only disguised as Source with only page/section — handle page validation
                    # If we have page_num, treat as page citation
                    if page_num is not None:
                        # Validate page against any evidence
                        if all_pages and page_num not in all_pages:
                            issues.append(f"citation {_sanitize_citation(cit_s)} references page {page_num} not in evidence")
                            invalid_citations.append({"citation": cit_s, "reason": "fake_page", "page": page_num})
                            continue
                        if not all_pages:
                            issues.append(f"citation {_sanitize_citation(cit_s)} references page {page_num} but no evidence has page numbers")
                            invalid_citations.append({"citation": cit_s, "reason": "fake_page_no_evidence", "page": page_num})
                            continue
                        # Find hit with that page
                        for h in authorized_hits:
                            if _get_page(_get_payload(h)) == page_num:
                                matched_hit = h
                                break
                        # Also check unauthorized leakage — already filtered
                        # chunk_text check
                        if matched_hit is not None:
                            payload = _get_payload(matched_hit)
                            if not str(payload.get("chunk_text", "") or "").strip():
                                issues.append(f"citation {_sanitize_citation(cit_s)} references chunk with empty text")
                                invalid_citations.append({"citation": cit_s, "reason": "empty_chunk_text"})
                                continue
                        valid_citations.append({"citation": cit_s, "page": page_num, "hit": matched_hit, "section": section})
                        continue
                    if section:
                        # Section-only citation — validate section exists in authorized hits if any hit has sections
                        any_section = any(str((_get_payload(h).get("section_title") or "")).strip() for h in authorized_hits)
                        if any_section:
                            sec_low = section.lower()
                            has_sec = False
                            for h in authorized_hits:
                                sec2 = str((_get_payload(h).get("section_title") or "")).lower()
                                if sec2 and (sec_low in sec2 or sec2 in sec_low):
                                    has_sec = True
                                    matched_hit = h
                                    break
                            if not has_sec:
                                issues.append(f"citation {_sanitize_citation(cit_s)} references unknown section")
                                invalid_citations.append({"citation": cit_s, "reason": "fake_section", "section": "***"})
                                continue
                        valid_citations.append({"citation": cit_s, "hit": matched_hit, "section": section})
                        continue
                    # Otherwise empty
                    issues.append(f"citation {_sanitize_citation(cit_s)} references unknown source")
                    invalid_citations.append({"citation": cit_s, "reason": "fake_filename", "filename": "***"})
                    continue

                # filename_candidate non-empty — try to match against known filenames
                # Use exact, partial containment, and basename matching
                matched = False
                matched_key: Optional[str] = None
                if filename_low in filenames:
                    matched = True
                    matched_key = filename_low
                else:
                    for known in filenames:
                        if filename_low in known or known in filename_low:
                            matched = True
                            matched_key = known
                            break
                        # basename comparison (handles paths)
                        known_base = known.split("/")[-1].split("\\")[-1]
                        cand_base = filename_low.split("/")[-1].split("\\")[-1]
                        if known_base and cand_base and known_base == cand_base:
                            matched = True
                            matched_key = known
                            break
                # Also allow matching doc_ids as fallback (if filename is actually a doc id)
                if not matched and filename_low in doc_ids:
                    matched = True
                    doc_id_candidate = filename_low
                    matched_key = None

                if not matched:
                    issues.append(f"citation {_sanitize_citation(cit_s)} references unknown source")
                    invalid_citations.append({"citation": cit_s, "reason": "fake_filename", "filename": "***"})
                    continue

                # At this point filename matched (or doc_id fallback)
                # If fallback doc_id, find hit
                if doc_id_candidate and doc_id_candidate in doc_ids and matched_key is None:
                    for h in authorized_hits:
                        if str(_get_payload(h).get("document_id", "") or "").lower() == doc_id_candidate:
                            if page_num is None:
                                matched_hit = h
                                break
                            if _get_page(_get_payload(h)) == page_num:
                                matched_hit = h
                                break
                else:
                    # filename path — find hit(s) with that filename
                    target_key = matched_key or filename_low
                    candidates: List[Any] = []
                    for h in authorized_hits:
                        payload = _get_payload(h)
                        fn = str(payload.get("filename", "") or "").lower()
                        # exact or containment or basename
                        is_match = (
                            fn == target_key
                            or target_key in fn
                            or fn in target_key
                            or fn.split("/")[-1].split("\\")[-1] == target_key.split("/")[-1].split("\\")[-1]
                        )
                        if is_match:
                            candidates.append(h)
                    if candidates:
                        if page_num is not None:
                            # filter by page
                            page_candidates = [h for h in candidates if _get_page(_get_payload(h)) == page_num]
                            if page_candidates:
                                matched_hit = page_candidates[0]
                            else:
                                # page mismatch
                                valid_pages: Set[int] = set()
                                for h in candidates:
                                    pg = _get_page(_get_payload(h))
                                    if pg is not None:
                                        valid_pages.add(pg)
                                if valid_pages and page_num not in valid_pages:
                                    issues.append(f"citation {_sanitize_citation(cit_s)} references page {page_num} not in evidence for source")
                                    invalid_citations.append({"citation": cit_s, "reason": "fake_page", "page": page_num})
                                    continue
                                if not valid_pages:
                                    issues.append(f"citation {_sanitize_citation(cit_s)} references page {page_num} but no evidence has page numbers for source")
                                    invalid_citations.append({"citation": cit_s, "reason": "fake_page_no_evidence", "page": page_num})
                                    continue
                                matched_hit = candidates[0]
                        else:
                            matched_hit = candidates[0]

                # If still no matched_hit but matched flag true, pick first candidate (should not happen)
                if not matched_hit and matched:
                    # Find any hit with that filename/doc
                    for h in authorized_hits:
                        payload = _get_payload(h)
                        fn = str(payload.get("filename", "") or "").lower()
                        if matched_key and (fn == matched_key or matched_key in fn or fn in matched_key):
                            matched_hit = h
                            break

            # Validate page if present and we matched via doc/filename but page check not yet done (for doc_id path)
            if page_num is not None and doc_id_candidate and doc_id_candidate in doc_ids:
                # Need to verify page exists for that doc
                valid_pages: Set[int] = set()
                for h in authorized_hits:
                    if str(_get_payload(h).get("document_id", "") or "").lower() == doc_id_candidate:
                        pg = _get_page(_get_payload(h))
                        if pg is not None:
                            valid_pages.add(pg)
                if valid_pages and page_num not in valid_pages:
                    issues.append(f"citation {_sanitize_citation(cit_s)} references page {page_num} not in evidence for document")
                    invalid_citations.append({"citation": cit_s, "reason": "fake_page", "page": page_num})
                    continue
                if not valid_pages:
                    issues.append(f"citation {_sanitize_citation(cit_s)} references page {page_num} but no evidence has page numbers")
                    invalid_citations.append({"citation": cit_s, "reason": "fake_page_no_evidence", "page": page_num})
                    continue

            # Validate chunk_text non-empty for matched hit
            if matched_hit is not None:
                payload = _get_payload(matched_hit)
                if not str(payload.get("chunk_text", "") or "").strip():
                    issues.append(f"citation {_sanitize_citation(cit_s)} references chunk with empty text")
                    invalid_citations.append({"citation": cit_s, "reason": "empty_chunk_text"})
                    continue
                if allowed_set is not None:
                    doc = str(payload.get("document_id", "") or "").lower()
                    if doc and doc not in allowed_set:
                        issues.append(f"citation {_sanitize_citation(cit_s)} references unauthorized document")
                        invalid_citations.append({"citation": cit_s, "reason": "unauthorized_document"})
                        continue

            # Validate section if present
            if section:
                sec_low = section.lower()
                any_section = any(str((_get_payload(h).get("section_title") or "")).strip() for h in authorized_hits)
                if any_section:
                    has_sec = False
                    # Check hits for same doc/filename
                    for h in authorized_hits:
                        payload = _get_payload(h)
                        # restrict to same doc/filename when possible
                        if doc_id_candidate and str(payload.get("document_id", "") or "").lower() == doc_id_candidate:
                            sec2 = str(payload.get("section_title", "") or "").lower()
                            if sec2 and (sec_low in sec2 or sec2 in sec_low):
                                has_sec = True
                                break
                        elif filename_low and str(payload.get("filename", "") or "").lower() in (filename_low, matched_key or "") or (matched_key and matched_key in str(payload.get("filename", "") or "").lower()):
                            sec2 = str(payload.get("section_title", "") or "").lower()
                            if sec2 and (sec_low in sec2 or sec2 in sec_low):
                                has_sec = True
                                break
                    # Fallback: check any hit if not found in restricted set
                    if not has_sec:
                        for h in authorized_hits:
                            sec2 = str((_get_payload(h).get("section_title") or "")).lower()
                            if sec2 and (sec_low in sec2 or sec2 in sec_low):
                                has_sec = True
                                break
                    if not has_sec:
                        issues.append(f"citation {_sanitize_citation(cit_s)} references unknown section")
                        invalid_citations.append({"citation": cit_s, "reason": "fake_section", "section": "***"})
                        continue

            # Passed all checks — valid
            valid_citations.append(
                {
                    "citation": cit_s,
                    "hit": matched_hit,
                    "filename": filename_candidate if filename_candidate else None,
                    "document_id": doc_id_candidate,
                    "page": page_num,
                    "section": section,
                }
            )
            continue

        # ---- Page-only citation: [p.12] ----
        if _PAGE_ONLY_RE.match(cit_s):
            pm = re.search(r"p\.\s*(\d+)", cit_s, re.IGNORECASE)
            page_num = int(pm.group(1)) if pm else None
            if page_num is None:
                issues.append(f"malformed page citation {_sanitize_citation(cit_s)}")
                invalid_citations.append({"citation": cit_s, "reason": "malformed_page"})
                continue
            if all_pages and page_num not in all_pages:
                issues.append(f"citation {_sanitize_citation(cit_s)} references page {page_num} not in any evidence")
                invalid_citations.append({"citation": cit_s, "reason": "fake_page", "page": page_num})
                continue
            if not all_pages:
                issues.append(f"citation {_sanitize_citation(cit_s)} references page {page_num} but no evidence has page numbers")
                invalid_citations.append({"citation": cit_s, "reason": "fake_page_no_evidence", "page": page_num})
                continue
            # Find hit with that page for chunk_text check
            matched_hit = None
            for h in authorized_hits:
                if _get_page(_get_payload(h)) == page_num:
                    matched_hit = h
                    break
            if matched_hit is not None and not str(_get_payload(matched_hit).get("chunk_text", "") or "").strip():
                issues.append(f"citation {_sanitize_citation(cit_s)} references chunk with empty text")
                invalid_citations.append({"citation": cit_s, "reason": "empty_chunk_text"})
                continue
            valid_citations.append({"citation": cit_s, "page": page_num, "hit": matched_hit})
            continue

        # ---- Numeric citation: [1] [1,2] [1-3] ----
        if _NUMERIC_RE.match(cit_s):
            inner = cit_s.strip()[1:-1].strip()  # remove brackets
            # Split by comma and whitespace, handle ranges
            raw_parts = re.split(r"[,\s]+", inner)
            nums: List[int] = []
            parse_error = False
            for part in raw_parts:
                part = part.strip()
                if not part:
                    continue
                # Handle range like 1-3 or 1–3
                if "-" in part or "–" in part:
                    sep = "-" if "-" in part else "–"
                    try:
                        a_str, b_str = part.split(sep, 1)
                        a = int(a_str.strip())
                        b = int(b_str.strip())
                        lo, hi = (a, b) if a <= b else (b, a)
                        for n in range(lo, hi + 1):
                            nums.append(n)
                    except Exception:
                        parse_error = True
                        break
                else:
                    try:
                        nums.append(int(part))
                    except Exception:
                        parse_error = True
                        break
            if parse_error or not nums:
                issues.append(f"malformed numeric citation {_sanitize_citation(cit_s)}")
                invalid_citations.append({"citation": cit_s, "reason": "malformed_numeric"})
                continue
            # Validate each index
            invalid_nums: List[int] = []
            matched_hits: List[Any] = []
            for n in nums:
                if n < 1 or n > len(authorized_hits):
                    invalid_nums.append(n)
                    continue
                hit = authorized_hits[n - 1]
                payload = _get_payload(hit)
                if not str(payload.get("chunk_text", "") or "").strip():
                    invalid_nums.append(n)
                    continue
                if allowed_set is not None:
                    doc = str(payload.get("document_id", "") or "").lower()
                    if doc and doc not in allowed_set:
                        invalid_nums.append(n)
                        continue
                matched_hits.append(hit)
            if invalid_nums:
                issues.append(f"citation {_sanitize_citation(cit_s)} references invalid index {invalid_nums}")
                invalid_citations.append(
                    {"citation": cit_s, "reason": "fake_id_or_out_of_range", "indices": nums, "invalid": invalid_nums}
                )
                continue
            # Also consider references to unauthorized hits that were filtered out — numeric indices refer to authorized list,
            # so an attempt to cite an unauthorized doc that was filtered would be out_of_range or wrong mapping, already flagged.
            # Valid
            valid_citations.append(
                {
                    "citation": cit_s,
                    "indices": nums,
                    "hit": matched_hits[0] if len(matched_hits) == 1 else matched_hits,
                }
            )
            continue

        # ---- Fallback: unrecognized citation form ----
        issues.append(f"unrecognized citation form {_sanitize_citation(cit_s)}")
        invalid_citations.append({"citation": cit_s, "reason": "unrecognized_form"})

    # ---- Missing citations for claims ----
    # Split original answer into sentences preserving citations for has_citation check
    original_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    if not original_sentences and answer.strip():
        original_sentences = [answer.strip()]

    missing_count = 0
    for sent in original_sentences:
        low_sent = sent.lower()
        # Skip boilerplate / very short non-factual sentences
        if len(sent) < 15:
            continue
        if any(low_sent.startswith(p) for p in _BOILERPLATE_PREFIXES):
            continue
        has_citation = bool(_CITATION_MASTER_RE.search(sent))
        if has_citation:
            continue
        # Check if sentence is factual with supporting evidence via keyword overlap
        sent_clean = _CITATION_MASTER_RE.sub("", sent)
        stoks = _token_set(sent_clean)
        if not stoks:
            continue
        # Require at least moderate length factual claim (at least 3 tokens)
        if len(stoks) < 3:
            continue
        best_overlap = 0
        for h in authorized_hits:
            payload = _get_payload(h)
            txt = str(payload.get("chunk_text", "") or "")
            htoks = _token_set(txt)
            if not htoks:
                continue
            overlap = len(stoks & htoks)
            if overlap > best_overlap:
                best_overlap = overlap
        threshold = max(2, int(len(stoks) * 0.35))
        ratio = best_overlap / len(stoks) if stoks else 0
        if best_overlap >= threshold or ratio >= 0.35:
            missing_count += 1
            truncated = sent[:80] + "..." if len(sent) > 80 else sent
            issues.append(f"missing citation for claim: {truncated}")
            invalid_citations.append({"citation": "", "reason": "missing_citation", "claim": truncated})

    # ---- Coverage ----
    # denominator includes missing as invalid; if no citations and no missing, coverage 1
    denom = len(valid_citations) + len(invalid_citations)
    if denom == 0:
        # No citations at all — coverage 1 if no issues, else 0 if we flagged empty chunk/hits
        # But if there were hits with empty chunk_text, issues already makes valid False, coverage reflects that
        if issues:
            # If issues are only about empty hits, coverage 0? Use 0 to indicate failure
            # Distinguish: if answer is empty/insufficient, coverage 1; else 0 when hits exist but no citations needed?
            # Check if answer has no factual claims -> coverage 1 else 0
            # Use missing_count to decide
            coverage = 0.0 if (missing_count > 0 or any("empty" in i or "missing" in i for i in issues)) else 1.0
        else:
            coverage = 1.0
    else:
        coverage = len(valid_citations) / denom if denom else 1.0

    coverage = round(float(max(0.0, min(1.0, coverage))), 3)
    valid = len(issues) == 0 and len(invalid_citations) == 0

    return CitationValidationResult(
        valid=valid,
        issues=issues,
        valid_citations=valid_citations,
        invalid_citations=invalid_citations,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Backward compat wrappers
# ---------------------------------------------------------------------------

def strict_validate_citations(
    answer: str,
    hits: List[Dict[str, Any]],
) -> CitationReport:
    """Legacy strict validation used by hybrid_service (no allowlist)."""
    result = validate_citations_strict(answer, hits, allowed_doc_ids=None)
    # Map to old CitationReport
    issues = [CitationIssue(type="validation", detail=d, citation_text="") for d in result.issues]
    total = len(extract_citations_from_answer(answer))
    mapped = len(result.valid_citations)
    # Preserve old valid semantics (no invalid)
    valid = result.valid
    return CitationReport(
        valid=valid,
        issues=issues,
        mapped_citations=mapped,
        total_citations=total,
        coverage=result.coverage,
    )


def citation_coverage_ratio(answer: str, hits: List[Dict[str, Any]]) -> float:
    """Fraction of claims/sentences with supporting evidence."""
    mapping = map_claims_to_citations(answer, hits)
    if not mapping:
        return 1.0 if not answer.strip() else 0.0
    supported = sum(1 for v in mapping.values() if v)
    return round(supported / len(mapping), 3)

