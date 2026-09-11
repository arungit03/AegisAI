"""Query decomposition — P4 atomic subqueries, deterministic, no LLM.

Splits complex multi-part queries into self-contained atomic subqueries
for independent retrieval. Caller handles retrieval per subquery and
combines evidence.

Heuristics (deterministic):
- Multiple "?" clauses
- Numbered lists (1. / 1) / 2. ...)
- Semicolon-separated clauses
- ", and" / " and " joining interrogative clauses
- Multiple wh-words (what, who, when, where, why, how, which)
- Comma-separated wh-clauses

Each subquery is made self-contained by propagating shared context
(e.g. "inspection") so it can be embedded independently.
"""

from __future__ import annotations

import re
from typing import List

_WH_RE = re.compile(r"\b(?:what|who|when|where|why|how|which|whose|whom)\b", re.IGNORECASE)
_NUMBERED_RE = re.compile(r"\b\d+[\.\)]\s*\w")
_AND_RE = re.compile(r"\band\b", re.IGNORECASE)
_COMMA_AND_RE = re.compile(r",\s*and\b", re.IGNORECASE)
# Bare adverbial fragments that unambiguously imply the shared subject
_BARE_FREQ_RE = re.compile(r"^\s*how\s+(?:frequently|often|frequent)\s*$", re.IGNORECASE)

# Stopwords for shared-phrase extraction
_STOPWORDS = frozenset(
    {
        "what", "who", "when", "where", "why", "how", "which", "whose", "whom",
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "will", "would", "can", "could", "should",
        "has", "have", "had", "may", "might", "must",
        "the", "a", "an", "and", "or", "this", "that", "these", "those", "it", "its",
        "requires", "require", "required", "needs", "need",
    }
)


def _count_wh(question: str) -> int:
    return len(_WH_RE.findall(question or ""))


def _first_clause(question: str) -> str:
    """Return the first clause before any delimiter."""
    q = (question or "").strip()
    q = re.sub(r"^\s*\d+[\.\)]\s*", "", q)
    parts = re.split(r"\s+and\s+|,|;|\?", q, flags=re.IGNORECASE)
    return parts[0].strip() if parts else q.strip()


def _extract_shared_phrase(question: str) -> str:
    """Trailing contiguous non-stopword sequence at end of first clause.

    "What equipment requires inspection" -> "inspection"
    "What is the remote work policy"   -> "remote work policy"

    When the first clause is truncated (e.g. numbered list fragment
    "How often"), we fall back to scanning numbered entries to find a
    more informative phrase, and down-rank fragile words like
    "responsible" as shared context.
    """
    # Weak singletons that should not be used as shared subject
    _WEAK = {"responsible", "applicable", "required", "relevant", "available"}
    first = _first_clause(question)
    if not first:
        return ""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", first)
    if not tokens:
        return ""
    phrase_tokens: List[str] = []
    for tok in reversed(tokens):
        if tok.lower() in _STOPWORDS:
            break
        phrase_tokens.append(tok)
    phrase = " ".join(reversed(phrase_tokens)) if phrase_tokens else ""
    if phrase and phrase.lower() not in _WEAK:
        return phrase
    if phrase:
        # Weak phrase — try to find a better candidate from numbered entries
        fallback = _extract_shared_from_numbered(question)
        if fallback:
            return fallback
    if phrase:
        return phrase
    for tok in reversed(tokens):
        if tok.lower() not in _STOPWORDS:
            if tok.lower() in _WEAK:
                fb = _extract_shared_from_numbered(question)
                if fb:
                    return fb
            return tok
    return tokens[-1] if tokens else ""


def _extract_shared_from_numbered(question: str) -> str:
    """Fallback: extract shared phrase from the longest numbered entry."""
    q = question.strip()
    if not _NUMBERED_RE.search(q):
        return ""
    raw = re.split(r"\s*\d+[\.\)]\s*", q)
    entries = [p.strip().rstrip(",;") for p in raw if p.strip()]
    best = ""
    for entry in entries:
        toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", entry)
        phrase_tokens2: List[str] = []
        for tok in reversed(toks):
            if tok.lower() in _STOPWORDS:
                break
            phrase_tokens2.append(tok)
        phrase2 = " ".join(reversed(phrase_tokens2)) if phrase_tokens2 else ""
        if phrase2 and phrase2.lower() not in {"responsible", "applicable", "required", "relevant", "available"}:
            if len(phrase2) > len(best):
                best = phrase2
    return best


def _capitalize_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:] if len(text) > 1 else text.upper()


def _enrich_fragment(fragment: str, shared: str) -> str:
    """Make a fragment self-contained, propagating shared context."""
    frag = fragment.strip().rstrip(",;").strip()
    if not frag:
        return frag

    base = frag.rstrip("?").strip()
    low = base.lower()
    shared_low = shared.lower() if shared else ""

    # Already contains shared context (substring match) — just normalize.
    if shared and shared_low and shared_low in low:
        out = _capitalize_first(base)
        if not out.endswith("?"):
            out += "?"
        return out
    # Token-overlap guard: if any token from shared already appears,
    # don't attach again (e.g. "inspections" shares "inspection" root)
    if shared:
        shared_tokens = [t.lower() for t in re.findall(r"[A-Za-z]{3,}", shared)]
        base_tokens = set(re.findall(r"[A-Za-z]{3,}", low))
        # stem-light: 5-char prefix match
        for st in shared_tokens:
            for bt in base_tokens:
                if st[:5] == bt[:5] and len(st) >= 5:
                    out = _capitalize_first(base)
                    if not out.endswith("?"):
                        out += "?"
                    return out

    # Bare adverbial like "how frequently" -> "How frequently is {shared} required?"
    if _BARE_FREQ_RE.match(base):
        cap = _capitalize_first(base)
        if shared:
            return f"{cap} is {shared} required?"
        return cap + ("?" if not cap.endswith("?") else "")

    # "who is responsible" -> "... for {shared}"
    if re.search(r"\bresponsible\b", base, re.IGNORECASE):
        cap = _capitalize_first(base)
        if shared:
            return f"{cap} for {shared}?"
        return cap + ("?" if not cap.endswith("?") else "")

    # "applies" -> "... to {shared}"
    if re.search(r"\bappl(?:y|ies)\b", base, re.IGNORECASE):
        cap = _capitalize_first(base)
        if shared:
            return f"{cap} to {shared}?"
        return cap + ("?" if not cap.endswith("?") else "")

    # Wh-prefixed fragment
    if re.search(r"^\s*(?:what|who|when|where|why|how|which)\b", base, re.IGNORECASE):
        # Short bare wh-fragments (<=2 words) e.g. "How often", "How many"
        # should not have shared context forced onto them via generic logic.
        # Bare freq is already handled above with " is {shared} required?".
        # Everything else short — just capitalize.
        if len(base.split()) <= 2:
            return _capitalize_first(base) + ("?" if not base.strip().endswith("?") else "")
        cap = _capitalize_first(base)
        has_verb = bool(
            re.search(
                r"\b(?:is|are|was|were|be|been|does|do|did|will|would|can|could|should|has|have|applies|required|responsible)\b",
                base, re.IGNORECASE,
            )
        )
        if has_verb:
            if shared:
                if re.search(r"\b(?:for|to|about|regarding|of)\b\s*$", base, re.IGNORECASE):
                    return f"{cap} {shared}?"
                if re.search(r"\bappl", base, re.IGNORECASE):
                    return f"{cap} to {shared}?"
                return f"{cap} for {shared}?"
            return cap + ("?" if not cap.endswith("?") else "")
        # wh-phrase without verb and longer — synthesize predicate
        if shared:
            return f"{cap} is {shared}?"
        return cap + ("?" if not cap.endswith("?") else "")

    # How-prefixed but not bare (e.g. "How often is cleaning done" with verb) -> shared handled via verb branch above
    # Fallback for non-wh fragments
    cap = _capitalize_first(base)
    if shared and shared_low and shared_low not in low:
        if re.search(r"\b(?:is|are|was|were)\b", cap, re.IGNORECASE):
            return f"{cap} for {shared}?" if not cap.endswith("?") else cap
        # Very short fragment — prepend verb
        if len(base.split()) <= 2:
            return f"{cap} is {shared}?" if not cap.endswith("?") else cap
        return f"{cap} for {shared}?" if not cap.endswith("?") else cap
    return cap + ("?" if not cap.endswith("?") else "")


def _split_raw(question: str) -> List[str]:
    """Split question into raw fragments before enrichment."""
    q = question.strip()

    # 1. Numbered list
    if _NUMBERED_RE.search(q):
        raw = re.split(r"\s*\d+[\.\)]\s*", q)
        parts = [p.strip().rstrip(",;") for p in raw if p.strip()]
        if len(parts) > 1:
            return parts

    # 2. Multiple "?" clauses
    if q.count("?") > 1:
        segs = [s.strip() for s in q.split("?") if s.strip()]
        return segs

    # 3. Semicolon
    if ";" in q:
        parts = [p.strip() for p in q.split(";") if p.strip()]
        if len(parts) > 1:
            return parts

    # 4. ", and" / " and "
    temp: List[str] = []
    found_and = False
    for seg in [q]:
        if re.search(r",\s*and\s+|\s+and\s+", seg, re.IGNORECASE):
            sub = re.split(r",\s*and\s+|\s+and\s+", seg, flags=re.IGNORECASE)
            cleaned = [s.strip().rstrip(",") for s in sub if s.strip()]
            if len(cleaned) > 1:
                temp.extend(cleaned)
                found_and = True
            else:
                temp.append(seg)
        else:
            temp.append(seg)
    if found_and and len(temp) > 1:
        # After splitting on "and", further split comma-wh inside each part
        merged: List[str] = []
        for seg in temp:
            if "," in seg and re.search(r",\s*(?:what|who|when|where|why|how|which)\b", seg, re.IGNORECASE):
                sub = re.split(r",\s*(?=(?:what|who|when|where|why|how|which)\b)", seg, flags=re.IGNORECASE)
                sub = [s.strip().rstrip(",") for s in sub if s.strip()]
                merged.extend(sub if len(sub) > 1 else [seg])
            else:
                merged.append(seg)
        return merged

    # 5. Comma-separated wh-clauses
    if "," in q and re.search(r",\s*(?:what|who|when|where|why|how|which)\b", q, re.IGNORECASE):
        sub = re.split(r",\s*(?=(?:what|who|when|where|why|how|which)\b)", q, flags=re.IGNORECASE)
        sub = [s.strip().rstrip(",") for s in sub if s.strip()]
        if len(sub) > 1:
            return sub

    # Generic comma split when multiple wh-words
    if "," in q and _count_wh(q) > 1:
        comma_parts = [p.strip() for p in q.split(",") if p.strip()]
        if len(comma_parts) > 1:
            return comma_parts

    # 6. Fallback: split on wh-word boundaries
    if _count_wh(q) > 1:
        sub = re.split(r"\s+(?=(?:what|who|when|where|why|how|which)\b)", q, flags=re.IGNORECASE)
        sub = [s.strip().rstrip(",;") for s in sub if s.strip()]
        if len(sub) > 1:
            return sub

    return [q]


def is_multi_part(question: str) -> bool:
    """Return True if question has multiple distinct information needs.

    Triggers (any is sufficient):
    - Multiple "?" marks
    - Numbered items (1. / 1))
    - Semicolon with multiple segments
    - ", and" joining clauses
    - Multiple wh-words
    - " and " joining where trailing clause is interrogative (wh-word or bare freq)
    - Comma followed by wh-word
    """
    if not question or not question.strip():
        return False
    q = question.strip()

    if q.count("?") > 1:
        return True
    if _NUMBERED_RE.search(q):
        return True
    if ";" in q and len([p for p in q.split(";") if p.strip()]) > 1:
        return True
    if _COMMA_AND_RE.search(q):
        return True
    if _count_wh(q) > 1:
        return True

    # " and " with interrogative continuation
    if _AND_RE.search(q):
        parts = [p.strip() for p in re.split(r"\s+and\s+", q, flags=re.IGNORECASE) if p.strip()]
        if len(parts) > 1:
            for part in parts[1:]:
                if re.search(r"^\s*(?:what|who|when|where|why|how|which)\b", part, re.IGNORECASE):
                    return True
                if _BARE_FREQ_RE.match(part):
                    return True
                if _WH_RE.search(part):
                    # Exclude bare noun conjunctions like "vacation and sick leave"
                    if "," in q:
                        return True
                    if re.search(r"\b(?:is|are|was|were|be|does|do|did|will|would|can|could|should)\b", part, re.IGNORECASE):
                        return True

    if re.search(r",\s*(?:what|who|when|where|why|how|which)\b", q, re.IGNORECASE):
        return True

    if "," in q and _count_wh(q) >= 1:
        comma_parts = [p.strip() for p in q.split(",") if p.strip()]
        if len(comma_parts) > 1 and any(len(p.split()) <= 6 for p in comma_parts[1:]):
            if any(re.search(r"^\s*(?:what|who|when|where|why|how|which)\b", p, re.IGNORECASE) for p in comma_parts[1:]):
                return True
            if q.strip().endswith("?"):
                return True

    return False


def should_decompose(question: str) -> bool:
    """Alias for is_multi_part — true when query has multiple information needs."""
    return is_multi_part(question)


def decompose_query(question: str) -> List[str]:
    """Split complex query into atomic self-contained subqueries.

    Each subquery is a complete question that can be embedded independently.
    Shared context (e.g. "inspection") is propagated to fragments that lack it.

    Args:
        question: Original user question.

    Returns:
        List of atomic subqueries. For simple queries, returns [question].
    """
    if not question or not question.strip():
        return [question] if question is not None else [""]

    if not should_decompose(question):
        return [question]

    q = question.strip()
    shared = _extract_shared_phrase(q)

    raw_parts = _split_raw(q)

    if len(raw_parts) <= 1:
        return [question]

    enriched: List[str] = []
    for part in raw_parts:
        p = part.strip()
        if not p:
            continue
        p = re.sub(r"^\s*\d+[\.\)]\s*", "", p).strip()
        if not p:
            continue
        enriched.append(_enrich_fragment(p, shared))

    cleaned: List[str] = []
    for p in enriched:
        p = p.strip()
        if not p:
            continue
        if not p.endswith("?"):
            p += "?"
        cleaned.append(p)

    seen: set[str] = set()
    out: List[str] = []
    for p in cleaned:
        low = p.lower()
        if low not in seen:
            seen.add(low)
            out.append(p)

    return out if out else [question]
