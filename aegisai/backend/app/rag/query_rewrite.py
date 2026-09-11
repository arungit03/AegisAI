"""Query rewriting — retrieval-oriented, never invents facts.

Local, deterministic, rule-based. No LLM or external API calls.

- Rewritten query is ONLY for retrieval; original is used for generation.
- Preserves meaning, exact identifiers (SIHxxxxx), technical terms,
  document names, dates, and intent.
- Resolves pronouns / demonstratives (this, that, it, they, these, those)
  using conversation_history or memory_summary.
- Never hallucinates: if no antecedent is available, the original question
  is returned unchanged.

Example:
    "What is policy for this?" + context "machine inspection"
    -> "What is machine inspection policy?"

Exports:
    needs_rewriting(question) -> bool
    rewrite_query(question, conversation_history, memory_summary) -> str
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional


# Pronouns / demonstratives that indicate an unresolved reference.
_PRONOUN_RE = re.compile(r"\b(this|that|it|they|these|those)\b", re.IGNORECASE)

# Exact identifiers that must be preserved verbatim (SIH + 5 digits, optional space/hyphen).
_IDENTIFIER_RE = re.compile(r"\bSIH[\s-]?\d{5}\b", re.IGNORECASE)

# Tokenizer that keeps SIH identifiers and hyphenated dates intact.
_TOKEN_RE = re.compile(r"\bSIH[\s-]?\d{5}\b|[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", re.IGNORECASE)

# Stopwords / filler words removed when extracting a noun phrase.
# Technical terms, document names, and identifiers are intentionally NOT in this set.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "it",
        "they",
        "these",
        "those",
        "them",
        "its",
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "whose",
        "whom",
        "for",
        "about",
        "on",
        "regarding",
        "with",
        "as",
        "by",
        "at",
        "from",
        "to",
        "of",
        "in",
        "into",
        "over",
        "under",
        "through",
        "between",
        "among",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "near",
        "per",
        "via",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "so",
        "because",
        "while",
        "also",
        "just",
        "only",
        "even",
        "still",
        "already",
        "yet",
        "very",
        "too",
        "me",
        "you",
        "we",
        "us",
        "i",
        "my",
        "your",
        "our",
        "he",
        "she",
        "him",
        "her",
        "his",
        "hers",
        "their",
        "theirs",
        "mine",
        "yours",
        "ours",
        "tell",
        "please",
        "explain",
        "describe",
        "give",
        "show",
        "provide",
        "need",
        "want",
        "know",
        "understand",
        "help",
        "assist",
        "like",
        "discussed",
        "discuss",
        "talked",
        "talk",
        "said",
        "told",
        "asked",
        "mention",
        "mentioned",
        "refer",
        "referring",
        "previous",
        "above",
        "same",
        "earlier",
        "recent",
        "last",
        "next",
        "first",
        "second",
        "hello",
        "hi",
        "thanks",
        "thank",
        "hey",
        "there",
        "greetings",
        "good",
        "morning",
        "afternoon",
        "evening",
        "more",
        "some",
        "any",
        "all",
        "each",
        "every",
        "few",
        "many",
        "much",
        "most",
        "other",
        "another",
        "such",
        "own",
    }
)

_GREETING_PHRASES = frozenset({"hello", "hi", "thanks", "thank you", "hey", "greetings"})

# Pattern for restructuring "noun for this" -> "antecedent noun"
# e.g., "policy for this" + antecedent "machine inspection" -> "machine inspection policy"
_RESTRUCTURE_RE = re.compile(
    r"\b((?:the|a|an)\s+)?(\w+)\s+(for|about|on|regarding|of)\s+(this|that|it|they|these|those)\b",
    re.IGNORECASE,
)

# "this/that/these/those <noun>" — demonstrative used as determiner.
# When antecedent already ends with <noun>, replace the whole phrase with
# antecedent to avoid duplication: "this deadline" + "submission deadline 2025-03-15"
# -> "submission deadline 2025-03-15" (not "submission deadline 2025-03-15 deadline").
_PRONOUN_NOUN_RE = re.compile(
    r"\b(this|that|these|those)\s+([A-Za-z0-9][A-Za-z0-9-]*)\b",
    re.IGNORECASE,
)


def _is_valid_antecedent(phrase: str) -> bool:
    """Check that an extracted phrase is not empty, greeting, or pronoun-only."""
    if not phrase or not phrase.strip():
        return False
    lowered = phrase.strip().lower()
    if lowered in _GREETING_PHRASES:
        return False
    # Single pronoun as antecedent is invalid
    if lowered in {"this", "that", "it", "they", "these", "those"}:
        return False
    # Very short generic phrases are not useful
    if len(lowered) < 2:
        return False
    return True


def _extract_candidate_phrase(text: str) -> Optional[str]:
    """Extract the most relevant noun phrase from a piece of text.

    Strategy:
      1. Tokenize preserving SIH identifiers and hyphenated tokens.
      2. Try contiguous trailing phrase (most recent noun phrase at end of text).
      3. Fallback to last 3 content words if trailing phrase is too short.
    Returns the phrase in its original casing, space-joined.
    """
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    # Remove surrounding quotes that may wrap a document name
    # but keep inner content.
    tokens = _TOKEN_RE.findall(cleaned)
    if not tokens:
        return None

    # --- Attempt 1: contiguous trailing content phrase ---
    trailing: List[str] = []
    # Walk backwards, collecting contiguous non-stopwords.
    for tok in reversed(tokens):
        low = tok.lower()
        if low in _STOPWORDS:
            if trailing:
                break
            continue
        # Skip isolated single chars (except 'a'/'i' already in stop)
        if len(tok) == 1:
            if trailing:
                break
            continue
        trailing.append(tok)
        if len(trailing) >= 3:
            break
    trailing.reverse()
    if trailing and len(trailing) >= 2:
        phrase = " ".join(trailing)
        if _is_valid_antecedent(phrase):
            return phrase
    if trailing and len(trailing) == 1:
        # Single trailing word may still be useful if it looks like a technical term
        # or identifier, but prefer fallback that may give richer phrase.
        # Keep it as candidate if it is an identifier or longer than 3 chars.
        if _IDENTIFIER_RE.search(trailing[0]) or len(trailing[0]) > 3:
            if _is_valid_antecedent(trailing[0]):
                # Don't return single generic word like "policy" alone if alternative exists
                # but allow if no better candidate
                pass

    # --- Attempt 2: fallback to last 3 content words (filtered, not necessarily contiguous) ---
    content = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 1]
    # Also remove pure pronoun tokens that slipped through
    content = [t for t in content if t.lower() not in {"this", "that", "it", "they", "these", "those"}]
    if not content:
        # If only trailing single word was available, return it
        if trailing:
            cand = " ".join(trailing)
            return cand if _is_valid_antecedent(cand) else None
        return None
    if len(content) >= 3:
        phrase = " ".join(content[-3:])
    elif len(content) >= 2:
        phrase = " ".join(content[-2:])
    else:
        phrase = content[-1]
    phrase = phrase.strip()
    if not _is_valid_antecedent(phrase):
        return None
    return phrase


def _find_antecedent(
    conversation_history: Optional[List[Dict]] = None,
    memory_summary: Optional[str] = None,
) -> Optional[str]:
    """Find the most recent relevant noun phrase from history or summary."""
    if conversation_history:
        for turn in reversed(conversation_history):
            if not isinstance(turn, dict):
                continue
            # Support multiple possible keys for the text content
            content = turn.get("content")
            if content is None:
                content = turn.get("message")
            if content is None:
                content = turn.get("text")
            if content is None:
                content = turn.get("body")
            if content is None:
                continue
            content = str(content).strip()
            if not content:
                continue
            if len(content.split()) < 2:
                continue
            # Skip turns that are clearly greetings or very short acknowledgements
            low = content.lower().strip()
            if low in _GREETING_PHRASES or low in {"hello", "hi", "thanks", "thank you", "ok", "okay"}:
                continue
            phrase = _extract_candidate_phrase(content)
            if phrase and _is_valid_antecedent(phrase):
                return phrase
    if memory_summary and isinstance(memory_summary, str) and memory_summary.strip():
        phrase = _extract_candidate_phrase(memory_summary)
        if phrase and _is_valid_antecedent(phrase):
            return phrase
    return None


def _try_restructure(question: str, antecedent: str) -> str:
    """Handle the 'noun for this' restructuring pattern.

    Example: "What is policy for this?" + "machine inspection"
             -> "What is machine inspection policy?"
    If the antecedent already ends with the noun, avoid duplication.
    Returns the restructured question if pattern matched, otherwise original.
    """

    def _repl(match: re.Match) -> str:
        article = match.group(1) or ""
        noun = match.group(2)
        # Guard against restructuring generic verbs/stopwords as 'noun'
        if noun.lower() in _STOPWORDS:
            return match.group(0)
        if noun.lower() in {"this", "that", "it", "they", "these", "those"}:
            return match.group(0)
        antecedent_clean = antecedent.strip()
        # Avoid "machine inspection policy policy" duplication
        if antecedent_clean.lower().endswith(noun.lower()):
            if article:
                return f"{article}{antecedent_clean}"
            return antecedent_clean
        # Normal case: antecedent + noun
        if article:
            # Preserve article spacing (article includes trailing space)
            return f"{article}{antecedent_clean} {noun}"
        return f"{antecedent_clean} {noun}"

    new_question, count = _RESTRUCTURE_RE.subn(_repl, question, count=1)
    return new_question if count > 0 else question


def needs_rewriting(question: str) -> bool:
    """Detect whether a question contains unresolved references.

    A question needs rewriting if it contains a pronoun / demonstrative
    (this, that, it, they, these, those) that would require context to
    resolve for retrieval. This check is purely local and rule-based.

    Args:
        question: The user question to inspect.

    Returns:
        True if the question contains a pronoun/demonstrative and is a
        candidate for rewriting, False otherwise.
    """
    if not question or not question.strip():
        return False
    return bool(_PRONOUN_RE.search(question))


def rewrite_query(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
    memory_summary: Optional[str] = None,
) -> str:
    """Rewrite a query by resolving pronouns using conversation history.

    Retrieval-oriented and never invents facts. The rewritten query is
    intended ONLY for retrieval; the original should be used for generation.

    - Preserves exact identifiers (SIHxxxxx), technical terms, document
      names, and dates.
    - If no rewriting is needed or no antecedent can be found, returns
      the original question unchanged (no hallucination).
    - Purely rule-based, local only, no LLM calls.

    Args:
        question: The current user question.
        conversation_history: Optional list of prior turns, each a dict
            with at least a 'content' (or 'message'/'text') key.
        memory_summary: Optional summarized memory string used as fallback
            when history yields no antecedent.

    Returns:
        The rewritten query for retrieval, or the original question if
        rewriting is not applicable.
    """
    if not question:
        return question
    if not needs_rewriting(question):
        return question

    antecedent = _find_antecedent(conversation_history, memory_summary)
    if not antecedent:
        return question

    # Clean antecedent: collapse whitespace, strip surrounding punctuation
    antecedent = re.sub(r"\s+", " ", antecedent).strip()
    antecedent = antecedent.strip(".,;!?\"'()[]{}")
    antecedent = antecedent.strip()
    if not antecedent or not _is_valid_antecedent(antecedent):
        return question

    # Preserve identifiers in the original question: our replacement only
    # touches pronouns, so identifiers remain untouched. As an extra guard,
    # snapshot identifiers before rewriting and ensure they survive.
    original_identifiers = _IDENTIFIER_RE.findall(question)

    # Handle "this/that + noun" demonstrative phrases first.
    # If the noun already appears in the antecedent, replace the whole phrase
    # with the antecedent alone (dedup), otherwise expand to "antecedent noun".
    def _pronoun_noun_repl(match: re.Match) -> str:
        noun = match.group(2)
        antecedent_clean = antecedent.strip()
        antecedent_words = {w.lower() for w in antecedent_clean.split()}
        if noun.lower() in antecedent_words:
            return antecedent_clean
        return f"{antecedent_clean} {noun}"

    question_after_noun = _PRONOUN_NOUN_RE.sub(_pronoun_noun_repl, question)
    if question_after_noun != question:
        # Remaining standalone pronouns (if any) still need replacement
        if needs_rewriting(question_after_noun):
            question_after_noun = _PRONOUN_RE.sub(antecedent, question_after_noun)
        for ident in original_identifiers:
            if ident.lower() not in question_after_noun.lower():
                return question
        return question_after_noun

    # Try the "policy for this" restructuring (matches spec example)
    restructured = _try_restructure(question, antecedent)
    if restructured != question:
        # If restructuring consumed a pronoun but others remain, resolve them too
        if needs_rewriting(restructured):
            restructured = _PRONOUN_RE.sub(antecedent, restructured)
        # Ensure identifiers preserved (they should be, but verify)
        for ident in original_identifiers:
            if ident not in restructured and ident.lower() not in restructured.lower():
                # Identifier lost — abort rewriting to preserve it
                return question
        return restructured

    # Fallback: replace all pronoun occurrences with the antecedent
    rewritten = _PRONOUN_RE.sub(antecedent, question)

    if rewritten == question:
        return question

    # Final identifier preservation check
    for ident in original_identifiers:
        # Case-insensitive check because SIH identifiers are case-insensitive
        if ident.lower() not in rewritten.lower():
            return question

    return rewritten
