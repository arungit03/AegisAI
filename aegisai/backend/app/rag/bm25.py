"""BM25 lexical retrieval — local only, permission-aware.

Uses an in-memory BM25 index built from authorized Qdrant scroll results.
All retrieval is filtered BEFORE scoring via Qdrant permission Filter;
no unauthorized chunk_text ever leaves this module.

Kept as a separate module so the legacy vector pipeline is untouched.
Activated only when HYBRID_RAG_ENABLED or RAGQuery.enable_hybrid is True.
"""

from __future__ import annotations

import re
import math
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from uuid import UUID

from qdrant_client.models import Filter

from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
    "of", "and", "or", "with", "as", "by", "it", "this", "that", "what", "which",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "from", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "among", "under", "over", "up", "down",
})


def _tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall((text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25Index:
    """In-memory BM25 scorer. Built per-query from authorized scroll results."""

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.docs = documents  # each has {id, payload}
        self.N = len(documents)
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.df: Dict[str, int] = defaultdict(int)
        self._build()

    def _build(self) -> None:
        for doc in self.docs:
            payload = doc.get("payload") or {}
            text = " ".join(str(payload.get(k, "")) for k in ("chunk_text", "title", "filename") if payload.get(k))
            tokens = _tokenize(text)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens) if tokens else 1)
            for term in set(tokens):
                self.df[term] += 1
        self.avgdl = sum(self.doc_lens) / max(1, self.N) if self.N else 1.0

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # Robertson/Sparck-Jones IDF with smoothing
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> List[tuple[int, float]]:
        q_tokens = _tokenize(query)
        if not q_tokens or not self.N:
            return []
        q_freq = Counter(q_tokens)
        scores: List[tuple[int, float]] = []
        for idx, tokens in enumerate(self.doc_tokens):
            if not tokens:
                continue
            tf = Counter(tokens)
            doc_len = self.doc_lens[idx]
            s = 0.0
            for term in q_freq:
                if term not in tf:
                    continue
                idf = self._idf(term)
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                s += idf * (f * (self.k1 + 1) / denom)
            if s > 0:
                scores.append((idx, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def bm25_search(
    qdrant_manager,
    query_text: str,
    filter_conditions: Optional[Filter] = None,
    limit: int = 20,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Dict[str, Any]]:
    """Lexical retrieval via BM25 over authorized Qdrant scroll.

    Security: scroll_filter=filter_conditions ensures only authorized
    documents are indexed. BM25 scoring happens locally on that subset.
    No external API is called.

    Args:
        qdrant_manager: QdrantManager instance.
        query_text: User query text.
        filter_conditions: Permission Filter from create_permission_filter.
        limit: Top-K to return.
        k1, b: BM25 hyperparameters.

    Returns:
        List of {id, score, payload} dicts sorted by BM25 score desc.
    """
    if not query_text or not query_text.strip():
        return []
    try:
        records, _ = qdrant_manager.client.scroll(
            collection_name=qdrant_manager.collection_name,
            scroll_filter=filter_conditions,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        logger.error("bm25_scroll_failed", error_type=type(e).__name__)
        return []

    if not records:
        return []

    docs = [{"id": str(r.id), "payload": r.payload or {}} for r in records]
    index = BM25Index(docs, k1=k1, b=b)
    ranked = index.score(query_text)
    results: List[Dict[str, Any]] = []
    for idx, score in ranked[:limit]:
        doc = docs[idx]
        results.append({"id": doc["id"], "score": float(score), "payload": doc["payload"]})
    logger.info("bm25_search_completed", query_len=len(query_text), candidates=len(docs), returned=len(results))
    return results
