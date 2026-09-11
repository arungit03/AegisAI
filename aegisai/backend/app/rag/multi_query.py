"""Multi-query retrieval orchestration for Phase 2 RAG.

Pipeline:
  Original + Rewritten + Subqueries + Optional expanded
  -> hybrid retrieval per variant (shared permission filter)
  -> dedup by id/text -> RRF fusion across variants -> MMR diversity -> rerank -> evidence selection

Security invariant:
  permission filter is created ONCE BEFORE any retrieval and reused for every
  variant (vector search and BM25). No unauthorized chunk ever reaches fusion/rerank.

Notes:
  - query_understanding, query_rewrite, query_expansion, query_decomposition are
    imported lazily inside functions to avoid circular dependencies.
  - Retrievals run sequentially per variant; one variant failing does not break others.
  - Uses dedup_hits / rrf_fuse / mmr_select / rerank_cross_encoder from fusion.py.
  - Settings VECTOR_TOP_K, BM25_TOP_K, FUSION_TOP_K, RRF_K, RERANK_TOP_K, RERANK_MODEL are read from app.core.config.settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_COMPANY_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Legacy bundle (kept for backward compatibility with hybrid_service.py)
# ---------------------------------------------------------------------------

@dataclass
class MultiQueryBundle:
    original: str
    rewritten: Optional[str] = None
    subqueries: List[str] = field(default_factory=list)
    expanded: List[str] = field(default_factory=list)

    def all_queries(self, deduplicate: bool = True) -> List[str]:
        out: List[str] = [self.original]
        if self.rewritten and self.rewritten.strip() and self.rewritten.strip() != self.original.strip():
            out.append(self.rewritten.strip())
        for sq in self.subqueries:
            sq = sq.strip()
            if sq and sq not in out:
                out.append(sq)
        for eq in self.expanded:
            eq = eq.strip()
            if eq and eq not in out:
                out.append(eq)
        if deduplicate:
            seen: Set[str] = set()
            deduped: List[str] = []
            for q in out:
                key = q.lower().strip()
                if key not in seen:
                    seen.add(key)
                    deduped.append(q)
            return deduped
        return out


@dataclass
class MultiQueryResult:
    """Result container for multi-query retrieval.

    New Phase 2 spec fields are required; legacy aliases (fused, per_query_hits, bundle)
    are retained for backward compatibility with hybrid_service.
    """

    original_query: str
    rewritten_query: Optional[str] = None
    subqueries: List[str] = field(default_factory=list)
    expanded_queries: List[str] = field(default_factory=list)
    all_hits: List[Dict[str, Any]] = field(default_factory=list)
    fused_hits: List[Dict[str, Any]] = field(default_factory=list)
    reranked_hits: List[Dict[str, Any]] = field(default_factory=list)
    final_hits: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_debug: Dict[str, Any] = field(default_factory=dict)
    # Legacy aliases (backward compat with hybrid_service's run_multi_query_retrieval)
    fused: List[Dict[str, Any]] = field(default_factory=list)
    per_query_hits: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    bundle: Optional[MultiQueryBundle] = None

    def __post_init__(self) -> None:
        # Keep legacy `fused` in sync with `fused_hits` when only one is set
        if not self.fused and self.fused_hits:
            self.fused = self.fused_hits
        elif self.fused and not self.fused_hits:
            self.fused_hits = self.fused
        # Ensure bundle exists for legacy access
        if self.bundle is None and self.original_query:
            try:
                # Avoid overwriting if all variant lists are empty and no query
                if self.rewritten_query is not None or self.subqueries or self.expanded_queries:
                    self.bundle = MultiQueryBundle(
                        original=self.original_query,
                        rewritten=self.rewritten_query,
                        subqueries=list(self.subqueries) if self.subqueries else [],
                        expanded=list(self.expanded_queries) if self.expanded_queries else [],
                    )
                else:
                    # Still create minimal bundle for observability
                    self.bundle = MultiQueryBundle(
                        original=self.original_query,
                        rewritten=self.rewritten_query,
                        subqueries=list(self.subqueries) if self.subqueries else [],
                        expanded=list(self.expanded_queries) if self.expanded_queries else [],
                    )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Legacy builders (kept for hybrid_service compatibility)
# ---------------------------------------------------------------------------

def build_multi_query_bundle(
    question: str,
    conversation_history=None,
    memory_summary: Optional[str] = None,
    enable_rewrite: bool = True,
    enable_decomposition: bool = True,
    enable_expansion: bool = False,
) -> MultiQueryBundle:
    """Build the multi-query bundle deterministically, local-only.

    Lazy imports avoid circular deps. Kept for backward compatibility; new code
    should use multi_query_retrieve which builds variants internally.
    """
    bundle = MultiQueryBundle(original=question)

    # Rewrite
    if enable_rewrite:
        try:
            from app.rag.query_rewrite import rewrite_query, needs_rewriting
            if needs_rewriting(question):
                rw = rewrite_query(question, conversation_history, memory_summary)
                if rw and rw.strip() != question.strip():
                    bundle.rewritten = rw
        except Exception as e:
            logger.warning("multi_query_rewrite_failed", error_type=type(e).__name__)

    # Decomposition
    if enable_decomposition:
        try:
            from app.rag.query_decomposition import should_decompose, decompose_query
            if should_decompose(question):
                subs = decompose_query(question)
                bundle.subqueries = [s for s in subs if s.strip().lower() != question.strip().lower()]
        except Exception as e:
            logger.warning("multi_query_decompose_failed", error_type=type(e).__name__)

    # Expansion (controlled, opt-in) — fix: do not pass max_variants kwarg (signature is expand_query(question))
    if enable_expansion:
        try:
            from app.rag.query_expansion import should_expand, expand_query
            target = bundle.rewritten or question
            if should_expand(target):
                exps = expand_query(target)
                # expand_query returns [original, variant1, ...]; keep only variants
                # but also handle if it already filtered
                variants = [e for e in exps if e.strip().lower() != question.strip().lower() and e.strip().lower() != (bundle.rewritten or "").strip().lower()]
                # Respect expanded deduplication and cap (caller expects at most a few)
                # Keep at most 2 as previously intended (controlled)
                bundle.expanded = variants[:2]
        except Exception as e:
            logger.warning("multi_query_expansion_failed", error_type=type(e).__name__)

    return bundle


def run_multi_query_retrieval(
    bundle: MultiQueryBundle,
    qdrant_manager,
    filter_conditions,
    embedding_model_name: str = "nomic-embed-text",
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    fusion_top_k: int = 12,
    rrf_k: int = 60,
    score_threshold: Optional[float] = None,
) -> MultiQueryResult:
    """Execute hybrid retrieval for each query variant and fuse (legacy API).

    Uses the SAME filter_conditions for every variant (security invariant).
    Retained for hybrid_service compatibility; delegates core logic but preserves
    legacy MultiQueryResult shape (fused/per_query_hits/bundle). New callers
    should use multi_query_retrieve instead.
    """
    from app.rag.embeddings import embed_text
    from app.rag.bm25 import bm25_search
    from app.rag.fusion import rrf_fuse, dedup_hits

    per_query: Dict[str, List[Dict[str, Any]]] = {}
    all_fused_candidates: List[List[Dict[str, Any]]] = []

    for q in bundle.all_queries():
        # Vector
        vector_hits: List[Dict[str, Any]] = []
        try:
            q_emb = embed_text(q, embedding_model_name)
            vector_hits = qdrant_manager.search(
                query_vector=q_emb,
                limit=vector_top_k,
                filter_conditions=filter_conditions,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,
            )
        except Exception as e:
            logger.warning("multi_query_vector_failed", error_type=type(e).__name__, query_preview=q[:60])
            vector_hits = []

        # BM25
        bm25_hits: List[Dict[str, Any]] = []
        try:
            bm25_hits = bm25_search(
                qdrant_manager=qdrant_manager,
                query_text=q,
                filter_conditions=filter_conditions,
                limit=bm25_top_k,
            )
        except Exception as e:
            logger.warning("multi_query_bm25_failed", error_type=type(e).__name__, query_preview=q[:60])
            bm25_hits = []

        # Per-query RRF (vector + lexical)
        if vector_hits or bm25_hits:
            ranked_lists = [vector_hits, bm25_hits] if bm25_hits else [vector_hits]
            try:
                per_fused = rrf_fuse(ranked_lists, k=rrf_k, top_k=fusion_top_k)
            except Exception as e:
                logger.warning("multi_query_rrf_failed", error_type=type(e).__name__)
                per_fused = (vector_hits + bm25_hits)[:fusion_top_k]
            per_fused = dedup_hits(per_fused)
            per_query[q] = per_fused
            if per_fused:
                all_fused_candidates.append(per_fused)
        else:
            per_query[q] = []

    if not all_fused_candidates:
        # Build legacy-shaped plus new-spec fields
        return MultiQueryResult(
            original_query=bundle.original,
            rewritten_query=bundle.rewritten,
            subqueries=list(bundle.subqueries),
            expanded_queries=list(bundle.expanded),
            all_hits=[],
            fused_hits=[],
            reranked_hits=[],
            final_hits=[],
            retrieval_debug={"variants": len(bundle.all_queries()), "per_query_counts": {q[:60]: len(h) for q, h in per_query.items()}},
            fused=[],
            per_query_hits=per_query,
            bundle=bundle,
        )

    # Global RRF across query variants
    try:
        global_fused = rrf_fuse(all_fused_candidates, k=rrf_k, top_k=fusion_top_k)
    except Exception as e:
        logger.warning("multi_query_global_rrf_failed", error_type=type(e).__name__)
        flat: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        for lst in all_fused_candidates:
            for h in lst:
                hid = str(h.get("id", ""))
                if hid and hid not in seen_ids:
                    seen_ids.add(hid)
                    flat.append(h)
        global_fused = flat[:fusion_top_k]

    global_fused = dedup_hits(global_fused)
    # Populate both new and legacy fields for compatibility
    all_hits_flat: List[Dict[str, Any]] = []
    for lst in all_fused_candidates:
        all_hits_flat.extend(lst)
    all_hits_flat = dedup_hits(all_hits_flat)
    return MultiQueryResult(
        original_query=bundle.original,
        rewritten_query=bundle.rewritten,
        subqueries=list(bundle.subqueries),
        expanded_queries=list(bundle.expanded),
        all_hits=all_hits_flat,
        fused_hits=global_fused,
        reranked_hits=global_fused,
        final_hits=global_fused,
        retrieval_debug={"variants": len(bundle.all_queries()), "per_query_counts": {q[:60]: len(h) for q, h in per_query.items()}, "fused": len(global_fused)},
        fused=global_fused,
        per_query_hits=per_query,
        bundle=bundle,
    )


# ---------------------------------------------------------------------------
# New Phase 2 spec API
# ---------------------------------------------------------------------------

def multi_query_retrieve(
    qdrant_manager,
    embedding_model_name: str,
    query,  # RAGQuery
    analysis=None,
) -> MultiQueryResult:
    """Multi-query retrieval orchestration per Phase 2 spec.

    Steps:
      1. Analyze query (or use provided analysis) to determine strategy
      2. Build query variants: original always, rewritten if needs_rewriting,
         subqueries if should_decompose, expanded if should_expand (controlled)
      3. Create permission filter ONCE before any retrieval (security invariant)
      4. For each variant: vector search + BM25 search (both filtered)
      5. Collect all hits, dedup efficiently (dedup_hits)
      6. RRF fusion across all variant result lists
      7. MMR diversity selection
      8. Cross-encoder reranking with visibility
      9. Return final evidence hits

    Args:
        qdrant_manager: QdrantManager instance.
        embedding_model_name: Embedding model name for vector search.
        query: RAGQuery with user context (question, user_role, user_id, department, conversation_history, memory_summary, intent).
        analysis: Optional QueryAnalysis; if None, analyzed lazily via query_understanding.

    Returns:
        MultiQueryResult with original_query, rewritten_query, subqueries,
        expanded_queries, all_hits, fused_hits, reranked_hits, final_hits, retrieval_debug.
    """
    # Lazily read settings (allows tests to patch)
    vector_top_k = int(getattr(settings, "VECTOR_TOP_K", 20))
    bm25_top_k = int(getattr(settings, "BM25_TOP_K", 20))
    fusion_top_k = int(getattr(settings, "FUSION_TOP_K", 12))
    rrf_k = int(getattr(settings, "RRF_K", 60))
    rerank_top_k = int(getattr(settings, "RERANK_TOP_K", 5))
    rerank_model = str(getattr(settings, "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    bm25_k1 = float(getattr(settings, "BM25_K1", 1.5))
    bm25_b = float(getattr(settings, "BM25_B", 0.75))

    # Normalize query input
    # query is expected to be RAGQuery, but handle plain string defensively
    if isinstance(query, str):
        original_query = query
        q_user_role = "employee"
        q_user_id = None
        q_department = None
        q_intent = None
        q_history = None
        q_memory = None
    else:
        original_query = str(getattr(query, "question", "") or "")
        q_user_role = getattr(query, "user_role", None)
        q_user_id = getattr(query, "user_id", None)
        q_department = getattr(query, "department", None)
        q_intent = getattr(query, "intent", None)
        q_history = getattr(query, "conversation_history", None)
        q_memory = getattr(query, "memory_summary", None)

    if not original_query or not original_query.strip():
        return MultiQueryResult(
            original_query=original_query or "",
            rewritten_query=None,
            subqueries=[],
            expanded_queries=[],
            all_hits=[],
            fused_hits=[],
            reranked_hits=[],
            final_hits=[],
            retrieval_debug={"error": "empty_query", "variants": 0},
            fused=[],
            per_query_hits={},
            bundle=MultiQueryBundle(original=original_query or ""),
        )

    # 1) Analyze query (or use provided)
    analysis_obj = analysis
    if analysis_obj is None:
        try:
            from app.rag.query_understanding import analyze_query  # lazy
            analysis_obj = analyze_query(original_query, q_history)
        except Exception as e:
            logger.warning("multi_query_analyze_failed", error_type=type(e).__name__)
            analysis_obj = None

    # 2) Build variants (lazy imports)
    rewritten_query: Optional[str] = None
    subqueries: List[str] = []
    expanded_queries: List[str] = []

    # Rewritten only if needs_rewriting and available
    try:
        from app.rag.query_rewrite import needs_rewriting, rewrite_query  # lazy
        if needs_rewriting(original_query):
            try:
                rw = rewrite_query(original_query, q_history, q_memory)
                if rw and rw.strip() and rw.strip().lower() != original_query.strip().lower():
                    rewritten_query = rw.strip()
            except Exception as e:
                logger.warning("multi_query_rewrite_query_failed", error_type=type(e).__name__)
    except Exception as e:
        logger.warning("multi_query_needs_rewriting_import_failed", error_type=type(e).__name__)

    # Subqueries only if should_decompose
    try:
        from app.rag.query_decomposition import should_decompose, decompose_query  # lazy
        if should_decompose(original_query):
            try:
                subs = decompose_query(original_query)
                # Filter trivial duplicates of original/rewritten and dedup case-insensitively
                seen_sq: Set[str] = {original_query.strip().lower()}
                if rewritten_query:
                    seen_sq.add(rewritten_query.strip().lower())
                filtered: List[str] = []
                for s in subs:
                    if not s or not s.strip():
                        continue
                    key = s.strip().lower()
                    if key in seen_sq:
                        continue
                    # Also avoid near-duplicates where subquery equals original lower
                    seen_sq.add(key)
                    filtered.append(s.strip())
                subqueries = filtered
            except Exception as e:
                logger.warning("multi_query_decompose_failed", error_type=type(e).__name__)
    except Exception as e:
        logger.warning("multi_query_decompose_import_failed", error_type=type(e).__name__)

    # Expanded only if should_expand (controlled)
    # Use rewritten or original as target, similar to legacy bundle
    try:
        from app.rag.query_expansion import should_expand, expand_query  # lazy
        expand_target = rewritten_query or original_query
        if should_expand(expand_target):
            try:
                exps = expand_query(expand_target)
                # expand_query returns [original, variant1, ...] capped at 3 alts
                # Keep only variants not duplicating original/rewritten/subqueries
                seen_exp: Set[str] = {original_query.strip().lower()}
                if rewritten_query:
                    seen_exp.add(rewritten_query.strip().lower())
                for sq in subqueries:
                    seen_exp.add(sq.strip().lower())
                filtered_exp: List[str] = []
                for e in exps:
                    if not e or not e.strip():
                        continue
                    key = e.strip().lower()
                    if key in seen_exp:
                        continue
                    # expand_target's first element is itself; skip if equals target
                    if key == expand_target.strip().lower():
                        continue
                    seen_exp.add(key)
                    filtered_exp.append(e.strip())
                # Controlled cap: at most 3, but keep as-is from expand_query (already capped)
                expanded_queries = filtered_exp[:3]
            except Exception as e:
                logger.warning("multi_query_expand_failed", error_type=type(e).__name__)
    except Exception as e:
        logger.warning("multi_query_expansion_import_failed", error_type=type(e).__name__)

    # Ordered, deduped variants for retrieval
    variants: List[str] = [original_query]
    if rewritten_query and rewritten_query.strip().lower() not in {v.strip().lower() for v in variants}:
        variants.append(rewritten_query)
    for sq in subqueries:
        if sq.strip().lower() not in {v.strip().lower() for v in variants}:
            variants.append(sq)
    for eq in expanded_queries:
        if eq.strip().lower() not in {v.strip().lower() for v in variants}:
            variants.append(eq)

    # Retrieval debug base
    retrieval_debug: Dict[str, Any] = {
        "original_query": original_query[:200],
        "variants": len(variants),
        "variant_queries": [v[:120] for v in variants],
        "rewritten": rewritten_query[:120] if rewritten_query else None,
        "subqueries_count": len(subqueries),
        "expanded_count": len(expanded_queries),
        "vector_top_k": vector_top_k,
        "bm25_top_k": bm25_top_k,
        "fusion_top_k": fusion_top_k,
        "rrf_k": rrf_k,
        "rerank_top_k": rerank_top_k,
        "rerank_model": rerank_model,
    }
    if analysis_obj is not None:
        try:
            retrieval_debug["query_types"] = [t.value for t in getattr(analysis_obj, "query_types", [])]
            retrieval_debug["is_simple"] = bool(getattr(analysis_obj, "is_simple", False))
            retrieval_debug["analysis_confidence"] = float(getattr(analysis_obj, "confidence", 0.0) or 0.0)
        except Exception:
            pass

    # Handle missing qdrant_manager
    if qdrant_manager is None:
        retrieval_debug["error"] = "qdrant_manager_missing"
        return MultiQueryResult(
            original_query=original_query,
            rewritten_query=rewritten_query,
            subqueries=subqueries,
            expanded_queries=expanded_queries,
            all_hits=[],
            fused_hits=[],
            reranked_hits=[],
            final_hits=[],
            retrieval_debug=retrieval_debug,
            fused=[],
            per_query_hits={},
            bundle=MultiQueryBundle(original=original_query, rewritten=rewritten_query, subqueries=subqueries, expanded=expanded_queries),
        )

    # 3) Security: permission filter created ONCE before all variant retrievals
    filter_conditions = None
    try:
        # Normalize role fallback
        role = q_user_role or "employee"
        filter_conditions = qdrant_manager.create_permission_filter(
            user_role=role,
            user_id=q_user_id,
            department=q_department,
        )
        retrieval_debug["permission_filter_created"] = True
        retrieval_debug["filter_role"] = role
    except Exception as e:
        logger.error("multi_query_permission_filter_failed", error_type=type(e).__name__)
        retrieval_debug["permission_filter_created"] = False
        retrieval_debug["permission_filter_error"] = type(e).__name__
        return MultiQueryResult(
            original_query=original_query,
            rewritten_query=rewritten_query,
            subqueries=subqueries,
            expanded_queries=expanded_queries,
            all_hits=[],
            fused_hits=[],
            reranked_hits=[],
            final_hits=[],
            retrieval_debug=retrieval_debug,
            fused=[],
            per_query_hits={},
            bundle=MultiQueryBundle(original=original_query, rewritten=rewritten_query, subqueries=subqueries, expanded=expanded_queries),
        )

    # Determine score threshold (company intent) — optional, respects security invariant unchanged
    score_threshold: Optional[float] = None
    try:
        intent_str = str(q_intent or "").lower()
        if intent_str == "company":
            score_threshold = _COMPANY_THRESHOLD
        else:
            # Also handle QueryIntent enum value if present
            from app.rag.routing import QueryIntent  # lazy, may not be needed
            if q_intent == QueryIntent.COMPANY.value or str(q_intent) == QueryIntent.COMPANY.value:
                score_threshold = _COMPANY_THRESHOLD
    except Exception:
        # If routing not available, keep None
        pass

    # 4) For each variant: vector + BM25 (both with permission filter) — sequential, errors isolated
    per_query_hits: Dict[str, List[Dict[str, Any]]] = {}
    per_variant_fused_lists: List[List[Dict[str, Any]]] = []
    all_hits_raw: List[Dict[str, Any]] = []
    per_variant_counts: Dict[str, int] = {}
    variant_errors: List[Dict[str, str]] = []

    # Lazy imports for retrieval primitives (avoid circular at top)
    try:
        from app.rag.embeddings import embed_text  # lazy but outside loop for efficiency
        _has_embed = True
    except Exception as e:
        logger.warning("multi_query_embed_import_failed", error_type=type(e).__name__)
        embed_text = None  # type: ignore
        _has_embed = False

    try:
        from app.rag.bm25 import bm25_search  # lazy
        _has_bm25 = True
    except Exception as e:
        logger.warning("multi_query_bm25_import_failed", error_type=type(e).__name__)
        bm25_search = None  # type: ignore
        _has_bm25 = False

    try:
        from app.rag.fusion import rrf_fuse, dedup_hits, mmr_select  # lazy for core fusion
        _has_fusion = True
    except Exception as e:
        logger.warning("multi_query_fusion_import_failed", error_type=type(e).__name__)
        rrf_fuse = None  # type: ignore
        dedup_hits = None  # type: ignore
        mmr_select = None  # type: ignore
        _has_fusion = False

    embedding_model = embedding_model_name or str(getattr(settings, "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))

    for variant_q in variants:
        vector_hits: List[Dict[str, Any]] = []
        bm25_hits: List[Dict[str, Any]] = []

        # Vector search
        if _has_embed and embed_text is not None:
            try:
                q_emb = embed_text(variant_q, embedding_model)  # type: ignore
                vector_hits = qdrant_manager.search(
                    query_vector=q_emb,
                    limit=vector_top_k,
                    filter_conditions=filter_conditions,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=score_threshold,
                ) or []
            except Exception as e:
                logger.warning("multi_query_variant_vector_failed", error_type=type(e).__name__, variant_preview=variant_q[:60])
                variant_errors.append({"variant": variant_q[:60], "stage": "vector", "error": type(e).__name__})
                vector_hits = []
        else:
            variant_errors.append({"variant": variant_q[:60], "stage": "vector", "error": "embed_unavailable"})

        # BM25 search
        if _has_bm25 and bm25_search is not None:
            try:
                bm25_hits = bm25_search(
                    qdrant_manager=qdrant_manager,
                    query_text=variant_q,
                    filter_conditions=filter_conditions,
                    limit=bm25_top_k,
                    k1=bm25_k1,
                    b=bm25_b,
                ) or []
            except Exception as e:
                logger.warning("multi_query_variant_bm25_failed", error_type=type(e).__name__, variant_preview=variant_q[:60])
                variant_errors.append({"variant": variant_q[:60], "stage": "bm25", "error": type(e).__name__})
                bm25_hits = []
        else:
            variant_errors.append({"variant": variant_q[:60], "stage": "bm25", "error": "bm25_unavailable"})

        # Collect raw hits
        if vector_hits:
            all_hits_raw.extend(vector_hits)
        if bm25_hits:
            all_hits_raw.extend(bm25_hits)

        # Per-variant hybrid RRF (vector + BM25) — prepares list for global fusion
        ranked_lists: List[List[Dict[str, Any]]] = []
        if vector_hits:
            ranked_lists.append(vector_hits)
        if bm25_hits:
            ranked_lists.append(bm25_hits)

        if not ranked_lists:
            per_query_hits[variant_q] = []
            per_variant_counts[variant_q[:60]] = 0
            continue

        per_fused: List[Dict[str, Any]] = []
        if _has_fusion and rrf_fuse is not None and dedup_hits is not None:
            try:
                per_fused = rrf_fuse(ranked_lists, k=rrf_k, top_k=fusion_top_k)
                per_fused = dedup_hits(per_fused)
            except Exception as e:
                logger.warning("multi_query_variant_rrf_failed", error_type=type(e).__name__, variant_preview=variant_q[:60])
                # Fallback: dedup concat
                try:
                    flat = vector_hits + bm25_hits
                    per_fused = dedup_hits(flat)[:fusion_top_k] if dedup_hits else flat[:fusion_top_k]
                except Exception:
                    per_fused = (vector_hits + bm25_hits)[:fusion_top_k]
        else:
            # Fallback without fusion helpers
            flat = vector_hits + bm25_hits
            # naive dedup by id if helpers unavailable
            seen_ids: Set[str] = set()
            deduped_flat: List[Dict[str, Any]] = []
            for h in flat:
                hid = str(h.get("id", ""))
                if hid and hid not in seen_ids:
                    seen_ids.add(hid)
                    deduped_flat.append(h)
                elif not hid:
                    deduped_flat.append(h)
            per_fused = deduped_flat[:fusion_top_k]

        per_query_hits[variant_q] = per_fused
        per_variant_counts[variant_q[:60]] = len(per_fused)
        if per_fused:
            per_variant_fused_lists.append(per_fused)

    # 5) Dedup all hits efficiently (prevent duplicate chunks from consuming context)
    if _has_fusion and dedup_hits is not None:
        try:
            all_hits_deduped = dedup_hits(all_hits_raw)
        except Exception as e:
            logger.warning("multi_query_all_dedup_failed", error_type=type(e).__name__)
            all_hits_deduped = all_hits_raw
    else:
        # Fallback dedup by id+text hash when fusion unavailable
        seen: Set[str] = set()
        all_hits_deduped = []
        import re, hashlib
        for h in all_hits_raw:
            payload = h.get("payload") or {}
            text = str(payload.get("chunk_text", "") or "").strip().lower()
            key = re.sub(r"\s+", " ", text)
            id_key = h.get("id", "")
            composite = f"{id_key}:{hashlib.sha256(key.encode()).hexdigest()[:16]}" if key else id_key
            if composite in seen or (key and key in seen):
                continue
            seen.add(composite)
            if key:
                seen.add(key)
            all_hits_deduped.append(h)

    retrieval_debug["all_hits_raw"] = len(all_hits_raw)
    retrieval_debug["all_hits_deduped"] = len(all_hits_deduped)
    retrieval_debug["per_variant_counts"] = per_variant_counts
    if variant_errors:
        retrieval_debug["variant_errors"] = variant_errors

    # 6) RRF fusion across all variant result lists
    fused_hits: List[Dict[str, Any]] = []
    if per_variant_fused_lists:
        if _has_fusion and rrf_fuse is not None and dedup_hits is not None:
            try:
                fused_hits = rrf_fuse(per_variant_fused_lists, k=rrf_k, top_k=fusion_top_k)
                fused_hits = dedup_hits(fused_hits)
            except Exception as e:
                logger.warning("multi_query_global_rrf_failed", error_type=type(e).__name__)
                # Fallback flatten dedup
                flat: List[Dict[str, Any]] = []
                seen_ids: Set[str] = set()
                for lst in per_variant_fused_lists:
                    for h in lst:
                        hid = str(h.get("id", ""))
                        if hid and hid not in seen_ids:
                            seen_ids.add(hid)
                            flat.append(h)
                try:
                    fused_hits = dedup_hits(flat)[:fusion_top_k] if dedup_hits else flat[:fusion_top_k]
                except Exception:
                    fused_hits = flat[:fusion_top_k]
        else:
            # Fallback without fusion helpers
            flat: List[Dict[str, Any]] = []
            seen_ids: Set[str] = set()
            for lst in per_variant_fused_lists:
                for h in lst:
                    hid = str(h.get("id", ""))
                    if hid and hid not in seen_ids:
                        seen_ids.add(hid)
                        flat.append(h)
            fused_hits = flat[:fusion_top_k]
    else:
        # No per-variant fused lists, fall back to deduped raw truncated
        fused_hits = all_hits_deduped[:fusion_top_k]

    retrieval_debug["fused_hits"] = len(fused_hits)

    # 7) MMR diversity selection (before expensive reranking)
    mmr_candidates: List[Dict[str, Any]] = fused_hits
    if _has_fusion and mmr_select is not None and len(fused_hits) > rerank_top_k:
        try:
            # Select up to 2x rerank pool for reranker to keep diversity but not explode cost
            candidates_k = min(len(fused_hits), max(rerank_top_k * 2, rerank_top_k))
            mmr_candidates = mmr_select(fused_hits, top_k=candidates_k, lambda_mult=0.65)
            retrieval_debug["mmr_applied"] = True
            retrieval_debug["mmr_candidates"] = len(mmr_candidates)
        except Exception as e:
            logger.warning("multi_query_mmr_failed", error_type=type(e).__name__)
            retrieval_debug["mmr_applied"] = False
            retrieval_debug["mmr_error"] = type(e).__name__
            mmr_candidates = fused_hits
    else:
        retrieval_debug["mmr_applied"] = False
        if len(fused_hits) <= rerank_top_k:
            retrieval_debug["mmr_skip_reason"] = "fused_lte_rerank_top_k"
        elif not _has_fusion:
            retrieval_debug["mmr_skip_reason"] = "fusion_unavailable"

    # 8) Cross-encoder reranking with visibility
    reranked_hits: List[Dict[str, Any]] = mmr_candidates[:rerank_top_k] if len(mmr_candidates) > rerank_top_k and not _has_fusion else mmr_candidates
    rerank_visibility: Dict[str, Any] = {"reranked": False, "model_loaded": False, "model_name": rerank_model, "fallback_reason": None}

    # Prefer reranking.py visibility wrapper when available
    try:
        from app.rag.reranking import rerank_with_visibility  # lazy
        try:
            vis = rerank_with_visibility(original_query, mmr_candidates, model_name=rerank_model, top_k=rerank_top_k)
            reranked_hits = vis.hits
            rerank_visibility = {
                "reranked": bool(vis.reranked),
                "model_loaded": bool(vis.model_loaded),
                "model_name": vis.model_name,
                "fallback_reason": vis.fallback_reason,
            }
        except Exception as e:
            logger.warning("multi_query_rerank_visibility_failed", error_type=type(e).__name__)
            rerank_visibility["fallback_reason"] = f"visibility_error:{type(e).__name__}"
            # Fallback to fusion reranker
            try:
                from app.rag.fusion import rerank_cross_encoder  # lazy

                fallback = rerank_cross_encoder(original_query, mmr_candidates, model_name=rerank_model, top_k=rerank_top_k)
                has_scores = any("rerank_score" in h for h in fallback)
                reranked_hits = fallback
                rerank_visibility["reranked"] = has_scores
                rerank_visibility["model_loaded"] = True
                if not has_scores:
                    rerank_visibility["fallback_reason"] = "rerank_failed_kept_order"
            except Exception as e2:
                logger.warning("multi_query_rerank_fallback_failed", error_type=type(e2).__name__)
                reranked_hits = mmr_candidates[:rerank_top_k]
                rerank_visibility["fallback_reason"] = f"rerank_error:{type(e2).__name__}"
    except Exception as e:
        # reranking.py not available — use fusion reranker directly
        if "reranking" not in str(type(e).__name__).lower():
            logger.warning("multi_query_reranking_import_failed", error_type=type(e).__name__)
        try:
            from app.rag.fusion import rerank_cross_encoder  # lazy

            fallback = rerank_cross_encoder(original_query, mmr_candidates, model_name=rerank_model, top_k=rerank_top_k)
            has_scores = any("rerank_score" in h for h in fallback)
            reranked_hits = fallback
            rerank_visibility["reranked"] = has_scores
            # rerank_cross_encoder handles missing model gracefully; infer model_loaded
            rerank_visibility["model_loaded"] = has_scores
            if not has_scores:
                rerank_visibility["fallback_reason"] = "rerank_failed_kept_order_or_unavailable"
        except Exception as e2:
            logger.warning("multi_query_rerank_failed", error_type=type(e2).__name__)
            reranked_hits = mmr_candidates[:rerank_top_k]
            rerank_visibility["fallback_reason"] = f"rerank_error:{type(e2).__name__}"

    retrieval_debug["rerank"] = rerank_visibility
    retrieval_debug["reranked_hits"] = len(reranked_hits)

    # 9) Final evidence hits (already capped by rerank_top_k, but ensure dedup not reintroduced)
    # Reranker returns top_k; final is that list deduped again to guarantee no duplicates in context
    if _has_fusion and dedup_hits is not None:
        try:
            final_hits = dedup_hits(reranked_hits)[:rerank_top_k]
        except Exception:
            final_hits = reranked_hits[:rerank_top_k]
    else:
        final_hits = reranked_hits[:rerank_top_k]

    retrieval_debug["final_hits"] = len(final_hits)
    retrieval_debug["all_hits"] = len(all_hits_deduped)
    retrieval_debug["fused"] = len(fused_hits)

    # Build bundle for legacy and debug
    bundle = MultiQueryBundle(
        original=original_query,
        rewritten=rewritten_query,
        subqueries=subqueries,
        expanded=expanded_queries,
    )

    result = MultiQueryResult(
        original_query=original_query,
        rewritten_query=rewritten_query,
        subqueries=subqueries,
        expanded_queries=expanded_queries,
        all_hits=all_hits_deduped,
        fused_hits=fused_hits,
        reranked_hits=reranked_hits,
        final_hits=final_hits,
        retrieval_debug=retrieval_debug,
        fused=fused_hits,
        per_query_hits=per_query_hits,
        bundle=bundle,
    )

    # Ensure legacy fused sync (post_init already does, but be explicit)
    if not result.fused:
        result.fused = result.fused_hits

    return result

