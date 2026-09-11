"""Hybrid RAG pipeline orchestrator - opt-in, permission-aware, local-only.

Security invariants (must not be violated):
- Authorization filter is created BEFORE any retrieval.
- Filter is applied via Qdrant's query_filter/scroll_filter - never post-filter.
- No chunk_text reaches the LLM without passing permission filtering.
- Hashing for observability when LOG_SENSITIVE_DATA is False.

Phase 1 Pipeline:
  permission filter -> [vector TOP_K + BM25 TOP_K] -> RRF fusion -> dedup
  -> MMR diversity -> cross-encoder rerank -> context (MAX_CONTEXT_CHARS)
  -> grounded prompt -> local LLM -> citation validation + confidence

Phase 2 Enhancements (all feature-flagged, backward compatible):
  query understanding -> rewrite/decompose/expand -> multi-query retrieval
  -> dedup -> RRF -> rerank with visibility -> doc diversity -> doc ranking
  -> evidence grouping -> context optimizer -> adjacent expansion
  -> table handling -> version handling -> conflict detection
  -> coverage -> calibrated confidence -> citation/claim validation
  -> prompt injection defense -> performance tracking

Falls back to legacy behavior for:
- GENERAL/MEMORY intents (no retrieval)
- Explicit SIH identifier/title/listing structured paths (kept deterministic)
- Any failure in hybrid stage (degrades gracefully, logs, still permission-aware)

New modules are separate files; legacy RAGService.query is untouched when
HYBRID_RAG_ENABLED is False.
"""

from __future__ import annotations

import re
import time
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

from app.core.config import settings
from app.core.logging import get_logger, log_rag_event
from app.rag.embeddings import embed_text
from app.rag.qdrant import QdrantManager
from app.rag.types import RAGQuery, RAGResult, SearchHit
from app.rag.routing import QueryIntent
from app.rag.bm25 import bm25_search
from app.rag.fusion import (
    rrf_fuse,
    dedup_hits,
    mmr_select,
    rerank_cross_encoder,
    build_context,
    validate_citations,
    confidence_score,
)

logger = get_logger(__name__)

_IDENTIFIER_RE = re.compile(r"\bSIH[\s-]*\d{5}\b", re.IGNORECASE)
INSUFFICIENT = "I couldn't find sufficient information in the authorized company knowledge base to answer this accurately."
COMPANY_THRESHOLD = 0.55  # kept aligned with service.py


def _should_use_hybrid(query: RAGQuery) -> bool:
    if query.enable_hybrid is not None:
        return bool(query.enable_hybrid)
    return bool(getattr(settings, "HYBRID_RAG_ENABLED", False))


def _is_phase2_enabled() -> bool:
    return bool(getattr(settings, "PHASE2_ENABLED", False))


def _identifier_terms(question: str) -> List[str]:
    return list(dict.fromkeys(
        re.sub(r"[^a-z0-9]", "", m.lower()) for m in _IDENTIFIER_RE.findall(question or "")
    ))


def _hits_to_search_hits(raw_hits: List[Dict[str, Any]]) -> List[SearchHit]:
    out: List[SearchHit] = []
    for r in raw_hits:
        payload = r.get("payload") or {}
        try:
            doc_id = UUID(str(payload.get("document_id", "00000000-0000-0000-0000-000000000000")))
        except Exception:
            doc_id = UUID("00000000-0000-0000-0000-000000000000")
        out.append(SearchHit(
            id=r.get("id", ""),
            score=float(r.get("rerank_score", r.get("rrf_score", r.get("score", 0.0)))),
            payload=payload,
            document_id=doc_id,
            filename=payload.get("filename", "unknown"),
            page_number=payload.get("page_number"),
            chunk_text=payload.get("chunk_text", "") or "",
            classification=payload.get("classification", "public_internal"),
            department=payload.get("department"),
            section_title=payload.get("section_title"),
            chunk_index=payload.get("chunk_index"),
            char_start=payload.get("char_start"),
            char_end=payload.get("char_end"),
            document_title=payload.get("document_title"),
            author=payload.get("author"),
        ))
    return out


class HybridRagPipeline:
    """Orchestrates the hybrid retrieval + generation pipeline."""

    def __init__(
        self,
        qdrant_manager: Optional[QdrantManager],
        embedding_model_name: str = "nomic-embed-text",
        llm_model_name: str = "qwen2.5:7b-instruct",
    ):
        self.qdrant = qdrant_manager
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name

    def query(self, query: RAGQuery, rag_service=None) -> Optional[RAGResult]:
        """Run hybrid pipeline. Returns None if hybrid not applicable (caller falls back).

        Args:
            query: RAGQuery with user context.
            rag_service: Existing RAGService instance for prompt/LLM reuse (avoids duplication).

        Returns:
            RAGResult with pipeline="hybrid" on success, None to signal fallback to legacy.
        """
        if not _should_use_hybrid(query):
            return None

        intent = query.intent or QueryIntent.COMPANY.value
        if intent in (QueryIntent.GENERAL.value, QueryIntent.MEMORY.value):
            return None  # No retrieval for these intents

        if self.qdrant is None:
            return None

        # Preserve deterministic structured paths (listing / identifier / title lookup)
        # These are already permission-aware and should not be hybridized.
        identifier_terms = _identifier_terms(query.question)
        is_listing = (
            not identifier_terms
            and re.search(r"\blist\b.*\b(?:sih|problem statements?)\b", query.question, re.IGNORECASE)
        )
        is_title_lookup = bool(re.search(
            r"\b(?:ps|problem statement)\s+(?:number|id)\b|\bwhat\s+title\s+(?:belongs|is associated)\b",
            query.question, re.IGNORECASE,
        ))
        if is_listing or identifier_terms or is_title_lookup:
            return None  # Defer to legacy deterministic path

        return self._run_hybrid(query, rag_service)

    def _run_hybrid(self, query: RAGQuery, rag_service) -> RAGResult:
        start_time = time.time()
        request_id = getattr(query, "request_id", None)
        question = query.question or ""
        phase2 = _is_phase2_enabled()

        # -- Performance tracking ---------------------------------------------
        tracker = None
        if phase2:
            try:
                from app.rag.performance import PerformanceTracker
                tracker = PerformanceTracker()
            except Exception:
                tracker = None

        # -- Query understanding (P1) -----------------------------------------
        analysis = None
        if phase2:
            try:
                from app.rag.query_understanding import analyze_query
                if tracker:
                    tracker.start("query_understanding")
                analysis = analyze_query(question, query.conversation_history)
                if tracker:
                    tracker.end("query_understanding")
                # Log analysis for observability (no raw text when LOG_SENSITIVE_DATA is False)
                log_rag_event(
                    logger, event_type="phase2_query_analysis",
                    user_id=str(query.user_id) if query.user_id else None,
                    request_id=request_id, user_role=query.user_role,
                    department=query.department, query=question,
                    details={
                        "query_types": [t.value for t in analysis.query_types] if analysis else [],
                        "is_simple": getattr(analysis, "is_simple", False),
                        "confidence": getattr(analysis, "confidence", 0),
                    },
                )
            except Exception as e:
                logger.warning("phase2_query_understanding_failed", error_type=type(e).__name__, request_id=request_id)

        # 1) Permission filter — BEFORE any retrieval (security invariant)
        if tracker:
            tracker.start("permission_filter")
        try:
            filter_conditions = self.qdrant.create_permission_filter(
                user_role=query.user_role or "employee",
                user_id=query.user_id,
                department=query.department,
            )
        except Exception as e:
            logger.error("hybrid_permission_filter_failed", error_type=type(e).__name__, request_id=request_id)
            return RAGResult(
                answer=INSUFFICIENT,
                sources=[],
                model_used=self.llm_model_name,
                processing_time_ms=int((time.time() - start_time) * 1000),
                question=question,
                pipeline="hybrid",
                confidence=0.0,
            )
        finally:
            if tracker:
                tracker.end("permission_filter")

        # 2-4) Retrieval — multi-query (Phase 2) or single-query (Phase 1)
        vector_hits: List[Dict[str, Any]] = []
        bm25_hits: List[Dict[str, Any]] = []
        fused: List[Dict[str, Any]] = []
        multi_query_bundle = None
        multi_query_result = None

        use_multi_query = (
            phase2
            and bool(getattr(settings, "PHASE2_MULTI_QUERY_ENABLED", False))
            and analysis is not None
            and not getattr(analysis, "is_simple", True)
        )

        if use_multi_query:
            # Phase 2 multi-query path
            if tracker:
                tracker.start("multi_query_retrieval")
            try:
                from app.rag.multi_query import build_multi_query_bundle, run_multi_query_retrieval
                bundle = build_multi_query_bundle(
                    question,
                    conversation_history=query.conversation_history,
                    memory_summary=getattr(query, "memory_summary", None),
                    enable_rewrite=bool(getattr(settings, "PHASE2_QUERY_REWRITE_ENABLED", False)),
                    enable_decomposition=bool(getattr(settings, "PHASE2_DECOMPOSITION_ENABLED", False)),
                    enable_expansion=bool(getattr(settings, "PHASE2_QUERY_EXPANSION_ENABLED", False)),
                )
                multi_query_bundle = bundle
                mqr = run_multi_query_retrieval(
                    bundle, self.qdrant, filter_conditions,
                    embedding_model_name=self.embedding_model_name,
                    vector_top_k=int(getattr(settings, "VECTOR_TOP_K", 20)),
                    bm25_top_k=int(getattr(settings, "BM25_TOP_K", 20)),
                    fusion_top_k=int(getattr(settings, "FUSION_TOP_K", 12)),
                    rrf_k=int(getattr(settings, "RRF_K", 60)),
                    score_threshold=COMPANY_THRESHOLD if query.intent == QueryIntent.COMPANY.value else None,
                )
                multi_query_result = mqr
                fused = mqr.fused
                # For observability, aggregate counts
                vector_hits = []  # multi-query fuses internally
                bm25_hits = []
                log_rag_event(
                    logger, event_type="phase2_multi_query_retrieval",
                    user_id=str(query.user_id) if query.user_id else None,
                    request_id=request_id, user_role=query.user_role,
                    department=query.department, query=question,
                    sources_count=len(fused),
                    details={
                        "variants": len(bundle.all_queries()),
                        "per_query_counts": {q[:60]: len(h) for q, h in mqr.per_query_hits.items()},
                        "fused": len(fused),
                    },
                )
            except Exception as e:
                logger.warning("phase2_multi_query_failed_falling_back", error_type=type(e).__name__, request_id=request_id)
                use_multi_query = False
            finally:
                if tracker:
                    tracker.end("multi_query_retrieval")

        if not use_multi_query:
            # Phase 1 single-query path (also fallback for multi-query failure)
            if tracker:
                tracker.start("vector_retrieval")
            try:
                query_embedding = embed_text(question, self.embedding_model_name)
                vector_hits = self.qdrant.search(
                    query_vector=query_embedding,
                    limit=int(getattr(settings, "VECTOR_TOP_K", 20)),
                    filter_conditions=filter_conditions,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=COMPANY_THRESHOLD if query.intent == QueryIntent.COMPANY.value else None,
                )
            except Exception as e:
                logger.warning("hybrid_vector_search_failed", error_type=type(e).__name__, request_id=request_id)
                vector_hits = []
            finally:
                if tracker:
                    tracker.end("vector_retrieval")

            if tracker:
                tracker.start("bm25_retrieval")
            try:
                bm25_hits = bm25_search(
                    qdrant_manager=self.qdrant,
                    query_text=question,
                    filter_conditions=filter_conditions,
                    limit=int(getattr(settings, "BM25_TOP_K", 20)),
                    k1=float(getattr(settings, "BM25_K1", 1.5)),
                    b=float(getattr(settings, "BM25_B", 0.75)),
                )
            except Exception as e:
                logger.warning("hybrid_bm25_failed", error_type=type(e).__name__, request_id=request_id)
                bm25_hits = []
            finally:
                if tracker:
                    tracker.end("bm25_retrieval")

            # Observability (sanitized: no raw query/chunk_text when LOG_SENSITIVE_DATA is False)
            log_rag_event(
                logger,
                event_type="hybrid_retrieval",
                user_id=str(query.user_id) if query.user_id else None,
                request_id=request_id,
                user_role=query.user_role,
                department=query.department,
                query=question,
                sources_count=len(vector_hits) + len(bm25_hits),
                details={
                    "vector_hits": len(vector_hits),
                    "bm25_hits": len(bm25_hits),
                    "pipeline": "hybrid",
                },
            )

            if not vector_hits and not bm25_hits:
                return RAGResult(
                    answer=INSUFFICIENT,
                    sources=[],
                    model_used=self.llm_model_name,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    question=question,
                    pipeline="hybrid",
                    confidence=0.0,
                )

            # 4) RRF fusion
            if tracker:
                tracker.start("rrf_fusion")
            try:
                fused = rrf_fuse(
                    [vector_hits, bm25_hits] if bm25_hits else [vector_hits],
                    k=int(getattr(settings, "RRF_K", 60)),
                    top_k=int(getattr(settings, "FUSION_TOP_K", 12)),
                )
            except Exception as e:
                logger.warning("hybrid_rrf_failed", error_type=type(e).__name__, request_id=request_id)
                fused = (vector_hits + bm25_hits)[: int(getattr(settings, "FUSION_TOP_K", 12))]
            finally:
                if tracker:
                    tracker.end("rrf_fusion")

            # 5) Dedup
            fused = dedup_hits(fused)

            # 6) Filter low-confidence Company hits (keep at least fused if threshold would empty)
            if query.intent == QueryIntent.COMPANY.value:
                # Only apply threshold to vector-originated scores; BM25 scores are on different scale
                # So we skip thresholding fused hybrid — confidence is computed later
                pass

            if not fused:
                return RAGResult(
                    answer=INSUFFICIENT,
                    sources=[],
                    model_used=self.llm_model_name,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    question=question,
                    pipeline="hybrid",
                    confidence=0.0,
                )

        # From here, `fused` is populated via either path.
        if not fused:
            return RAGResult(
                answer=INSUFFICIENT,
                sources=[],
                model_used=self.llm_model_name,
                processing_time_ms=int((time.time() - start_time) * 1000),
                question=question,
                pipeline="hybrid",
                confidence=0.0,
            )

        # 7) Diversity (MMR) — select diverse subset before expensive reranking
        # Keep fusion pool at FUSION_TOP_K, but narrow to RERANK_TOP_K via MMR then rerank
        fusion_top_k = int(getattr(settings, "FUSION_TOP_K", 12))
        rerank_top_k = int(getattr(settings, "RERANK_TOP_K", 5))
        # If fusion produced more than rerank pool, use MMR to pick candidates for reranking
        rerank_candidates = fused
        if len(fused) > rerank_top_k:
            if tracker:
                tracker.start("mmr_diversity")
            try:
                rerank_candidates = mmr_select(fused, top_k=min(len(fused), max(rerank_top_k * 2, rerank_top_k)), lambda_mult=0.65)
            finally:
                if tracker:
                    tracker.end("mmr_diversity")

        # 8) Cross-encoder reranking (local, graceful fallback) with visibility
        reranked = None
        rerank_visibility = None
        if tracker:
            tracker.start("reranking")
        try:
            # Prefer reranking.py visibility wrapper when Phase 2 is on
            if phase2 and bool(getattr(settings, "PHASE2_RERANK_VISIBILITY_ENABLED", True)):
                try:
                    from app.rag.reranking import rerank_with_visibility
                    vis = rerank_with_visibility(
                        question, rerank_candidates,
                        model_name=str(getattr(settings, "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
                        top_k=rerank_top_k,
                    )
                    reranked = vis.hits
                    rerank_visibility = vis
                except Exception:
                    reranked = rerank_cross_encoder(
                        query=question,
                        hits=rerank_candidates,
                        model_name=str(getattr(settings, "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
                        top_k=rerank_top_k,
                    )
            else:
                reranked = rerank_cross_encoder(
                    query=question,
                    hits=rerank_candidates,
                    model_name=str(getattr(settings, "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
                    top_k=rerank_top_k,
                )
        finally:
            if tracker:
                tracker.end("reranking")

        # 9) Final diversity pass on reranked output (small pool, light)
        # Phase 2 document diversity (relevance-aware per-doc capping)
        final_hits = reranked  # reranker already selected top rerank_top_k
        if phase2 and bool(getattr(settings, "PHASE2_DIVERSITY_ENABLED", False)):
            if tracker:
                tracker.start("document_diversity")
            try:
                from app.rag.document_diversity import diversify_by_document
                # Apply doc diversity as post-rerank filter (keeps rerank order primary)
                diversified = diversify_by_document(
                    final_hits,
                    top_k=len(final_hits),
                    max_per_doc=int(getattr(settings, "PHASE2_DIVERSITY_MAX_PER_DOC", 3)),
                )
                if diversified:
                    final_hits = diversified
            except Exception as e:
                logger.warning("phase2_diversity_failed", error_type=type(e).__name__, request_id=request_id)
            finally:
                if tracker:
                    tracker.end("document_diversity")

        # Document-level ranking (for debug/observability, not filtering)
        doc_ranking = None
        if phase2:
            try:
                from app.rag.document_ranking import rank_documents
                doc_ranking = rank_documents(final_hits, top_docs=5)
            except Exception:
                pass

        # Adjacent chunk expansion (opt-in, permission-filtered)
        if phase2 and bool(getattr(settings, "PHASE2_ADJACENT_EXPANSION_ENABLED", False)):
            if tracker:
                tracker.start("adjacent_expansion")
            try:
                from app.rag.adjacent_chunks import expand_adjacent_chunks
                expanded = expand_adjacent_chunks(
                    final_hits,
                    qdrant_manager=self.qdrant,
                    filter_conditions=filter_conditions,
                    score_threshold=float(getattr(settings, "PHASE2_ADJACENT_THRESHOLD", 0.7)),
                )
                # Only keep expansion if it added something useful and within budget
                if len(expanded) > len(final_hits):
                    final_hits = expanded[: rerank_top_k + 3]  # cap growth
            except Exception as e:
                logger.warning("phase2_adjacent_expansion_failed", error_type=type(e).__name__, request_id=request_id)
            finally:
                if tracker:
                    tracker.end("adjacent_expansion")

        # Table handling — annotate hits where chunk is tabular
        if phase2 and bool(getattr(settings, "PHASE2_TABLE_HANDLING_ENABLED", True)):
            try:
                from app.rag.table_handling import annotate_chunks_with_table_info
                annotate_chunks_with_table_info(final_hits)
            except Exception:
                pass

        # Version handling — prefer latest unless historical query
        if phase2 and bool(getattr(settings, "PHASE2_VERSION_HANDLING_ENABLED", False)):
            if tracker:
                tracker.start("version_handling")
            try:
                from app.rag.version_handling import resolve_document_versions
                versioned = resolve_document_versions(final_hits, question=question)
                if versioned:
                    final_hits = versioned
            except Exception as e:
                logger.warning("phase2_version_handling_failed", error_type=type(e).__name__, request_id=request_id)
            finally:
                if tracker:
                    tracker.end("version_handling")

        # 10) Context construction under MAX_CONTEXT_CHARS
        max_chars = int(getattr(settings, "MAX_CONTEXT_CHARS", 12000))
        context = ""
        included_raw: List[Dict[str, Any]] = []
        context_stats = None

        if phase2:
            # Try grouped context first (better multi-doc presentation)
            try:
                # Use grouped context for multi-doc, optimizer for single-doc
                is_multi = getattr(analysis, "is_multi_document", False) if analysis else False
                if is_multi and len({str((h.get("payload") or {}).get("document_id", "")) for h in final_hits}) > 1:
                    from app.rag.evidence_grouping import group_evidence_by_document, build_grouped_context

                    if tracker:
                        tracker.start("grouped_context")
                    groups = group_evidence_by_document(final_hits)
                    context, included_raw = build_grouped_context(groups, max_chars=max_chars)
                    if tracker:
                        tracker.end("grouped_context")
                else:
                    from app.rag.context_optimizer import optimize_context_with_stats

                    if tracker:
                        tracker.start("context_optimization")
                    context, included_raw, context_stats = optimize_context_with_stats(final_hits, max_chars=max_chars, query=question)
                    if tracker:
                        tracker.end("context_optimization")
                # Fallback to legacy build_context if phase2 produced empty
                if not context.strip():
                    context, included_raw = build_context(final_hits, max_chars=max_chars)
            except Exception as e:
                logger.warning("phase2_context_failed_falling_back", error_type=type(e).__name__, request_id=request_id)
                context, included_raw = build_context(final_hits, max_chars=max_chars)
        else:
            context, included_raw = build_context(final_hits, max_chars=max_chars)

        if not context.strip():
            return RAGResult(
                answer=INSUFFICIENT,
                sources=[],
                model_used=self.llm_model_name,
                processing_time_ms=int((time.time() - start_time) * 1000),
                question=question,
                pipeline="hybrid",
                confidence=0.0,
            )

        # 11) Evidence coverage, conflict detection, confidence
        coverage_report = None
        conflicts = []
        if phase2:
            # Coverage — measure per sub-question (or overall)
            if bool(getattr(settings, "PHASE2_COVERAGE_ENABLED", True)):
                try:
                    from app.rag.evidence_coverage import assess_coverage
                    sub_qs = getattr(analysis, "sub_questions", []) if analysis else []
                    # Also consider multi_query subqueries if available
                    if multi_query_bundle and multi_query_bundle.subqueries:
                        sub_qs = multi_query_bundle.subqueries
                    coverage_report = assess_coverage(sub_qs, final_hits)
                except Exception:
                    pass

            # Conflict detection
            if bool(getattr(settings, "PHASE2_CONFLICT_DETECTION_ENABLED", True)):
                try:
                    from app.rag.conflict_detection import detect_conflicts
                    conflicts = detect_conflicts(final_hits, question=question)
                    if conflicts:
                        log_rag_event(
                            logger, event_type="phase2_conflicts_detected",
                            user_id=str(query.user_id) if query.user_id else None,
                            request_id=request_id, query=question,
                            details={"conflicts": len(conflicts), "topics": [c.topic for c in conflicts]},
                        )
                except Exception:
                    pass

        # Confidence — calibrated (Phase 2) or legacy tanh
        conf = 0.0
        confidence_breakdown = None
        if phase2:
            try:
                from app.rag.confidence import compute_confidence
                coverage_ratio = getattr(coverage_report, "coverage_ratio", 1.0) if coverage_report else 1.0
                cb = compute_confidence(
                    final_hits,
                    reranked=bool(rerank_visibility.reranked) if rerank_visibility else False,
                    vector_hits=vector_hits if not use_multi_query else None,
                    bm25_hits=bm25_hits if not use_multi_query else None,
                    coverage_ratio=coverage_ratio,
                    citation_valid=True,
                    conflicts=conflicts,
                )
                conf = cb.final
                confidence_breakdown = cb
            except Exception:
                conf = confidence_score(final_hits)
        else:
            conf = confidence_score(final_hits)

        citation_check = validate_citations("", final_hits) if getattr(settings, "CITATION_VALIDATION_ENABLED", True) else {"valid": True, "issues": []}

        # Prompt injection defense — detect injections in retrieved chunks
        injection_result = None
        if phase2 and bool(getattr(settings, "PHASE2_PROMPT_DEFENSE_ENABLED", True)):
            try:
                from app.rag.prompt_defense import detect_injection_in_hits
                injection_result = detect_injection_in_hits(final_hits)
                if injection_result and injection_result.has_injection:
                    log_rag_event(
                        logger, event_type="phase2_injection_detected",
                        user_id=str(query.user_id) if query.user_id else None,
                        request_id=request_id, query=question,
                        details={"findings": len(injection_result.findings)},
                    )
            except Exception:
                pass

        # 12) Grounded generation — reuse existing prompt/LLM if available, else direct
        # Build prompt — use secure prompt when Phase 2 defense is on
        prompt = None
        if phase2 and bool(getattr(settings, "PHASE2_PROMPT_DEFENSE_ENABLED", True)):
            try:
                from app.rag.prompt_defense import build_secure_prompt
                prompt = build_secure_prompt(question, context, query.conversation_history)
            except Exception:
                prompt = None

        if prompt is None:
            if rag_service is not None and hasattr(rag_service, "_build_prompt") and hasattr(rag_service, "_call_llm"):
                prompt = rag_service._build_prompt(question, context, query.conversation_history)
                if getattr(query, "memory_summary", None):
                    pass
                llm_result = rag_service._call_llm(prompt)
                answer = (llm_result.get("response") or "").strip() or INSUFFICIENT
                tokens_used = llm_result.get("tokens_used")
                model_used = rag_service.llm_model_name
            else:
                from app.rag.service import RAGService as _RAGService
                tmp = _RAGService(
                    qdrant_manager=self.qdrant,
                    embedding_model_name=self.embedding_model_name,
                    llm_model_name=self.llm_model_name,
                )
                prompt = tmp._build_prompt(question, context, query.conversation_history)
                llm_result = tmp._call_llm(prompt)
                answer = (llm_result.get("response") or "").strip() or INSUFFICIENT
                tokens_used = llm_result.get("tokens_used")
                model_used = self.llm_model_name
        else:
            # Secure prompt already built — call LLM
            if tracker:
                tracker.start("llm_generation")
            try:
                if rag_service is not None and hasattr(rag_service, "_call_llm"):
                    llm_result = rag_service._call_llm(prompt)
                    answer = (llm_result.get("response") or "").strip() or INSUFFICIENT
                    tokens_used = llm_result.get("tokens_used")
                    model_used = rag_service.llm_model_name
                else:
                    from app.rag.service import RAGService as _RAGService
                    tmp = _RAGService(
                        qdrant_manager=self.qdrant,
                        embedding_model_name=self.embedding_model_name,
                        llm_model_name=self.llm_model_name,
                    )
                    llm_result = tmp._call_llm(prompt)
                    answer = (llm_result.get("response") or "").strip() or INSUFFICIENT
                    tokens_used = llm_result.get("tokens_used")
                    model_used = self.llm_model_name
            finally:
                if tracker:
                    tracker.end("llm_generation")

        # If we used the non-secure path, llm already called; if secure path, also done.
        # For non-secure path where prompt was built via rag_service._build_prompt, llm already called above.
        # Need to handle the case where prompt was built securely but we already handled llm.
        # Actually both paths above already call LLM. So answer/tokens_used/model_used are set.

        # 13) Post-generation validation
        search_hits = _hits_to_search_hits(included_raw if context else final_hits)
        if getattr(settings, "CITATION_VALIDATION_ENABLED", True):
            post_check = validate_citations(answer, [h for h in final_hits if h in included_raw])
            if not post_check["valid"]:
                logger.warning("hybrid_citation_validation_issues", issues=post_check["issues"], request_id=request_id)

        # Strict citation validation (Phase 2)
        strict_citation_report = None
        if phase2:
            try:
                from app.rag.citation_validation import strict_validate_citations
                strict_citation_report = strict_validate_citations(answer, [h for h in final_hits if h in included_raw] or final_hits)
                if strict_citation_report and not strict_citation_report.valid:
                    logger.warning("phase2_strict_citation_issues", issues=[i.detail for i in strict_citation_report.issues], request_id=request_id)
            except Exception:
                pass

        # Claim validation (Phase 2)
        claim_report = None
        if phase2:
            try:
                from app.rag.claim_validation import extract_claims, validate_claims
                claims = extract_claims(answer)
                if claims:
                    claim_report = validate_claims(claims, [h for h in final_hits if h in included_raw] or final_hits)
            except Exception:
                pass

        processing_ms = int((time.time() - start_time) * 1000)

        # Only return grounded answer if we have sources; otherwise insufficient
        if not search_hits:
            answer = INSUFFICIENT
            conf = 0.0

        # Build retrieval_debug
        retrieval_debug: Dict[str, Any] = {
            "vector_hits": len(vector_hits),
            "bm25_hits": len(bm25_hits),
            "fused": len(fused),
            "reranked": len(final_hits),
            "included": len(search_hits),
            "max_context_chars": max_chars,
            "citation_valid": citation_check.get("valid", True),
        }
        if phase2:
            retrieval_debug["phase2"] = True
            if analysis:
                retrieval_debug["query_types"] = [t.value for t in analysis.query_types]
                retrieval_debug["is_simple"] = analysis.is_simple
            if multi_query_bundle:
                retrieval_debug["multi_query_variants"] = len(multi_query_bundle.all_queries())
                retrieval_debug["multi_query_bundle"] = {
                    "original": multi_query_bundle.original[:120],
                    "rewritten": (multi_query_bundle.rewritten or "")[:120],
                    "subqueries": multi_query_bundle.subqueries,
                    "expanded": multi_query_bundle.expanded,
                }
            if rerank_visibility:
                retrieval_debug["rerank"] = {
                    "reranked": rerank_visibility.reranked,
                    "model_loaded": rerank_visibility.model_loaded,
                    "model_name": rerank_visibility.model_name,
                    "fallback_reason": rerank_visibility.fallback_reason,
                }
            if doc_ranking:
                retrieval_debug["doc_ranking"] = [
                    {"document_id": d.document_id, "filename": d.filename, "combined_score": d.combined_score, "chunk_count": d.chunk_count}
                    for d in doc_ranking[:5]
                ]
            if coverage_report:
                retrieval_debug["coverage"] = {
                    "ratio": coverage_report.coverage_ratio,
                    "covered": coverage_report.covered_count,
                    "total": coverage_report.total_sub_questions,
                    "gaps": coverage_report.gaps,
                }
            if conflicts:
                retrieval_debug["conflicts"] = [
                    {"topic": c.topic, "values": c.values, "severity": c.severity, "description": c.description}
                    for c in conflicts
                ]
            if confidence_breakdown:
                retrieval_debug["confidence_breakdown"] = {
                    "final": confidence_breakdown.final,
                    "reranker_component": confidence_breakdown.reranker_component,
                    "retrieval_agreement": confidence_breakdown.retrieval_agreement,
                    "supporting_chunks": confidence_breakdown.supporting_chunks,
                    "supporting_docs": confidence_breakdown.supporting_docs,
                    "query_coverage": confidence_breakdown.query_coverage,
                    "citation_coverage": confidence_breakdown.citation_coverage,
                    "source_agreement": confidence_breakdown.source_agreement,
                    "conflict_penalty": confidence_breakdown.conflict_penalty,
                    "details": confidence_breakdown.details,
                }
            if strict_citation_report:
                retrieval_debug["strict_citation"] = {
                    "valid": strict_citation_report.valid,
                    "mapped": strict_citation_report.mapped_citations,
                    "total": strict_citation_report.total_citations,
                    "coverage": strict_citation_report.coverage,
                    "issues": [i.detail for i in strict_citation_report.issues[:5]],
                }
            if claim_report:
                retrieval_debug["claim_validation"] = {
                    "total_claims": claim_report.total_claims,
                    "grounded": claim_report.grounded_count,
                    "ratio": claim_report.grounded_ratio,
                    "ungrounded": claim_report.ungrounded_claims[:3],
                }
            if injection_result and injection_result.has_injection:
                retrieval_debug["injection"] = {
                    "has_injection": True,
                    "findings": [{"pattern": f.pattern, "matched": f.matched_text, "source": f.source} for f in injection_result.findings[:5]],
                }
            if context_stats:
                retrieval_debug["context_stats"] = {
                    "total_chars": context_stats.total_chars,
                    "included": context_stats.included_count,
                    "truncated": context_stats.truncated_count,
                    "avg_relevance": context_stats.avg_relevance,
                    "utilization": context_stats.utilization,
                }
            if tracker:
                retrieval_debug["performance"] = tracker.to_dict()

        result = RAGResult(
            answer=answer,
            sources=search_hits,
            model_used=model_used,
            processing_time_ms=processing_ms,
            tokens_used=tokens_used,
            question=question,
            confidence=conf,
            pipeline="hybrid",
            retrieval_debug=retrieval_debug,
        )

        log_rag_event(
            logger,
            event_type="hybrid_query_completed",
            user_id=str(query.user_id) if query.user_id else None,
            request_id=request_id,
            user_role=query.user_role,
            department=query.department,
            query=question,
            sources_count=len(search_hits),
            answer_length=len(answer),
            processing_time_ms=processing_ms,
            details={
                "pipeline": "hybrid",
                "confidence": conf,
                "model": model_used,
                **(result.retrieval_debug or {}),
            },
        )
        return result
