# AegisAI — Advanced Hybrid RAG Implementation

Status: Incremental, opt-in, backward compatible. All 141 existing tests pass with `HYBRID_RAG_ENABLED=false` (default).

## 1) What changed (summary)

**Goal:** Enterprise-grade multi-document hybrid retrieval without breaking JWT/RBAC/department/classification/audit/Docker/Postgres/Qdrant/Ollama/frontend contracts, and without external APIs.

**Architecture:**
```
Question → classify intent → [GENERAL/MEMORY: conversational LLM, no retrieval]
                          → [DOCUMENT_AWARE: DB-scoped document listing, no Qdrant/LLM]
                          → [COMPANY: permission Filter BEFORE retrieval]
                                     → hybrid (when enabled): vector TOP_K + BM25 TOP_K
                                       → RRF fusion (k=60) → dedup → MMR diversity
                                       → local cross-encoder rerank → context MAX_CHARS
                                       → grounded prompt → local Ollama → citation validation + confidence
                                     → legacy (default): single-vector filtered search → top-K → prompt
```

**Security invariants (enforced):**
- Authorization is server-side Qdrant `query_filter` / `scroll_filter` created via `QdrantManager.create_permission_filter(...)` BEFORE any vector or BM25 retrieval.
- `admin` bypasses filter; `manager`/`engineer` get department OR `public_internal` (OR via `min_should`); `employee` gets `public_internal`; all roles additionally get `uploaded_by_id` / `allowed_users` via `allowed_users` match.
- No unauthorized `chunk_text` is ever sent to the LLM.
- Observability respects `LOG_SENSITIVE_DATA=false` (hashes/fingerprints via `log_rag_event`).

## 2) Files added

- `aegisai/backend/app/rag/bm25.py` — Local BM25 (k1/b configurable) over authorized Qdrant `scroll`. No external call. Tokenization strips stopwords/punctuation; IDF is Robertson/Sparck-Jones with smoothing.
- `aegisai/backend/app/rag/fusion.py` — `rrf_fuse` (RRF k=60), `dedup_hits` (normalized chunk_text + id), `mmr_select` (token Jaccard MMR, λ≈0.65), `rerank_cross_encoder` (local `sentence-transformers` CrossEncoder, graceful fallback if model/dep unavailable), `build_context` (hard cap `MAX_CONTEXT_CHARS`), `validate_citations`, `confidence_score` (tanh of top score + gap, penalize single-source).
- `aegisai/backend/app/rag/hybrid_service.py` — `HybridRagPipeline` orchestrator. Enforces filter-before-retrieval, preserves deterministic structured SIH paths (listing / exact identifier / title reverse lookup) by returning `None` to defer to legacy, and degrades gracefully (hybrid failure → caller falls back to legacy).

## 3) Files modified

- `aegisai/backend/app/core/config.py` — Added `HYBRID_RAG_ENABLED`, `BM25_K1`, `BM25_B`, `VECTOR_TOP_K`, `BM25_TOP_K`, `FUSION_TOP_K`, `RRF_K`, `RERANK_TOP_K`, `RERANK_MODEL`, `MAX_CONTEXT_CHARS`, `CITATION_VALIDATION_ENABLED` (all backward compatible, defaults keep legacy behavior).
- `aegisai/backend/app/rag/types.py` — `RAGQuery.memory_summary`, `RAGQuery.enable_hybrid`/`hybrid_top_k`; `RAGResult.confidence`, `RAGResult.pipeline`, `RAGResult.retrieval_debug`.
- `aegisai/backend/app/schemas/chat.py` — `ChatRequest.enable_hybrid`; `ChatResponse.confidence`/`pipeline`.
- `aegisai/backend/app/rag/service.py` — `RAGService.query` now tries `HybridRagPipeline` first when opt-in is true; hybrid returning `None` falls through to the untouched legacy pipeline. Failures in hybrid delegation log a warning and fall through.
- `aegisai/backend/app/api/v1/chat.py` — Both `POST /chat` and `POST /chat/stream` now pass `enable_hybrid` into `RAGQuery`; responses include `confidence`/`pipeline`; stream final `data: {done:true, ...}` event includes `confidence`/`pipeline`; `memory_summary` is threaded through (fixes prior streaming omission).
- `aegisai/frontend/src/types/index.ts` — `ChatRequest.enable_hybrid`; `ChatResponse.confidence`/`pipeline`.
- `aegisai/frontend/src/pages/ChatPage.tsx` — Assistant message actions show `conf XX%` and `hybrid` badge when present.
- `aegisai/backend/app/services/api.ts` — No hard-coded change required; `sendMessageStream` generically forwards `enable_hybrid` if caller sets it (via `ChatRequest`).
- `aegisai/docker-compose.yml` — Backend now forwards `HYBRID_RAG_ENABLED`, `BM25_K1/B`, `VECTOR/BM25/FUSION/RERANK` tunables, `RERANK_MODEL`, `MAX_CONTEXT_CHARS`, `CITATION_VALIDATION_ENABLED`, `LOG_SENSITIVE_DATA`.
- `.env.production.example` — Documents all hybrid env vars.

## 4) Configuration

| Var | Default | Purpose |
|-----|---------|---------|
| `HYBRID_RAG_ENABLED` | `false` | Global kill-switch for hybrid pipeline |
| `BM25_K1` | `1.5` | BM25 term-frequency saturation |
| `BM25_B` | `0.75` | BM25 length normalization |
| `VECTOR_TOP_K` | `20` | Vector candidates before fusion |
| `BM25_TOP_K` | `20` | BM25 candidates before fusion |
| `FUSION_TOP_K` | `12` | Fused pool size (post-RRF, pre-rerank) |
| `RRF_K` | `60` | RRF rank discount constant |
| `RERANK_TOP_K` | `5` | Final sources after cross-encoder |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder id |
| `MAX_CONTEXT_CHARS` | `12000` | Hard cap on LLM context string |
| `CITATION_VALIDATION_ENABLED` | `true` | Validate non-empty chunk_text/document_id |

Per-request override: `POST /api/v1/chat` and `/chat/stream` accept `"enable_hybrid": true|false`. `null`/absent → global default.

## 5) Enabling hybrid

- **Globally (Docker):** Set `HYBRID_RAG_ENABLED=true` in env / `.env` and redeploy. Tune `VECTOR_TOP_K`/`BM25_TOP_K`/`FUSION_TOP_K`/`MAX_CONTEXT_CHARS` as needed.
- **Per-request:** Frontend or API caller sets `enable_hybrid: true` on `ChatRequest`. Useful for A/B or canary without flipping global flag.
- **Model:** `RERANK_MODEL` requires the cross-encoder weights to be available in the backend image/environment. If unavailable, reranking gracefully falls back to fused order (no external call, no failure).

## 6) Verification

- **Existing suite:** `141 passed` (`backend/tests -q`) with default `HYBRID_RAG_ENABLED=false`. Hybrid does not alter GENERAL/MEMORY/DOCUMENT_AWARE paths or structured SIH exact-match paths.
- **Smoke checks performed:**
  - `bm25` / `fusion` / `hybrid_service` / `service` import successfully.
  - Frontend `npm run build` succeeds (≈458 kB / 143 kB gzip).
  - Legacy query with default flag returns `pipeline=legacy`; hybrid-enabled generic company query returns `pipeline=hybrid` with `confidence`, filter creation verified (`create_permission_filter` called), and no leakage of unauthorized chunks.
- **Docker:** Backend `environment` in `docker-compose.yml` updated; no new external service added. Rebuild: `docker compose build backend`.

## 7) Observability

- Hybrid logs `RAG_EVENT` with `event_type=hybrid_retrieval` and `hybrid_query_completed`, including `vector_hits`/`bm25_hits`/`fused`/`reranked`/`included` and `confidence` — sanitized to fingerprint when `LOG_SENSITIVE_DATA=false`.
- Degradation (vector/BM25/RRF/rerank failures) emits `warning` with `error_type` and falls through or falls back in-pool; never leaks exceptions to caller.

## 8) Limitations & next steps (not in this increment)

- No automated Recall@K/Precision@K/MRR evaluation harness or security/prompt-injection regression pack added in this increment (would be a separate test suite).
- Chunking `section_title` extraction is not yet implemented (citations carry the field but ingestion does not populate it per-section).
- Streaming path still does two logical retrievals internally (one inside `RAGService.query` and context rebuild for `_call_llm_stream`); a follow-up can thread the exact hybrid `context` into the stream to avoid recomputation.
- Cross-encoder model weights should be baked into the Docker image or a local volume for fully offline reranking in air-gapped deployments.

## 9) Rollback

Set `HYBRID_RAG_ENABLED=false` (and omit `enable_hybrid` per-request) to restore pure legacy behavior instantly with no code change.
