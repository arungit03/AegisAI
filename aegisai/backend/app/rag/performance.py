"""Performance tracking — per-stage latency measurement (Phase 2).

Lightweight wall-clock timing for each pipeline stage. No profiling
overhead, no external calls, just time.perf_counter() deltas.

Spec contracts:
  - StageLatency(stage, latency_ms, enabled)
  - PerformanceReport(total_ms, stages, breakdown, overhead_vs_legacy_ms)
  - PerformanceTracker(start_stage, end_stage, report, reset)
  - get_performance_tracker() -> PerformanceTracker

Backward compatible with the previous ad-hoc interface:
  - PerformanceTracker.start / end / measure / to_dict
  - StageLatency.ms / detail aliases
  - PerformanceReport.to_dict / summary / int-ms coercion
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# ---------------------------------------------------------------------------
# Full pipeline stage inventory (Phase 2). Trackers may only touch a
# subset per request; report() still accounts for all stages via KNOWN_STAGES.
# ---------------------------------------------------------------------------

KNOWN_STAGES: tuple[str, ...] = (
    "query_analysis",
    "query_rewriting",
    "query_expansion",
    "query_decomposition",
    "permission_filter",
    "vector_retrieval",
    "bm25_retrieval",
    "fusion",
    "dedup",
    "mmr",
    "reranking",
    "diversity",
    "doc_ranking",
    "evidence_grouping",
    "context_building",
    "version_handling",
    "conflict_detection",
    "coverage_check",
    "confidence_scoring",
    "citation_validation",
    "claim_validation",
    "llm_generation",
)

# Canonical aliases from legacy / hybrid_service stage names to KNOWN_STAGES
_STAGE_ALIASES: Dict[str, str] = {
    "query_understanding": "query_analysis",
    "query_rewrite": "query_rewriting",
    "query_expand": "query_expansion",
    "expansion": "query_expansion",
    "rewrite": "query_rewriting",
    "decomposition": "query_decomposition",
    "vector_search": "vector_retrieval",
    "bm25_search": "bm25_retrieval",
    "rrf_fusion": "fusion",
    "rrf": "fusion",
    "dedup_hits": "dedup",
    "mmr_diversity": "mmr",
    "mmr_select": "mmr",
    "diversity_mmr": "mmr",
    "rerank": "reranking",
    "cross_encoder_rerank": "reranking",
    "document_diversity": "diversity",
    "doc_diversity": "diversity",
    "document_ranking": "doc_ranking",
    "rank_documents": "doc_ranking",
    "evidence_group": "evidence_grouping",
    "grouped_context": "evidence_grouping",
    "context_optimization": "context_building",
    "context_optimizer": "context_building",
    "build_context": "context_building",
    "context": "context_building",
    "version": "version_handling",
    "conflict": "conflict_detection",
    "coverage": "coverage_check",
    "evidence_coverage": "coverage_check",
    "confidence": "confidence_scoring",
    "citation": "citation_validation",
    "claim": "claim_validation",
    "llm": "llm_generation",
    "llm_call": "llm_generation",
    # multi_query_retrieval is an aggregated stage that spans vector+bm25+fusion
    # internally; keep it as-is but also count it toward fusion for breakdown
    # completeness — do not alias, but allow it to appear in reports.
}

_LEGACY_BASELINE_MS: float = 0.0  # caller may override per tracker


def _canonical_stage(name: str) -> str:
    return _STAGE_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Dataclasses — spec fields are primary; legacy aliases via properties
# ---------------------------------------------------------------------------

@dataclass
class StageLatency:
    """Latency for a single pipeline stage."""

    stage: str
    latency_ms: float = 0.0
    enabled: bool = True

    # Optional detail string kept for backward compatibility / observability
    detail: Optional[str] = None

    # --- legacy aliases ----------------------------------------------------

    @property
    def ms(self) -> int:  # noqa: D102
        return int(round(self.latency_ms))

    @ms.setter
    def ms(self, value: int | float) -> None:  # noqa: D102
        self.latency_ms = float(value)


@dataclass
class PerformanceReport:
    """Aggregated latency report for one request.

    Spec fields:
        total_ms, stages, breakdown, overhead_vs_legacy_ms

    Also exposes legacy to_dict/summary and keeps total_ms as float.
    """

    total_ms: float = 0.0
    stages: List[StageLatency] = field(default_factory=list)
    breakdown: Dict[str, float] = field(default_factory=dict)
    overhead_vs_legacy_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ms": self.total_ms,
            "stages": [
                {
                    "stage": s.stage,
                    "latency_ms": s.latency_ms,
                    "enabled": s.enabled,
                    **({"detail": s.detail} if s.detail else {}),
                    # also include legacy `ms` for callers that read it
                    "ms": s.ms,
                }
                for s in self.stages
            ],
            "breakdown": dict(self.breakdown),
            "overhead_vs_legacy_ms": self.overhead_vs_legacy_ms,
        }

    def summary(self) -> str:
        lines = [f"Total: {self.total_ms:.1f}ms (overhead vs legacy: {self.overhead_vs_legacy_ms:+.1f}ms)"]
        for s in self.stages:
            flag = "" if s.enabled else " [skipped]"
            detail = f" ({s.detail})" if s.detail else ""
            lines.append(f"  {s.stage}: {s.latency_ms:.1f}ms{flag}{detail}")
        if self.breakdown:
            lines.append("  breakdown: " + ", ".join(f"{k}={v:.1f}ms" for k, v in self.breakdown.items()))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """Lightweight per-request latency tracker.

    Thread-safe enough for single-request tracking (no concurrent stage
    overlap within one request). Uses a lock for correctness when helper
    threads are involved.

    Spec methods:
        start_stage(name), end_stage(name), report() -> PerformanceReport, reset()

    Legacy compat:
        start(name), end(name, detail?), measure(name, detail?), to_dict(), stages prop
    """

    def __init__(self, baseline_ms: float = _LEGACY_BASELINE_MS) -> None:
        self._lock = threading.Lock()
        self._starts: Dict[str, float] = {}
        self._latencies: Dict[str, float] = {}
        self._enabled: Dict[str, bool] = {}
        self._details: Dict[str, Optional[str]] = {}
        self._order: List[str] = []  # completion order for stages list
        self._t0: float = time.perf_counter()
        self._baseline_ms: float = float(baseline_ms)

    # -- spec API ----------------------------------------------------------

    def start_stage(self, name: str) -> None:
        cname = _canonical_stage(name)
        with self._lock:
            self._starts[cname] = time.perf_counter()
            # also store under raw name so end_stage(raw) finds it
            if cname != name:
                self._starts[name] = self._starts[cname]

    def end_stage(self, name: str) -> float:
        cname = _canonical_stage(name)
        now = time.perf_counter()
        with self._lock:
            t0 = self._starts.pop(cname, None)
            if t0 is None:
                # try raw alias key
                t0 = self._starts.pop(name, None)
            if t0 is None:
                return 0.0
            # clean up alias entry if present
            if cname != name:
                self._starts.pop(name, None)
            dt_ms = (now - t0) * 1000.0
            # accumulate if stage was entered multiple times
            prev = self._latencies.get(cname, 0.0)
            self._latencies[cname] = prev + dt_ms
            if cname not in self._enabled:
                self._enabled[cname] = True
            if cname not in self._order:
                self._order.append(cname)
            # raw-name alias details also map to canonical
            if name in self._details and cname not in self._details:
                self._details[cname] = self._details.pop(name)
            return dt_ms

    def report(self) -> PerformanceReport:
        with self._lock:
            # per-stage StageLatency in completion order, then any remaining
            # known stages that were never touched are omitted (not penalized)
            stages: List[StageLatency] = []
            breakdown: Dict[str, float] = {}
            for name in self._order:
                ms = float(self._latencies.get(name, 0.0))
                enabled = bool(self._enabled.get(name, True))
                detail = self._details.get(name)
                stages.append(StageLatency(stage=name, latency_ms=ms, enabled=enabled, detail=detail))
                breakdown[name] = ms
            # also include any stages that have latency but somehow not in order
            for name, ms in self._latencies.items():
                if name not in breakdown:
                    stages.append(StageLatency(stage=name, latency_ms=float(ms), enabled=bool(self._enabled.get(name, True)), detail=self._details.get(name)))
                    breakdown[name] = float(ms)

            total_ms = sum(breakdown.values())
            # If nothing was tracked yet, derive total from wall clock so
            # report() is still meaningful mid-request
            if total_ms == 0.0:
                total_ms = (time.perf_counter() - self._t0) * 1000.0

            overhead = total_ms - float(self._baseline_ms)
            return PerformanceReport(
                total_ms=float(total_ms),
                stages=stages,
                breakdown=dict(breakdown),
                overhead_vs_legacy_ms=float(overhead),
            )

    def reset(self) -> None:
        with self._lock:
            self._starts.clear()
            self._latencies.clear()
            self._enabled.clear()
            self._details.clear()
            self._order.clear()
            self._t0 = time.perf_counter()

    # -- legacy / convenience API (kept for hybrid_service + existing callers)

    def start(self, stage: str) -> None:  # noqa: D102
        self.start_stage(stage)

    def end(self, stage: str, detail: str | None = None) -> int:  # noqa: D102
        if detail is not None:
            with self._lock:
                cname = _canonical_stage(stage)
                self._details[cname] = detail
                if cname != stage:
                    self._details[stage] = detail
        ms = self.end_stage(stage)
        return int(round(ms))

    @property
    def stages(self) -> List[StageLatency]:  # noqa: D102
        # Return snapshot in completion order
        with self._lock:
            out: List[StageLatency] = []
            for name in self._order:
                out.append(StageLatency(
                    stage=name,
                    latency_ms=float(self._latencies.get(name, 0.0)),
                    enabled=bool(self._enabled.get(name, True)),
                    detail=self._details.get(name),
                ))
            # include any extra latencies not yet in order
            seen = set(self._order)
            for name, ms in self._latencies.items():
                if name not in seen:
                    out.append(StageLatency(stage=name, latency_ms=float(ms), enabled=bool(self._enabled.get(name, True)), detail=self._details.get(name)))
            return out

    @contextmanager
    def measure(self, stage: str, detail: str | None = None):  # noqa: D102
        self.start(stage)
        try:
            yield
        finally:
            self.end(stage, detail=detail)

    def to_dict(self) -> Dict[str, Any]:  # noqa: D102
        return self.report().to_dict()

    # Optional: mark a stage as disabled / skipped without timing it
    def mark_disabled(self, stage: str) -> None:  # noqa: D102
        cname = _canonical_stage(stage)
        with self._lock:
            if cname not in self._latencies:
                self._latencies[cname] = 0.0
                self._enabled[cname] = False
                if cname not in self._order:
                    self._order.append(cname)


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_tracker_singleton: Optional[PerformanceTracker] = None
_singleton_lock = threading.Lock()


def get_performance_tracker() -> PerformanceTracker:
    """Return a module-level PerformanceTracker singleton.

    Thread-safe lazy initialization. Callers that need per-request
    isolation should construct PerformanceTracker() directly and call
    reset() between requests; this singleton is for convenience in
    single-request or script contexts.
    """
    global _tracker_singleton
    if _tracker_singleton is None:
        with _singleton_lock:
            if _tracker_singleton is None:
                _tracker_singleton = PerformanceTracker()
    return _tracker_singleton
