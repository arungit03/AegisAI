"""Test configuration and fixtures for Phase 22 integration tests.

Provides mock implementations of QdrantManager and RAGService that allow
testing the full RAG pipeline without requiring running infrastructure.
"""
import pytest
import asyncio
import os
import tempfile
from uuid import uuid4, UUID
from typing import List, Optional, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure we can import from app
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")


# ─── Mock Embedding Provider ──────────────────────────────────────────────────

class MockEmbeddingProvider:
    """Deterministic mock embedding provider for testing."""

    def __init__(self, dimension: int = 768):
        self._dimension = dimension
        self.model_name = "mock-embedding"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """Generate a deterministic fake embedding based on text hash.

        Expands the SHA256 hash to fill the required dimension by
        repeating hash blocks. This ensures all vectors have the same
        dimension as configured (default 768).
        """
        import hashlib
        h = hashlib.sha256(text.encode()).hexdigest()
        # Each 2 hex chars = 1 float; we get 32 floats per hash
        hashes_needed = (self._dimension + 31) // 32  # ceil division
        expanded = (h * hashes_needed)[:self._dimension * 2]
        result = [float(int(expanded[i:i+2], 16)) / 255.0 for i in range(0, len(expanded), 2)]
        return result[:self._dimension]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


# ─── Mock Qdrant Manager ──────────────────────────────────────────────────────

class MockQdrantManager:
    """Mock Qdrant manager for testing without a running Qdrant instance.

    Stores vectors in-memory and applies permission filters client-side,
    mimicking real Qdrant behavior for integration testing.
    """

    def __init__(self, collection_name: str = "test_documents", vector_size: int = 768):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._points: Dict[str, Dict[str, Any]] = {}
        self._collection_exists = False
        self.search_calls: List[Dict] = []
        self.add_vectors_calls: List[List] = []

    def ensure_collection(self) -> bool:
        self._collection_exists = True
        return True

    def add_vectors(self, points: List, wait: bool = True) -> bool:
        self.add_vectors_calls.append(points)
        for point in points:
            pid = str(point.id)
            self._points[pid] = {
                "id": pid,
                "vector": point.vector,
                "payload": point.payload,
            }
        return True

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filter_conditions: Optional[Any] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search stored points, applying filter conditions.

        Records the search call for inspection in tests.
        """
        self.search_calls.append({
            "query_vector": query_vector,
            "limit": limit,
            "filter_conditions": filter_conditions,
            "with_payload": with_payload,
            "with_vectors": with_vectors,
        })

        # Get all points and apply filter (simulating Qdrant's server-side filter)
        candidates = list(self._points.values())

        # Apply filter by simulating what Qdrant would do
        if filter_conditions is not None:
            candidates = self._apply_filter(candidates, filter_conditions)

        # Simple similarity by dot product (normalized cosine approximation)
        scored = []
        for point in candidates:
            vec = point["vector"]
            score = sum(a * b for a, b in zip(query_vector, vec)) / (len(vec) + 1e-10)
            scored.append((score, point))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply limit
        results = scored[:limit]
        if score_threshold is not None:
            results = [item for item in results if item[0] >= score_threshold]

        return [
            {
                "id": p["id"],
                "score": score,
                "payload": p["payload"] if with_payload else {},
            }
            for score, p in results
        ]

    def _apply_filter(self, points: List[Dict], filter_conditions) -> List[Dict[str, Any]]:
        """Apply Qdrant Filter conditions client-side for mock testing."""
        if filter_conditions is None:
            return points

        # Handle empty filter (admin - no restrictions)
        must = getattr(filter_conditions, "must", None)
        min_should = getattr(filter_conditions, "min_should", None)
        should = getattr(filter_conditions, "should", None)

        # If must is empty/None and min_should is None, return all (admin bypass)
        if (must is None or must == []) and min_should is None:
            return points

        # Check must conditions (all must match)
        if must:
            filtered = []
            for point in points:
                if self._check_conditions(point, must):
                    filtered.append(point)
            points = filtered

        # Check min_should conditions (at least min_count must match)
        if min_should is not None:
            conditions = getattr(min_should, "conditions", None)
            min_count = getattr(min_should, "min_count", 1)
            if conditions:
                matches_should = []
                for point in points:
                    should_matches = self._check_conditions(point, conditions, any_match=True)
                    if should_matches:
                        matches_should.append(point)
                points = matches_should

        # Check legacy should conditions (for backward compatibility)
        if should:
            matches_should = []
            for point in points:
                should_matches = self._check_conditions(point, should, any_match=True)
                if should_matches:
                    matches_should.append(point)
            points = matches_should

        return points

    def _check_conditions(self, point: Dict, conditions, any_match: bool = False) -> bool:
        """Check if a point matches the given conditions."""
        payload = point.get("payload", {})
        results = []

        for cond in conditions:
            if hasattr(cond, "should") or hasattr(cond, "conditions"):
                # Nested filter (Filter or Conditions object)
                sub_conditions = cond.should if hasattr(cond, "should") else cond.conditions
                if sub_conditions:
                    sub_results = self._check_conditions(point, sub_conditions, any_match=True)
                    results.append(sub_results)
                else:
                    results.append(False)
            else:
                # FieldCondition
                key = getattr(cond, "key", None)
                match = getattr(cond, "match", None)

                if key is None or match is None:
                    results.append(False)
                    continue

                # Extract match value
                if hasattr(match, "value"):
                    match_value = match.value
                elif hasattr(match, "any"):
                    match_any = match.any
                    results.append(payload.get(key) in match_any if payload.get(key) else False)
                    continue
                else:
                    results.append(False)
                    continue

                payload_value = payload.get(key)
                if isinstance(match_value, list):
                    results.append(payload_value in match_value)
                else:
                    results.append(payload_value == match_value)

        if any_match:
            return any(results)
        return all(results)

    def delete_document_vectors(self, document_id: UUID) -> bool:
        doc_id_str = str(document_id)
        to_delete = [pid for pid, pt in self._points.items() if pt.get("payload", {}).get("document_id") == doc_id_str]
        for pid in to_delete:
            del self._points[pid]
        return True

    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        return {
            "vectors_count": len(self._points),
            "config": {"size": self.vector_size, "distance": "COSINE"},
        }

    def create_permission_filter(
        self,
        user_role: str,
        user_id: Optional[UUID] = None,
        department: Optional[str] = None,
        allowed_document_ids: Optional[List[UUID]] = None,
    ) -> Any:
        """Create a permission filter using the real QdrantManager logic."""
        from app.rag.qdrant import QdrantManager
        temp_manager = QdrantManager.__new__(QdrantManager)
        return temp_manager.create_permission_filter(
            user_role=user_role,
            user_id=user_id,
            department=department,
            allowed_document_ids=allowed_document_ids,
        )

    def reset(self):
        """Clear all stored points and call history."""
        self._points.clear()
        self.search_calls.clear()
        self.add_vectors_calls.clear()


# ─── Mock LLM Service ─────────────────────────────────────────────────────────

class MockLLMService:
    """Mock LLM service that simulates Ollama responses.

    Can be configured with canned responses or to echo context verification.
    Useful for data-leakage prevention tests.
    """

    def __init__(self):
        self.call_log: List[Dict[str, Any]] = []
        self.responses: List[Dict[str, Any]] = []
        self.default_response = "I'm sorry, but I couldn't find sufficient information to answer this question accurately."
        self.call_count = 0

    def set_responses(self, responses: List[Dict[str, Any]]):
        """Set a queue of responses to return."""
        self.responses = list(responses)

    def set_default_response(self, response: str):
        """Set the default response when no queued responses remain."""
        self.default_response = response

    def call(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Simulate an LLM call, recording the prompt for inspection."""
        self.call_count += 1
        self.call_log.append({
            "prompt": prompt,
            "prompt_length": len(prompt),
            "call_number": self.call_count,
        })

        if self.responses:
            resp = self.responses.pop(0)
            return resp
        return {"response": self.default_response, "tokens_used": 100}

    def reset(self):
        """Reset call log and responses."""
        self.call_log.clear()
        self.responses.clear()
        self.call_count = 0


# ─── Test Document Fixtures ───────────────────────────────────────────────────

TEST_DOCUMENTS = {
    "public_internal": {
        "text": "AegisAI is a secure AI assistant for organizations. Remote work policy allows up to 3 days per week.",
        "filename": "company_policy.txt",
        "classification": "public_internal",
        "department": "engineering",
    },
    "confidential": {
        "text": "CONFIDENTIAL: Q4 2025 budget allocation: Marketing $2M, R&D $5M. Revenue forecast $50M.",
        "filename": "confidential_budget.docx",
        "classification": "confidential",
        "department": "finance",
    },
    "restricted": {
        "text": "RESTRICTED: Internal restructuring plan. Department heads only. Merge timeline Q2 2026.",
        "filename": "restructuring_plan.pdf",
        "classification": "restricted",
        "department": "hr",
    },
    "highly_restricted": {
        "text": "HIGHLY RESTRICTED: Executive salary data. CEO compensation $5M, CFO $3M. SSN numbers redacted.",
        "filename": "executive_compensation.xlsx",
        "classification": "highly_restricted",
        "department": "executive",
    },
}


# ─── RAG Service Factory ──────────────────────────────────────────────────────

def create_mock_rag_service(
    qdrant_manager: Optional[MockQdrantManager] = None,
    embedding_provider: Optional[MockEmbeddingProvider] = None,
    llm_service: Optional[MockLLMService] = None,
    top_k: int = 5,
) -> Any:
    """Create a RAGService with mock dependencies for integration testing.

    Patches the real QdrantManager, embedding provider, and LLM calls
    so tests can run without Docker or external services.
    """
    from app.rag.service import RAGService

    if qdrant_manager is None:
        qdrant_manager = MockQdrantManager()
    if embedding_provider is None:
        embedding_provider = MockEmbeddingProvider()
    if llm_service is None:
        llm_service = MockLLMService()

    qdrant_manager.ensure_collection()

    service = RAGService(
        qdrant_manager=qdrant_manager,
        embedding_model_name="mock-embedding",
        llm_model_name="qwen2.5:7b-instruct",
        top_k=top_k,
        chunk_size=512,
        chunk_overlap=50,
    )

    # Replace the embedding provider with our mock
    service.embedding_provider = embedding_provider

    # Patch the LLM call to use our mock
    original_llm = service._call_llm

    def mocked_call_llm(prompt: str) -> Dict[str, Any]:
        return llm_service.call(prompt)

    service._call_llm = mocked_call_llm
    service._original_llm = original_llm

    return service, qdrant_manager, embedding_provider, llm_service


# ─── Temp File Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def temp_txt_file():
    """Create a temporary TXT file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is test content for document ingestion. It needs to be long enough for chunking to work properly.")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing."""
    # Create a minimal valid PDF
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 24 Tf 100 700 Td (Test PDF) Tj ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000190 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
283
%%EOF"""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
        f.write(pdf_content)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def temp_docx_file():
    """Create a temporary DOCX file for testing."""
    import docx
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        path = f.name
    doc = docx.Document()
    doc.add_paragraph("This is test content for DOCX document.")
    doc.save(path)
    yield path
    os.unlink(path)


# ─── Patch Embed Text ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_embed_text():
    """Patch embed_text to use our mock provider."""
    from app.rag.embeddings import set_embedding_provider
    provider = MockEmbeddingProvider()
    set_embedding_provider(provider)
    yield provider
    # Reset global provider after test
    set_embedding_provider(MagicMock())
