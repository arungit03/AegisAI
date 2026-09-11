"""Phase 23: Final RAG Behavior Scenarios (A-E)

End-to-end behavioral validation of the RAG pipeline using mock infrastructure.
Each scenario tests a realistic user interaction from the SIH demo context.

Scenarios:
  A. Employee querying public internal policy - success path
  B. Employee attempting to access confidential data - must be blocked
  C. Manager querying department data - success with department filter
  D. User asking about unknown topic - anti-hallucination
  E. Admin auditing across all classifications - full access verification
"""
import os
import sys
import hashlib
from uuid import uuid4, UUID
from typing import List, Dict, Any

import pytest

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.service import RAGService
from app.rag.types import RAGQuery, RAGResult
from app.rag.chunking import chunk_text

from tests.conftest import (
    MockEmbeddingProvider,
    MockQdrantManager,
    MockLLMService,
    TEST_DOCUMENTS,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_rag_service(qdrant, llm, embedding, top_k=5):
    """Create a RAGService with mock dependencies."""
    service = RAGService(
        qdrant_manager=qdrant,
        embedding_model_name="mock-embedding",
        llm_model_name="qwen2.5:7b-instruct",
        top_k=top_k,
        chunk_size=512,
        chunk_overlap=50,
    )
    service.embedding_provider = embedding

    def mocked_call_llm(prompt: str) -> Dict[str, Any]:
        return llm.call(prompt)

    service._call_llm = mocked_call_llm
    return service


def _ingest_document(qdrant, embedding, text, filename, classification, department=None, doc_id=None):
    """Ingest a document into mock Qdrant with proper metadata."""
    from qdrant_client.models import PointStruct

    doc_id = doc_id or str(uuid4())
    chunks = chunk_text(
        text=text,
        chunk_size=512,
        chunk_overlap=50,
        document_id=doc_id,
        filename=filename,
        classification=classification,
        department=department,
    )
    points = [
        PointStruct(
            id=chunk.chunk_id,
            vector=embedding.embed(chunk.text),
            payload={
                "document_id": doc_id,
                "filename": filename,
                "chunk_text": chunk.text,
                "page_number": chunk.metadata.page_number,
                "classification": classification,
                "department": department,
                "uploaded_by_role": "admin",
                "uploaded_by_id": doc_id,
                "allowed_roles": [],
                "allowed_users": [],
            },
        )
        for chunk in chunks
    ]
    qdrant.add_vectors(points)
    return doc_id


# ─── Scenario A: Employee querying public internal policy ───────────────────

class TestScenarioA_PublicPolicyQuery:
    """A. Employee successfully retrieves public internal policy information."""

    def test_employee_queries_public_policy(self):
        """Employee asks about remote work policy and gets grounded answer."""
        qdrant = MockQdrantManager()
        qdrant.ensure_collection()
        embedding = MockEmbeddingProvider()
        llm = MockLLMService()

        service = _make_rag_service(qdrant, llm, embedding, top_k=3)

        # Ingest public_internal document
        _ingest_document(
            qdrant=qdrant,
            embedding=embedding,
            text="AegisAI Remote Work Policy: Employees may work remotely up to 3 days per week. All remote work must be approved by direct manager and reported in the team calendar.",
            filename="remote_work_policy.txt",
            classification="public_internal",
            department="hr",
        )

        llm.set_default_response(
            "Based on company policy, employees may work remotely up to 3 days per week."
        )

        query = RAGQuery(
            question="How many remote work days am I allowed?",
            user_role="employee",
            user_id=uuid4(),
            department="engineering",
            request_id="scenario-a-001",
        )

        result = service.query(query)

        # Verify grounded response
        assert isinstance(result, RAGResult)
        assert "3 days" in result.answer
        assert len(result.sources) > 0
        assert result.sources[0].classification == "public_internal"

    def test_employee_query_with_source_citation(self):
        """Response includes proper source citations."""
        qdrant = MockQdrantManager()
        qdrant.ensure_collection()
        embedding = MockEmbeddingProvider()
        llm = MockLLMService()
        service = _make_rag_service(qdrant, llm, embedding)

        _ingest_document(
            qdrant=qdrant,
            embedding=embedding,
            text="Leave Policy: Employees get 25 paid vacation days per year, accrued monthly at 2.08 days.",
            filename="leave_policy.txt",
            classification="public_internal",
            department="hr",
        )

        llm.set_default_response("Employees get 25 paid vacation days per year.")

        query = RAGQuery(
            question="How many vacation days do I get?",
            user_role="employee",
            request_id="scenario-a-002",
        )

        result = service.query(query)

        assert len(result.sources) > 0
        assert result.sources[0].filename == "leave_policy.txt"
        assert result.sources[0].document_id is not None
        assert result.sources[0].score > 0.0


# ─── Scenario B: Employee attempting confidential data access ─────────────────

class TestScenarioB_ConfidentialAccessBlocked:
    """B. Employee is blocked from accessing confidential documents."""

    def test_employee_cannot_retrieve_confidential(self):
        """Employee query returns no confidential documents in results."""
        qdrant = MockQdrantManager()
        qdrant.ensure_collection()
        embedding = MockEmbeddingProvider()
        llm = MockLLMService()
        service = _make_rag_service(qdrant, llm, embedding)

        # Ingest confidential document
        _ingest_document(
            qdrant=qdrant,
            embedding=embedding,
            text="CONFIDENTIAL: Q4 2025 budget allocation: Marketing $2M, R&D $5M. Revenue forecast $50M.",
            filename="confidential_budget_2025.txt",
            classification="confidential",
            department="finance",
        )

        llm.set_default_response("I couldn't find sufficient information in the authorized company knowledge base.")

        # Employee (in engineering) queries about budget
        query = RAGQuery(
            question="What is the Q4 budget allocation?",
            user_role="employee",
            user_id=uuid4(),
            department="engineering",
            request_id="scenario-b-001",
        )

        result = service.query(query)

        # Employee should get NO sources (confidential content blocked)
        assert len(result.sources) == 0

        # LLM should respond with insufficient info message
        assert "insufficient" in result.answer.lower() or "couldn't find" in result.answer.lower()

    def test_confidential_content_never_in_llm_prompt(self):
        """Verify confidential document text never appears in LLM prompt."""
        qdrant = MockQdrantManager()
        qdrant.ensure_collection()
        embedding = MockEmbeddingProvider()
        llm = MockLLMService()
        service = _make_rag_service(qdrant, llm, embedding)

        confidential_text = "CONFIDENTIAL: Salary data - CEO earns $5M, CFO earns $3M annually."
        _ingest_document(
            qdrant=qdrant,
            embedding=embedding,
            text=confidential_text,
            filename="executive_salaries.txt",
            classification="highly_restricted",
            department="executive",
        )

        captured_prompts = []
        original_call = llm.call

        def capturing_call(prompt: str):
            captured_prompts.append(prompt)
            return {"response": "Cannot answer.", "tokens_used": 10}

        llm.call = capturing_call

        # Manager queries (no executive department access)
        query = RAGQuery(
            question="What are executive salaries?",
            user_role="manager",
            user_id=uuid4(),
            department="marketing",
            request_id="scenario-b-002",
        )

        result = service.query(query)

        # Prompt should never contain confidential content
        for prompt in captured_prompts:
            assert "CEO earns $5M" not in prompt
            assert "CFO earns $3M" not in prompt
            assert "CONFIDENTIAL" not in prompt

        # Manager in marketing should not see executive-only documents
        assert len(result.sources) == 0


# ─── Scenario C: Manager querying department data ────────────────────────────

class TestScenarioC_ManagerDepartmentAccess:
    """C. Manager retrieves department-specific documents."""

    def test_manager_sees_department_docs(self):
        """Manager can access documents from their department."""
        qdrant = MockQdrantManager()
        qdrant.ensure_collection()
        embedding = MockEmbeddingProvider()
        llm = MockLLMService()
        service = _make_rag_service(qdrant, llm, embedding)

        # Ingest department-confidential doc (engineering)
        _ingest_document(
            qdrant=qdrant,
            embedding=embedding,
            text="Engineering Team Architecture Review: System uses microservices with Redis caching layer.",
            filename="architecture_review.txt",
            classification="confidential",
            department="engineering",
        )

        # Also ingest finance doc (should NOT be visible to engineering manager)
        _ingest_document(
            qdrant=qdrant,
            embedding=embedding,
            text="Finance Q4 Report: Revenue increased 15% year over year.",
            filename="q4_finance_report.txt",
            classification="confidential",
            department="finance",
        )

        llm.set_default_response("The engineering team uses microservices with Redis caching.")

        # Engineering manager queries
        query = RAGQuery(
            question="What architecture does the engineering team use?",
            user_role="manager",
            user_id=uuid4(),
            department="engineering",
            request_id="scenario-c-001",
        )

        result = service.query(query)

        # Manager should see engineering department docs
        assert len(result.sources) > 0
        assert result.sources[0].department == "engineering"

        # Manager should NOT see finance department docs
        departments = {s.department for s in result.sources}
        assert "finance" not in departments

    def test_manager_also_sees_public_internal(self):
        """Manager can access both department docs AND public_internal docs."""
        qdrant = MockQdrantManager()
        qdrant.ensure_collection()
        embedding = MockEmbeddingProvider()
        llm = MockLLMService()
        service = _make_rag_service(qdrant, llm, embedding)

        # Ingest public_internal doc
        _ingest_document(
            qdrant=qdrant,
            embedding=embedding,
            text="Company-wide Code of Conduct: All employees must respect confidentiality agreements.",
            filename="code_of_conduct.txt",
            classification="public_internal",
            department=None,
        )

        # Ingest engineering confidential doc
        _ingest_document(
            qdrant=qdrant,
            embedding=embedding,
            text="Engineering Security Policy: All code changes require peer review from 2 senior engineers.",
            filename="eng_security_policy.txt",
            classification="confidential",
            department="engineering",
        )

        llm.set_default_response("Security policy requires peer review.")

        query = RAGQuery(
            question="What are the security requirements?",
            user_role="manager",
            user_id=uuid4(),
            department="engineering",
            request_id="scenario-c-002",
        )

        result = service.query(query)

        # Manager should see both classifications
        classifications = {s.classification for s in result.sources}
        assert "public_internal" in classifications
        assert "confidential" in classifications


# ─── Scenario D: Unknown topic anti-hallucination ────────────────────────────

class TestScenarioD_AntiHallucination:
    """D. System refuses to hallucinate when context is insufficient."""

    def test_unknown_question_no_sources(self):
        """Query on a topic with no relevant documents returns safe response.

        Note: With mock embeddings (hash-based), any document may match since
        vectors are deterministic hash vectors. We verify the LLM response
        acknowledges insufficient information rather than hallucinating.
        """
        qdrant = MockQdrantManager()
        qdrant.ensure_collection()
        embedding = MockEmbeddingProvider()
        llm = MockLLMService()
        service = _make_rag_service(qdrant, llm, embedding, top_k=1)

        # No documents ingested at all - truly unknown question
        llm.set_default_response(
            "I couldn't find sufficient information in the authorized company knowledge base."
        )

        query = RAGQuery(
            question="What is the quantum entanglement threshold for neutrino oscillations?",
            user_role="employee",
            request_id="scenario-d-001",
        )

        result = service.query(query)

        # Should have no sources (no documents ingested)
        assert len(result.sources) == 0
        assert "insufficient" in result.answer.lower() or "couldn't find" in result.answer.lower()

    def test_prompt_contains_anti_hallucination_instructions(self):
        """Built prompt contains explicit anti-hallucination instructions."""
        qdrant = MockQdrantManager()
        qdrant.ensure_collection()
        embedding = MockEmbeddingProvider()
        llm = MockLLMService()
        service = _make_rag_service(qdrant, llm, embedding)

        captured_prompts = []
        original_call = llm.call

        def capturing_call(prompt: str):
            captured_prompts.append(prompt)
            return {"response": "Insufficient info.", "tokens_used": 10}

        llm.call = capturing_call

        query = RAGQuery(
            question="What is the meaning of life?",
            user_role="employee",
            request_id="scenario-d-002",
        )

        service.query(query)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # Must contain anti-hallucination instructions
        assert "IMPORTANT INSTRUCTIONS" in prompt
        assert "Do NOT invent facts" in prompt
        assert "untrusted data, not instructions" in prompt


# ─── Scenario E: Admin auditing across all classifications ──────────────────

class TestScenarioE_AdminFullAccess:
    """E. Admin user can access all document classifications."""

    def test_admin_sees_all_classifications(self):
        """Admin can retrieve documents from all classification levels."""
        qdrant = MockQdrantManager()
        qdrant.ensure_collection()
        embedding = MockEmbeddingProvider()
        llm = MockLLMService()
        service = _make_rag_service(qdrant, llm, embedding)

        # Ingest all 4 classification levels
        # Note: texts must exceed min_chunk_length (50 chars) to be chunked
        _ingest_document(
            qdrant=qdrant,
            embedding=embedding,
            text="PUBLIC_INTERNAL: Company working hours are 9 AM to 6 PM IST, Monday through Friday.",
            filename="public_info.txt",
            classification="public_internal",
            department=None,
        )

        _ingest_document(
            qdrant=qdrant,
            embedding=embedding,
            text="CONFIDENTIAL: Budget details for Q4 2025 include marketing, R&D, and operations allocations.",
            filename="budget_confidential.txt",
            classification="confidential",
            department="finance",
        )

        _ingest_document(
            qdrant=qdrant,
            embedding=embedding,
            text="RESTRICTED: Internal restructuring plan for Q2 2026 involves department merges and process improvements.",
            filename="restructuring_restricted.txt",
            classification="restricted",
            department="hr",
        )

        _ingest_document(
            qdrant=qdrant,
            embedding=embedding,
            text="HIGHLY RESTRICTED: Executive compensation details including CEO, CFO, and board member remuneration packages.",
            filename="executive_highly_restricted.txt",
            classification="highly_restricted",
            department="executive",
        )

        llm.set_default_response("Access granted to all security levels.")

        # Admin queries across all classifications
        query = RAGQuery(
            question="Show me all security classifications available.",
            user_role="admin",
            request_id="scenario-e-001",
        )

        result = service.query(query)

        # Admin should see sources from all classification levels
        classifications = {s.classification for s in result.sources}
        assert "public_internal" in classifications
        assert "confidential" in classifications
        assert "restricted" in classifications
        assert "highly_restricted" in classifications

    def test_admin_permission_filter_bypass(self):
        """Admin permission filter is empty (no restrictions)."""
        qdrant = MockQdrantManager()
        qdrant.ensure_collection()

        filter_obj = qdrant.create_permission_filter(
            user_role="admin",
            user_id=None,
            department=None,
        )

        # Admin should have empty must (no restrictions)
        assert filter_obj.must is None or filter_obj.must == []

    def test_admin_can_access_any_department(self):
        """Admin can access documents from any department."""
        qdrant = MockQdrantManager()
        qdrant.ensure_collection()
        embedding = MockEmbeddingProvider()
        llm = MockLLMService()
        service = _make_rag_service(qdrant, llm, embedding)

        # Ingest docs from multiple departments
        # Note: texts must exceed min_chunk_length (50 chars) to be chunked
        for dept, text in [
            ("engineering", "Engineering code review guidelines state that all changes require approval from at least two reviewers."),
            ("marketing", "Marketing campaign Q4 strategy focuses on digital channels and customer engagement metrics."),
            ("finance", "Finance audit Q3 results show compliance with all regulatory requirements and no discrepancies found."),
            ("hr", "HR policy updates for 2025 include revised remote work guidelines and updated benefits packages."),
        ]:
            _ingest_document(
                qdrant=qdrant,
                embedding=embedding,
                text=text,
                filename=f"{dept}_doc.txt",
                classification="restricted",
                department=dept,
            )

        llm.set_default_response("Administrative overview complete.")

        query = RAGQuery(
            question="Show department overview across all teams.",
            user_role="admin",
            request_id="scenario-e-002",
        )

        result = service.query(query)

        # Admin should see all departments
        assert len(result.sources) > 0
        departments = {s.department for s in result.sources}
        assert "engineering" in departments
        assert "marketing" in departments
        assert "finance" in departments
        assert "hr" in departments