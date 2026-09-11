"""Tests for conversational routing and secure chat context handling."""
from uuid import UUID, uuid4
from unittest.mock import patch
import pytest

from app.rag.ingestion import IngestionResult

from app.rag.routing import QueryIntent, classify_query
from app.rag.service import (
    COMPANY_RELEVANCE_THRESHOLD,
    INSUFFICIENT_COMPANY_KNOWLEDGE,
    RAGService,
)
from app.rag.types import RAGQuery
from app.rag.chunking import chunk_text

from tests.conftest import MockEmbeddingProvider, MockQdrantManager


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    def call(self, prompt):
        self.prompts.append(prompt)
        if "What is my name?" in prompt and "My name is Arun." in prompt:
            answer = "Your name is Arun."
        elif "Explain Python." in prompt:
            answer = "Python is a general-purpose programming language."
        else:
            answer = "Thanks for sharing."
        return {"response": answer, "tokens_used": 10}


class StubQdrant(MockQdrantManager):
    def __init__(self, results=None):
        super().__init__()
        self.results = results or []

    def search(self, *args, **kwargs):
        self.search_calls.append(kwargs)
        return self.results


class ExactMatchQdrant(StubQdrant):
    def __init__(self, results=None, exact_results=None):
        super().__init__(results=results)
        self.exact_results = exact_results or []
        self.keyword_calls = []

    def keyword_search(self, terms, **kwargs):
        self.keyword_calls.append((terms, kwargs))
        return self.exact_results


class StructuredLookupQdrant(StubQdrant):
    def __init__(self, records=None):
        super().__init__()
        self.records = records or []
        self.structured_calls = []
        self.list_calls = []

    def search_sih_records(self, **kwargs):
        self.structured_calls.append(kwargs)
        identifier = kwargs.get("identifier")
        if identifier:
            return [record for record in self.records
                    if record.get("payload", {}).get("problem_statement_id", "").lower() == identifier.lower()]
        return self.records

    def list_sih_records(self, **kwargs):
        self.list_calls.append(kwargs)
        return self.records


class FailingQdrant(MockQdrantManager):
    def add_vectors(self, points, wait=True):
        return False


def make_service(qdrant=None, llm=None):
    llm = llm or RecordingLLM()
    service = RAGService(
        qdrant_manager=qdrant,
        embedding_model_name="mock-embedding",
        llm_model_name="qwen2.5:7b-instruct",
    )
    service._call_llm = llm.call
    return service, llm


def test_query_classifier_separates_general_memory_and_company_questions():
    assert classify_query("Explain Python.") == QueryIntent.GENERAL
    assert classify_query("What is my name?") == QueryIntent.MEMORY
    assert classify_query("What is our remote work policy?") == QueryIntent.COMPANY
    assert classify_query("What are the office working hours?") == QueryIntent.COMPANY
    assert classify_query("What is the problem statement in my SIH document?") == QueryIntent.COMPANY
    assert classify_query("What is the PS number for Multiple intermediaries?") == QueryIntent.COMPANY
    assert classify_query("List the SIH problem statements in my uploaded documents?") == QueryIntent.DOCUMENT_AWARE
    assert classify_query("What title belongs to Automated Urban Parcel Mapping?") == QueryIntent.COMPANY


def test_memory_uses_previous_messages_from_the_same_conversation():
    service, llm = make_service()

    first = service.query(RAGQuery(
        question="My name is Arun.",
        intent=QueryIntent.GENERAL.value,
    ))
    second = service.query(RAGQuery(
        question="What is my name?",
        intent=QueryIntent.MEMORY.value,
        conversation_history=[
            {"role": "user", "content": "My name is Arun."},
            {"role": "assistant", "content": first.answer},
        ],
    ))

    assert "Thanks" in first.answer
    assert second.answer == "Your name is Arun."
    assert "User: My name is Arun." in llm.prompts[-1]


def test_general_question_does_not_require_qdrant():
    service, llm = make_service(qdrant=None)

    result = service.query(RAGQuery(
        question="Explain Python.",
        intent=QueryIntent.GENERAL.value,
    ))

    assert result.answer.startswith("Python is")
    assert len(llm.prompts) == 1


def test_company_question_uses_authorized_retrieval_and_citation():
    document_id = uuid4()
    qdrant = StubQdrant(results=[{
        "id": str(uuid4()),
        "score": 0.92,
        "payload": {
            "document_id": str(document_id),
            "filename": "remote_work_policy.txt",
            "page_number": 1,
            "chunk_text": "Remote work is allowed up to three days per week.",
            "classification": "public_internal",
            "department": "engineering",
        },
    }])
    service, llm = make_service(qdrant=qdrant)
    llm.call = lambda prompt: {"response": "Three days per week.", "tokens_used": 10}
    service._call_llm = llm.call

    result = service.query(RAGQuery(
        question="What is our remote work policy?",
        intent=QueryIntent.COMPANY.value,
        user_role="employee",
        user_id=uuid4(),
        department="engineering",
    ))

    assert result.answer == "Three days per week."
    assert result.sources[0].filename == "remote_work_policy.txt"
    assert len(qdrant.search_calls) == 1


def test_company_question_without_authorized_evidence_refuses_safely():
    qdrant = StubQdrant()
    service, llm = make_service(qdrant=qdrant)

    result = service.query(RAGQuery(
        question="What is our remote work policy?",
        intent=QueryIntent.COMPANY.value,
        user_role="employee",
        user_id=uuid4(),
    ))

    assert result.answer == INSUFFICIENT_COMPANY_KNOWLEDGE
    assert result.sources == []
    assert llm.prompts == []


def test_explicit_sih_identifier_uses_authorized_exact_match():
    document_id = uuid4()
    qdrant = ExactMatchQdrant(exact_results=[{
        "id": str(uuid4()),
        "score": 1.0,
        "payload": {
            "document_id": str(document_id),
            "filename": "PS SIH 26 -1.pdf",
            "page_number": 10,
            "chunk_text": "SIH26010: AI-Based early warning and landslide Risk Monitoring System in NER Disaster Management",
        },
    }])
    service, llm = make_service(qdrant=qdrant)
    llm.call = lambda prompt: {"response": "The SIH26010 title is present.", "tokens_used": 10}
    service._call_llm = llm.call

    result = service.query(RAGQuery(
        question="What problem statement title is associated with SIH26010?",
        intent=QueryIntent.COMPANY.value,
        user_role="admin",
        user_id=uuid4(),
    ))

    assert result.answer == "The SIH26010 title is present."
    assert result.sources[0].page_number == 10
    assert qdrant.keyword_calls[0][0] == ["sih26010"]


def test_explicit_missing_sih_identifier_does_not_use_neighboring_vector_hit():
    qdrant = ExactMatchQdrant(results=[{
        "id": str(uuid4()),
        "score": 0.91,
        "payload": {
            "document_id": str(uuid4()),
            "filename": "PS SIH 26 -1.pdf",
            "page_number": 10,
            "chunk_text": "SIH26001 is a different problem statement.",
        },
    }])
    service, llm = make_service(qdrant=qdrant)

    result = service.query(RAGQuery(
        question="What problem statement title is associated with SIH26010?",
        intent=QueryIntent.COMPANY.value,
        user_role="admin",
        user_id=uuid4(),
    ))

    assert result.answer == "I couldn't find SIH26010 in the authorized documents."
    assert result.sources == []
    assert llm.prompts == []


@pytest.mark.parametrize("question", [
    "What is SIH26003?",
    "What is sih26003?",
    "What is SIH-26003?",
    "What is SIH 26003?",
])
def test_sih_identifier_normalization_is_canonical(question):
    assert RAGService._identifier_terms(question) == ["sih26003"]


def test_structured_sih_record_title_is_returned_without_neighbor_fields():
    qdrant = ExactMatchQdrant(exact_results=[{
        "id": str(uuid4()),
        "score": 1.0,
        "payload": {
            "document_id": str(uuid4()),
            "filename": "PS SIH 26 -1.pdf",
            "page_number": 2,
            "record_type": "sih_record",
            "problem_statement_id": "SIH26003",
            "title": "AI-Based Cognitive Gaming and Memory Assistance Platform for Elderly Dementia Patients in North Eastern Region (NER)",
            "chunk_text": "Problem Statement ID: SIH26003\nProblem Statement Title: AI-Based Cognitive Gaming and Memory Assistance Platform for Elderly Dementia Patients in North Eastern Region (NER)",
        },
    }])
    service, llm = make_service(qdrant=qdrant)

    result = service.query(RAGQuery(
        question="What problem statement title is associated with SIH26003?",
        intent=QueryIntent.COMPANY.value,
        user_role="admin",
        user_id=uuid4(),
    ))

    assert result.answer.startswith("AI-Based Cognitive Gaming")
    assert "SIH26002" not in result.answer
    assert "SIH26004" not in result.answer
    assert result.sources[0].page_number == 2
    assert llm.prompts == []


def test_exact_identifier_skips_semantic_search_and_returns_structured_title():
    document_id = uuid4()
    qdrant = StructuredLookupQdrant(records=[{
        "id": str(uuid4()),
        "score": 1.0,
        "payload": {
            "document_id": str(document_id),
            "filename": "PS SIH 26 -1.pdf",
            "page_number": 2,
            "record_type": "sih_record",
            "problem_statement_id": "SIH26012",
            "title": "AI-Based Automated Urban Parcel Mapping",
            "chunk_text": "Problem Statement ID: SIH26012\nProblem Statement Title: AI-Based Automated Urban Parcel Mapping",
        },
    }])
    service, llm = make_service(qdrant=qdrant)

    result = service.query(RAGQuery(
        question="What is the title of SIH26012?",
        intent=QueryIntent.COMPANY.value,
        user_role="admin",
        user_id=uuid4(),
    ))

    assert result.answer == "AI-Based Automated Urban Parcel Mapping"
    assert qdrant.search_calls == []
    assert len(qdrant.structured_calls) == 1
    assert llm.prompts == []


def test_reverse_title_lookup_returns_exact_structured_identifier():
    qdrant = StructuredLookupQdrant(records=[{
        "id": str(uuid4()),
        "score": 1.0,
        "payload": {
            "document_id": str(uuid4()),
            "filename": "PS SIH 26 -1.pdf",
            "page_number": 3,
            "record_type": "sih_record",
            "problem_statement_id": "SIH26033",
            "title": "Multiple intermediaries reduce farmers earnings and increase consumer prices.",
            "chunk_text": "Problem Statement ID: SIH26033",
        },
    }])
    service, llm = make_service(qdrant=qdrant)

    result = service.query(RAGQuery(
        question="What is the problem statement number for 'Multiple intermediaries reduce farmers earnings and increase consumer prices'?",
        intent=QueryIntent.COMPANY.value,
        user_role="admin",
        user_id=uuid4(),
    ))

    assert result.answer == "SIH26033"
    assert result.sources[0].page_number == 3
    assert qdrant.search_calls == []
    assert llm.prompts == []


def test_multiple_explicit_identifiers_are_all_retrieved_before_generation():
    records = []
    for identifier, title, page in (
        ("SIH26012", "Urban Parcel Mapping", 2),
        ("SIH26013", "Community Health Platform", 2),
    ):
        records.append({
            "id": str(uuid4()),
            "score": 1.0,
            "payload": {
                "document_id": str(uuid4()),
                "filename": "PS SIH 26 -1.pdf",
                "page_number": page,
                "record_type": "sih_record",
                "problem_statement_id": identifier,
                "title": title,
                "chunk_text": f"Problem Statement ID: {identifier}\nProblem Statement Title: {title}",
            },
        })
    qdrant = StructuredLookupQdrant(records=records)
    service, llm = make_service(qdrant=qdrant)

    result = service.query(RAGQuery(
        question="Compare SIH26012 and SIH26013.",
        intent=QueryIntent.COMPANY.value,
        user_role="admin",
        user_id=uuid4(),
    ))

    assert result.sources and {
        source.payload["problem_statement_id"] for source in result.sources
    } == {"SIH26012", "SIH26013"}
    assert len(qdrant.structured_calls) == 2
    assert qdrant.search_calls == []
    assert "SIH26012" in llm.prompts[-1] and "SIH26013" in llm.prompts[-1]


def test_structured_sih_listing_does_not_use_semantic_search():
    qdrant = StructuredLookupQdrant(records=[{
        "id": str(uuid4()),
        "score": 1.0,
        "payload": {
            "document_id": str(uuid4()),
            "filename": "PS SIH 26 -1.pdf",
            "page_number": 2,
            "record_type": "sih_record",
            "problem_statement_id": "SIH26012",
            "title": "Urban Parcel Mapping",
            "chunk_text": "Problem Statement ID: SIH26012",
        },
    }])
    service, llm = make_service(qdrant=qdrant)

    result = service.query(RAGQuery(
        question="List SIH problem statements in the document.",
        intent=QueryIntent.COMPANY.value,
        user_role="admin",
        user_id=uuid4(),
    ))

    assert "SIH26012: Urban Parcel Mapping" in result.answer
    assert qdrant.list_calls
    assert qdrant.search_calls == []
    assert llm.prompts == []


def test_explicit_sih_presence_question_returns_boolean_from_structured_record():
    record = {
        "id": str(uuid4()),
        "score": 1.0,
        "payload": {
            "document_id": str(uuid4()),
            "filename": "PS SIH 26 -1.pdf",
            "page_number": 2,
            "record_type": "sih_record",
            "problem_statement_id": "SIH26012",
            "title": "Urban Parcel Mapping",
            "chunk_text": "Problem Statement ID: SIH26012",
        },
    }
    qdrant = StructuredLookupQdrant(records=[record])
    service, llm = make_service(qdrant=qdrant)

    result = service.query(RAGQuery(
        question="Does my uploaded SIH document contain SIH26012?",
        intent=QueryIntent.COMPANY.value,
        user_role="admin",
        user_id=uuid4(),
    ))

    assert result.answer == "YES"
    assert result.sources[0].payload["problem_statement_id"] == "SIH26012"
    assert llm.prompts == []


def test_pdf_ingestion_chunks_preserve_source_page_numbers(tmp_path):
    service, _ = make_service(qdrant=StubQdrant())
    service.embedding_provider = MockEmbeddingProvider()
    document_id = str(uuid4())
    page_one = "Page one contains the first SIH problem statement and enough text to create a valid chunk."
    page_two = "Page two contains the second SIH problem statement and enough text to create a valid chunk."
    ingestion = IngestionResult(
        success=True,
        status="processed",
        text=f"{page_one}\n{page_two}",
        page_count=2,
        page_texts=[page_one, page_two],
    )
    document_path = tmp_path / "sih.pdf"
    document_path.write_bytes(b"placeholder")

    with patch("app.rag.service.ingest_document", return_value=ingestion):
        result = service.process_document_upload(
            file_path=str(document_path),
            filename="sih.pdf",
            user_role="admin",
            document_id=document_id,
        )

    assert result.success is True
    points = service.qdrant.add_vectors_calls[0]
    assert {point.payload["page_number"] for point in points} == {1, 2}


def test_low_confidence_company_hit_is_not_cited_or_sent_to_llm():
    qdrant = StubQdrant(results=[{
        "id": str(uuid4()),
        "score": COMPANY_RELEVANCE_THRESHOLD - 0.01,
        "payload": {
            "document_id": str(uuid4()),
            "filename": "unrelated.txt",
            "chunk_text": "This document does not contain the requested policy.",
        },
    }])
    service, llm = make_service(qdrant=qdrant)

    result = service.query(RAGQuery(
        question="What is the company maternity leave policy?",
        intent=QueryIntent.COMPANY.value,
        user_role="admin",
        user_id=uuid4(),
    ))

    assert result.answer == INSUFFICIENT_COMPANY_KNOWLEDGE
    assert result.sources == []
    assert llm.prompts == []


def test_conversation_history_does_not_leak_between_queries():
    service, llm = make_service()

    service.query(RAGQuery(
        question="What is my name?",
        intent=QueryIntent.MEMORY.value,
        conversation_history=[{"role": "user", "content": "My name is Arun."}],
    ))
    service.query(RAGQuery(
        question="What is my name?",
        intent=QueryIntent.MEMORY.value,
        conversation_history=[{"role": "user", "content": "My name is Priya."}],
    ))

    assert "Arun" in llm.prompts[0]
    assert "Arun" not in llm.prompts[1]
    assert "Priya" in llm.prompts[1]


def test_chunk_ids_are_valid_qdrant_point_ids():
    chunks = chunk_text(
        "AegisAI employees can work remotely every Friday under this policy.",
        document_id=str(uuid4()),
        filename="policy.txt",
    )

    assert chunks
    for chunk in chunks:
        assert UUID(chunk.chunk_id)


def test_failed_vector_insert_does_not_mark_ingestion_processed(tmp_path):
    document_path = tmp_path / "policy.txt"
    document_path.write_text(
        "AegisAI employees can work remotely every Friday under this policy.",
        encoding="utf-8",
    )
    service, _ = make_service(qdrant=FailingQdrant())
    service.embedding_provider = MockEmbeddingProvider()

    result = service.process_document_upload(
        file_path=str(document_path),
        filename=document_path.name,
        user_role="admin",
        document_id=str(uuid4()),
        uploaded_by_id=str(uuid4()),
    )

    assert result.success is False
    assert result.status == "failed"
    assert result.error == "Failed to store vectors in Qdrant"
