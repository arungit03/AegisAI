"""Validate SIH extraction, authorized Qdrant records, and exact retrieval.

Run inside the backend container, for example:
    python -m app.rag.verify_sih --document-id <uuid> --file /app/uploads/<file>.pdf --check-retrieval
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from typing import Any, Dict, List, Optional
from uuid import UUID

from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.document import Document
from app.rag.ingestion import extract_sih_records_pdf
from app.rag.qdrant import QdrantManager
from app.rag.service import RAGService
from app.rag.types import RAGQuery
from app.rag.routing import QueryIntent


async def _load_document(document_id: UUID) -> Optional[Dict[str, Any]]:
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with async_session() as session:
            result = await session.execute(select(Document).where(Document.id == document_id))
            document = result.scalar_one_or_none()
            if not document:
                return None
            return {
                "id": str(document.id),
                "original_filename": document.original_filename,
                "status": getattr(document.status, "value", str(document.status)),
                "page_count": document.page_count,
                "chunk_count": document.chunk_count,
                "uploaded_by_id": str(document.uploaded_by_id),
            }
    finally:
        await engine.dispose()


def _scroll_document_points(client: QdrantClient, document_id: str) -> List[Any]:
    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=True,
    )
    return [p for p in points if (p.payload or {}).get("document_id") == document_id]


def validate(document_id: UUID, file_path: str, check_retrieval: bool) -> Dict[str, Any]:
    parsed = extract_sih_records_pdf(file_path)
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
    points = _scroll_document_points(client, str(document_id))
    structured = [p for p in points if (p.payload or {}).get("record_type") == "sih_record"]

    parsed_by_id = {r["problem_statement_id"]: r for r in parsed}
    point_by_id = {
        (p.payload or {}).get("problem_statement_id"): p
        for p in structured
    }
    parsed_ids = set(parsed_by_id)
    point_ids = set(point_by_id)
    duplicate_ids = [key for key, count in Counter(
        (p.payload or {}).get("problem_statement_id") for p in structured
    ).items() if count > 1]
    missing_ids = sorted(parsed_ids - point_ids)
    unexpected_ids = sorted(point_ids - parsed_ids)
    mismatches = []
    rows = []
    for record_id in sorted(parsed_ids):
        expected = parsed_by_id[record_id]
        point = point_by_id.get(record_id)
        payload = point.payload if point else {}
        mismatch_fields = {
            field: {"extracted": expected.get(field), "qdrant": payload.get(field)}
            for field in ("title", "organization", "category", "theme", "deadline", "page_number")
            if expected.get(field) != payload.get(field)
        }
        if mismatch_fields:
            mismatches.append({"problem_statement_id": record_id, "fields": mismatch_fields})
        rows.append({
            "problem_statement_id": record_id,
            "title": expected.get("title"),
            "page": expected.get("page_number"),
            "qdrant_point": str(point.id) if point else None,
            "status": "ok" if point and not mismatch_fields else "mismatch",
        })

    bad_dimensions = [
        {"point_id": str(point.id), "dimension": len(point.vector)}
        for point in points
        if isinstance(point.vector, list) and len(point.vector) != settings.EMBEDDING_DIMENSION
    ]
    db_document = asyncio.run(_load_document(document_id))

    retrieval_failures = []
    if check_retrieval and not missing_ids and not mismatches:
        rag = RAGService(
            qdrant_manager=QdrantManager(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vector_size=settings.EMBEDDING_DIMENSION,
            ),
            embedding_model_name=settings.OLLAMA_EMBEDDING_MODEL,
            llm_model_name=settings.OLLAMA_MODEL,
            top_k=5,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        for record in parsed:
            result = rag.query(RAGQuery(
                question=f"What is the problem statement title for {record['problem_statement_id']}?",
                user_role="admin",
                user_id=UUID(db_document["uploaded_by_id"]) if db_document else None,
                intent=QueryIntent.COMPANY.value,
            ))
            source_id = result.sources[0].payload.get("problem_statement_id") if result.sources else None
            if result.answer != record["title"] or source_id != record["problem_statement_id"]:
                retrieval_failures.append({
                    "problem_statement_id": record["problem_statement_id"],
                    "expected_title": record["title"],
                    "answer": result.answer,
                    "source_id": source_id,
                })

    report = {
        "database_document": db_document,
        "extracted_record_count": len(parsed),
        "qdrant_point_count": len(points),
        "qdrant_structured_record_count": len(structured),
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "unexpected_ids": unexpected_ids,
        "mismatches": mismatches,
        "bad_vector_dimensions": bad_dimensions,
        "retrieval_checked": check_retrieval,
        "retrieval_failures": retrieval_failures,
        "records": rows,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", required=True, type=UUID)
    parser.add_argument("--file", required=True, dest="file_path")
    parser.add_argument("--check-retrieval", action="store_true")
    args = parser.parse_args()
    report = validate(args.document_id, args.file_path, args.check_retrieval)
    print(json.dumps(report, indent=2))
    failures = (
        report["duplicate_ids"]
        or report["missing_ids"]
        or report["unexpected_ids"]
        or report["mismatches"]
        or report["bad_vector_dimensions"]
        or report["retrieval_failures"]
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
