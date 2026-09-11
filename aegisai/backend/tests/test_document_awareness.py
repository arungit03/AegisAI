"""Tests for database-backed document existence and listing answers."""

from types import SimpleNamespace
from uuid import uuid4

from app.models.document import DocumentStatus
from app.rag.routing import QueryIntent, classify_query
from app.services.document_awareness import (
    answer_document_awareness,
    build_authorized_document_query,
)


def make_document(filename, owner_id, status=DocumentStatus.PROCESSED, title=None):
    return SimpleNamespace(
        original_filename=filename,
        filename=filename,
        title=title,
        description=None,
        tags=None,
        status=status,
        uploaded_by_id=owner_id,
    )


def test_user_has_sih_document_returns_yes():
    user_id = uuid4()
    document = make_document("PS SIH 26 -1.pdf", user_id)

    assert answer_document_awareness(
        "Can you see the SIH documents I added? Say only yes or no",
        [document],
    ) == "YES"


def test_user_with_no_documents_returns_no():
    assert answer_document_awareness(
        "Do I have any documents? Say only yes or no",
        [],
    ) == "NO"


def test_another_users_sih_document_is_not_in_the_authorized_input():
    user_id = uuid4()
    other_document = make_document("SIH26117.pdf", uuid4())
    query = build_authorized_document_query(
        SimpleNamespace(id=user_id, role_id=uuid4()), owner_only=True
    )

    assert query.compile().params["uploaded_by_id_1"] == user_id
    assert answer_document_awareness(
        "Can you see the SIH documents I added? Say only yes or no",
        [],
    ) == "NO"
    assert other_document.uploaded_by_id != user_id


def test_user_documents_without_sih_match_returns_no():
    user_id = uuid4()
    document = make_document("engineering-handbook.pdf", user_id)

    assert answer_document_awareness(
        "Are my SIH documents uploaded? Say only yes or no",
        [document],
    ) == "NO"


def test_list_my_documents_only_formats_the_authorized_records():
    user_id = uuid4()
    answer = answer_document_awareness(
        "List my uploaded documents",
        [make_document("owned-policy.txt", user_id)],
    )

    assert "owned-policy.txt" in answer
    assert "other-user" not in answer


def test_sih_content_question_remains_on_rag_route():
    assert classify_query("What is the problem statement in my SIH document?") == QueryIntent.COMPANY
    assert classify_query("What technology is required?") == QueryIntent.COMPANY


def test_sih_upload_status_question_is_document_awareness():
    assert classify_query("Are my SIH documents uploaded?") == QueryIntent.DOCUMENT_AWARE
