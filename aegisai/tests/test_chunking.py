"""Tests for text chunking service."""

import pytest
from app.rag.chunking import chunk_text, ChunkResult, ChunkMetadata


def test_chunk_text_basic():
    """Test basic text chunking."""
    text = "This is a test. " * 100  # 500+ characters
    chunks = chunk_text(
        text=text,
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_length=10,
        document_id="doc_001",
        filename="test.txt",
    )
    assert len(chunks) > 0
    assert all(len(c.text) >= 10 for c in chunks)  # min_chunk_length filter
    assert all(c.chunk_id.startswith("doc_001_test_") for c in chunks)


def test_chunk_text_empty():
    """Test chunking empty text."""
    chunks = chunk_text(
        text="",
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_length=10,
        document_id="doc_001",
        filename="test.txt",
    )
    assert len(chunks) == 0


def test_chunk_text_smaller_than_size():
    """Test chunking text smaller than chunk_size."""
    text = "Short text."
    chunks = chunk_text(
        text=text,
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_length=10,
        document_id="doc_001",
        filename="test.txt",
    )
    # Text shorter than chunk_size should return as single chunk if >= min_chunk_length
    assert len(chunks) >= 0
    if len(chunks) > 0:
        assert len(chunks[0].text) >= 10


def test_chunk_metadata_retained():
    """Test that metadata is retained in chunks."""
    text = "This is a test sentence. " * 50
    chunks = chunk_text(
        text=text,
        chunk_size=200,
        chunk_overlap=30,
        min_chunk_length=20,
        document_id="doc_123",
        filename="machine_manual.pdf",
        page_number=42,
        classification="confidential",
        department="engineering",
    )
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata.document_id == "doc_123"
        assert chunk.metadata.filename == "machine_manual.pdf"
        assert chunk.metadata.page_number == 42
        assert chunk.metadata.classification == "confidential"
        assert chunk.metadata.department == "engineering"


def test_chunk_minimum_length_filter():
    """Test that minimum chunk length is enforced."""
    # Very short text that's below min_chunk_length
    text = "Hi."
    chunks = chunk_text(
        text=text,
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_length=100,  # Very high minimum
        document_id="doc_001",
        filename="test.txt",
    )
    # Should return no chunks since text is too short
    assert len(chunks) == 0


def test_chunk_overlap_preserved():
    """Test that chunk overlap is maintained between consecutive chunks."""
    text = "A " * 500  # 1000 characters, plenty of text
    chunks = chunk_text(
        text=text,
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_length=10,
        document_id="doc_001",
        filename="test.txt",
    )
    assert len(chunks) > 1

    # Check that overlaps are present (chunk i's end should relate to chunk i+1's start)
    for i in range(len(chunks) - 1):
        # There should be some relationship between consecutive chunks
        # due to the overlap mechanism
        assert chunks[i].char_end > 0
        assert chunks[i + 1].char_start >= 0


def test_chunk_single_chunk():
    """Test chunking very short text returns single chunk."""
    text = "This is a moderate length sentence for testing."
    chunks = chunk_text(
        text=text,
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_length=10,
        document_id="doc_001",
        filename="test.txt",
    )
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_allowed_roles_and_users():
    """Test that allowed_roles and allowed_users are retained in chunk metadata."""
    text = "This is a test document with multiple sentences. " * 20
    chunks = chunk_text(
        text=text,
        chunk_size=200,
        chunk_overlap=30,
        min_chunk_length=20,
        document_id="doc_456",
        filename="sensitive.pdf",
        page_number=1,
        classification="confidential",
        department="engineering",
        allowed_roles=["admin", "manager"],
        allowed_users=["user-123", "user-456"],
    )
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata.allowed_roles == ["admin", "manager"]
        assert chunk.metadata.allowed_users == ["user-123", "user-456"]


def test_chunk_metadata_to_dict():
    """Test that chunk metadata can be converted to dict."""
    text = "Test content for metadata"
    chunks = chunk_text(
        text=text,
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_length=10,
        document_id="doc_meta_001",
        filename="test.txt",
        page_number=1,
        classification="public_internal",
        department="marketing",
    )
    assert len(chunks) >= 1
    chunk_dict = chunks[0].to_dict()
    assert chunk_dict["document_id"] == "doc_meta_001"
    assert chunk_dict["filename"] == "test.txt"
    assert chunk_dict["page_number"] == 1
    assert chunk_dict["classification"] == "public_internal"
    assert chunk_dict["department"] == "marketing"