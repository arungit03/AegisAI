"""Tests for document ingestion service."""

import os
import tempfile
import pytest
from pathlib import Path

from app.rag.ingestion import ingest_document, validate_file, IngestionResult


def test_validate_file_valid():
    """Test validation of a valid file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Hello World")
        tmp_path = f.name

    try:
        is_valid, error = validate_file(tmp_path)
        assert is_valid is True
        assert error is None
    finally:
        os.unlink(tmp_path)


def test_validate_file_nonexistent():
    """Test validation of a nonexistent file."""
    is_valid, error = validate_file("/nonexistent/path.txt")
    assert is_valid is False
    assert error is not None


def test_validate_file_empty():
    """Test validation of an empty file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        tmp_path = f.name

    try:
        is_valid, error = validate_file(tmp_path)
        assert is_valid is False
        assert error == "File is empty"
    finally:
        os.unlink(tmp_path)


def test_validate_file_oversized():
    """Test validation of an oversized file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("X" * (51 * 1024 * 1024))  # 51MB, over 50MB limit
        tmp_path = f.name

    try:
        is_valid, error = validate_file(tmp_path, max_size=50 * 1024 * 1024)
        assert is_valid is False
        assert "exceeds limit" in error
    finally:
        os.unlink(tmp_path)


def test_ingest_txt_file():
    """Test ingesting a TXT file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\nIt has multiple lines.\nThird line here.")
        tmp_path = f.name

    try:
        result: IngestionResult = ingest_document(tmp_path)
        assert result.success is True
        assert result.filename == "test.txt"
        assert result.text is not None
        assert "This is a test document" in result.text
        assert result.status == "processed"
    finally:
        os.unlink(tmp_path)


def test_ingest_pdf_file():
    """Test ingesting a PDF file."""
    # Create a simple PDF using pypdf
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        tmp_path = f.name

    try:
        # Use pypdf to create a simple PDF
        from PyPDF2 import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(100, 100)  # 100x100 page
        writer.add_blank_page(100, 100)
        with open(tmp_path, "wb") as output_file:
            writer.write(output_file)

        # Even an empty PDF should be valid (no text extraction error)
        result: IngestionResult = ingest_document(tmp_path)
        # PDF with no text content - should handle gracefully
        assert result is not None
    finally:
        os.unlink(tmp_path)


def test_ingest_docx_file():
    """Test ingesting a DOCX file."""
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
        tmp_path = f.name

    try:
        from docx import Document
        doc = Document()
        doc.add_paragraph("Test DOCX content line 1")
        doc.add_paragraph("Test DOCX content line 2")
        doc.save(tmp_path)

        result: IngestionResult = ingest_document(tmp_path)
        assert result.success is True
        assert result.filename == "test.docx"
        assert result.text is not None
        assert "Test DOCX content" in result.text
        assert result.status == "processed"
    finally:
        os.unlink(tmp_path)


def test_ingest_unsupported_file():
    """Test ingesting an unsupported file type."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        tmp_path = f.name

    try:
        result: IngestionResult = ingest_document(tmp_path)
        assert result.success is False
        assert result.error == "Unsupported file extension: .jpg"
        assert result.status == "failed"
    finally:
        os.unlink(tmp_path)


def test_ingest_nonexistent_file():
    """Test ingesting a nonexistent file."""
    result: IngestionResult = ingest_document("/nonexistent/file.pdf")
    assert result.success is False
    assert result.error == "File does not exist"
    assert result.status == "failed"


def test_ingest_empty_txt():
    """Test ingesting an empty TXT file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        tmp_path = f.name

    try:
        result: IngestionResult = ingest_document(tmp_path)
        assert result.success is False
        assert "No text could be extracted" in result.error
        assert result.status == "failed"
    finally:
        os.unlink(tmp_path)


def test_ingestion_result_to_dict():
    """Test IngestionResult can be serialized to dict."""
    import uuid
    result = IngestionResult(
        success=True,
        document_id=uuid.uuid4(),
        filename="test.txt",
        page_count=3,
        text="sample text",
        status="processed",
        chunk_count=5,
    )
    d = result.to_dict()
    assert d["success"] is True
    assert d["filename"] == "test.txt"
    assert d["page_count"] == 3
    assert d["status"] == "processed"
    assert d["chunk_count"] == 5


def test_ingest_docx_with_tables():
    """Test ingesting a DOCX file with tables."""
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
        tmp_path = f.name

    try:
        from docx import Document
        doc = Document()
        doc.add_paragraph("Table content:")
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "A1"
        table.cell(0, 1).text = "B1"
        table.cell(1, 0).text = "A2"
        table.cell(1, 1).text = "B2"
        doc.save(tmp_path)

        result: IngestionResult = ingest_document(tmp_path)
        assert result.success is True
        assert result.text is not None
        # Table content should be extracted
        assert "A1" in result.text or "B1" in result.text
    finally:
        os.unlink(tmp_path)