"""Local document ingestion service for PDF, DOCX, and TXT files.

Handles file validation, text extraction, page information where available,
metadata, errors, empty documents, unsupported files, and file size limits.
"""

import os
import structlog
import uuid
import hashlib
import re
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field

import docx
from pypdf import PdfReader
import fitz

logger = structlog.get_logger(__name__)


def _sanitize_filename_for_log(filename: str) -> str:
    """Hash filename for logging to avoid exposing sensitive names."""
    name, ext = os.path.splitext(filename)
    name_hash = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"{name_hash}{ext}"


def _sanitize_file_path_for_log(file_path: str) -> str:
    """Extract only the filename from a file path for logging."""
    return os.path.basename(file_path)


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    success: bool = False
    document_id: Optional[uuid.UUID] = None
    filename: Optional[str] = None
    page_count: Optional[int] = None
    text: Optional[str] = None
    # Page-preserving text used for chunking and citation metadata.  The
    # legacy ``text`` field remains the complete extracted document.
    page_texts: Optional[List[str]] = None
    sih_records: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    status: str = "failed"  # uploaded, processing, processed, failed
    chunk_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "document_id": str(self.document_id) if self.document_id else None,
            "filename": self.filename,
            "page_count": self.page_count,
            "error": self.error,
            "status": self.status,
            "chunk_count": self.chunk_count,
        }


def validate_file(file_path: str, max_size: int = 50 * 1024 * 1024) -> Tuple[bool, Optional[str]]:
    """Validate file before processing.

    Args:
        file_path: Path to the file.
        max_size: Maximum file size in bytes (default 50MB).

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not os.path.exists(file_path):
        return False, "File does not exist"

    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        return False, f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({max_size / 1024 / 1024:.1f}MB)"

    if file_size == 0:
        return False, "File is empty"

    return True, None


def extract_text_pdf(file_path: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Tuple of (extracted_text, page_count).
    """
    try:
        page_texts = extract_page_texts_pdf(file_path)
        return "\n".join(page_texts), len(page_texts)

    except Exception as e:
        logger.error("PDF extraction failed", filename=_sanitize_file_path_for_log(file_path), error_type=type(e).__name__)
        return None, None


def extract_page_texts_pdf(file_path: str) -> List[str]:
    """Extract PDF text while retaining one string per source page."""
    reader = PdfReader(file_path)
    page_texts: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            page_texts.append(page.extract_text() or "")
        except Exception as e:
            logger.warning("PDF page extraction failed", page=i + 1, error_type=type(e).__name__)
            page_texts.append("")
    return page_texts


def _clean_sih_text(value: str) -> str:
    """Clean coordinate-extracted table text without changing its meaning."""
    text = " ".join(value.replace("\n", " ").split())
    # pypdf occasionally separates a single OCR word into ``an d`` or
    # ``P owered``. Join only clear one-letter fragments; ordinary word
    # boundaries such as ``for policy`` remain untouched.
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\b([A-Za-z]{2,})\s+([a-z])\b", r"\1\2", text)
        text = re.sub(r"\b([A-Z])\s+([a-z]{2,})\b", r"\1\2", text)
        text = re.sub(r"\b([A-Z][a-z]{1,2})\s+([a-z]{2,})\b", r"\1\2", text)
    # The source PDF uses a font encoding that makes a small set of capital
    # I/A glyphs look like lowercase l/A in extracted text. Correct only these
    # known OCR artifacts; do not fuzzy-match identifiers or titles.
    replacements = {
        "Al-": "AI-",
        "lmagery": "Imagery",
        "lma ges": "Images",
        "lntelligent": "Intelligent",
        "lntegration": "Integration",
        "lnfrastructure": "Infrastructure",
        "lnnovation": "Innovation",
        "SYstem": "System",
        "Anlntegrated": "An Integrated",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    # The rendered table repeats a page marker and URL after the final row.
    # Keep those footer tokens out of record metadata (especially deadline).
    text = re.sub(r"\s+https?://\S+\s*$", "", text)
    text = re.sub(r"\s+\d{1,3}/\d{1,3}\s*$", "", text)
    return text


def extract_sih_records_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract SIH table rows using PDF coordinates.

    The SIH PDF is a rendered HTML table. Plain text extraction interleaves
    columns, so rows are rebuilt from the shared vertical span of each serial
    number and columns are selected by their x-coordinate. This keeps the ID
    and title from one visual row rather than independently searching text.
    """
    records: List[Dict[str, Any]] = []
    pdf = fitz.open(file_path)
    for page_number, page in enumerate(pdf, start=1):
        words = page.get_text("words")
        row_starts = sorted({
            (word[1], word[4])
            for word in words
            if word[0] <= 100 and re.fullmatch(r"\d{1,3}", word[4])
        })
        for row_index, (start_y, serial) in enumerate(row_starts):
            end_y = row_starts[row_index + 1][0] if row_index + 1 < len(row_starts) else float("inf")
            row_words = [word for word in words if start_y <= word[1] < end_y]
            ordered_words = sorted(row_words, key=lambda item: (item[1], item[0]))
            row_text = " ".join(word[4] for word in ordered_words)
            canonical_text = re.sub(r"[^A-Z0-9]", "", row_text.upper())
            identifier_match = re.search(r"SIH(\d{5})", canonical_text)
            if not identifier_match:
                continue

            # On the SIH table, organization is the first column and title is
            # the second. The table's x-column boundaries are stable across
            # the uploaded PDF and are preserved by PyMuPDF word coordinates.
            def column_text(left: float, right: float) -> str:
                return _clean_sih_text(" ".join(
                    word[4] for word in ordered_words if left <= word[0] < right
                ))

            title_parts = [
                word[4] for word in sorted(row_words, key=lambda item: (item[1], item[0]))
                if 210 <= word[0] < 400
            ]
            record_id = f"SIH{identifier_match.group(1)}"
            title = _clean_sih_text(" ".join(title_parts))
            if not title:
                continue
            records.append({
                "problem_statement_id": record_id,
                "title": title,
                "description": None,
                "organization": column_text(110, 210),
                "category": column_text(400, 480),
                "submitted_ideas_count": column_text(560, 610),
                "theme": column_text(610, 720),
                "deadline": column_text(720, float("inf")),
                "page_number": page_number,
                "record_type": "sih_record",
                "raw_text": _clean_sih_text(row_text),
                "serial_number": serial,
            })
    return records


def extract_text_docx(file_path: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract text from a DOCX file.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        Tuple of (extracted_text, page_count estimate).
    """
    try:
        doc = docx.Document(file_path)

        # Extract tables text as well
        table_texts = []
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    table_texts.append(cell.text)

        # Extract regular paragraphs
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        full_text = "\n".join(paragraphs + table_texts)
        # Rough page estimate: assume ~500 words per page
        word_count = len(full_text.split())
        page_count = max(1, (word_count // 500) + 1) if word_count else 1

        return full_text, page_count

    except Exception as e:
        logger.error("DOCX extraction failed", filename=_sanitize_file_path_for_log(file_path), error_type=type(e).__name__)
        return None, None


def extract_text_txt(file_path: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract text from a TXT file.

    Args:
        file_path: Path to the TXT file.

    Returns:
        Tuple of (extracted_text, page_count estimate).
    """
    try:
        # Try UTF-8 first, then fall back to other encodings
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]

        full_text = None
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    full_text = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if full_text is None:
            logger.error("TXT file encoding error", filename=_sanitize_file_path_for_log(file_path))
            return None, None

        # Rough page estimate: assume ~500 words per page
        word_count = len(full_text.split())
        page_count = max(1, (word_count // 500) + 1) if word_count else 1

        return full_text, page_count

    except Exception as e:
        logger.error("TXT extraction failed", filename=_sanitize_file_path_for_log(file_path), error_type=type(e).__name__)
        return None, None


def ingest_document(file_path: str, max_size: int = 50 * 1024 * 1024) -> IngestionResult:
    """Ingest and extract text from a local document file.

    Supports: PDF, DOCX, TXT

    Args:
        file_path: Path to the document file.
        max_size: Maximum file size in bytes (default 50MB).

    Returns:
        IngestionResult with extraction results.
    """
    # Validate file
    is_valid, error = validate_file(file_path, max_size)
    if not is_valid:
        result = IngestionResult(
            success=False,
            filename=os.path.basename(file_path),
            error=error,
            status="failed",
        )
        logger.warning("File validation failed", filename=_sanitize_file_path_for_log(file_path), error_category="validation_failed")
        return result

    # Determine file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Extract text based on type
    result = IngestionResult(
        filename=os.path.basename(file_path),
        status="uploaded",
    )

    if ext == ".pdf":
        page_texts = extract_page_texts_pdf(file_path)
        full_text = "\n".join(page_texts)
        page_count = len(page_texts)
        result.text = full_text
        result.page_texts = page_texts
        result.sih_records = extract_sih_records_pdf(file_path)
        result.page_count = page_count
        if full_text and full_text.strip():
            result.status = "processed"
            result.success = True
        else:
            result.error = "No text could be extracted from the PDF"
            result.status = "failed"

    elif ext == ".docx":
        full_text, page_count = extract_text_docx(file_path)
        result.text = full_text
        result.page_texts = [full_text] if full_text else []
        result.page_count = page_count
        if full_text and full_text.strip():
            result.status = "processed"
            result.success = True
        else:
            result.error = "No text could be extracted from the DOCX file"
            result.status = "failed"

    elif ext == ".txt":
        full_text, page_count = extract_text_txt(file_path)
        result.text = full_text
        result.page_texts = [full_text] if full_text else []
        result.page_count = page_count
        if full_text and full_text.strip():
            result.status = "processed"
            result.success = True
        else:
            result.error = "No text could be extracted from the TXT file"
            result.status = "failed"

    else:
        result.error = f"Unsupported file extension: {ext}"
        result.status = "failed"

    if result.success:
        logger.info(
            "document_ingested",
            filename=_sanitize_filename_for_log(result.filename),
            page_count=result.page_count,
            char_count=len(result.text) if result.text else 0,
        )
    else:
        logger.warning(
            "document_ingestion_failed",
            filename=_sanitize_filename_for_log(result.filename),
            error_category="extraction_failed",
        )

    return result
