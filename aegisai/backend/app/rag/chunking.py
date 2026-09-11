"""Text chunking service for RAG documents.

Implements configurable chunking with support for chunk size, overlap,
and minimum chunk length. Each chunk retains metadata including document_id,
filename, page number, classification, and department.
"""

import structlog
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = structlog.get_logger(__name__)


def _make_chunk_id(document_id: str, filename: str, chunk_index: object) -> str:
    """Return a deterministic UUID accepted by Qdrant as a point ID."""
    identity = f"{document_id}:{filename}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


@dataclass
class ChunkMetadata:
    """Metadata retained for each text chunk."""
    document_id: str
    filename: str
    page_number: Optional[int] = None
    classification: str = "public_internal"
    department: Optional[str] = None
    uploader_role: Optional[str] = None
    allowed_roles: List[str] = field(default_factory=list)
    allowed_users: List[str] = field(default_factory=list)


@dataclass
class ChunkResult:
    """Result of chunking a text."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    chunk_index: int
    char_start: int
    char_end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_id": self.metadata.document_id,
            "filename": self.metadata.filename,
            "page_number": self.metadata.page_number,
            "classification": self.metadata.classification,
            "department": self.metadata.department,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_length: int = 50,
    document_id: str = "",
    filename: str = "",
    page_number: Optional[int] = None,
    chunk_index_start: int = 0,
    classification: str = "public_internal",
    department: Optional[str] = None,
    allowed_roles: List[str] = None,
    allowed_users: List[str] = None,
) -> List[ChunkResult]:
    """Split text into overlapping chunks.

    Each chunk retains metadata for authorization and retrieval.

    Args:
        text: The full text to chunk.
        chunk_size: Size of each chunk in characters (default 512).
        chunk_overlap: Number of characters to overlap between chunks (default 50).
        min_chunk_length: Minimum chunk length in characters to keep (default 50).
        document_id: ID of the source document.
        filename: Filename of the source document.
        page_number: Page number if available.
        chunk_index_start: Global chunk index offset when chunking pages separately.
        classification: Document classification level.
        department: Document department.

    Returns:
        List of ChunkResult objects.
    """
    if not text or not text.strip():
        logger.warning("Attempted to chunk empty text")
        return []

    chunks = []
    text_length = len(text)

    # If text is smaller than chunk_size, return as single chunk
    if text_length <= chunk_size:
        if text_length >= min_chunk_length:
            chunk_index = chunk_index_start
            chunk_id = _make_chunk_id(document_id, filename, chunk_index)
            chunk = ChunkResult(
                chunk_id=chunk_id,
                text=text,
                metadata=ChunkMetadata(
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    classification=classification,
                    department=department,
                    allowed_roles=allowed_roles or [],
                    allowed_users=allowed_users or [],
                ),
                chunk_index=chunk_index,
                char_start=0,
                char_end=text_length,
            )
            chunks.append(chunk)
        return chunks

    # Chunk the text with overlap
    # P13 Table handling: avoid breaking rows mid-row when table markers present
    def _snap_to_row_boundary(s: int, e: int, txt: str) -> int:
        """If window contains table markers, snap end back to previous row boundary (\n).

        Keeps row/col relationships intact. Falls back to original end if snapping
        would make chunk too small.
        """
        window = txt[s:e]
        # quick check: table-like markers in this window
        has_table = ("|" in window and window.count("|") >= 2) or ("\t" in window) or ("```table" in window)
        if not has_table:
            return e
        if e >= len(txt):
            return e
        # Don't snap if we'd cut into a very short chunk
        # Prefer previous newline within window as row boundary
        # Search backwards from e for \n
        last_nl = txt.rfind("\n", s, e)
        if last_nl == -1 or last_nl <= s:
            return e
        # Ensure snapped chunk still meets min length and doesn't lose too much
        snapped_len = last_nl + 1 - s  # include newline
        # Keep at least 60% of requested size or min_chunk_length, whichever larger
        min_keep = max(min_chunk_length, int(chunk_size * 0.6))
        if snapped_len < min_keep:
            return e
        # Also avoid snapping inside a fenced table block mid-marker
        # If we cut right after ```table opening, don't snap there
        preview = txt[last_nl + 1 : e]
        if preview.strip().startswith("```") or txt[max(s, last_nl - 10) : last_nl + 10].count("```") % 2 == 1:
            # Inside fence marker — keep original boundary instead
            return e
        return last_nl + 1

    start = 0
    chunk_index = 0

    while start < text_length:
        # End of current chunk
        end = min(start + chunk_size, text_length)

        # Table-aware: prefer row boundaries over arbitrary mid-row cuts
        if end < text_length:
            end = _snap_to_row_boundary(start, end, text)

        # Get the chunk text
        chunk_text = text[start:end]

        # If chunk is too small and not at the end, extend it
        if len(chunk_text) < min_chunk_length and end < text_length:
            # Try to extend to next word boundary
            remaining = text_length - end
            if remaining > 0:
                # Look ahead for word boundary
                search_end = min(end + chunk_overlap * 2, text_length)
                next_chunk = text[end:search_end]
                # Find a good breaking point (sentence or paragraph)
                break_pos = next_chunk.find("\n\n")
                if break_pos == -1:
                    break_pos = next_chunk.find(". ")
                if break_pos == -1:
                    break_pos = next_chunk.find("!")
                if break_pos == -1:
                    break_pos = next_chunk.find("?")
                if break_pos == -1:
                    break_pos = next_chunk.find(" ")
                if break_pos > 0:
                    extension = next_chunk[:break_pos + 1]
                    chunk_text = chunk_text + extension
                    end = start + len(chunk_text)

        # Only create chunk if it meets minimum length
        if len(chunk_text) >= min_chunk_length:
            global_chunk_index = chunk_index_start + chunk_index
            chunk_id = _make_chunk_id(document_id, filename, global_chunk_index)

            chunk = ChunkResult(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=ChunkMetadata(
                    document_id=document_id,
                    filename=filename,
                    page_number=page_number,
                    classification=classification,
                    department=department,
                    allowed_roles=allowed_roles or [],
                    allowed_users=allowed_users or [],
                ),
                chunk_index=global_chunk_index,
                char_start=start,
                char_end=end,
            )
            chunks.append(chunk)

        # Move start position, accounting for overlap
        start = start + chunk_size - chunk_overlap

        # Prevent infinite loop
        if chunk_index > 0 and start <= chunks[-1].char_start:
            # We've reached or passed the previous chunk start; advance by one character
            start = chunks[-1].char_end

        chunk_index += 1

        # Safety limit
        if chunk_index > text_length + 10:
            logger.warning(f"Chunking loop detected at index {chunk_index}, stopping")
            break

    # Deduplicate very similar adjacent chunks
    logger.info(f"Chunked text into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
