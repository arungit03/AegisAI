"""Table / structured content handling — preserve row/col relationships.

P13 Table/Structured Content:
- preserve row/col relationships, page/section
- avoid destroying meaning
- no OCR unless necessary (ingestion.py handles SIH coordinate extraction)

Exports:
- is_table_content(text: str) -> bool
- preserve_table_structure(text: str) -> str
- extract_table_rows(text: str) -> List[Dict[str, str]]
- format_table_for_context(rows: List[Dict], max_chars: int = 2000) -> str
"""

from __future__ import annotations

import re
from typing import List, Dict, Any


_TABLE_MARKERS = re.compile(r"(\|.*\|)|(\t)|(\s{2,}.*\s{2,})")
_SEPARATOR_RE = re.compile(r"^\s*[\|\-\+\=\:]+\s*$")
_PIPE_SEP_RE = re.compile(r"^\s*\|?(\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$")
# fallback separator-ish row (markdown or ascii)
_GENERIC_SEP_RE = re.compile(r"^\s*[\|\-\+\=\:\s]+\s*$")


def is_table_content(text: str) -> bool:
    """Heuristic: does text look like a table or structured grid?

    Detects:
    - pipe-separated tables (| col | col |)
    - tab-separated with consistent columns
    - CSV-like with consistent commas
    - whitespace-aligned columns (multiple spaces)
    - repeated column-like structure

    Avoids OCR/fuzzy fixes — purely structural.
    """
    if not text or len(text.strip()) < 20:
        return False
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    # Pipe table: at least 2 rows with >=2 pipes -> strong signal
    pipe_lines = sum(1 for line in lines if line.count("|") >= 2)
    if pipe_lines >= 2:
        return True
    # Looser pipe: >=60% of lines contain pipe
    pipe_any = sum(1 for line in lines if "|" in line)
    if pipe_any >= 2 and pipe_any / len(lines) >= 0.6:
        return True

    # Separator row (---|---|---|  or  |---|---)
    if any(_SEPARATOR_RE.match(line) for line in lines):
        return True
    if any(_PIPE_SEP_RE.match(line) for line in lines):
        return True

    # Tab-separated with consistent columns
    tab_lines = [line for line in lines if "\t" in line]
    if len(tab_lines) >= 2 and len({line.count("\t") for line in tab_lines}) == 1:
        return True
    # Tab present in majority of lines even if counts vary slightly
    if len(tab_lines) >= 2 and len(tab_lines) / len(lines) >= 0.6:
        # require at least 2 tabs overall to avoid false positive
        if any(line.count("\t") >= 1 for line in tab_lines):
            # check column consistency within 1
            counts = [line.count("\t") for line in tab_lines]
            if max(counts) - min(counts) <= 1 and max(counts) >= 1:
                return True

    # CSV-like: consistent comma counts (e.g., 3+ lines with same comma count >=2)
    comma_lines = [line for line in lines if line.count(",") >= 2]
    if len(comma_lines) >= 2:
        comma_counts = {line.count(",") for line in comma_lines}
        if len(comma_counts) == 1:
            return True
        # allow slight variance for quoted commas? already strict, keep 1-variance
        if len(comma_lines) / len(lines) >= 0.6 and max(comma_counts) - min(comma_counts) <= 1:
            return True

    # Multiple lines with aligned whitespace columns (e.g., "col  col  col")
    multi_space = sum(1 for line in lines if re.search(r"\S\s{2,}\S", line))
    if multi_space >= 2 and multi_space / len(lines) >= 0.6:
        return True

    # Repeated column-like structure via pipe-cell count consistency
    # e.g., every line split by "|" gives same number of cells (within 1)
    pipe_splits = []
    for line in lines:
        if "|" in line:
            cells = [c for c in line.strip().strip("|").split("|")]
            # count non-trivial splits
            if len(cells) >= 2:
                pipe_splits.append(len(cells))
    if len(pipe_splits) >= 2 and max(pipe_splits) - min(pipe_splits) <= 1:
        # also require at least 2 such lines covering majority
        if len(pipe_splits) / len(lines) >= 0.5:
            return True

    return False


def preserve_table_structure(text: str, page: int | None = None, section: str | None = None) -> str:
    """Wrap table-like text to keep row/col alignment intact.

    - Preserves row/col relationships and row boundaries with clear delimiters.
    - Preserves page/section header when provided.
    - Avoids destroying meaning: never reorders columns, never OCR-corrects.
    - No LLM, no cloud, no OCR.

    Args:
        text: Raw extracted text (may contain a table).
        page: Optional page number to preserve.
        section: Optional section title to preserve.

    Returns:
        If table detected, text wrapped with ````table`` fence and header so
        downstream chunking/LLM sees explicit row delimiters. Else original text.
    """
    if not is_table_content(text):
        return text

    header = ""
    if page is not None:
        header += f"[Table p.{page}]"
    if section:
        header += f" [{section}]" if header else f"[{section}]"
    if header:
        header += "\n"

    # Normalize row boundaries: strip trailing whitespace per row, drop
    # completely empty rows inside table but keep row delimiters (\n).
    # Do NOT reflow or wrap rows — each visual row stays on its own line.
    raw = text.strip("\n")
    # Preserve internal empty lines as row separators only if they are
    # meaningful (single empty line); collapse 3+ consecutive empties.
    lines: List[str] = []
    empty_run = 0
    for line in raw.splitlines():
        if not line.strip():
            empty_run += 1
            if empty_run <= 1:
                lines.append("")
            continue
        empty_run = 0
        # rstrip keeps column alignment intent but removes trailing noise
        lines.append(line.rstrip())

    # Remove leading/trailing empty lines added above
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    normalized = "\n".join(lines).strip()
    # Use markdown fence to signal structure to LLM without altering content.
    # The fence itself acts as a clear row/col delimiter block.
    return f"{header}```table\n{normalized}\n```"


def _is_separator_row(line: str) -> bool:
    """True if line is a markdown/ascii separator row (---|---)."""
    stripped = line.strip()
    if not stripped:
        return False
    if _PIPE_SEP_RE.match(stripped):
        return True
    if _SEPARATOR_RE.match(stripped) and set(stripped) <= set("|-+: ="):
        # require at least 3 dashes/colons to avoid false positive on " - "
        if stripped.count("-") >= 3:
            return True
    return False


def _detect_delimiter(lines: List[str]) -> str:
    """Detect column delimiter among |, tab, comma, 2+ spaces."""
    # Prefer pipe if strong signal
    pipe_lines = sum(1 for line in lines if "|" in line)
    if pipe_lines >= 2:
        # confirm consistent cell counts
        counts = []
        for line in lines:
            if _is_separator_row(line):
                continue
            if "|" in line:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 2:
                    counts.append(len(cells))
        if len(counts) >= 2 and max(counts) - min(counts) <= 1:
            return "|"
        if pipe_lines / len(lines) >= 0.5:
            return "|"

    tab_lines = [line for line in lines if "\t" in line]
    if len(tab_lines) >= 2:
        counts = {line.count("\t") for line in tab_lines}
        if len(counts) == 1 or (max(counts) - min(counts) <= 1 and len(tab_lines) / len(lines) >= 0.5):
            return "\t"

    comma_lines = [line for line in lines if line.count(",") >= 2]
    if len(comma_lines) >= 2 and len({line.count(",") for line in comma_lines}) == 1:
        return ","

    multi = sum(1 for line in lines if re.search(r"\S\s{2,}\S", line))
    if multi >= 2 and multi / len(lines) >= 0.5:
        return "ws2"

    # fallback: pick strongest single marker
    if "|" in "".join(lines):
        return "|"
    if "\t" in "".join(lines):
        return "\t"
    if any("," in line for line in lines):
        return ","
    return "ws2"


def _split_row(line: str, delimiter: str) -> List[str]:
    """Split a single row by detected delimiter, stripping cells."""
    if delimiter == "|":
        # handle leading/trailing pipes gracefully
        stripped = line.strip()
        # remove outer pipes then split
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]
    if delimiter == "\t":
        return [cell.strip() for cell in line.split("\t")]
    if delimiter == ",":
        return [cell.strip() for cell in line.split(",")]
    if delimiter == "ws2":
        # 2+ whitespace as column separator — preserve single spaces inside cells
        return [cell.strip() for cell in re.split(r"\s{2,}", line.strip())]
    return [line.strip()]


def extract_table_rows(text: str) -> List[Dict[str, str]]:
    """Parse table rows into structured dicts if possible.

    - Detects delimiter (|, tab, comma, 2+ spaces) from repeated structure.
    - Skips separator rows (---|---).
    - First non-separator row is treated as header; subsequent rows mapped
      to header keys. Duplicate/empty headers become col_0, col_1...
    - Pads short rows with "" and truncates long rows to header width.
    - If no header can be determined, keys are col_0...col_n.

    Args:
        text: Raw table text ( ideally already preserve_table_structure output
              or raw extracted text). Fence markers ```table are stripped.

    Returns:
        List of row dicts. Empty list if not tabular or not parseable.
    """
    if not text or not text.strip():
        return []

    # Strip fence if present
    stripped = text.strip()
    if stripped.startswith("```"):
        # remove first fence line and last fence line
        fence_lines = stripped.splitlines()
        # drop opening ```table or ``` line
        if fence_lines and fence_lines[0].strip().startswith("```"):
            fence_lines = fence_lines[1:]
        if fence_lines and fence_lines[-1].strip().startswith("```"):
            fence_lines = fence_lines[:-1]
        stripped = "\n".join(fence_lines)

    # Remove page/section header like "[Table p.3]" injected by preserve_table_structure
    lines = [line.rstrip() for line in stripped.splitlines()]
    # filter leading header lines like "[Table p.2]" or "[Section]"
    filtered: List[str] = []
    for line in lines:
        if re.match(r"^\s*\[Table p\.\d+\]", line):
            # may be "[Table p.2] [Section]" on same line — drop only that prefix
            # keep remainder if any table content on same line (unlikely)
            remainder = re.sub(r"^\s*\[Table p\.\d+\]\s*(\[.*?\])?\s*", "", line)
            if remainder.strip():
                filtered.append(remainder)
            continue
        if re.match(r"^\s*\[.*?\]\s*$", line) and "|" not in line and "\t" not in line:
            # standalone bracket header — skip
            continue
        filtered.append(line)

    # Keep only non-empty lines for structure detection, but track original
    non_empty = [line for line in filtered if line.strip()]
    if len(non_empty) < 2:
        return []

    # Remove separator rows for delimiter detection and header selection
    data_lines = [line for line in non_empty if not _is_separator_row(line)]
    if len(data_lines) < 2:
        return []

    delimiter = _detect_delimiter(data_lines)

    # Header is first non-separator line
    header_cells = _split_row(data_lines[0], delimiter)
    # Clean header: empty -> col_i, duplicates -> disambiguate
    headers: List[str] = []
    seen: Dict[str, int] = {}
    for idx, cell in enumerate(header_cells):
        key = cell.strip() if cell.strip() else f"col_{idx}"
        # disambiguate duplicates
        if key in seen:
            seen[key] += 1
            key = f"{key}_{seen[key]}"
        else:
            seen[key] = 0
        # sanitize key: keep readable but safe
        if not key:
            key = f"col_{idx}"
        headers.append(key)

    if not headers:
        return []

    rows: List[Dict[str, str]] = []
    for line in data_lines[1:]:
        cells = _split_row(line, delimiter)
        # Normalize width to headers
        if len(cells) < len(headers):
            cells = cells + [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[: len(headers)]
        # Skip rows that are entirely empty
        if not any(c.strip() for c in cells):
            continue
        row = {headers[i]: cells[i] for i in range(len(headers))}
        rows.append(row)

    return rows


def format_table_for_context(rows: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    """Format table rows for LLM context with row/col headers preserved.

    - Preserves column headers and row boundaries (each row is a separate line).
    - Uses markdown pipe table for compact, LLM-friendly presentation.
    - Respects max_chars strictly and truncates at row boundaries (never mid-row).
    - Escapes internal pipes in cell values to avoid breaking column structure.

    Args:
        rows: List of dicts as returned by extract_table_rows. All dicts should
              share the same keys (column headers). Mixed keys are unioned.
        max_chars: Hard cap on returned string length.

    Returns:
        Formatted string. Empty string if rows is empty.
    """
    if not rows:
        return ""

    # Union headers in order of first appearance
    headers: List[str] = []
    seen_h = set()
    for row in rows:
        for key in row.keys():
            if key not in seen_h:
                seen_h.add(key)
                headers.append(str(key))

    if not headers:
        return ""

    def escape_cell(value: Any) -> str:
        s = str(value) if value is not None else ""
        # escape pipe to keep column structure
        s = s.replace("|", "\\|")
        # collapse newlines inside a cell to space (row boundary is line)
        s = s.replace("\n", " ").replace("\r", " ")
        # trim and normalize whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Build markdown table lines
    header_line = "| " + " | ".join(escape_cell(h) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    lines: List[str] = [header_line, sep_line]

    # Pre-compute header overhead
    base = "\n".join(lines)
    # If even header doesn't fit, truncate header itself (unlikely)
    if len(base) > max_chars:
        return base[: max_chars - 3] + "..."

    for row in rows:
        cells = [escape_cell(row.get(h, "")) for h in headers]
        row_line = "| " + " | ".join(cells) + " |"
        candidate = "\n".join(lines + [row_line])
        if len(candidate) > max_chars:
            # Row doesn't fit — try to truncate cell contents to fit this one row
            remaining = max_chars - len("\n".join(lines)) - 1  # -1 for newline
            if remaining > 20 and len(row_line) > remaining:
                # Proportionally truncate each cell so row fits within remaining
                # Keep per-cell budget
                per_cell_budget = max(5, (remaining - (len(headers) * 3 + 1)) // len(headers))
                truncated_cells = []
                for c in cells:
                    if len(c) > per_cell_budget:
                        truncated_cells.append(c[: per_cell_budget - 3] + "...")
                    else:
                        truncated_cells.append(c)
                truncated_line = "| " + " | ".join(truncated_cells) + " |"
                if len("\n".join(lines + [truncated_line])) <= max_chars:
                    lines.append(truncated_line)
            break
        lines.append(row_line)

    result = "\n".join(lines)
    # Final safety: truncate at last row boundary if somehow over
    if len(result) > max_chars:
        # cut and ensure we end at a row boundary (newline)
        cut = result[:max_chars]
        last_nl = cut.rfind("\n")
        if last_nl > len(header_line):
            result = cut[:last_nl]
        else:
            result = cut[: max_chars - 3] + "..."
    return result


def annotate_chunks_with_table_info(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add is_table flag to hits where chunk_text looks tabular.

    Backward-compatible helper used by hybrid_service.py.
    Mutates hits in place and returns them.
    """
    for hit in hits:
        payload = hit.get("payload") or {}
        txt = str(payload.get("chunk_text", ""))
        hit["is_table"] = is_table_content(txt)
        if hit["is_table"]:
            hit["table_preserved"] = preserve_table_structure(
                txt,
                page=payload.get("page_number"),
                section=payload.get("section_title"),
            )
            # Also expose structured rows for downstream consumers when parseable
            try:
                rows = extract_table_rows(txt)
                if rows:
                    hit["table_rows"] = rows
                    hit["table_context"] = format_table_for_context(rows)
            except Exception:
                pass
    return hits
