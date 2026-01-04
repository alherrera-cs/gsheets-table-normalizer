"""
Table extraction from OCR text blocks and candidates.

This module converts detected table candidates into structured row/column format
suitable for the normalization pipeline.
"""

from typing import List, Dict, Any, Optional
import logging

from .models import TextBlock, TableCandidate, OCRResult, OCRMetadata

logger = logging.getLogger(__name__)


def extract_tables_from_blocks(
    text_blocks: List[TextBlock],
    table_candidates: Optional[List[TableCandidate]] = None,
    metadata: Optional[OCRMetadata] = None
) -> OCRResult:
    """
    Extract structured tables from OCR text blocks.
    
    Converts text blocks and table candidates into a format compatible with
    the normalization pipeline (List[Dict[str, Any]]).
    
    Args:
        text_blocks: Parsed text blocks from OCR
        table_candidates: Detected table regions (optional)
        metadata: OCR processing metadata (optional)
    
    Returns:
        OCRResult containing rows, raw_text, and metadata
    """
    if not text_blocks:
        logger.warning("No text blocks provided for table extraction")
        return OCRResult(
            rows=[],
            raw_text=None,
            metadata=metadata or OCRMetadata(),
            text_blocks=text_blocks,
            table_candidates=table_candidates or []
        )
    
    # Extract raw text
    raw_text = _extract_raw_text(text_blocks)
    logger.debug(f"[table_extract] Extracted raw text ({len(raw_text)} chars), preview: {raw_text[:300]}")
    
    # If table candidates are provided, use them
    if table_candidates:
        rows = _extract_rows_from_candidates(table_candidates)
        logger.debug(f"[table_extract] Extracted {len(rows)} rows from {len(table_candidates)} table candidates")
    else:
        # Fallback: attempt to extract tables from text blocks directly
        rows = _extract_rows_from_blocks(text_blocks)
        logger.debug(f"[table_extract] Extracted {len(rows)} rows directly from {len(text_blocks)} text blocks (no table candidates)")
        if rows:
            logger.debug(f"[table_extract] First row sample: {list(rows[0].keys()) if rows[0] else 'empty'}")
    
    result = OCRResult(
        rows=rows,
        raw_text=raw_text,
        metadata=metadata or OCRMetadata(),
        text_blocks=text_blocks,
        table_candidates=table_candidates or []
    )
    
    logger.debug(f"[table_extract] Final result: {len(rows)} rows from {len(text_blocks)} text blocks")
    return result


def _extract_raw_text(text_blocks: List[TextBlock]) -> str:
    """
    Extract full raw text from text blocks, preserving order.
    
    Args:
        text_blocks: List of text blocks
    
    Returns:
        Concatenated text string
    """
    if not text_blocks:
        return ""
    
    # Sort by line number and position
    sorted_blocks = sorted(
        text_blocks,
        key=lambda b: (b.line_number or 0, b.bbox[0])
    )
    
    lines = []
    current_line = None
    
    for block in sorted_blocks:
        line_num = block.line_number or 0
        if line_num != current_line:
            if current_line is not None:
                lines.append("")  # New line
            current_line = line_num
        lines.append(block.text)
    
    full_text = "\n".join(lines)
    # Log if text seems suspiciously short or fragmented
    if len(full_text) < 200 and len(sorted_blocks) > 20:
        sample_blocks = [b.text[:20] for b in sorted_blocks[:10]]
        logger.debug(f"[table_extract] Suspiciously short text ({len(full_text)} chars) from {len(sorted_blocks)} blocks. Sample blocks: {sample_blocks}")
    
    return full_text


def _extract_rows_from_candidates(candidates: List[TableCandidate]) -> List[Dict[str, Any]]:
    """
    Extract rows from table candidates.
    
    Converts TableCandidate.cells into list of dictionaries with column headers.
    
    Args:
        candidates: List of detected table candidates
    
    Returns:
        List of row dictionaries
    """
    if not candidates:
        return []
    
    # Use first candidate (can be extended to handle multiple tables)
    candidate = candidates[0]
    
    if not candidate.cells or len(candidate.cells) == 0:
        return []
    
    rows: List[Dict[str, Any]] = []
    
    # Determine header row
    start_row = 1 if candidate.has_header and len(candidate.cells) > 1 else 0
    headers = candidate.cells[0] if candidate.has_header else None
    
    # Convert cells to row dictionaries
    for i in range(start_row, len(candidate.cells)):
        row_cells = candidate.cells[i]
        
        if headers:
            # Use headers as keys
            row_dict = {}
            for j, header in enumerate(headers):
                value = row_cells[j] if j < len(row_cells) else ""
                # Clean header for use as key
                clean_header = _clean_column_name(header)
                row_dict[clean_header] = value
            rows.append(row_dict)
        else:
            # No headers - use column indices
            row_dict = {}
            for j, value in enumerate(row_cells):
                row_dict[f"column_{j+1}"] = value
            rows.append(row_dict)
    
    return rows


def _extract_rows_from_blocks(text_blocks: List[TextBlock]) -> List[Dict[str, Any]]:
    """
    Attempt to extract table rows directly from text blocks.
    
    Fallback method when table candidates are not available.
    Uses heuristics to identify row/column structure.
    
    Args:
        text_blocks: List of text blocks to analyze
    
    Returns:
        List of row dictionaries (may be empty if structure unclear)
    """
    if not text_blocks:
        return []
    
    logger.debug("[OCR] Extracting rows directly from text blocks (no table candidates)")
    
    # Group blocks by rows
    from .parser import _group_blocks_by_row
    
    rows_blocks = _group_blocks_by_row(text_blocks, row_threshold=20.0)
    
    if len(rows_blocks) < 2:
        logger.debug("[OCR] Not enough rows for table extraction")
        return []
    
    # Detect column structure using helper functions
    column_positions = _detect_column_positions_internal(rows_blocks)
    
    if len(column_positions) < 2:
        logger.debug("[OCR] Not enough columns for table extraction")
        return []
    
    # Extract cells for each row
    all_cells = []
    for row_blocks in rows_blocks:
        row_cells = _extract_cells_from_row_internal(row_blocks, column_positions)
        if row_cells:
            all_cells.append(row_cells)
    
    if not all_cells:
        return []
    
    # Convert to row dictionaries
    # Use first row as headers if it looks like headers
    has_header = _detect_header_row_internal(all_cells[0]) if all_cells else False
    
    rows: List[Dict[str, Any]] = []
    start_idx = 1 if has_header and len(all_cells) > 1 else 0
    headers = all_cells[0] if has_header else None
    
    for i in range(start_idx, len(all_cells)):
        row_cells = all_cells[i]
        row_dict = {}
        
        if headers:
            # Use headers as keys
            for j, header in enumerate(headers):
                value = row_cells[j] if j < len(row_cells) else ""
                clean_header = _clean_column_name(header)
                row_dict[clean_header] = value
        else:
            # Use column indices
            for j, value in enumerate(row_cells):
                row_dict[f"column_{j+1}"] = value
        
        rows.append(row_dict)
    
    logger.debug(f"[OCR] Extracted {len(rows)} rows directly from {len(text_blocks)} text blocks")
    return rows


def _clean_column_name(header: str) -> str:
    """
    Clean column header for use as dictionary key.
    
    Removes special characters, normalizes whitespace, converts to lowercase.
    
    Args:
        header: Raw column header text
    
    Returns:
        Cleaned header string
    """
    if not header:
        return ""
    
    # Basic cleaning: lowercase, strip, replace spaces with underscores
    cleaned = header.strip().lower()
    cleaned = cleaned.replace(" ", "_")
    cleaned = "".join(c for c in cleaned if c.isalnum() or c == "_")
    
    # Ensure it doesn't start with a number
    if cleaned and cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    
    return cleaned or "unnamed_column"


def _detect_column_positions_internal(rows: List[List[TextBlock]], min_gap: float = 30.0) -> List[float]:
    """Internal helper to detect column positions (duplicated from parser for table_extract use)."""
    if not rows:
        return []
    
    all_x_positions = []
    for row in rows:
        for block in row:
            all_x_positions.append(block.bbox[0])
            all_x_positions.append(block.bbox[2])
    
    if not all_x_positions:
        return []
    
    sorted_x = sorted(set(all_x_positions))
    column_positions = []
    current_cluster = [sorted_x[0]]
    
    for x in sorted_x[1:]:
        if x - current_cluster[-1] < min_gap:
            current_cluster.append(x)
        else:
            column_positions.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [x]
    
    if current_cluster:
        column_positions.append(sum(current_cluster) / len(current_cluster))
    
    return sorted(set(column_positions))


def _extract_cells_from_row_internal(row: List[TextBlock], column_positions: List[float]) -> List[str]:
    """Internal helper to extract cells from a row (duplicated from parser for table_extract use)."""
    if not row or not column_positions:
        return []
    
    sorted_row = sorted(row, key=lambda b: b.bbox[0])
    cells = []
    col_idx = 0
    
    for block in sorted_row:
        block_center_x = (block.bbox[0] + block.bbox[2]) / 2
        
        while col_idx < len(column_positions) and block_center_x > column_positions[col_idx]:
            if len(cells) <= col_idx:
                cells.append("")
            col_idx += 1
        
        if col_idx < len(column_positions):
            while len(cells) <= col_idx:
                cells.append("")
            cells[col_idx] += (" " if cells[col_idx] else "") + block.text
        else:
            if cells:
                cells[-1] += " " + block.text
    
    while len(cells) < len(column_positions):
        cells.append("")
    
    return cells


def _detect_header_row_internal(first_row: List[str]) -> bool:
    """Internal helper to detect header row (duplicated from parser for table_extract use)."""
    if not first_row:
        return False
    
    short_cells = sum(1 for cell in first_row if len(cell.strip()) < 30)
    if short_cells / len(first_row) > 0.7:
        return True
    
    caps_or_title = sum(1 for cell in first_row if cell.strip().isupper() or cell.strip().istitle())
    if caps_or_title / len(first_row) > 0.6:
        return True
    
    return False

