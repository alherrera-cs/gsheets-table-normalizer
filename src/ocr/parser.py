"""
Text block parsing and table candidate detection.

This module processes raw OCR text blocks to:
- Identify text structure (lines, paragraphs, headings)
- Detect potential table regions
- Group related text blocks
"""

from typing import List, Optional
import logging

from .models import TextBlock, TableCandidate

logger = logging.getLogger(__name__)


def parse_text_blocks(
    text_blocks: List[TextBlock],
    min_confidence: float = 0.5
) -> List[TextBlock]:
    """
    Parse and clean text blocks from OCR output.
    
    Filters low-confidence blocks and organizes text by spatial position.
    
    Args:
        text_blocks: Raw text blocks from OCR engine
        min_confidence: Minimum confidence threshold (0.0-1.0)
    
    Returns:
        Filtered and sorted list of TextBlock objects
    """
    if not text_blocks:
        return []
    
    # Filter by confidence
    filtered = [block for block in text_blocks if block.confidence >= min_confidence]
    filtered_count = len(filtered)
    total_count = len(text_blocks)
    if filtered_count < total_count:
        avg_confidence = sum(b.confidence for b in text_blocks) / total_count if text_blocks else 0.0
        logger.debug(f"[parser] Filtered {total_count - filtered_count} blocks (confidence < {min_confidence}), avg confidence: {avg_confidence:.2f}")
    
    # Sort by position (top-to-bottom, left-to-right)
    # Primary sort: y_min (top to bottom)
    # Secondary sort: x_min (left to right)
    sorted_blocks = sorted(
        filtered,
        key=lambda b: (b.bbox[1], b.bbox[0])  # (y_min, x_min)
    )
    
    # Assign line numbers based on y-position clustering
    _assign_line_numbers(sorted_blocks)
    
    logger.debug(f"[parser] Parsed {len(sorted_blocks)} text blocks from {len(text_blocks)} raw blocks")
    return sorted_blocks


def detect_table_candidates(
    text_blocks: List[TextBlock],
    min_rows: int = 2,
    min_cols: int = 2
) -> List[TableCandidate]:
    """
    Detect potential table regions from text blocks.
    
    Analyzes spatial arrangement of text blocks to identify table-like structures.
    
    Args:
        text_blocks: Parsed text blocks to analyze
        min_rows: Minimum number of rows to consider a table
        min_cols: Minimum number of columns to consider a table
    
    Returns:
        List of TableCandidate objects representing detected tables
    """
    if not text_blocks:
        return []
    
    logger.debug(f"[OCR] Detecting table candidates from {len(text_blocks)} text blocks")
    
    # Group blocks by rows
    rows = _group_blocks_by_row(text_blocks, row_threshold=20.0)
    
    if len(rows) < min_rows:
        logger.debug(f"[OCR] Not enough rows ({len(rows)} < {min_rows}) for table detection")
        return []
    
    # Analyze rows to detect table structure
    candidates: List[TableCandidate] = []
    
    # Find rows with consistent column structure
    column_positions = _detect_column_positions(rows)
    
    if len(column_positions) < min_cols:
        logger.debug(f"[OCR] Not enough columns ({len(column_positions)} < {min_cols}) for table detection")
        return []
    
    # Build cells matrix
    cells = []
    for row in rows:
        row_cells = _extract_cells_from_row(row, column_positions)
        if row_cells:
            cells.append(row_cells)
    
    if len(cells) >= min_rows:
        # Calculate bounding box
        all_x = [b.bbox[0] for row in rows for b in row] + [b.bbox[2] for row in rows for b in row]
        all_y = [b.bbox[1] for row in rows for b in row] + [b.bbox[3] for row in rows for b in row]
        
        bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
        
        # Detect if first row is a header (heuristic: different formatting or all caps)
        has_header = _detect_header_row(cells[0]) if cells else False
        
        # Calculate confidence based on regularity
        confidence = _calculate_table_confidence(cells, column_positions)
        
        candidate = TableCandidate(
            cells=cells,
            bbox=bbox,
            confidence=confidence,
            row_count=len(cells),
            column_count=len(column_positions),
            has_header=has_header
        )
        
        candidates.append(candidate)
        logger.debug(f"[OCR] Detected table candidate: {len(cells)} rows x {len(column_positions)} cols (confidence: {confidence:.2f})")
    
    return candidates


def _assign_line_numbers(text_blocks: List[TextBlock], line_threshold: float = 10.0) -> None:
    """
    Assign line numbers to text blocks based on y-position clustering.
    
    Modifies text_blocks in place by setting line_number attribute.
    
    Args:
        text_blocks: List of text blocks to process
        line_threshold: Maximum y-distance to consider same line (in pixels)
    """
    if not text_blocks:
        return
    
    current_line = 1
    last_y = None
    
    for block in text_blocks:
        y_min = block.bbox[1]
        
        if last_y is None:
            # First block
            block.line_number = current_line
            last_y = y_min
        elif abs(y_min - last_y) <= line_threshold:
            # Same line as previous block
            block.line_number = current_line
        else:
            # New line
            current_line += 1
            block.line_number = current_line
            last_y = y_min


def _group_blocks_by_row(
    text_blocks: List[TextBlock],
    row_threshold: float = 20.0
) -> List[List[TextBlock]]:
    """
    Group text blocks into rows based on y-position.
    
    Args:
        text_blocks: List of text blocks to group
        row_threshold: Maximum y-distance to consider same row (in pixels)
    
    Returns:
        List of rows, where each row is a list of TextBlock objects
    """
    if not text_blocks:
        return []
    
    rows: List[List[TextBlock]] = []
    current_row: List[TextBlock] = []
    last_y = None
    
    for block in sorted(text_blocks, key=lambda b: (b.bbox[1], b.bbox[0])):
        y_min = block.bbox[1]
        
        if last_y is None:
            # First block
            current_row.append(block)
            last_y = y_min
        elif abs(y_min - last_y) <= row_threshold:
            # Same row
            current_row.append(block)
        else:
            # New row
            if current_row:
                rows.append(sorted(current_row, key=lambda b: b.bbox[0]))  # Sort by x
            current_row = [block]
            last_y = y_min
    
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b.bbox[0]))
    
    return rows


def _detect_column_positions(rows: List[List[TextBlock]], min_gap: float = 30.0) -> List[float]:
    """
    Detect column boundaries from rows of text blocks.
    
    Args:
        rows: List of rows, each containing text blocks
        min_gap: Minimum gap between columns (in pixels)
    
    Returns:
        List of x-positions representing column boundaries
    """
    if not rows:
        return []
    
    # Collect all x-positions from all blocks
    all_x_positions = []
    for row in rows:
        for block in row:
            all_x_positions.append(block.bbox[0])  # x_min
            all_x_positions.append(block.bbox[2])  # x_max
    
    if not all_x_positions:
        return []
    
    # Find clusters of x-positions (columns)
    sorted_x = sorted(set(all_x_positions))
    column_positions = []
    current_cluster = [sorted_x[0]]
    
    for x in sorted_x[1:]:
        if x - current_cluster[-1] < min_gap:
            current_cluster.append(x)
        else:
            # End of cluster - use median as column position
            column_positions.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [x]
    
    if current_cluster:
        column_positions.append(sum(current_cluster) / len(current_cluster))
    
    return sorted(set(column_positions))


def _extract_cells_from_row(row: List[TextBlock], column_positions: List[float]) -> List[str]:
    """
    Extract cell values from a row of text blocks based on column positions.
    
    Args:
        row: List of text blocks in a row
        column_positions: List of x-positions for column boundaries
    
    Returns:
        List of cell text values
    """
    if not row or not column_positions:
        return []
    
    # Sort row blocks by x-position
    sorted_row = sorted(row, key=lambda b: b.bbox[0])
    
    cells = []
    col_idx = 0
    
    for block in sorted_row:
        block_center_x = (block.bbox[0] + block.bbox[2]) / 2
        
        # Find which column this block belongs to
        while col_idx < len(column_positions) and block_center_x > column_positions[col_idx]:
            # Add empty cell for skipped columns
            if len(cells) <= col_idx:
                cells.append("")
            col_idx += 1
        
        # Add block text to current column
        if col_idx < len(column_positions):
            while len(cells) <= col_idx:
                cells.append("")
            cells[col_idx] += (" " if cells[col_idx] else "") + block.text
        else:
            # Block beyond last column - append to last cell
            if cells:
                cells[-1] += " " + block.text
    
    # Ensure we have cells for all columns
    while len(cells) < len(column_positions):
        cells.append("")
    
    return cells


def _detect_header_row(first_row: List[str]) -> bool:
    """
    Detect if first row is likely a header row.
    
    Heuristic: Check if row contains mostly short, capitalized, or distinct text.
    
    Args:
        first_row: List of cell values from first row
    
    Returns:
        True if likely a header row
    """
    if not first_row:
        return False
    
    # Check if most cells are short (likely headers)
    short_cells = sum(1 for cell in first_row if len(cell.strip()) < 30)
    if short_cells / len(first_row) > 0.7:
        return True
    
    # Check if most cells are all caps or title case
    caps_or_title = sum(1 for cell in first_row if cell.strip().isupper() or cell.strip().istitle())
    if caps_or_title / len(first_row) > 0.6:
        return True
    
    return False


def _calculate_table_confidence(cells: List[List[str]], column_positions: List[float]) -> float:
    """
    Calculate confidence that detected structure is a valid table.
    
    Args:
        cells: 2D list of cell values
        column_positions: List of column x-positions
    
    Returns:
        Confidence score (0.0-1.0)
    """
    if not cells or not column_positions:
        return 0.0
    
    # Base confidence on regularity
    # Check if all rows have same number of columns
    col_counts = [len(row) for row in cells]
    if len(set(col_counts)) == 1:
        # All rows have same column count - high confidence
        regularity_score = 0.8
    else:
        # Varying column counts - lower confidence
        regularity_score = 0.4
    
    # Check if cells are not all empty
    non_empty_ratio = sum(1 for row in cells for cell in row if cell.strip()) / (len(cells) * len(column_positions))
    content_score = min(non_empty_ratio, 1.0)
    
    # Combined confidence
    confidence = (regularity_score * 0.6) + (content_score * 0.4)
    
    return min(max(confidence, 0.0), 1.0)

