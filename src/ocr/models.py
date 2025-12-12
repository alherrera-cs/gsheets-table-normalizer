"""
Data models for OCR processing pipeline.

Defines the structure for OCR results, text blocks, table candidates,
and metadata used throughout the OCR extraction process.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OCRMetadata:
    """
    Metadata about the OCR extraction process.
    
    Attributes:
        confidence: Overall confidence score (0.0-1.0)
        page_number: Page number if from multi-page document (1-indexed)
        page_count: Total number of pages in document
        engine: OCR engine used (e.g., "tesseract", "easyocr", "vision")
        language: Detected or specified language code
        processing_time: Time taken for OCR in seconds
        image_dimensions: Tuple of (width, height) in pixels
        dpi: Resolution of source image/document
        vision_model: Vision API model used (e.g., "gpt-4o", "gpt-4-turbo") if Vision engine was used
    """
    confidence: float = 0.0
    page_number: Optional[int] = None
    page_count: Optional[int] = None
    engine: str = "placeholder"
    language: str = "eng"
    processing_time: float = 0.0
    image_dimensions: Optional[Tuple[int, int]] = None
    dpi: Optional[int] = None
    vision_model: Optional[str] = None


@dataclass
class TextBlock:
    """
    A single block of extracted text with spatial information.
    
    Attributes:
        text: The extracted text content
        bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)
        confidence: Confidence score for this block (0.0-1.0)
        line_number: Approximate line number in document (for ordering)
        block_type: Type of block ("paragraph", "heading", "table", "list", etc.)
    """
    text: str
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    confidence: float = 1.0
    line_number: Optional[int] = None
    block_type: str = "text"


@dataclass
class TableCandidate:
    """
    A detected table region with extracted cells.
    
    Attributes:
        cells: 2D list of cell text values [row][column]
        bbox: Bounding box of entire table (x_min, y_min, x_max, y_max)
        confidence: Confidence that this is a valid table (0.0-1.0)
        row_count: Number of rows detected
        column_count: Number of columns detected
        has_header: Whether first row appears to be a header
    """
    cells: List[List[str]]
    bbox: Tuple[float, float, float, float]
    confidence: float = 0.0
    row_count: int = 0
    column_count: int = 0
    has_header: bool = False


@dataclass
class OCRResult:
    """
    Complete OCR extraction result.
    
    Attributes:
        rows: List of dictionaries representing extracted table rows
        raw_text: Full extracted text from document (optional)
        metadata: OCR processing metadata
        text_blocks: List of detected text blocks (optional)
        table_candidates: List of detected table regions (optional)
    """
    rows: List[Dict[str, Any]]
    raw_text: Optional[str] = None
    metadata: OCRMetadata = field(default_factory=OCRMetadata)
    text_blocks: Optional[List[TextBlock]] = None
    table_candidates: Optional[List[TableCandidate]] = None

