"""
OCR module for extracting structured data from images and PDFs.

This module provides OCR capabilities for the table normalizer, supporting:
- Image files (PNG, JPG, JPEG)
- PDF documents
- Text block extraction and table detection
- Conversion to structured row/column format
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

from .reader import extract_text_from_image, extract_text_from_pdf
from .parser import parse_text_blocks, detect_table_candidates
from .table_extract import extract_tables_from_blocks
from .models import OCRResult, TextBlock, TableCandidate, OCRMetadata

__all__ = [
    "extract_text_from_image",
    "extract_text_from_pdf",
    "parse_text_blocks",
    "detect_table_candidates",
    "extract_tables_from_blocks",
    "OCRResult",
    "TextBlock",
    "TableCandidate",
    "OCRMetadata",
]

