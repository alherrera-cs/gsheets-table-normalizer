"""
Fallback inference module for extracting vehicle data from messy/broken files.

This module provides inference functions that can extract key vehicle fields
even when headers are missing, wrong, or the file structure is broken.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Known vehicle makes for detection
KNOWN_MAKES = {
    "toyota", "honda", "ford", "chevrolet", "nissan", "bmw", "mercedes", 
    "audi", "volkswagen", "hyundai", "kia", "mazda", "subaru", "jeep",
    "dodge", "chrysler", "gmc", "cadillac", "lexus", "acura", "infiniti",
    "tesla", "volvo", "porsche", "jaguar", "land rover", "mini", "fiat",
    "ram", "buick", "lincoln", "genesis", "mitsubishi", "alfa romeo"
}

# VIN pattern: 17 characters, A-Z and 0-9, excluding I, O, Q
VIN_PATTERN = re.compile(r'^[A-HJ-NPR-Z0-9]{17}$')

# Year pattern: 4 digits between 1900-2035
YEAR_PATTERN = re.compile(r'^(19[0-9]{2}|20[0-2][0-9]|203[0-5])$')


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance (number of character changes needed)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def fuzzy_match_header(header: str, target: str, threshold: float = 0.75) -> bool:
    """
    Check if header fuzzy matches target using Levenshtein distance.
    
    Args:
        header: Actual header from file
        target: Target header to match
        threshold: Similarity threshold (0.0-1.0)
        
    Returns:
        True if similarity >= threshold
    """
    if not header or not target:
        return False
    
    # Normalize both
    header_norm = header.lower().strip()
    target_norm = target.lower().strip()
    
    # Exact match
    if header_norm == target_norm:
        return True
    
    # Calculate similarity
    max_len = max(len(header_norm), len(target_norm))
    if max_len == 0:
        return False
    
    distance = levenshtein_distance(header_norm, target_norm)
    similarity = 1.0 - (distance / max_len)
    
    return similarity >= threshold


def detect_vin_in_row(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """
    Scan all cells in a row to find a VIN.
    
    Args:
        row: Row dictionary with all cell values
        
    Returns:
        Tuple of (vin_value, cell_key) if found, None otherwise
    """
    for key, value in row.items():
        if value is None:
            continue
        
        value_str = str(value).strip()
        
        # Remove common prefixes/suffixes that might be in the cell
        # e.g., "VIN: HBUSRJGF4CBFPR9BN" or "HBUSRJGF4CBFPR9BN (VIN)"
        value_clean = value_str.upper()
        
        # Try to extract VIN from the value (might have extra text)
        # Look for 17-character alphanumeric sequences
        vin_match = re.search(r'[A-HJ-NPR-Z0-9]{17}', value_clean)
        if vin_match:
            vin_value = vin_match.group(0)
            if VIN_PATTERN.match(vin_value):
                logger.debug(f"[VIN Inference] Found VIN in cell '{key}': {vin_value} (from '{value_str}')")
                return (vin_value, key)
        
        # Also check if the whole value matches
        if VIN_PATTERN.match(value_clean):
            logger.debug(f"[VIN Inference] Found VIN in cell '{key}': {value_clean}")
            return (value_clean, key)
    
    return None


def detect_year_in_row(row: Dict[str, Any]) -> Optional[Tuple[int, str]]:
    """
    Scan all cells in a row to find a valid year (1900-2035).
    Ignores values that look like mileage or weight.
    Handles OCR-extracted text like "Year: 2024 Year: 2020..."
    
    Args:
        row: Row dictionary with all cell values
        
    Returns:
        Tuple of (year_value, cell_key) if found, None otherwise
    """
    for key, value in row.items():
        if value is None:
            continue
        
        value_str = str(value).strip()
        if not value_str:
            continue
        
        # First, try exact match (simple 4-digit number)
        if YEAR_PATTERN.match(value_str):
            try:
                year = int(value_str)
                if 1900 <= year <= 2035:
                    logger.debug(f"[Year Inference] Found year in cell '{key}': {year}")
                    return (year, key)
            except (ValueError, TypeError):
                pass
        
        # Try to extract year from OCR text like "Year: 2024" or "2024 Year"
        # Look for 4-digit year patterns in the text
        year_matches = re.findall(r'\b(19[0-9]{2}|20[0-2][0-9]|203[0-5])\b', value_str)
        for year_str in year_matches:
            try:
                year = int(year_str)
                if 1900 <= year <= 2035:
                    # Prefer years that appear after "Year:" or "year"
                    if re.search(r'year\s*:?\s*' + year_str, value_str, re.IGNORECASE):
                        logger.debug(f"[Year Inference] Found year in cell '{key}': {year} (from OCR text)")
                        return (year, key)
                    # Otherwise, take the first valid year found
                    logger.debug(f"[Year Inference] Found year in cell '{key}': {year} (from OCR text)")
                    return (year, key)
            except (ValueError, TypeError):
                continue
    
    return None


def detect_make_model_in_row(row: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    """
    Scan all cells to detect make and model using known make list.
    Handles OCR-extracted text like "Make: Toyota Make: Ford..." or "Model: Camry Model: F-150..."
    
    Args:
        row: Row dictionary with all cell values
        
    Returns:
        Tuple of (make, model, cell_key) if found, None otherwise
    """
    for key, value in row.items():
        if value is None:
            continue
        
        value_str = str(value).strip()
        if not value_str:
            continue
        
        # Check for exact make match
        value_lower = value_str.lower()
        for known_make in KNOWN_MAKES:
            if value_lower == known_make:
                logger.debug(f"[Make Inference] Found make in cell '{key}': {known_make}")
                return (known_make.title(), None, key)
        
        # Check for "Make Model" pattern (e.g., "Toyota Camry")
        words = value_str.split()
        if len(words) >= 2:
            first_word = words[0].lower()
            if first_word in KNOWN_MAKES:
                make = first_word.title()
                model = ' '.join(words[1:])
                logger.debug(f"[Make/Model Inference] Found in cell '{key}': make={make}, model={model}")
                return (make, model, key)
        
        # Handle OCR format: "Make: Toyota Make: Ford..." or "Model: Camry Model: F-150..."
        # Try to extract first make/model from patterns like "Make: Toyota" or "Model: Camry"
        make_match = re.search(r'Make\s*:?\s*([A-Za-z]+)', value_str, re.IGNORECASE)
        model_match = re.search(r'Model\s*:?\s*([A-Za-z0-9\s\-]+?)(?:\s+Model\s*:|$)', value_str, re.IGNORECASE)
        
        if make_match:
            make_candidate = make_match.group(1).strip().lower()
            if make_candidate in KNOWN_MAKES:
                make = make_candidate.title()
                model = None
                
                # Try to find corresponding model
                if model_match:
                    model_candidate = model_match.group(1).strip()
                    # Check if model appears after the make
                    make_pos = value_str.lower().find(f"make: {make_candidate}")
                    model_pos = value_str.lower().find("model:")
                    if model_pos > make_pos or model_pos == -1:
                        model = model_candidate
                
                logger.debug(f"[Make/Model Inference] Found in OCR text '{key}': make={make}, model={model}")
                return (make, model, key)
    
    return None


def is_headerless_file(raw_data: List[List[Any]], header_row_index: int = 0) -> bool:
    """
    Detect if the file appears to be headerless (first row looks like data).
    
    Args:
        raw_data: 2D list of raw data
        header_row_index: Index of supposed header row
        
    Returns:
        True if file appears headerless
    """
    if not raw_data or len(raw_data) <= header_row_index:
        return False
    
    header_row = raw_data[header_row_index]
    
    # Check if header row looks like data:
    # - All values are numbers
    # - Contains VIN-like strings
    # - No text that looks like column names
    
    numeric_count = 0
    vin_like_count = 0
    text_like_count = 0
    
    for cell in header_row:
        if cell is None:
            continue
        
        cell_str = str(cell).strip()
        if not cell_str:
            continue
        
        # Check if numeric
        try:
            float(cell_str)
            numeric_count += 1
            continue
        except (ValueError, TypeError):
            pass
        
        # Check if VIN-like
        if VIN_PATTERN.match(cell_str.upper()):
            vin_like_count += 1
            continue
        
        # Check if text-like (contains letters and looks like a header)
        if any(c.isalpha() for c in cell_str) and len(cell_str) > 2:
            # Check for common header words
            cell_lower = cell_str.lower()
            header_keywords = ['vin', 'make', 'model', 'year', 'color', 'mileage', 
                             'trim', 'body', 'fuel', 'transmission', 'email', 'owner']
            if any(keyword in cell_lower for keyword in header_keywords):
                text_like_count += 1
    
    total_cells = len([c for c in header_row if c is not None and str(c).strip()])
    
    if total_cells == 0:
        return False
    
    # If most cells are numeric or VIN-like, and few are text-like, it's probably headerless
    data_like_ratio = (numeric_count + vin_like_count) / total_cells if total_cells > 0 else 0
    text_like_ratio = text_like_count / total_cells if total_cells > 0 else 0
    
    is_headerless = data_like_ratio > 0.7 and text_like_ratio < 0.3
    
    if is_headerless:
        logger.debug(f"[Headerless Detection] File appears headerless: numeric/VIN-like={data_like_ratio:.2f}, text-like={text_like_ratio:.2f}")
    
    return is_headerless


def apply_fallback_inference(
    row: Dict[str, Any],
    row_result: Dict[str, Any],
    raw_row_values: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Apply fallback inference to extract fields that mapping missed.
    
    Only infers fields that are currently None in row_result.
    
    Args:
        row: Row dictionary (may have numeric keys if headerless)
        row_result: Current row result with mapped fields
        raw_row_values: Optional raw row values for headerless files
        
    Returns:
        Updated row_result with inferred fields
    """
    inferred_fields = {}
    
    # VIN inference
    if row_result.get("vin") is None or row_result.get("vin") == "":
        vin_result = detect_vin_in_row(row)
        if vin_result:
            vin_value, cell_key = vin_result
            inferred_fields["vin"] = vin_value
            logger.debug(f"[Fallback Inference] Inferred vin={vin_value} from cell '{cell_key}'")
    
    # Year inference
    if row_result.get("year") is None:
        year_result = detect_year_in_row(row)
        if year_result:
            year_value, cell_key = year_result
            inferred_fields["year"] = year_value
            logger.debug(f"[Fallback Inference] Inferred year={year_value} from cell '{cell_key}'")
    
    # Make/Model inference
    if row_result.get("make") is None or row_result.get("make") == "":
        make_model_result = detect_make_model_in_row(row)
        if make_model_result:
            make, model, cell_key = make_model_result
            if make:
                inferred_fields["make"] = make
                logger.debug(f"[Fallback Inference] Inferred make={make} from cell '{cell_key}'")
            if model and (row_result.get("model") is None or row_result.get("model") == ""):
                inferred_fields["model"] = model
                logger.debug(f"[Fallback Inference] Inferred model={model} from cell '{cell_key}'")
    
    # Update row_result with inferred fields
    for field, value in inferred_fields.items():
        if row_result.get(field) is None or row_result.get(field) == "":
            row_result[field] = value
    
    return row_result
