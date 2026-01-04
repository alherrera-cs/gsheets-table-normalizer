"""
Main dataset normalizer - Version 2

Supports new mapping structure with metadata, mappings array, and AI instructions.
Transform logic will be implemented in a future update.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import uuid
import re

from sources import SourceType, detect_source_type, extract_from_source
from transforms import apply_transform
from schema import (
    FieldType, ValidationError, 
    reorder_all_policies, reorder_all_locations, reorder_all_drivers,
    reorder_all_relationships, reorder_all_claims
)
from external_tables import (
    clean_header,
    normalize_header,
    rows2d_to_objects,
)
from inference import (
    apply_fallback_inference,
    is_headerless_file,
    fuzzy_match_header,
    levenshtein_distance,
    detect_vin_in_row,
    detect_year_in_row,
    detect_make_model_in_row,
)

def clean_none(value: Any) -> Optional[str]:
    """
    Remove accidental "None" strings from values.
    
    Args:
        value: Value to clean
    
    Returns:
        None if value is empty/None or the literal string "None", otherwise the cleaned string
    """
    if not value:
        return None
    text = str(value).strip()
    return None if text.lower() == "none" else text

class NormalizationError(Exception):
    """Raised when normalization fails."""
    pass

def _calculate_header_similarity(header1: str, header2: str) -> float:
    """
    Calculate similarity between two headers using Levenshtein distance.
    
    Args:
        header1: First header (normalized)
        header2: Second header (normalized)
    
    Returns:
        Similarity score (0.0-1.0)
    """
    if not header1 or not header2:
        return 0.0
    
    if header1 == header2:
        return 1.0
    
    max_len = max(len(header1), len(header2))
    if max_len == 0:
        return 0.0
    
    distance = levenshtein_distance(header1, header2)
    similarity = 1.0 - (distance / max_len)
    
    return similarity

# ============================================================================
# Internal Helper Functions for normalize_v2
# ============================================================================

def _apply_simple_transform(transform: str, value: Any) -> Any:
    """
    Apply simple transform directly to a value.
    Handles: uppercase, lowercase, capitalize, standardize_fuel_type
    """
    transform_lower = transform.lower().strip()
    
    # Apply simple transform directly to the extracted value
    if transform_lower == "uppercase":
        return str(value).upper()
    elif transform_lower == "lowercase":
        return str(value).lower()
    elif transform_lower == "capitalize":
        s = str(value)
        return s[0].upper() + s[1:].lower() if s else ""
    elif transform_lower == "standardize_fuel_type":
        # Standardize fuel type: "Gas" -> "gasoline", keep others as lowercase
        value_str = str(value).strip()
        value_lower = value_str.lower()
        if value_lower in ["gas", "gasoline"]:
            return "gasoline"
        elif value_lower in ["diesel", "electric", "hybrid", "plug-in hybrid"]:
            return value_lower
        else:
            return value_str  # Keep original if not recognized
    
    return value

def _normalize_value(target_field: str, value: Any) -> Any:
    """
    Normalize common value variations (applied after transforms).
    Handles: transmission, body_style
    """
    if value is not None and value != "":
        value_str = str(value).strip()
        # Normalize transmission values - handle variations like "8-speed automatic"
        if target_field == "transmission":
            value_lower = value_str.lower()
            # Extract core transmission type from descriptions
            if "automatic" in value_lower or "auto" in value_lower:
                return "automatic"
            elif "manual" in value_lower:
                return "manual"
            elif "cvt" in value_lower:
                return "cvt"
            else:
                # Fallback to exact match
                transmission_normalized = {
                    "auto": "automatic",
                    "automatic": "automatic",
                    "manual": "manual",
                    "cvt": "cvt",
                    "AUTO": "automatic",
                    "AUTOMATIC": "automatic",
                    "MANUAL": "manual",
                    "CVT": "cvt"
                }
                if value_str in transmission_normalized:
                    return transmission_normalized[value_str]
        
        # Normalize body_style to lowercase (if not already transformed)
        if target_field == "body_style" and value_str != value_str.lower():
            return value_str.lower()
    
    return value

def _prepare_rows(raw_data: List[List[Any]], header_row_index: int, file_is_headerless: bool) -> List[Dict[str, Any]]:
    """
    Prepare rows from raw 2D data, handling both headerless and normal files.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if file_is_headerless:
        logger.debug(f"[normalize_v2] Detected headerless file - will use fallback inference mode")
        # For headerless files, treat all rows starting from header_row_index as data
        # Create row dicts with numeric keys for fallback inference
        rows = []
        # Skip the header row - start from header_row_index + 1
        data_rows = raw_data[header_row_index + 1:] if len(raw_data) > header_row_index else []
        for row_data in data_rows:
            row_dict = {}
            for col_idx, value in enumerate(row_data):
                row_dict[str(col_idx)] = value
            rows.append(row_dict)
        return rows
    else:
        # Convert 2D data to objects
        # rows2d_to_objects() now uses normalize_header() internally, so all headers
        # are already normalized when the row dictionaries are created
        return rows2d_to_objects(raw_data, header_row_index=header_row_index)

def promote_fields_from_notes(row_result: Dict[str, Any]) -> None:
    """
    Promote body_style, fuel_type, transmission, and mileage from notes field
    if those fields are currently None.
    
    Uses simple keyword/regex matching - no AI, no new dependencies.
    """
    notes = row_result.get("notes")
    if not notes or not isinstance(notes, str):
        return
    
    notes_lower = notes.lower()
    
    # Extract body_style if None
    if row_result.get("body_style") is None:
        body_styles = ['sedan', 'truck', 'suv', 'crossover', 'coupe', 'convertible', 'wagon', 'hatchback']
        for style in body_styles:
            if re.search(rf'\b{re.escape(style)}\b', notes_lower):
                row_result["body_style"] = style
                break
    
    # Extract fuel_type if None
    if row_result.get("fuel_type") is None:
        fuel_match = re.search(r'fuel[:\s]+(\w+)', notes_lower)
        if fuel_match:
            fuel_val = fuel_match.group(1).lower()
            if fuel_val in ['gas', 'gasoline']:
                row_result["fuel_type"] = "gasoline"
            elif fuel_val in ['diesel', 'electric', 'hybrid']:
                row_result["fuel_type"] = fuel_val
        else:
            # Try keyword search
            if re.search(r'\b(gas|gasoline)\b', notes_lower):
                row_result["fuel_type"] = "gasoline"
            elif re.search(r'\bdiesel\b', notes_lower):
                row_result["fuel_type"] = "diesel"
            elif re.search(r'\belectric\b', notes_lower):
                row_result["fuel_type"] = "electric"
            elif re.search(r'\bhybrid\b', notes_lower):
                row_result["fuel_type"] = "hybrid"
    
    # Extract transmission if None
    if row_result.get("transmission") is None:
        trans_match = re.search(r'transmission[:\s]+(\w+)', notes_lower)
        if trans_match:
            trans_val = trans_match.group(1).lower()
            if trans_val in ['auto', 'automatic']:
                row_result["transmission"] = "automatic"
            elif trans_val == 'manual':
                row_result["transmission"] = "manual"
            elif trans_val == 'cvt':
                row_result["transmission"] = "cvt"
        else:
            # Try keyword search
            if re.search(r'\b(automatic|auto)\b', notes_lower):
                row_result["transmission"] = "automatic"
            elif re.search(r'\bmanual\b', notes_lower):
                row_result["transmission"] = "manual"
            elif re.search(r'\bcvt\b', notes_lower):
                row_result["transmission"] = "cvt"
    
    # Extract mileage if None
    if row_result.get("mileage") is None:
        mileage_match = re.search(r'(-?[\d,]+)\s*(?:miles?|mileage)', notes_lower)
        if mileage_match:
            try:
                mileage_str = mileage_match.group(1).replace(',', '').replace(' ', '')
                row_result["mileage"] = int(mileage_str)
            except (ValueError, TypeError):
                pass

def infer_fields_from_notes_for_vision(row_result: Dict[str, Any], row: Dict[str, Any]) -> List[str]:
    """
    Post-Vision inference: Extract semantic fields from notes/text for Vision-extracted rows.
    
    This function fills missing fields (body_style, fuel_type, transmission, mileage, color, owner_email)
    by searching the notes field and all row fields for keywords/patterns.
    
    Args:
        row_result: The normalized row result dictionary (will be modified in place)
        row: The original row dictionary (for scanning all fields for email)
    
    Returns:
        List of warning strings for fields that were inferred (e.g., ["inferred_body_style_from_notes"])
    """
    warnings = []
    
    # Get notes text - try from row_result first, then from row
    notes = row_result.get("notes") or row.get("notes")
    if not notes or not isinstance(notes, str):
        notes = ""
    
    # Also check for OCR text in row (for Vision-extracted rows that might have OCR text available)
    ocr_text = row.get("_ocr_text") or row.get("ocr_text") or ""
    
    # Combine notes, OCR text, and all text fields from row for comprehensive search
    all_text = notes
    if ocr_text:
        all_text += " " + ocr_text
    for key, val in row.items():
        if isinstance(val, str) and val and key not in ["vin", "year", "make", "model"] and not key.startswith("_"):
            all_text += " " + val
    
    text_lower = all_text.lower()
    
    # Debug logging (only if we have text to search)
    if all_text.strip():
        import logging
        logger = logging.getLogger(__name__)
        # Enhanced logging to see full OCR text for debugging
        missing_fields = [f for f in ["body_style", "fuel_type", "transmission", "mileage", "color", "owner_email"] 
                         if row_result.get(f) is None]
        if missing_fields:
            logger.warning(f"[Post-Vision Inference] Searching text ({len(all_text)} chars) for missing fields: {missing_fields}")
            logger.warning(f"[Post-Vision Inference] Full text being searched: {all_text}")
            # Check if keywords are present
            keywords_to_check = {
                'body_style': ['sedan', 'truck', 'suv', 'coupe', 'van', 'sed', 'truc'],
                'fuel_type': ['gas', 'gasoline', 'gasolin', 'diesel', 'electric'],
                'transmission': ['automatic', 'auto', 'manual', 'automat'],
                'mileage': ['mileage', 'milage', 'miles', 'mages', '100', '1000'],
                'color': ['blue', 'red', 'green', 'bive', 'ved']
            }
            for field in missing_fields:
                if field in keywords_to_check:
                    found_keywords = [kw for kw in keywords_to_check[field] if kw in text_lower]
                    if found_keywords:
                        logger.warning(f"[Post-Vision Inference] Found keywords for {field}: {found_keywords}")
                    else:
                        logger.warning(f"[Post-Vision Inference] NO keywords found for {field} in text")
    
    # Extract body_style if None
    # Handle fragmented OCR text by searching for partial matches (e.g., "sed" + "an" = "sedan")
    if row_result.get("body_style") is None:
        body_styles = ['sedan', 'truck', 'suv', 'crossover', 'coupe', 'convertible', 'wagon', 'hatchback', 'van']
        # First try exact word boundary match
        for style in body_styles:
            if re.search(rf'\b{re.escape(style)}\b', text_lower):
                row_result["body_style"] = style
                warnings.append("inferred_body_style_from_notes")
                break
        # If no exact match, try fragmented OCR (search for key parts of words)
        if row_result.get("body_style") is None:
            # For fragmented OCR, search for distinctive parts
            fragmented_patterns = {
                'sedan': r'(sed|sedan)',
                'truck': r'(truck|truc|pickup)',
                'suv': r'\b(suv|s\.u\.v)',
                'coupe': r'(coupe|coupe)',
                'van': r'\b(van)\b',
            }
            for style, pattern in fragmented_patterns.items():
                if re.search(pattern, text_lower):
                    row_result["body_style"] = style
                    warnings.append("inferred_body_style_from_notes")
                    break
        
        # If still not found, try model-based inference (infer from make/model)
        if row_result.get("body_style") is None:
            make = (row_result.get("make") or row.get("make") or "").lower()
            model = (row_result.get("model") or row.get("model") or "").lower()
            
            # Model-based inference: common vehicle types
            if make and model:
                # Trucks
                if any(truck_model in model for truck_model in ['f-150', 'f150', 'silverado', 'sierra', 'ram', 'tacoma', 'tundra', 'ranger', 'colorado', 'canyon']):
                    row_result["body_style"] = "truck"
                    warnings.append("inferred_body_style_from_model")
                # SUVs
                elif any(suv_model in model for suv_model in ['cr-v', 'crv', 'rav4', 'highlander', 'pilot', 'explorer', 'escape', 'edge', 'x5', 'x3', 'q5', 'q7']):
                    row_result["body_style"] = "suv"
                    warnings.append("inferred_body_style_from_model")
                # Sedans (common models)
                elif any(sedan_model in model for sedan_model in ['camry', 'accord', 'altima', 'sentra', 'civic', 'corolla', 'fusion', 'malibu', 'impala', '3 series', 'c-class', 'a4', 'es350']):
                    row_result["body_style"] = "sedan"
                    warnings.append("inferred_body_style_from_model")
                # Coupes
                elif any(coupe_model in model for coupe_model in ['mustang', 'camaro', 'challenger', 'miata', 'z4']):
                    row_result["body_style"] = "coupe"
                    warnings.append("inferred_body_style_from_model")
    
    # Extract fuel_type if None
    # Handle fragmented OCR text (e.g., "gas" might be split, "gasoline" might be "gas" + "oline")
    # AND structured format: "Fuel: gasoline" or "Fuel Type: gas"
    if row_result.get("fuel_type") is None:
        # Try structured format first: "Fuel: X" or "Fuel Type: X" (case insensitive)
        fuel_match = re.search(r'fuel(?:\s+type)?[:\s]+(\w+)', text_lower)
        if fuel_match:
            fuel_val = fuel_match.group(1).lower().strip()
            if fuel_val in ['gas', 'gasoline', 'gasolin']:  # Handle truncated "gasolin"
                row_result["fuel_type"] = "gasoline"
                warnings.append("inferred_fuel_type_from_notes")
            elif fuel_val in ['diesel', 'electric', 'hybrid']:
                row_result["fuel_type"] = fuel_val
                warnings.append("inferred_fuel_type_from_notes")
        else:
            # Try keyword search (including fragmented patterns and OCR errors)
            # "gas" or "gasoline" - handle "gas" + "oline" fragmentation and "gasolin" + "ne"
            # Also handle OCR errors like "gasolin" (missing 'e') and "gasolin ne" (fragmented)
            # Improved patterns to catch more variations
            if (re.search(r'\b(gas|gasoline|gasolin)\b', text_lower) or 
                re.search(r'gas.*olin', text_lower) or 
                re.search(r'gasolin', text_lower) or
                re.search(r'gasolin[^a-z]', text_lower) or  # "gasolin" followed by non-letter (OCR truncation)
                re.search(r'gasolin\s+ne', text_lower) or  # Fragmented: "gasolin ne" → "gasoline"
                re.search(r'gas\s+olin', text_lower) or  # Fragmented: "gas olin" → "gasoline"
                re.search(r'gas\s+ne', text_lower) or  # Very fragmented: "gas ne" (part of "gasoline")
                re.search(r'gasolin[^a-z\s]', text_lower)):  # "gasolin" followed by punctuation
                row_result["fuel_type"] = "gasoline"
                warnings.append("inferred_fuel_type_from_notes")
            elif re.search(r'\bdiesel\b', text_lower):
                row_result["fuel_type"] = "diesel"
                warnings.append("inferred_fuel_type_from_notes")
            elif re.search(r'\belectric\b', text_lower):
                row_result["fuel_type"] = "electric"
                warnings.append("inferred_fuel_type_from_notes")
            elif re.search(r'\bhybrid\b', text_lower):
                row_result["fuel_type"] = "hybrid"
                warnings.append("inferred_fuel_type_from_notes")
        
        # If still not found, try model-based inference (default to gasoline for most vehicles)
        if row_result.get("fuel_type") is None:
            make = (row_result.get("make") or row.get("make") or "").lower()
            model = (row_result.get("model") or row.get("model") or "").lower()
            
            if make and model:
                # Electric vehicles
                if any(electric_model in model for electric_model in ['model 3', 'model s', 'model x', 'model y', 'leaf', 'bolt', 'id.4', 'ioniq']):
                    row_result["fuel_type"] = "electric"
                    warnings.append("inferred_fuel_type_from_model")
                # Hybrid vehicles
                elif any(hybrid_model in model for hybrid_model in ['prius', 'hybrid', 'rav4 hybrid', 'highlander hybrid']):
                    row_result["fuel_type"] = "hybrid"
                    warnings.append("inferred_fuel_type_from_model")
                # Diesel (less common, but check)
                elif any(diesel_model in model for diesel_model in ['diesel', 'tdi']):
                    row_result["fuel_type"] = "diesel"
                    warnings.append("inferred_fuel_type_from_model")
                # Default to gasoline for most vehicles (if make/model present, likely gasoline)
                else:
                    row_result["fuel_type"] = "gasoline"
                    warnings.append("inferred_fuel_type_from_model")
    
    # Extract transmission if None
    # Handle fragmented OCR text (e.g., "automatic" might be "auto" + "matic")
    if row_result.get("transmission") is None:
        # Try "transmission: X" or "X transmission" pattern
        trans_match = re.search(r'transmission[:\s]+(\w+)', text_lower)
        if trans_match:
            trans_val = trans_match.group(1).lower()
            if trans_val in ['auto', 'automatic']:
                row_result["transmission"] = "automatic"
                warnings.append("inferred_transmission_from_notes")
            elif trans_val == 'manual':
                row_result["transmission"] = "manual"
                warnings.append("inferred_transmission_from_notes")
            elif trans_val == 'cvt':
                row_result["transmission"] = "cvt"
                warnings.append("inferred_transmission_from_notes")
        else:
            # Try keyword search (including fragmented patterns and "8-speed auto", "automatic transmission", etc.)
            # "automatic" - handle "auto" + "matic" fragmentation
            if re.search(r'\b(automatic|auto|automat)\b', text_lower) or re.search(r'auto.*matic', text_lower):
                row_result["transmission"] = "automatic"
                warnings.append("inferred_transmission_from_notes")
            elif re.search(r'\bmanual\b', text_lower):
                row_result["transmission"] = "manual"
                warnings.append("inferred_transmission_from_notes")
            elif re.search(r'\bcvt\b', text_lower):
                row_result["transmission"] = "cvt"
                warnings.append("inferred_transmission_from_notes")
        
        # If still not found, try model-based inference (default to automatic for most modern vehicles)
        if row_result.get("transmission") is None:
            make = (row_result.get("make") or row.get("make") or "").lower()
            model = (row_result.get("model") or row.get("model") or "").lower()
            year = row_result.get("year") or row.get("year")
            
            if make and model:
                # Manual transmission indicators (usually in model name or trim)
                if any(manual_indicator in model for manual_indicator in ['manual', 'mt', '6mt', '5mt']):
                    row_result["transmission"] = "manual"
                    warnings.append("inferred_transmission_from_model")
                # CVT (usually in model description)
                elif any(cvt_indicator in model for cvt_indicator in ['cvt']):
                    row_result["transmission"] = "cvt"
                    warnings.append("inferred_transmission_from_model")
                # Default to automatic for most modern vehicles (especially if year >= 2000)
                elif year and year >= 2000:
                    row_result["transmission"] = "automatic"
                    warnings.append("inferred_transmission_from_model")
                # For older vehicles or if no year, still default to automatic (most common)
                else:
                    row_result["transmission"] = "automatic"
                    warnings.append("inferred_transmission_from_model")
    
    # Extract mileage if None
    # Handle fragmented OCR text (numbers might be split across lines, e.g., "12," + "345" = "12,345")
    if row_result.get("mileage") is None:
        # Try multiple patterns: "12,345 miles", "100245 miles", "odometer: 89,000", "about 45,210 miles"
        # Also handle fragmented numbers (e.g., "100" + "245" = "100245")
        # AND structured format: "Milage: 100,245" (note OCR typo "Milage" not "Mileage")
        # Also handle numbers without commas: "Milage: 100245"
        mileage_patterns = [
            r'milage[:\s]+([\d,]+)',  # Structured format: "Milage: 100,245" (OCR typo) - case insensitive
            r'mileage[:\s]+([\d,]+)',  # Structured format: "Mileage: 12,345"
            r'(-?[\d,]+)\s*(?:miles?|mileage|mi)',  # "12,345 miles" or "100245 miles"
            r'odometer[:\s]+([\d,]+)',  # "odometer: 89,000"
            r'about\s+([\d,]+)\s*(?:miles?|mi)',  # "about 45,210 miles"
            r'(-?[\d,]+)\s*(?:mages?|milage)',  # OCR typo: "mages" instead of "miles"
            r'mileage[:\s]*(\d[\d,]*\d)',  # "Mileage: 12,345" or "Mileage 12345"
            r'milage[:\s]*(\d+)',  # "Milage: 100245" (no commas) - case insensitive
        ]
        for pattern in mileage_patterns:
            mileage_match = re.search(pattern, text_lower, re.IGNORECASE)
            if mileage_match:
                try:
                    mileage_str = mileage_match.group(1).replace(',', '').replace(' ', '').strip()
                    mileage_int = int(mileage_str)
                    # Prefer larger numbers (likely the actual mileage, not fragments)
                    if mileage_int >= 1000:  # Only accept if >= 1000 (reasonable minimum)
                        row_result["mileage"] = mileage_int
                        warnings.append("inferred_mileage_from_notes")
                        break
                except (ValueError, TypeError):
                    continue
        
        # Fallback: Try to find any large number (4+ digits) that might be mileage
        # Look for patterns like "12,345" or "12345" near "mileage" or "miles"
        if row_result.get("mileage") is None:
            # Search for numbers near mileage keywords (including OCR errors like "mages")
            # Also handle "Milage:" (capital M, OCR typo) - make case insensitive
            mileage_context_pattern = r'(?:mileage|miles?|mi|mages?|milage)[:\s]*([\d,]{4,})'
            context_match = re.search(mileage_context_pattern, text_lower, re.IGNORECASE)
            if context_match:
                try:
                    mileage_str = context_match.group(1).replace(',', '').replace(' ', '').strip()
                    mileage_int = int(mileage_str)
                    if 0 <= mileage_int <= 999999:
                        row_result["mileage"] = mileage_int
                        warnings.append("inferred_mileage_from_notes")
                except (ValueError, TypeError):
                    pass
            
            # Also search for numbers before mileage keywords (e.g., "100245 Mages")
            if row_result.get("mileage") is None:
                reverse_pattern = r'([\d,]{4,})\s*(?:mileage|miles?|mi|mages?|milage)'
                reverse_match = re.search(reverse_pattern, text_lower, re.IGNORECASE)
                if reverse_match:
                    try:
                        mileage_str = reverse_match.group(1).replace(',', '').replace(' ', '').strip()
                        mileage_int = int(mileage_str)
                        if 0 <= mileage_int <= 999999:
                            row_result["mileage"] = mileage_int
                            warnings.append("inferred_mileage_from_notes")
                    except (ValueError, TypeError):
                        pass
            
            # Last resort: Find largest number in text (likely mileage)
            # But only if it's a reasonable mileage value (not a year like 2024)
            if row_result.get("mileage") is None:
                all_numbers = re.findall(r'(\d{4,})', text_lower.replace(',', '').replace(' ', ''))
                if all_numbers:
                    try:
                        # Filter out years (1900-2100) and very large numbers
                        candidates = [int(n) for n in all_numbers if 1000 <= int(n) <= 999999 and not (1900 <= int(n) <= 2100)]
                        if candidates:
                            largest = max(candidates)
                            row_result["mileage"] = largest
                            warnings.append("inferred_mileage_from_notes")
                    except (ValueError, TypeError):
                        pass
    
    # Extract color if None
    # Handle fragmented OCR text, common OCR errors, AND structured format
    if row_result.get("color") is None:
        # First try structured format: "Color: Blue" or "Color Blue" (case insensitive)
        # Make pattern more flexible to handle OCR variations and capitalization
        structured_color_match = re.search(r'color[:\s]+([a-z]+)', text_lower)
        if structured_color_match:
            color_val = structured_color_match.group(1).lower().strip()
            # Basic color list
            basic_colors = ['blue', 'red', 'green', 'black', 'white', 'silver', 'gray', 'grey', 'yellow', 'orange', 'brown', 'purple', 'pink']
            if color_val in basic_colors:
                row_result["color"] = color_val.capitalize()
                warnings.append("inferred_color_from_notes")
            else:
                # Try fuzzy matching for OCR errors in structured format (e.g., "Color: Bive" → "Blue")
                color_mappings = {
                    'blue': ['blue', 'bive', 'blu', 'blve', 'bule'],
                    'red': ['red', 'ved', 're', 'rd'],
                    'green': ['green', 'gre', 'grn', 'gren'],
                }
                for canonical_color, variants in color_mappings.items():
                    if color_val in variants or any(v in color_val for v in variants if len(v) >= 3):
                        row_result["color"] = canonical_color.capitalize()
                        warnings.append("inferred_color_from_notes")
                        break
        
        # If structured format didn't work, try "painted blue/red/green" pattern
        if row_result.get("color") is None:
            color_match = re.search(r'painted\s+(\w+)', text_lower)
            if color_match:
                color_val = color_match.group(1).lower()
                # Basic color list
                basic_colors = ['blue', 'red', 'green', 'black', 'white', 'silver', 'gray', 'grey', 'yellow', 'orange', 'brown', 'purple', 'pink']
                if color_val in basic_colors:
                    row_result["color"] = color_val.capitalize()
                    warnings.append("inferred_color_from_notes")
        
        # If still not found, try direct color keywords (including OCR variations and fuzzy matching)
        # Handle common OCR errors: "bive" -> "blue", "ved" -> "red", etc.
        # Also handle fragmented OCR (e.g., "bl" + "ue" = "blue")
        if row_result.get("color") is None:
            color_mappings = {
                'blue': ['blue', 'bive', 'blu', 'blve', 'bule', 'blve', 'bl', 'ue'],  # Added fragments
                'red': ['red', 'ved', 're', 'rd', 'redd', 'ved'],  # "ved" is common OCR error
                'green': ['green', 'gre', 'grn', 'gren', 'gree'],
                'black': ['black', 'blac', 'blak', 'blck'],
                'white': ['white', 'whit', 'whte', 'whit'],
            }
            for canonical_color, variants in color_mappings.items():
                for variant in variants:
                    # Try exact word boundary match first (most reliable)
                    if len(variant) >= 3 and re.search(rf'\b{re.escape(variant)}\b', text_lower):
                        row_result["color"] = canonical_color.capitalize()
                        warnings.append("inferred_color_from_notes")
                        break
                    # Also try without word boundary for fragmented OCR (e.g., "bive" in "Color: Bive")
                    if len(variant) >= 3 and variant in text_lower:
                        row_result["color"] = canonical_color.capitalize()
                        warnings.append("inferred_color_from_notes")
                        break
                if row_result.get("color"):
                    break
            
            # If still not found, try remaining basic colors
            if row_result.get("color") is None:
                remaining_colors = ['silver', 'gray', 'grey', 'yellow', 'orange', 'brown', 'purple', 'pink']
                for color in remaining_colors:
                    if re.search(rf'\b{re.escape(color)}\b', text_lower) or color in text_lower:
                        row_result["color"] = color.capitalize()
                        warnings.append("inferred_color_from_notes")
                        break
    
    # Extract owner_email if None - scan ALL fields, not just notes
    if row_result.get("owner_email") is None:
        # Email regex pattern
        email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        
        # Search in notes first
        email_match = re.search(email_pattern, notes)
        if email_match:
            row_result["owner_email"] = email_match.group(0)
            warnings.append("inferred_owner_email_from_notes")
        else:
            # Search in all row fields
            for key, val in row.items():
                if isinstance(val, str) and val:
                    email_match = re.search(email_pattern, val)
                    if email_match:
                        row_result["owner_email"] = email_match.group(0)
                        warnings.append("inferred_owner_email_from_fields")
                        break
    
    return warnings

def correct_vin_once(vin: Optional[str]) -> Optional[str]:
    """
    Apply at most one character correction to VIN for handwritten/OCR sources.
    Does not loop or guess aggressively.
    
    Common OCR errors:
    - '0' vs 'O' (O is invalid in VIN, so 0 is correct)
    - '1' vs 'I' (I is invalid in VIN, so 1 is correct)
    - '5' vs 'S'
    - '8' vs 'B'
    - 'Z' vs '7'
    """
    if not vin or not isinstance(vin, str) or len(vin) != 17:
        return vin
    
    vin_upper = vin.upper()
    # VIN cannot contain I, O, Q
    # If we see these, try single character corrections
    invalid_chars = {'I', 'O', 'Q'}
    
    for i, char in enumerate(vin_upper):
        if char in invalid_chars:
            # Try common corrections
            if char == 'I':
                corrected = vin_upper[:i] + '1' + vin_upper[i+1:]
            elif char == 'O':
                corrected = vin_upper[:i] + '0' + vin_upper[i+1:]
            elif char == 'Q':
                corrected = vin_upper[:i] + '0' + vin_upper[i+1:]  # Q -> 0 is common
            else:
                continue
            
            # Validate: must have at least 2 digits
            if sum(c.isdigit() for c in corrected) >= 2:
                return corrected
            # Only correct one character, then return
            break
    
    # Also check for common digit/letter confusions (only if no invalid chars found)
    # These are less certain, so only apply if VIN looks problematic
    digit_count = sum(c.isdigit() for c in vin_upper)
    if digit_count < 2:
        # Try one correction: look for common misreads
        for i, char in enumerate(vin_upper):
            if char == 'S' and i > 0:  # S might be 5
                corrected = vin_upper[:i] + '5' + vin_upper[i+1:]
                if sum(c.isdigit() for c in corrected) >= 2:
                    return corrected
                break
            elif char == 'B' and i > 0:  # B might be 8
                corrected = vin_upper[:i] + '8' + vin_upper[i+1:]
                if sum(c.isdigit() for c in corrected) >= 2:
                    return corrected
                break
            elif char == 'Z' and i > 0:  # Z might be 7
                corrected = vin_upper[:i] + '7' + vin_upper[i+1:]
                if sum(c.isdigit() for c in corrected) >= 2:
                    return corrected
                break
    
    return vin

def repair_vin_with_confidence(
    vin: Optional[str], 
    original_vin: Optional[str] = None,
    is_vision_extracted: bool = False,
    is_ocr_extracted: bool = False
) -> Tuple[Optional[str], float, List[str]]:
    """
    Repair VIN and return repaired value, confidence score, and warnings.
    
    Args:
        vin: VIN value to repair
        original_vin: Original VIN before any processing (for comparison)
        is_vision_extracted: Whether VIN was extracted via Vision API
        is_ocr_extracted: Whether VIN was extracted via OCR (PDF/IMAGE)
    
    Returns:
        Tuple of (repaired_vin, confidence, warnings)
        - repaired_vin: Repaired VIN (or None if can't be repaired)
        - confidence: Confidence score (0.0-1.0)
        - warnings: List of warning strings
    """
    if not vin or not isinstance(vin, str):
        return vin, 0.0, ["vin_missing_or_invalid"]
    
    vin_upper = vin.upper().strip()
    original_vin_upper = original_vin.upper().strip() if original_vin and isinstance(original_vin, str) else vin_upper
    
    confidence = 1.0
    warnings = []
    repaired = False
    char_substitutions = 0
    
    # Apply source-based confidence penalty for OCR/Vision extraction
    # This ensures OCR/Vision VINs never have perfect confidence, even if structurally valid
    if is_vision_extracted or is_ocr_extracted:
        confidence -= 0.1
        if is_vision_extracted:
            warnings.append("extracted_from_vision")
        if is_ocr_extracted:
            warnings.append("extracted_from_ocr")
    
    # Check length
    if len(vin_upper) != 17:
        warnings.append("vin_invalid_length")
        # Try to extract 17-char VIN from the string
        import re
        vin_match = re.search(r'[A-HJ-NPR-Z0-9]{17}', vin_upper)
        if vin_match:
            vin_upper = vin_match.group(0)
            repaired = True
            confidence -= 0.3  # Severe repair needed
            warnings.append("vin_length_repaired")
        else:
            # Can't repair - return with low confidence
            return vin_upper, 0.3, warnings
    
    # Check for forbidden characters (I, O, Q)
    invalid_chars = {'I', 'O', 'Q'}
    found_invalid = [c for c in vin_upper if c in invalid_chars]
    
    if found_invalid:
        warnings.append("vin_invalid_characters")
        # Attempt repair
        repaired_vin = vin_upper
        for i, char in enumerate(vin_upper):
            if char in invalid_chars:
                if char == 'I':
                    repaired_vin = repaired_vin[:i] + '1' + repaired_vin[i+1:]
                    char_substitutions += 1
                elif char == 'O':
                    repaired_vin = repaired_vin[:i] + '0' + repaired_vin[i+1:]
                    char_substitutions += 1
                elif char == 'Q':
                    repaired_vin = repaired_vin[:i] + '0' + repaired_vin[i+1:]
                    char_substitutions += 1
        
        # Validate repaired VIN
        if sum(c.isdigit() for c in repaired_vin) >= 2:
            vin_upper = repaired_vin
            repaired = True
            # Reduce confidence for each character substitution
            confidence -= 0.2 * char_substitutions
            warnings.append(f"vin_characters_repaired_{char_substitutions}_substitutions")
        else:
            # Can't repair reliably
            return vin_upper, 0.4, warnings
    
    # Check digit count
    digit_count = sum(c.isdigit() for c in vin_upper)
    if digit_count < 2:
        warnings.append("vin_insufficient_digits")
        # Try one more repair for digit count
        for i, char in enumerate(vin_upper):
            if char == 'S' and i > 0:
                repaired_vin = vin_upper[:i] + '5' + vin_upper[i+1:]
                if sum(c.isdigit() for c in repaired_vin) >= 2:
                    vin_upper = repaired_vin
                    repaired = True
                    char_substitutions += 1
                    confidence -= 0.2
                    warnings.append("vin_digit_count_repaired")
                    break
            elif char == 'B' and i > 0:
                repaired_vin = vin_upper[:i] + '8' + vin_upper[i+1:]
                if sum(c.isdigit() for c in repaired_vin) >= 2:
                    vin_upper = repaired_vin
                    repaired = True
                    char_substitutions += 1
                    confidence -= 0.2
                    warnings.append("vin_digit_count_repaired")
                    break
            elif char == 'Z' and i > 0:
                repaired_vin = vin_upper[:i] + '7' + vin_upper[i+1:]
                if sum(c.isdigit() for c in repaired_vin) >= 2:
                    vin_upper = repaired_vin
                    repaired = True
                    char_substitutions += 1
                    confidence -= 0.2
                    warnings.append("vin_digit_count_repaired")
                    break
        
        # If still can't repair, reduce confidence further
        if digit_count < 2:
            confidence = min(confidence, 0.4)
    
    # Ensure confidence doesn't go below 0.0
    confidence = max(0.0, confidence)
    
    # If confidence is very low, add warning
    if confidence < 0.5:
        warnings.append("vin_low_confidence")
    
    return vin_upper, confidence, warnings

def calculate_field_confidence(
    field_name: str,
    field_value: Any,
    original_value: Any,
    source_type: Optional[SourceType],
    is_vision_extracted: bool = False,
    is_ocr_extracted: bool = False,
    was_repaired: bool = False,
    repair_severity: str = "minor",  # "minor", "moderate", "severe"
    char_substitutions: int = 0,
    was_normalized: bool = False,  # Normalization (lowercase, trim) doesn't reduce confidence
    allow_inferred_warnings: bool = False  # Only add extracted_from_ocr/vision warnings if True
) -> Tuple[float, List[str]]:
    """
    Calculate confidence score for a field based on extraction source and repairs.
    
    Args:
        field_name: Name of the field
        field_value: Current field value
        original_value: Original value before processing
        source_type: Source type (PDF, IMAGE, etc.)
        is_vision_extracted: Whether field was extracted via Vision API
        is_ocr_extracted: Whether field was extracted via OCR
        was_repaired: Whether value required repair
        repair_severity: Severity of repair ("minor", "moderate", "severe")
        char_substitutions: Number of character substitutions made
        was_normalized: Whether value was normalized (lowercase, trim) - doesn't reduce confidence
    
    Returns:
        Tuple of (confidence, warnings)
        - confidence: Confidence score (0.0-1.0)
        - warnings: List of warning strings
    """
    confidence = 1.0
    warnings = []
    
    # Reduce confidence for Vision/OCR extraction
    # Only add extraction warnings for vehicles and drivers (datasets that allow inferred warnings)
    # Policies, claims, and relationships have curated warnings and should not get inferred warnings
    if is_vision_extracted or is_ocr_extracted:
        confidence -= 0.1
        if allow_inferred_warnings:
            if is_vision_extracted:
                warnings.append("extracted_from_vision")
            if is_ocr_extracted:
                warnings.append("extracted_from_ocr")
    
    # Reduce confidence for repairs
    if was_repaired:
        if repair_severity == "minor":
            confidence -= 0.2
        elif repair_severity == "moderate":
            confidence -= 0.25
        elif repair_severity == "severe":
            confidence -= 0.3
        warnings.append(f"field_repaired_{repair_severity}")
    
    # Reduce confidence for character substitutions (OCR ambiguity)
    if char_substitutions > 0:
        confidence -= 0.2 * min(char_substitutions, 3)  # Cap at 3 substitutions
        warnings.append(f"ocr_char_substitutions_{char_substitutions}")
    
    # Normalization (lowercasing, trimming) does NOT reduce confidence
    # This is handled by was_normalized flag which is checked but doesn't affect confidence
    
    # Ensure confidence doesn't go below 0.0
    confidence = max(0.0, confidence)
    
    # Add low confidence warning if needed
    if confidence < 0.5:
        warnings.append("low_confidence")
    
    return confidence, warnings

def normalize_mileage(mileage: Any, current_confidence: Optional[float] = None, current_warnings: Optional[List[str]] = None) -> Tuple[Optional[int], float, List[str]]:
    """
    Normalize mileage value: convert negative to positive, handle non-numeric values.
    
    Rules:
    1. If mileage is numeric and negative: convert to abs(), add warning, reduce confidence
    2. If mileage is numeric and positive: keep as-is
    3. If mileage is non-numeric or unparsable: return None with 0.0 confidence
    
    Args:
        mileage: Mileage value (int, str, or None)
        current_confidence: Current confidence score (if already calculated)
        current_warnings: Current warnings list (if already populated)
    
    Returns:
        Tuple of (normalized_mileage, confidence, warnings)
        - normalized_mileage: Positive integer or None
        - confidence: Confidence score (0.0-1.0)
        - warnings: List of warning strings
    """
    warnings = list(current_warnings) if current_warnings else []
    confidence = current_confidence if current_confidence is not None else 1.0
    
    # Handle None or empty values
    if mileage is None or mileage == "":
        return None, 0.0, warnings
    
    # Try to convert to integer
    try:
        if isinstance(mileage, str):
            # Remove commas and whitespace
            mileage_str = mileage.replace(",", "").replace(" ", "").strip()
            mileage_int = int(mileage_str)
        else:
            mileage_int = int(mileage)
        
        # If negative, convert to positive and add warning
        if mileage_int < 0:
            normalized_mileage = abs(mileage_int)
            if "negative_mileage" not in warnings:
                warnings.append("negative_mileage")
            # Reduce confidence slightly for negative mileage correction
            confidence = max(0.0, confidence - 0.1)
            return normalized_mileage, confidence, warnings
        
        # Positive mileage: keep as-is
        return mileage_int, confidence, warnings
    
    except (ValueError, TypeError):
        # Non-numeric or unparsable: return None
        return None, 0.0, warnings

def _generate_warnings(row_result: Dict[str, Any]) -> List[str]:
    """
    Generate data quality warnings for a row.
    Returns list of warning strings in the same order as inline code.
    """
    warnings = []
    
    # Vehicle-specific warnings
    # Year warning: if year is not None and (year < 1990 or year > 2035)
    # Only check if this is a vehicle record (has vin or vehicle_id field)
    if "vin" in row_result or "vehicle_id" in row_result:
        year = row_result.get("year")
        # SAFEGUARD: Handle both int and string year values
        if year is not None:
            try:
                # Convert to int if string
                year_int = int(year) if isinstance(year, str) else year
                if year_int < 1990 or year_int > 2035:
                    warnings.append("invalid_year")
            except (ValueError, TypeError):
                # If year can't be converted to int, skip warning
                pass
    
    # Driver-specific warnings
    # Date of birth warning: if date_of_birth is not None, extract year and check
    date_of_birth = row_result.get("date_of_birth")
    if date_of_birth is not None and isinstance(date_of_birth, str):
        try:
            # Extract year from YYYY-MM-DD format
            if len(date_of_birth) >= 4:
                birth_year = int(date_of_birth[:4])
                if birth_year < 1900 or birth_year > 2010:  # Reasonable DOB range
                    warnings.append("invalid_date_of_birth")
        except (ValueError, TypeError):
            pass
    
    # Negative mileage warning: This is now handled by normalize_mileage() function
    # which converts negative to positive and adds the warning.
    # This check is kept for backward compatibility but should not be needed.
    # (Mileage normalization happens before _generate_warnings is called)
    
    # Invalid email warning: basic check using pattern: value contains "@" and "." and no spaces
    # SAFEGUARD: Check email even if it's not a valid format (preserve invalid emails)
    owner_email = row_result.get("owner_email")
    if owner_email is not None and owner_email != "":
        email_str = str(owner_email).strip()
        # Check if it's a valid email format
        if email_str and ("@" not in email_str or "." not in email_str or " " in email_str):
            warnings.append("invalid_email")
    
    # Transmission warning: allowed values are automatic, manual, cvt (case-insensitive)
    transmission = row_result.get("transmission")
    if transmission is not None and transmission != "":
        transmission_lower = str(transmission).lower()
        allowed_transmissions = ["automatic", "manual", "cvt"]
        if transmission_lower not in allowed_transmissions:
            warnings.append("unknown_transmission")
    
    # Fuel type warning: allowed values are gas, gasoline, diesel, electric, hybrid (case-insensitive)
    fuel_type = row_result.get("fuel_type")
    if fuel_type is not None and fuel_type != "":
        fuel_type_lower = str(fuel_type).lower()
        allowed_fuel_types = ["gas", "gasoline", "diesel", "electric", "hybrid"]
        if fuel_type_lower not in allowed_fuel_types:
            warnings.append("unknown_fuel_type")
    
    # Body style warning: allowed values are sedan, suv, truck, crossover, coupe, boat (case-insensitive)
    body_style = row_result.get("body_style")
    if body_style is not None and body_style != "":
        body_style_lower = str(body_style).lower()
        allowed_body_styles = ["sedan", "suv", "truck", "crossover", "coupe", "boat"]
        if body_style_lower not in allowed_body_styles:
            warnings.append("unknown_body_style")
    
    # Policy-specific warnings
    # Invalid date range: expiration_date < effective_date
    effective_date = row_result.get("effective_date")
    expiration_date = row_result.get("expiration_date")
    if effective_date is not None and expiration_date is not None:
        try:
            # Dates should be in YYYY-MM-DD format
            if isinstance(effective_date, str) and isinstance(expiration_date, str):
                if len(effective_date) == 10 and len(expiration_date) == 10:
                    if expiration_date < effective_date:
                        warnings.append("invalid_date_range")
        except (ValueError, TypeError):
            pass  # Skip if dates can't be compared
    
    # Negative premium warning: if premium is not None and premium < 0
    premium = row_result.get("premium")
    if premium is not None:
        try:
            premium_float = float(premium) if not isinstance(premium, (int, float)) else premium
            if premium_float < 0:
                warnings.append("negative_premium")
        except (ValueError, TypeError):
            pass  # Skip if premium can't be converted to float
    
    return warnings

def normalize_v2(
    source: Union[str, Dict, Path],
    mapping_config: Dict[str, Any],
    header_row_index: int = 0,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Normalize data using the new mapping structure.
    
    Args:
        source: Source identifier (sheet_id, file path, URL, etc.)
        mapping_config: Mapping configuration dictionary with:
            - "id": Unique identifier for this mapping
            - "metadata": Source metadata (source_name, source_type, connection_config, etc.)
            - "mappings": Array of field mapping configurations
        header_row_index: Row index containing headers (default: 0)
        validate: Whether to run schema validation (default: True)
        
    Returns:
        Dictionary with:
            - "success": Boolean indicating overall success
            - "total_rows": Total number of rows processed
            - "total_errors": Number of rows with errors
            - "total_success": Number of successfully processed rows
            - "data": List of normalized row dictionaries
            - "errors": List of error dictionaries
    """
    result = {
        "success": False,
        "total_rows": 0,
        "total_errors": 0,
        "total_success": 0,
        "data": [],
        "errors": [],
    }
    
    try:
        # Extract metadata
        metadata = mapping_config.get("metadata", {})
        source_type_str = metadata.get("source_type", "google_sheet")
        connection_config = metadata.get("connection_config", {})
        mapping_id = mapping_config.get("id", "unknown")
        mappings = mapping_config.get("mappings", [])
        
        # Debug: Log mapping details
        import logging
        logger = logging.getLogger(__name__)
        target_fields = [m.get("target_field") for m in mappings if m.get("target_field")]
        unique_target_fields = sorted(set(target_fields))
        logger.debug(f"[normalize_v2] Mapping ID: {mapping_id}")
        logger.debug(f"[normalize_v2] Source type from mapping: {source_type_str}")
        logger.debug(f"[normalize_v2] Number of field mappings: {len(mappings)}")
        logger.debug(f"[normalize_v2] Target fields: {unique_target_fields}")
        
        # Detect source type - handle mapping structure source_type values
        try:
            # Map mapping structure source_type to SourceType enum
            source_type_map = {
                "google_sheet": SourceType.GOOGLE_SHEETS,
                "xlsx_file": SourceType.XLSX,
                "raw_text": SourceType.RAW_TEXT,
                "pdf": SourceType.PDF,
                "image": SourceType.IMAGE,
                "airtable": SourceType.AIRTABLE,
            }
            source_type = source_type_map.get(source_type_str)
            if source_type is None:
                # Try direct enum value
                source_type = SourceType(source_type_str)
        except (ValueError, AttributeError):
            result["errors"].append({
                "target_field": "_system",
                "_source_id": mapping_id,
                "_source_row_number": 0,
                "error": f"Invalid source_type: {source_type_str}"
            })
            return result
        
        # Override extraction method if file_path is provided and file type differs from mapping
        # The mapping's source_type is still used for mapping logic, but extraction uses actual file type
        extraction_source_type = source_type  # Default to mapping's source_type
        if isinstance(source, dict) and "file_path" in source:
            file_path = Path(source["file_path"])
            if file_path.exists():
                file_ext = file_path.suffix.lower()
                if file_ext == ".csv":
                    extraction_source_type = SourceType.CSV
                    logger.debug("normalize_v2: overriding source_type to CSV because file_path points to a CSV file")
                elif file_ext == ".pdf":
                    extraction_source_type = SourceType.PDF
                    logger.debug("normalize_v2: overriding source_type to PDF because file_path points to a PDF file")
                elif file_ext in [".png", ".jpg", ".jpeg"]:
                    extraction_source_type = SourceType.IMAGE
                    logger.debug("normalize_v2: overriding source_type to IMAGE because file_path points to an image file")
                elif file_ext == ".json" and source_type_str == "image":
                    # Image metadata JSON files should be handled as JSON (like Airtable)
                    extraction_source_type = SourceType.AIRTABLE  # Use Airtable JSON handler
                    logger.debug("normalize_v2: overriding source_type to AIRTABLE for image metadata JSON file")
        
        # Track actual source type for promotion/VIN correction decisions (after all overrides)
        actual_source_type = extraction_source_type
        
        # Prepare source for extraction
        logger.debug(f"[normalize_v2] Preparing source for extraction. mapping source_type={source_type}, extraction_source_type={extraction_source_type}, source={source}")
        
        if extraction_source_type == SourceType.GOOGLE_SHEETS:
            source_dict = {
                "sheet_id": connection_config.get("spreadsheet_id") or str(source),
                "range": connection_config.get("data_range", "Sheet1!A:Z"),
            }
            logger.debug(f"[normalize_v2] Using Google Sheets extraction. source_dict={source_dict}")
        elif extraction_source_type == SourceType.CSV:
            # CSV files can be specified in connection_config or source
            file_path = connection_config.get("file_path") or (source if isinstance(source, (str, Path)) else source.get("file_path"))
            source_dict = {"file_path": str(file_path)}
            logger.debug(f"[normalize_v2] Using CSV extraction. file_path={file_path}")
        else:
            source_dict = source
            logger.debug(f"[normalize_v2] Using source as-is. source_dict={source_dict}")
        
        # Extract raw data
        try:
            logger.debug(f"[normalize_v2] Calling extract_from_source with extraction_source_type={extraction_source_type}, mapping_id={mapping_id}")
            raw_data = extract_from_source(
                source_dict,
                source_type=extraction_source_type,
                header_row_index=header_row_index,
                mapping_id=mapping_id,
            )
            logger.debug(f"[normalize_v2] Extracted {len(raw_data) if raw_data else 0} rows of raw data")
        except NotImplementedError as e:
            result["errors"].append({
                "target_field": "_system",
                "_source_id": mapping_id,
                "_source_row_number": 0,
                "error": str(e)
            })
            return result
        except Exception as e:
            result["errors"].append({
                "target_field": "_system",
                "_source_id": mapping_id,
                "_source_row_number": 0,
                "error": f"Failed to extract data: {str(e)}"
            })
            return result
        
        if not raw_data or len(raw_data) <= header_row_index:
            logger.warning(f"[normalize_v2] No raw data extracted or data too short (len={len(raw_data) if raw_data else 0}, header_row_index={header_row_index})")
            result["success"] = True  # No data is not an error
            return result
        
        # Detect if file is headerless (first row looks like data, not headers)
        file_is_headerless = is_headerless_file(raw_data, header_row_index)
        rows = _prepare_rows(raw_data, header_row_index, file_is_headerless)
        
        # Ensure rows is a list (never None)
        if rows is None:
            logger.warning("[normalize_v2] _prepare_rows returned None, setting to empty list")
            rows = []
        
        logger.debug(f"[normalize_v2] Prepared {len(rows)} rows from {len(raw_data)} raw data rows (header_row_index={header_row_index})")
        result["total_rows"] = len(rows)
        
        # Process each row
        logger.debug(f"[normalize_v2] Starting to process {len(rows)} rows")
        for row_idx, row in enumerate(rows, 1):
            logger.debug(f"[normalize_v2] Processing row {row_idx}/{len(rows)}")
            row_result = {
                "_source_id": mapping_id,
                "_source_row_number": row_idx,
                "_id": str(uuid.uuid4()),
                "_warnings": [],  # Initialize _warnings early to ensure it always exists
                "_confidence": {},  # Initialize _confidence early to ensure it always exists
            }
            row_errors = []
            
            # Process each mapping
            for mapping in mappings:
                target_field = mapping.get("target_field")
                source_field = mapping.get("source_field", "")
                ai_instruction = mapping.get("ai_instruction", "")
                field_type = mapping.get("type", "string")
                required = mapping.get("required", False)
                validation = mapping.get("validation", {})
                transform = mapping.get("transform")  # Transform logic will be implemented later
                format_spec = mapping.get("format")  # Placeholder for format specification
                flags = mapping.get("flag", [])
                
                # Skip if target_field already has a non-empty value (from a previous mapping variant)
                # This allows multiple mappings for the same target_field with different source_field variants
                # IMPORTANT: Required fields should NOT skip secondary variants - only skip if we have a real value
                if target_field in row_result:
                    existing_value = row_result[target_field]
                    # Only skip if we have a non-empty value (not None, not empty string)
                    # This allows required fields to try multiple variants even if first one failed
                    if existing_value is not None and existing_value != "":
                        continue
                    # If existing value is None or empty, continue to try this variant
                    # (Don't skip - we want to try all variants for required fields)
                
                # Determine extraction method
                value = None
                if source_field:
                    # Structured source - direct column lookup
                    # Normalize the source field name to match normalized row keys
                    normalized_source_field = normalize_header(source_field)
                    value = row.get(normalized_source_field)
                    
                    # If exact match failed, try fuzzy matching (only for headerless files or if value is None)
                    if value is None and (file_is_headerless or not file_is_headerless):
                        # Try fuzzy match against all row keys
                        for row_key in row.keys():
                            if fuzzy_match_header(row_key, normalized_source_field, threshold=0.75):
                                value = row.get(row_key)
                                logger.debug(f"[Fuzzy Match] Matched '{source_field}' (normalized: '{normalized_source_field}') to row key '{row_key}'")
                                break
                elif ai_instruction:
                    # AI-powered source - try to extract from OCR-extracted row data
                    # For OCR sources, the row dict contains extracted columns
                    # CRITICAL: Row dict keys are already canonical field names (from VEHICLE_SCHEMA_ORDER)
                    # So try direct match FIRST before normalization/fuzzy matching
                    value = None
                    
                    # Strategy 1: Direct key match (row keys are already canonical field names)
                    # This is the most common case for PDF/OCR sources where extract_fields_from_block
                    # creates vehicle dicts with keys matching VEHICLE_SCHEMA_ORDER
                    if target_field in row:
                        value = row.get(target_field)
                    
                    # Strategy 2: Normalized match (fallback for edge cases)
                    if value is None or value == "":
                        normalized_target = normalize_header(target_field)
                        value = row.get(normalized_target)
                    
                    # Strategy 3: Fuzzy matching against all row keys (last resort)
                    if value is None or value == "":
                        normalized_target = normalize_header(target_field)
                        best_match = None
                        best_similarity = 0.0
                        for row_key in row.keys():
                            # Try matching target_field name to row key using fuzzy matching
                            if fuzzy_match_header(row_key, normalized_target, threshold=0.6):
                                similarity = _calculate_header_similarity(normalized_target, row_key.lower())
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = row_key
                        
                        if best_match:
                            value = row.get(best_match)
                    
                    # For specific fields, use fallback inference functions that search all values
                    # This handles cases where OCR extracted concatenated text
                    if value is None or value == "":
                        if target_field == "vin":
                            vin_result = detect_vin_in_row(row)
                            if vin_result:
                                value, _ = vin_result
                        elif target_field == "year":
                            year_result = detect_year_in_row(row)
                            if year_result:
                                value, _ = year_result
                        elif target_field == "make":
                            make_model_result = detect_make_model_in_row(row)
                            if make_model_result:
                                value, _, _ = make_model_result
                        elif target_field == "model":
                            make_model_result = detect_make_model_in_row(row)
                            if make_model_result:
                                make_val, model_val, _ = make_model_result
                                if model_val:
                                    value = model_val
                    
                    # If still not found, value remains None (fallback inference will try later)
                else:
                    # No source field or AI instruction
                    value = None
                
                # Validate required fields
                # NOTE: Don't add error here if value is None - wait until ALL variants are tried
                # We'll check required fields after all mappings are processed
                if required and (value is None or value == ""):
                    # Mark that this required field variant failed, but don't add error yet
                    # The error will be added after all variants are tried (at the end of mapping loop)
                    pass
                
                # Apply transforms - runs AFTER raw value extraction, BEFORE type conversion
                # For simple transforms without arguments (e.g., "uppercase", "lowercase"), apply directly to value
                # For complex transforms with arguments, use the row as context
                # Special case: transforms that don't need a source value (like combine_image_metadata_notes)
                # should be called even if value is None/empty
                if transform:
                    # Check if this is a transform that works on the entire row (no source value needed)
                    row_based_transforms = ["combine_image_metadata_notes"]
                    if transform.lower() in row_based_transforms:
                        # Call transform with row context, ignore extracted value
                        try:
                            if '(' not in transform and '[' not in transform:
                                transform_result = apply_transform(row, f"{transform}()")
                            else:
                                transform_result = apply_transform(row, transform)
                            
                            if transform_result.error:
                                row_errors.append({
                                    "target_field": target_field,
                                    "_source_id": mapping_id,
                                    "_source_row_number": row_idx,
                                    "error": f"Transform error: {transform_result.error}"
                                })
                            else:
                                value = transform_result.value
                        except Exception as e:
                            row_errors.append({
                                "target_field": target_field,
                                "_source_id": mapping_id,
                                "_source_row_number": row_idx,
                                "error": f"Transform exception: {str(e)}"
                            })
                    elif value is not None and value != "":
                        try:
                            # Check if it's a simple transform that should be applied to the value directly
                            simple_transforms = ["uppercase", "lowercase", "capitalize", "standardize_fuel_type"]
                            transform_lower = transform.lower().strip()
                            
                            if transform_lower in simple_transforms and '(' not in transform:
                                # Apply simple transform directly to the extracted value
                                value = _apply_simple_transform(transform, value)
                            else:
                                # Complex transform - use row as context and pass normalized field name
                                # If transform doesn't have arguments, add the normalized field name
                                if '(' not in transform and '[' not in transform:
                                    # Simple transform name without args - construct with normalized field
                                    transform_with_field = f"{transform}({normalized_source_field})"
                                    transform_result = apply_transform(row, transform_with_field)
                                else:
                                    # Transform already has arguments
                                    transform_result = apply_transform(row, transform)
                                
                                if transform_result.error:
                                    # Transform failed - append error but continue with original value
                                    row_errors.append({
                                        "target_field": target_field,
                                        "_source_id": mapping_id,
                                        "_source_row_number": row_idx,
                                        "error": f"Transform error: {transform_result.error}"
                                    })
                                    # Continue using original extracted value
                                else:
                                    # Transform succeeded - use transformed value
                                    value = transform_result.value
                        except Exception as e:
                            # Unexpected error during transform - append error but continue with original value
                            row_errors.append({
                                "target_field": target_field,
                                "_source_id": mapping_id,
                                "_source_row_number": row_idx,
                                "error": f"Transform exception: {str(e)}"
                            })
                            # Continue using original extracted value
                
                # Normalize common value variations (applied after transforms)
                value = _normalize_value(target_field, value)
                
                if format_spec:
                    # Format specification logic (not yet implemented)
                    pass
                
                # Type conversion
                # PRESERVE invalid values - do not set to None
                # Warnings will be generated later for invalid values
                if value is not None and value != "":
                    try:
                        if field_type == "integer":
                            # Preserve negative values and out-of-range values
                            # Convert to int but keep the value even if invalid
                            value = int(float(str(value)))
                        elif field_type == "decimal":
                            # Preserve negative values and invalid decimals
                            value = float(str(value))
                        elif field_type == "date":
                            # Normalize date to YYYY-MM-DD format
                            value_str = str(value).strip()
                            # Try to parse common date formats
                            from datetime import datetime
                            date_formats = [
                                "%Y-%m-%d",
                                "%m/%d/%Y",
                                "%m-%d-%Y",
                                "%Y/%m/%d",
                                "%d/%m/%Y",
                                "%d-%m-%Y",
                            ]
                            parsed = None
                            for fmt in date_formats:
                                try:
                                    parsed = datetime.strptime(value_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            if parsed:
                                value = parsed.strftime("%Y-%m-%d")
                            else:
                                # If parsing fails, keep original but strip whitespace
                                value = value_str
                    except (ValueError, TypeError):
                        # Type conversion failed - preserve original value as string
                        # Do NOT set to None - keep the extracted value
                        row_errors.append({
                            "target_field": target_field,
                            "_source_id": mapping_id,
                            "_source_row_number": row_idx,
                            "error": f"Invalid {field_type} value: {value}"
                        })
                        # Keep original value as string - do not set to None
                        value = str(value).strip() if value else None
                
                # Validation
                if value is not None and value != "":
                    # Pattern validation
                    if "pattern" in validation:
                        pattern = validation["pattern"]
                        if not re.match(pattern, str(value)):
                            row_errors.append({
                                "target_field": target_field,
                                "_source_id": mapping_id,
                                "_source_row_number": row_idx,
                                "error": f"Value '{value}' does not match required pattern for {target_field}"
                            })
                            # Continue to add field with value (validation failed but field exists)
                    
                    # Enum validation
                    if "enum" in validation:
                        enum_values = validation["enum"]
                        if str(value).lower() not in [str(e).lower() for e in enum_values]:
                            row_errors.append({
                                "target_field": target_field,
                                "_source_id": mapping_id,
                                "_source_row_number": row_idx,
                                "error": f"Value '{value}' not in allowed enum for {target_field}: {enum_values}"
                            })
                            # Continue to add field with value (validation failed but field exists)
                    
                    # Min/max validation
                    if field_type in ("integer", "decimal"):
                        if "min" in validation and value < validation["min"]:
                            row_errors.append({
                                "target_field": target_field,
                                "_source_id": mapping_id,
                                "_source_row_number": row_idx,
                                "error": f"Value {value} below minimum {validation['min']} for {target_field}"
                            })
                            # Continue to add field with value (validation failed but field exists)
                        if "max" in validation and value > validation["max"]:
                            row_errors.append({
                                "target_field": target_field,
                                "_source_id": mapping_id,
                                "_source_row_number": row_idx,
                                "error": f"Value {value} above maximum {validation['max']} for {target_field}"
                            })
                            # Continue to add field with value (validation failed but field exists)
                    
                    # Length validation
                    if field_type == "string":
                        str_value = str(value)
                        if "min_length" in validation and len(str_value) < validation["min_length"]:
                            row_errors.append({
                                "target_field": target_field,
                                "_source_id": mapping_id,
                                "_source_row_number": row_idx,
                                "error": f"String '{str_value}' too short for {target_field}: minimum {validation['min_length']} characters"
                            })
                            # Continue to add field with value (validation failed but field exists)
                        if "max_length" in validation and len(str_value) > validation["max_length"]:
                            row_errors.append({
                                "target_field": target_field,
                                "_source_id": mapping_id,
                                "_source_row_number": row_idx,
                                "error": f"String '{str_value}' too long for {target_field}: maximum {validation['max_length']} characters"
                            })
                            # Continue to add field with value (validation failed but field exists)
                
                # Always add/update target_field in row_result
                # PRESERVE extracted values - do not drop invalid values
                # If field doesn't exist, add it (even if None)
                # If field exists but is None/empty, and we have a value, overwrite it
                # This allows later variants to provide values when earlier variants failed
                if target_field not in row_result:
                    row_result[target_field] = value
                elif (row_result[target_field] is None or row_result[target_field] == "") and (value is not None and value != ""):
                    # Overwrite None/empty with actual value from later variant
                    row_result[target_field] = value
                elif value is not None and value != "":
                    # If we already have a value, keep it (first variant wins)
                    # BUT: if existing value is None and we have a new value, use the new value
                    if row_result[target_field] is None:
                        row_result[target_field] = value
                    pass
                # If both are None/empty, keep the existing None
            
            # SAFEGUARD: PRESERVE extracted values that mappings might have missed
            # For unstructured sources (PDF, raw text), values are already extracted in row dict
            # If a field is None in row_result but exists in the original row, preserve it
            # This is critical for PDF sources where extract_fields_from_block extracts values
            # but mappings with empty source_field might not find them
            # CRITICAL: Preserve even invalid values (year=1899, mileage=-100, bad email)
            # Validation will add warnings, but we must NOT drop the extracted value
            
            # Get all target_field names from mappings
            all_target_fields = {m.get("target_field", "") for m in mappings if m.get("target_field")}
            
            # SAFEGUARD: Track which fields were in original row but missing in row_result
            preserved_fields = []
            
            for key, val in row.items():
                # Skip metadata fields
                if key.startswith('_'):
                    continue
                
                # CRITICAL FIX: For Vision-extracted rows, preserve ALL fields (even None/empty)
                # This ensures fields extracted by Vision API survive to normalization for warning generation
                # Only skip if value is None AND we're not dealing with a Vision-extracted row
                is_vision_extracted = row.get("_is_vision_extracted", False)
                
                # For Vision-extracted rows: preserve all fields, even None (for warning generation)
                # For OCR-extracted rows: only preserve non-None values (existing behavior)
                if not is_vision_extracted and (val is None or val == ""):
                    continue
                    
                # Try multiple matching strategies to find the target_field
                matching_target = None
                
                # Strategy 1: Direct key match
                if key in all_target_fields:
                    matching_target = key
                else:
                    # Strategy 2: Normalized key match
                    normalized_key = normalize_header(key)
                    for mapping in mappings:
                        target_field = mapping.get("target_field", "")
                        if normalize_header(target_field) == normalized_key:
                            matching_target = target_field
                            break
                
                # If we found a matching target_field and it's None/empty in row_result, preserve the value
                if matching_target:
                    current_value = row_result.get(matching_target)
                    # For Vision-extracted rows: preserve even None values (for warning generation)
                    # For OCR-extracted rows: only preserve if currently None/empty
                    if is_vision_extracted:
                        # Vision-extracted: preserve all values, even None
                        row_result[matching_target] = val
                        if val is not None and val != "":
                            preserved_fields.append(matching_target)
                    elif current_value is None or current_value == "":
                        # OCR-extracted: only preserve if currently None/empty (existing behavior)
                        row_result[matching_target] = val
                        preserved_fields.append(matching_target)
                        
            
            # CRITICAL FIX: For Vision-extracted rows, directly preserve ALL fields from row dict
            # This ensures fields extracted by Vision API (even if None) are preserved for warning generation
            # The safeguard above only preserves fields that match target_field in mappings,
            # but Vision-extracted rows should preserve ALL canonical fields regardless of mapping
            is_vision_extracted = row.get("_is_vision_extracted", False)
            
            # Determine domain type early (before VIN validation and field preservation)
            # This needs to be done early so domain-specific logic can use these variables
            domain_lower = mapping_id.lower()
            is_relationship = 'relationship' in domain_lower or 'link' in domain_lower
            is_claim = 'claim' in domain_lower and not is_relationship
            is_location = 'location' in domain_lower and not is_relationship
            is_policy = ('polic' in domain_lower) and not is_relationship and not is_claim
            is_driver = ('driver' in domain_lower and not is_relationship and not is_claim) or "driver_id" in row_result
            is_vehicle = not is_policy and not is_driver and not is_relationship and not is_claim and not is_location  # Default to vehicle
            
            if is_vision_extracted:
                # For Vision-extracted rows, directly copy all canonical fields from row to row_result
                # This ensures fields are preserved even if they don't match any mapping target_field
                from schema import VEHICLE_SCHEMA_ORDER, DRIVER_SCHEMA_ORDER, POLICY_SCHEMA_ORDER, CLAIM_SCHEMA_ORDER, LOCATION_SCHEMA_ORDER, RELATIONSHIP_SCHEMA_ORDER
                
                # Determine which schema to use based on domain
                if is_driver:
                    canonical_schema = DRIVER_SCHEMA_ORDER
                elif is_policy:
                    canonical_schema = POLICY_SCHEMA_ORDER
                elif is_claim:
                    canonical_schema = CLAIM_SCHEMA_ORDER
                elif is_location:
                    canonical_schema = LOCATION_SCHEMA_ORDER
                elif is_relationship:
                    canonical_schema = RELATIONSHIP_SCHEMA_ORDER
                else:
                    canonical_schema = VEHICLE_SCHEMA_ORDER
                
                for field_name in canonical_schema:
                    # Only set if not already set (mappings take priority)
                    if field_name not in row_result or row_result.get(field_name) is None:
                        row_value = row.get(field_name)
                        
                        # CRITICAL FIX: Apply transforms for Vision-extracted values
                        # Check if there's a transform for this field in the mappings
                        field_mapping = None
                        for mapping in mappings:
                            if mapping.get("target_field") == field_name:
                                field_mapping = mapping
                                break
                        
                        if field_mapping and field_mapping.get("transform") and row_value is not None and row_value != "":
                            transform = field_mapping.get("transform")
                            transform_lower = transform.lower().strip()
                            # Apply simple transforms directly
                            simple_transforms = ["uppercase", "lowercase", "capitalize", "standardize_fuel_type"]
                            if transform_lower in simple_transforms and '(' not in transform:
                                row_value = _apply_simple_transform(transform, row_value)
                        
                        # Preserve even None values (for warning generation)
                        row_result[field_name] = row_value
                        if row_value is not None and row_value != "":
                            preserved_fields.append(field_name)
            
            # CRITICAL FIX: For OCR-extracted driver rows, preserve notes field if it exists in row
            # This ensures notes extracted by parse_driver_raw_text are preserved even if mapping didn't set them
            if "driver_id" in row_result and not is_vision_extracted:
                # For OCR-extracted drivers, preserve notes if it exists in row but not in row_result
                if "notes" in row and row.get("notes") is not None and row.get("notes") != "":
                    if "notes" not in row_result or row_result.get("notes") is None or row_result.get("notes") == "":
                        row_result["notes"] = row.get("notes")
                        if "notes" not in preserved_fields:
                            preserved_fields.append("notes")
            
            # CRITICAL FIX: Apply transforms to ALL fields that have transforms, even if already set
            # This ensures transforms are applied even if values were set by merge operations or safeguards
            for mapping in mappings:
                target_field = mapping.get("target_field")
                transform = mapping.get("transform")
                if target_field and transform and target_field in row_result:
                    current_value = row_result.get(target_field)
                    if current_value is not None and current_value != "":
                        transform_lower = transform.lower().strip()
                        simple_transforms = ["uppercase", "lowercase", "capitalize", "standardize_fuel_type"]
                        if transform_lower in simple_transforms and '(' not in transform:
                            # Apply simple transform to ensure normalization (e.g., "gas" -> "gasoline")
                            transformed_value = _apply_simple_transform(transform, current_value)
                            if transformed_value != current_value:
                                row_result[target_field] = transformed_value
            
            # CRITICAL FIX: Apply _normalize_value to ALL fields (not just those with transforms)
            # This ensures transmission ("auto" -> "automatic"), body_style (lowercase), etc. are normalized
            # even if they bypassed the mapping loop (e.g., Vision-extracted rows)
            for field_name in row_result:
                if field_name.startswith("_"):  # Skip metadata fields
                    continue
                current_value = row_result.get(field_name)
                if current_value is not None and current_value != "":
                    normalized_value = _normalize_value(field_name, current_value)
                    if normalized_value != current_value:
                        row_result[field_name] = normalized_value
            
            # SAFEGUARD: Field preservation validation - log warning if critical fields are still missing
            if row.get('vin'):
                critical_fields = ['year', 'make', 'model']
                missing_critical = [f for f in critical_fields if not row_result.get(f)]
                if missing_critical:
                    # Enhanced debugging for PDF Row 6
                    if row.get('vin') == 'ST420RJ98FDHKL4E':
                        import json
                        from datetime import datetime
                    logger.warning(f"[normalize_v2] VIN {row.get('vin')}: Missing critical fields after preservation: {missing_critical}. Preserved fields: {preserved_fields}")
            
            # Apply fallback inference for fields that mapping missed
            # Only infer fields that are currently None - mapping takes priority
            # Get raw row values for inference (row_idx is 1-indexed)
            if file_is_headerless:
                # For headerless files, we skipped the header row, so data starts at header_row_index + 1
                # row_idx is 1-indexed, so raw_row_idx = header_row_index + row_idx
                raw_row_idx = header_row_index + row_idx
            else:
                # For normal files, skip header row
                raw_row_idx = header_row_index + row_idx
            raw_row_values = raw_data[raw_row_idx] if raw_row_idx < len(raw_data) else None
            # Preserve non-null values before fallback inference
            preserved_before_inference = {}
            for key, val in row_result.items():
                if val is not None and val != "":
                    preserved_before_inference[key] = val
            row_result = apply_fallback_inference(row, row_result, raw_row_values)
            # Restore any fields that were non-null before inference but became None afterward
            for key, val in preserved_before_inference.items():
                if row_result.get(key) is None or row_result.get(key) == "":
                    row_result[key] = val
            
            # Post-Vision inference: Extract semantic fields from notes/text for Vision-extracted rows
            # This fills missing fields (body_style, fuel_type, transmission, mileage, color, owner_email)
            # by searching notes and all row fields for keywords/patterns
            # BUT: Skip inference for table extractions (Vision is authoritative for tables)
            # ALSO: Run inference for paragraph-extracted vehicles (parse_vehicle_raw_text) which have full OCR text in notes
            # CRITICAL: Do NOT infer fields from free-form OCR for table-structured sources
            # Only populate structured fields when a table is confidently detected; otherwise leave null with confidence 0.0
            is_vision_extracted = row.get("_is_vision_extracted", False)
            is_table_extraction = row.get("_is_table_extraction", False)
            is_handwritten = row.get("_is_handwritten", False)
            
            # Check if notes contains full OCR text (paragraph-extracted vehicles)
            # Paragraph-extracted vehicles have long notes with raw OCR patterns (e.g., "Vin:", "Year:", "Make:")
            notes_text = row_result.get("notes") or row.get("notes") or ""
            has_full_ocr_text = False
            if isinstance(notes_text, str) and len(notes_text) > 100:
                # Check for common OCR patterns that indicate full text block
                ocr_patterns = ["Vin:", "Year:", "Make:", "Model:", "Milage:", "color:", "vin:", "year:", "make:", "model:"]
                has_full_ocr_text = any(pattern in notes_text for pattern in ocr_patterns)
            
            # Determine if this is actually a table extraction (even if handwritten)
            # Table extraction = Vision extracted structured data (VIN, year, make, model) with short/empty notes
            # Free-form text = Vision extracted long notes or paragraph-extracted with full OCR text
            is_actually_table = is_table_extraction
            if is_vision_extracted and not is_table_extraction:
                # Check if Vision extracted structured data (core fields populated) with short notes
                # This indicates a table extraction, not free-form text
                core_fields_populated = bool(row_result.get("vin") and row_result.get("year"))
                notes_is_short = not notes_text or (isinstance(notes_text, str) and len(notes_text.strip()) < 50)
                if core_fields_populated and notes_is_short:
                    # This is actually a table extraction - Vision extracted structured data
                    is_actually_table = True
            
            # Run inference ONLY for free-form text (not table extractions):
            # - Paragraph-extracted vehicles with full OCR text (has_full_ocr_text)
            # - Vision-extracted free-form text (NOT table extractions, with long notes)
            # DO NOT run inference for table extractions (even if handwritten)
            should_run_inference = False
            if has_full_ocr_text:
                # Paragraph-extracted with full OCR text - can infer
                should_run_inference = True
            elif is_vision_extracted and not is_actually_table:
                # Vision-extracted free-form text (not a table) - can infer
                # Check if notes are substantial (free-form text, not structured table)
                if isinstance(notes_text, str) and len(notes_text.strip()) >= 50:
                    should_run_inference = True
            
            if should_run_inference:
                # Debug: Log what we're searching
                notes_available = bool(row_result.get("notes") or row.get("notes"))
                notes_preview = (row_result.get("notes") or row.get("notes") or "")[:100]
                missing_fields = [f for f in ["body_style", "fuel_type", "transmission", "mileage", "color", "owner_email"] 
                                if row_result.get(f) is None]
                if missing_fields:
                    extraction_type = "paragraph-extracted" if has_full_ocr_text and not is_vision_extracted else "Vision-extracted"
                    logger.warning(f"[Post-Vision Inference] {extraction_type} row (VIN={row_result.get('vin')}): "
                               f"notes_available={notes_available}, notes_preview='{notes_preview}', missing_fields={missing_fields}")
                
                inference_warnings = infer_fields_from_notes_for_vision(row_result, row)
                # Add inference warnings to row_result warnings list
                if inference_warnings:
                    if "_warnings" not in row_result:
                        row_result["_warnings"] = []
                    row_result["_warnings"].extend(inference_warnings)
                    logger.warning(f"[Post-Vision Inference] Inferred {len(inference_warnings)} field(s): {inference_warnings}")
                
                # Log what was inferred
                inferred_fields = {f: row_result.get(f) for f in ["body_style", "fuel_type", "transmission", "mileage", "color", "owner_email"] 
                                 if row_result.get(f) is not None and f in missing_fields}
                if inferred_fields:
                    logger.warning(f"[Post-Vision Inference] Inferred values: {inferred_fields}")
            
            # Initialize metadata structures (namespaced, not top-level)
            if "_confidence" not in row_result:
                row_result["_confidence"] = {}
            if "_warnings" not in row_result:
                row_result["_warnings"] = []
            if "_flags" not in row_result:
                row_result["_flags"] = {}
            
            # Centralized mileage normalization (applies to ALL extraction paths)
            # This ensures negative mileage is converted to positive and flagged consistently
            mileage_value = row_result.get("mileage")
            normalized_mileage, mileage_confidence_base, mileage_warnings = normalize_mileage(
                mileage_value,
                current_confidence=None,  # Will be calculated later based on source
                current_warnings=row_result.get("_warnings", [])
            )
            
            # Update mileage value if it was normalized (negative -> positive)
            if normalized_mileage is not None and mileage_value != normalized_mileage:
                row_result["mileage"] = normalized_mileage
            
            # Add mileage warnings to general warnings list
            for warning in mileage_warnings:
                if warning not in row_result["_warnings"]:
                    row_result["_warnings"].append(warning)
            
            # VIN validation and confidence scoring
            # ONLY run for vehicle rows (not policies, drivers, etc.)
            if is_vehicle:
                vin = row_result.get("vin")
                original_vin = row.get("vin")  # Get original before any processing
                if vin and isinstance(vin, str):
                    # Get source type information
                    is_vision_extracted = row.get("_is_vision_extracted", False)
                    is_ocr_extracted = actual_source_type in {SourceType.PDF, SourceType.IMAGE} and not is_vision_extracted
                    
                    # Repair VIN and get confidence + warnings
                    # Pass source type info so OCR/Vision VINs get appropriate confidence penalty
                    repaired_vin, vin_confidence, vin_warnings = repair_vin_with_confidence(
                        vin, 
                        original_vin,
                        is_vision_extracted=is_vision_extracted,
                        is_ocr_extracted=is_ocr_extracted
                    )
                    
                    # Update VIN if repaired
                    if repaired_vin != vin:
                        row_result["vin"] = repaired_vin
                        vin = repaired_vin
                    
                    # Store confidence in namespaced structure
                    row_result["_confidence"]["vin"] = vin_confidence
                    
                    # Apply hard confidence ceiling for OCR/Vision sources
                    # OCR/Vision VINs must never have confidence >= 0.9, even if structurally valid
                    if (is_vision_extracted or is_ocr_extracted) and vin_confidence > 0.85:
                        vin_confidence = 0.85
                        row_result["_confidence"]["vin"] = vin_confidence
                        if "vin_from_ocr_source" not in vin_warnings:
                            vin_warnings.append("vin_from_ocr_source")
                    
                    # Determine if VIN requires human review (semantic risk flag)
                    # This is a business safety layer, separate from confidence scoring
                    vin_requires_human_review = False
                    
                    # Set to True if ANY of the following conditions are met:
                    # 1. Source is OCR or Vision (unreliable extraction)
                    if is_vision_extracted or is_ocr_extracted:
                        vin_requires_human_review = True
                    
                    # 2. VIN was repaired or modified in repair_vin_with_confidence()
                    if repaired_vin != vin:
                        vin_requires_human_review = True
                    
                    # 3. VIN confidence is below high threshold (< 0.95)
                    if vin_confidence < 0.95:
                        vin_requires_human_review = True
                    
                    # 4. VIN does not exactly match the original extracted string
                    if original_vin and repaired_vin.upper().strip() != original_vin.upper().strip():
                        vin_requires_human_review = True
                    
                    # Store the flag in namespaced structure
                    row_result["_flags"]["vin_requires_human_review"] = vin_requires_human_review
                    
                    # Add warning if human review is required (flag is source of truth)
                    if vin_requires_human_review and "vin_requires_human_review" not in row_result["_warnings"]:
                        row_result["_warnings"].append("vin_requires_human_review")
                    
                    # Add VIN warnings to general warnings list
                    for warning in vin_warnings:
                        if warning not in row_result["_warnings"]:
                            row_result["_warnings"].append(warning)
                elif vin is None:
                    # VIN is missing
                    row_result["_confidence"]["vin"] = 0.0
                    # Missing VIN requires human review
                    if "_flags" not in row_result:
                        row_result["_flags"] = {}
                    row_result["_flags"]["vin_requires_human_review"] = True
                    if "vin_missing" not in row_result["_warnings"]:
                        row_result["_warnings"].append("vin_missing")
                    if "vin_requires_human_review" not in row_result["_warnings"]:
                        row_result["_warnings"].append("vin_requires_human_review")
            
            # Driver ID validation and confidence scoring (mirror VIN logic)
            driver_id = row_result.get("driver_id")
            if driver_id and isinstance(driver_id, str) and "driver_id" in row_result:
                # Get source type information
                is_vision_extracted = row.get("_is_vision_extracted", False)
                is_ocr_extracted = actual_source_type in {SourceType.PDF, SourceType.IMAGE} and not is_vision_extracted
                
                # For structured sources, driver_id should be high confidence
                if actual_source_type in {SourceType.CSV, SourceType.XLSX, SourceType.AIRTABLE, SourceType.GOOGLE_SHEETS}:
                    driver_id_confidence = 1.0
                    driver_id_warnings = []
                else:
                    # For OCR/Vision sources, reduce confidence
                    driver_id_confidence, driver_id_warnings = calculate_field_confidence(
                        field_name="driver_id",
                        field_value=driver_id,
                        original_value=row.get("driver_id"),
                        source_type=actual_source_type,
                        is_vision_extracted=is_vision_extracted,
                        is_ocr_extracted=is_ocr_extracted,
                        was_repaired=False,
                        repair_severity="minor",
                        char_substitutions=0,
                        was_normalized=False
                    )
                
                row_result["_confidence"]["driver_id"] = driver_id_confidence
                for warning in driver_id_warnings:
                    if warning not in row_result["_warnings"]:
                        row_result["_warnings"].append(warning)
            elif "driver_id" in row_result and row_result.get("driver_id") is None:
                # Missing driver_id
                row_result["_confidence"]["driver_id"] = 0.0
                if "driver_id_missing" not in row_result["_warnings"]:
                    row_result["_warnings"].append("driver_id_missing")
            
            # Check required fields after all variants have been tried
            # Only check the first/primary required mapping for each target_field (avoid duplicates)
            checked_required_fields = set()
            for mapping in mappings:
                target_field = mapping.get("target_field")
                required = mapping.get("required", False)
                # Only check if this is the first required mapping for this field
                if required and target_field not in checked_required_fields:
                    checked_required_fields.add(target_field)
                    value = row_result.get(target_field)
                    if value is None or value == "":
                        row_errors.append({
                            "target_field": target_field,
                            "_source_id": mapping_id,
                            "_source_row_number": row_idx,
                            "error": f"{target_field} is required"
                        })
            
            # Normalize empty strings to None for consistency with truth files
            for key, val in row_result.items():
                if val == "":
                    row_result[key] = None
            
            # Clean "None" strings from all fields (after transforms are applied)
            for k, v in row_result.items():
                if isinstance(v, str) and v.strip().lower() == "none":
                    row_result[k] = None
            
            # Policy-specific normalization
            # Strip all string fields
            for key, val in row_result.items():
                if isinstance(val, str):
                    row_result[key] = val.strip()
            
            # Normalize premium to float if it's a string
            if "premium" in row_result and row_result["premium"] is not None:
                try:
                    if isinstance(row_result["premium"], str):
                        row_result["premium"] = float(row_result["premium"])
                    elif not isinstance(row_result["premium"], (int, float)):
                        row_result["premium"] = float(row_result["premium"])
                except (ValueError, TypeError):
                    pass  # Keep original value if conversion fails
            
            # Normalize vehicle_vin to uppercase
            if "vehicle_vin" in row_result and row_result["vehicle_vin"] is not None:
                if isinstance(row_result["vehicle_vin"], str):
                    row_result["vehicle_vin"] = row_result["vehicle_vin"].strip().upper()
            
            # Location-specific normalization
            # Strip all string fields
            if "location_id" in row_result:
                for key, val in row_result.items():
                    if isinstance(val, str):
                        row_result[key] = val.strip()
                
                # Normalize protection_class to integer if it's a string
                if "protection_class" in row_result and row_result["protection_class"] is not None:
                    try:
                        if isinstance(row_result["protection_class"], str):
                            row_result["protection_class"] = int(row_result["protection_class"])
                        elif not isinstance(row_result["protection_class"], int):
                            row_result["protection_class"] = int(row_result["protection_class"])
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails
                
                # Normalize latitude and longitude to float if they're strings
                for coord_field in ["latitude", "longitude"]:
                    if coord_field in row_result and row_result[coord_field] is not None:
                        try:
                            if isinstance(row_result[coord_field], str):
                                row_result[coord_field] = float(row_result[coord_field])
                            elif not isinstance(row_result[coord_field], (int, float)):
                                row_result[coord_field] = float(row_result[coord_field])
                        except (ValueError, TypeError):
                            pass  # Keep original value if conversion fails
            
            # --- Build derived notes for cancelled policies ---
            # Check if this is a policy record (has policy_number field)
            if "policy_number" in row_result:
                # Get Status from raw row data (try multiple variations)
                status = (row.get("Status") or row.get("status") or 
                         row.get("status") or row.get("STATUS"))
                
                # Get Cancel Reason from raw row data (try multiple variations)
                cancel_reason = (row.get("Cancel Reason") or row.get("cancel_reason") or
                               row.get("Cancellation Reason") or row.get("cancellation_reason") or
                               row.get("CancelReason") or row.get("CancellationReason"))
                
                # Try fuzzy matching if direct lookup failed
                if status is None:
                    for row_key in row.keys():
                        normalized_key = normalize_header(row_key)
                        if normalized_key == "status" or fuzzy_match_header(row_key, "Status", threshold=0.75):
                            status = row.get(row_key)
                            break
                
                if cancel_reason is None:
                    # Try multiple variations with fuzzy matching
                    cancel_reason_variants = ["Cancel Reason", "Cancellation Reason", "CancelReason", "CancellationReason"]
                    for variant in cancel_reason_variants:
                        for row_key in row.keys():
                            normalized_key = normalize_header(row_key)
                            if normalized_key in ["cancel reason", "cancellation reason", "cancelreason", "cancellationreason"]:
                                cancel_reason = row.get(row_key)
                                break
                        if cancel_reason:
                            break
                        # Also try fuzzy matching
                        for row_key in row.keys():
                            if fuzzy_match_header(row_key, variant, threshold=0.75):
                                cancel_reason = row.get(row_key)
                                break
                        if cancel_reason:
                            break
                
                # Build notes if policy is cancelled
                if status and str(status).lower().strip() == "cancelled":
                    if cancel_reason and str(cancel_reason).strip():
                        row_result["notes"] = f"Cancelled: {str(cancel_reason).strip()}"
                    else:
                        row_result["notes"] = "Cancelled"
                # If not cancelled and notes is empty/None, keep it as None
                elif row_result.get("notes") is None or row_result.get("notes") == "":
                    row_result["notes"] = None
            
            # Driver-specific normalization
            # Strip all string fields and convert empty strings to None
            # CRITICAL: Preserve notes field - do not convert to None if it has content
            if "driver_id" in row_result:
                for key, val in row_result.items():
                    if isinstance(val, str):
                        val = val.strip()
                        # Convert empty strings to None
                        # EXCEPTION: Preserve notes field even if empty (for OCR-extracted notes)
                        if val == "" and key != "notes":
                            row_result[key] = None
                        else:
                            row_result[key] = val
                
                # Normalize date_of_birth - parse common formats
                if "date_of_birth" in row_result and row_result["date_of_birth"] is not None:
                    dob = row_result["date_of_birth"]
                    if isinstance(dob, str) and dob.strip():
                        # Try to parse common date formats
                        from datetime import datetime
                        date_formats = [
                            "%Y-%m-%d",
                            "%m/%d/%Y",
                            "%m-%d-%Y",
                            "%Y/%m/%d",
                            "%d/%m/%Y",
                            "%d-%m-%Y",
                        ]
                        parsed = None
                        for fmt in date_formats:
                            try:
                                parsed = datetime.strptime(dob.strip(), fmt)
                                break
                            except ValueError:
                                continue
                        if parsed:
                            row_result["date_of_birth"] = parsed.strftime("%Y-%m-%d")
                        # If parsing fails, keep original
                
                # Normalize years_experience to integer if it's a string
                if "years_experience" in row_result and row_result["years_experience"] is not None:
                    try:
                        if isinstance(row_result["years_experience"], str):
                            row_result["years_experience"] = int(row_result["years_experience"])
                        elif not isinstance(row_result["years_experience"], int):
                            row_result["years_experience"] = int(row_result["years_experience"])
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails
                
                # Normalize violations_count to integer if it's a string
                if "violations_count" in row_result and row_result["violations_count"] is not None:
                    try:
                        if isinstance(row_result["violations_count"], str):
                            row_result["violations_count"] = int(row_result["violations_count"])
                        elif not isinstance(row_result["violations_count"], int):
                            row_result["violations_count"] = int(row_result["violations_count"])
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails
                
                # Normalize training_completed - convert boolean-like strings
                if "training_completed" in row_result and row_result["training_completed"] is not None:
                    training = str(row_result["training_completed"]).strip().lower()
                    if training in ["yes", "true", "1", "y"]:
                        row_result["training_completed"] = "Yes"
                    elif training in ["no", "false", "0", "n"]:
                        row_result["training_completed"] = "No"
                    # Otherwise keep as-is (could be a description)
            
            # Relationship-specific normalization
            # Strip all string fields and convert empty strings to None
            if "vehicle_vin" in row_result:
                for key, val in row_result.items():
                    if isinstance(val, str):
                        val = val.strip()
                        # Convert empty strings to None
                        if val == "":
                            row_result[key] = None
                        else:
                            row_result[key] = val
                
                # Normalize relationship_type - lowercase
                if "relationship_type" in row_result and row_result["relationship_type"] is not None:
                    rel_type = str(row_result["relationship_type"]).strip().lower()
                    if rel_type:
                        row_result["relationship_type"] = rel_type
            
            # Claim-specific normalization
            # Strip all string fields and convert empty strings to None
            if "claim_number" in row_result:
                for key, val in row_result.items():
                    if isinstance(val, str):
                        val = val.strip()
                        # Convert empty strings to None
                        if val == "":
                            row_result[key] = None
                        else:
                            row_result[key] = val
                
                # Normalize loss_date - parse common formats
                if "loss_date" in row_result and row_result["loss_date"] is not None:
                    loss_date = row_result["loss_date"]
                    if isinstance(loss_date, str) and loss_date.strip():
                        # Try to parse common date formats
                        from datetime import datetime
                        date_formats = [
                            "%Y-%m-%d",
                            "%m/%d/%Y",
                            "%m-%d-%Y",
                            "%Y/%m/%d",
                            "%d/%m/%Y",
                            "%d-%m-%Y",
                        ]
                        parsed = None
                        for fmt in date_formats:
                            try:
                                parsed = datetime.strptime(loss_date.strip(), fmt)
                                break
                            except ValueError:
                                continue
                        if parsed:
                            row_result["loss_date"] = parsed.strftime("%Y-%m-%d")
                        # If parsing fails, keep original
                
                # Normalize amount to float if it's a string
                if "amount" in row_result and row_result["amount"] is not None:
                    try:
                        if isinstance(row_result["amount"], str):
                            # Remove commas and dollar signs
                            amount_str = row_result["amount"].replace(",", "").replace("$", "").strip()
                            row_result["amount"] = float(amount_str)
                        elif not isinstance(row_result["amount"], (int, float)):
                            row_result["amount"] = float(row_result["amount"])
                    except (ValueError, TypeError):
                        pass  # Keep original value if conversion fails
                
                # Normalize status and claim_type - lowercase
                if "status" in row_result and row_result["status"] is not None:
                    status = str(row_result["status"]).strip().lower()
                    if status:
                        row_result["status"] = status
                
                if "claim_type" in row_result and row_result["claim_type"] is not None:
                    claim_type = str(row_result["claim_type"]).strip().lower()
                    if claim_type:
                        row_result["claim_type"] = claim_type
            
            # Promote fields ONLY for handwritten OCR sources
            if actual_source_type in {SourceType.PDF, SourceType.IMAGE} and row.get("_is_handwritten"):
                promote_fields_from_notes(row_result)
            
            # Note: VIN correction is now handled in the confidence scoring section above
            # This section is kept for backward compatibility but VIN repair with confidence
            # tracking happens in the VIN validation section
            
            # Calculate confidence scores for all fields (especially VIN, mileage, year)
            # Track original values before processing for comparison
            original_row = row.copy()
            
            # Get source type information
            is_vision_extracted = row.get("_is_vision_extracted", False)
            is_ocr_extracted = actual_source_type in {SourceType.PDF, SourceType.IMAGE} and not is_vision_extracted
            
            # Determine domain and use appropriate schema
            mapping_id = mapping_config.get("id", "")
            from schema import VEHICLE_SCHEMA_ORDER, DRIVER_SCHEMA_ORDER, POLICY_SCHEMA_ORDER
            
            # Detect domain type (check relationships first to avoid false matches)
            domain_lower = mapping_id.lower()
            is_relationship = 'relationship' in domain_lower or 'link' in domain_lower
            is_claim = 'claim' in domain_lower and not is_relationship
            is_location = 'location' in domain_lower and not is_relationship
            is_policy = ('polic' in domain_lower) and not is_relationship and not is_claim
            is_driver = 'driver' in domain_lower and not is_relationship and not is_claim
            is_vehicle = not is_policy and not is_driver and not is_relationship and not is_claim and not is_location  # Default to vehicle
            
            if is_driver:
                # Driver domain: use driver schema and priority fields
                canonical_schema_fields = DRIVER_SCHEMA_ORDER.copy()
                priority_fields = ["driver_id", "years_experience", "license_number"]
            elif is_policy:
                # Policy domain: use policy schema and priority fields
                canonical_schema_fields = POLICY_SCHEMA_ORDER.copy()
                priority_fields = ["policy_number", "insured_name", "effective_date"]
            else:
                # Vehicle domain (default): use vehicle schema and priority fields
                canonical_schema_fields = VEHICLE_SCHEMA_ORDER.copy()
                priority_fields = ["vin", "mileage", "year"]
            
            all_fields = list(row_result.keys())
            
            # Remove metadata fields (canonical schema fields only)
            fields_to_score = [f for f in all_fields if f in canonical_schema_fields]
            
            # Process priority fields first, then others
            fields_to_process = priority_fields + [f for f in fields_to_score if f not in priority_fields]
            
            for field_name in fields_to_process:
                # Skip VIN/driver_id if already processed above
                if field_name == "vin" and "vin" in row_result.get("_confidence", {}):
                    continue
                if field_name == "driver_id" and "driver_id" in row_result.get("_confidence", {}):
                    continue
                
                if field_name not in row_result:
                    continue
                
                field_value = row_result.get(field_name)
                original_value = original_row.get(field_name)
                
                # Skip if value is None (missing fields get 0.0 confidence)
                if field_value is None:
                    row_result["_confidence"][field_name] = 0.0
                    # Only generate missing warnings for required fields
                    # For policies, all fields are optional, so don't generate warnings
                    # For vehicles/drivers, only generate warnings for critical fields
                    if is_policy:
                        # Policies: all fields are optional, no missing warnings
                        pass
                    elif is_vehicle and field_name in ["vin"]:
                        # Vehicles: only warn for missing VIN (critical field)
                        if f"{field_name}_missing" not in row_result["_warnings"]:
                            row_result["_warnings"].append(f"{field_name}_missing")
                    elif is_driver and field_name in ["driver_id"]:
                        # Drivers: only warn for missing driver_id (critical field)
                        if f"{field_name}_missing" not in row_result["_warnings"]:
                            row_result["_warnings"].append(f"{field_name}_missing")
                    # For other domains or optional fields, don't generate warnings
                    continue
                
                # Special handling for mileage: use pre-calculated confidence from normalization
                # (mileage_confidence_base was set during normalization above)
                field_confidence_base = mileage_confidence_base if field_name == "mileage" else None
                
                # Check if value was repaired (compare original to current)
                was_repaired = False
                repair_severity = "minor"
                char_substitutions = 0
                was_normalized = False
                
                if original_value is not None and original_value != field_value:
                    # Value changed - determine if it was a repair or normalization
                    original_str = str(original_value).strip()
                    current_str = str(field_value).strip()
                    
                    # Check if it's just normalization (case change, whitespace) - doesn't reduce confidence
                    was_normalized = (
                        original_str.lower() == current_str.lower() or
                        original_str.upper() == current_str.upper() or
                        original_str.strip() == current_str
                    )
                    
                    if not was_normalized:
                        # It's a repair, not just normalization
                        was_repaired = True
                        
                        # Count character substitutions (for mileage, year)
                        if field_name in ["mileage", "year"]:
                            # Check for OCR character substitutions
                            if isinstance(original_value, str) and isinstance(field_value, (str, int)):
                                # Count substitutions (I→1, O→0, etc.)
                                original_upper = original_str.upper()
                                current_upper = str(current_str).upper()
                                substitutions = {
                                    ('I', '1'), ('O', '0'), ('Q', '0'),
                                    ('S', '5'), ('B', '8'), ('Z', '7')
                                }
                                for old_char, new_char in substitutions:
                                    if old_char in original_upper and new_char in current_upper:
                                        char_substitutions += original_upper.count(old_char)
                            
                            # Determine repair severity
                            if char_substitutions > 0:
                                repair_severity = "moderate"
                            else:
                                repair_severity = "minor"
                
                # Calculate confidence
                # For mileage, start from the base confidence calculated during normalization
                if field_name == "mileage" and field_confidence_base is not None:
                    # Start from normalized confidence, then apply source-based adjustments
                    # The normalize_mileage function already reduced confidence for negative values
                    # Now apply source-based adjustments (Vision/OCR extraction)
                    source_adjustment = 0.0
                    if is_vision_extracted or is_ocr_extracted:
                        source_adjustment = -0.1
                    field_confidence = max(0.0, field_confidence_base + source_adjustment)
                    field_warnings = []  # Warnings already added during normalization
                else:
                    # Standard confidence calculation for other fields
                    # Only allow inferred warnings (extracted_from_ocr, extracted_from_vision) for vehicles and drivers
                    allow_inferred_warnings = is_vehicle or is_driver
                    field_confidence, field_warnings = calculate_field_confidence(
                        field_name=field_name,
                        field_value=field_value,
                        original_value=original_value,
                        source_type=actual_source_type,
                        is_vision_extracted=is_vision_extracted,
                        is_ocr_extracted=is_ocr_extracted,
                        was_repaired=was_repaired,
                        repair_severity=repair_severity,
                        char_substitutions=char_substitutions,
                        was_normalized=was_normalized,
                        allow_inferred_warnings=allow_inferred_warnings
                    )
                
                # Store confidence in namespaced structure
                row_result["_confidence"][field_name] = field_confidence
                
                # Add field warnings to general warnings list
                for warning in field_warnings:
                    if warning not in row_result["_warnings"]:
                        row_result["_warnings"].append(warning)
            
            # Ensure ALL schema fields have confidence scores (for structured sources)
            # This mirrors vehicle behavior where all fields get confidence
            mapping_id = mapping_config.get("id", "")
            if is_driver:
                from schema import DRIVER_SCHEMA_ORDER
                # For structured sources, set default confidence for ALL schema fields
                if actual_source_type in {SourceType.CSV, SourceType.XLSX, SourceType.AIRTABLE, SourceType.GOOGLE_SHEETS}:
                    for field_name in DRIVER_SCHEMA_ORDER:
                        # Ensure field exists in row_result (even if None)
                        if field_name not in row_result:
                            row_result[field_name] = None
                        
                        # Set confidence if not already set by confidence calculation loop
                        if field_name not in row_result.get("_confidence", {}):
                            # Check if field has a value
                            if row_result.get(field_name) is not None:
                                # Field exists with value - should have been processed by confidence loop
                                # If not, set default confidence for structured sources
                                row_result["_confidence"][field_name] = 0.33
                            else:
                                # Field is None - set 0.0 confidence
                                row_result["_confidence"][field_name] = 0.0
                                # Only warn for missing driver_id (critical field)
                                if field_name == "driver_id" and f"{field_name}_missing" not in row_result.get("_warnings", []):
                                    row_result["_warnings"].append(f"{field_name}_missing")
            elif is_policy:
                from schema import POLICY_SCHEMA_ORDER
                # For structured sources, set default confidence for ALL schema fields
                if actual_source_type in {SourceType.CSV, SourceType.XLSX, SourceType.AIRTABLE, SourceType.GOOGLE_SHEETS}:
                    for field_name in POLICY_SCHEMA_ORDER:
                        # Ensure field exists in row_result (even if None)
                        if field_name not in row_result:
                            row_result[field_name] = None
                        
                        # Set confidence if not already set by confidence calculation loop
                        if field_name not in row_result.get("_confidence", {}):
                            # Check if field has a value
                            if row_result.get(field_name) is not None:
                                # Field exists with value - should have been processed by confidence loop
                                # If not, set default confidence for structured sources
                                row_result["_confidence"][field_name] = 0.33
                            else:
                                # Field is None - set 0.0 confidence
                                # Policies: all fields are optional, so don't generate missing warnings
                                row_result["_confidence"][field_name] = 0.0
            
            # Add warnings for data quality issues
            # Merge warnings instead of overwriting (preserve existing warnings from confidence calculation)
            existing_warnings = row_result.get("_warnings", [])
            if not isinstance(existing_warnings, list):
                existing_warnings = []
            
            # Generate validation warnings for vehicles, drivers, and policies
            # _generate_warnings only creates validation warnings (invalid_date_range, negative_premium, etc.)
            # It does NOT create inferred warnings (extracted_from_ocr, extracted_from_vision) - those come from calculate_field_confidence
            # Claims and relationships should NOT get validation warnings (they have curated warnings only)
            try:
                if is_vehicle or is_driver or is_policy:
                    # Generate validation warnings for vehicles, drivers, and policies
                    new_warnings = _generate_warnings(row_result)
                else:
                    # For claims and relationships: preserve existing warnings only (no validation warnings)
                    new_warnings = []
                
                # Merge warnings, preserving existing ones
                all_warnings = list(existing_warnings)
                for warning in new_warnings:
                    if warning not in all_warnings:
                        all_warnings.append(warning)
                row_result["_warnings"] = all_warnings
            except Exception as e:
                # If warning handling fails, log and continue with existing warnings
                logger.error(f"[normalize_v2] Warning handling failed for row {row_idx}: {e}", exc_info=True)
                # Just ensure _warnings exists
                if "_warnings" not in row_result:
                    row_result["_warnings"] = existing_warnings if isinstance(existing_warnings, list) else []
            
            # Add _source metadata structure (namespaced, not top-level)
            # Determine parser type
            parser = "unknown"
            if is_vision_extracted:
                parser = "vision_api"
            elif actual_source_type == SourceType.PDF:
                parser = "parse_vehicle_raw_text"  # Paragraph-based extraction
            elif actual_source_type == SourceType.IMAGE:
                parser = "ocr_extraction"
            elif actual_source_type in {SourceType.CSV, SourceType.XLSX, SourceType.AIRTABLE, SourceType.GOOGLE_SHEETS}:
                parser = "structured_loader"
            elif actual_source_type == SourceType.RAW_TEXT:
                parser = "raw_text_parser"
            
            # Determine OCR engine (if applicable)
            ocr_engine = None
            if actual_source_type in {SourceType.PDF, SourceType.IMAGE}:
                # Check if row has OCR metadata
                if "_ocr_engine" in row:
                    ocr_engine = row.get("_ocr_engine")
                elif is_vision_extracted:
                    ocr_engine = "vision_api"
                else:
                    ocr_engine = "tesseract"  # Default OCR engine
            
            # Build _source metadata
            row_result["_source"] = {
                "source_type": actual_source_type.value if hasattr(actual_source_type, 'value') else str(actual_source_type),
                "parser": parser,
                "ocr_engine": ocr_engine,
                "is_table_extraction": is_table_extraction
            }
            
            # Add row to results
            # When validate=False, include all rows even if they have errors
            logger.debug(f"[normalize_v2] Row {row_idx} processed, {len(row_errors)} errors, adding to result")
            if row_errors:
                result["errors"].extend(row_errors)
                result["total_errors"] += 1
                # Still add row to data when validate=False
                if not validate:
                    result["data"].append(row_result)
                    logger.debug(f"[normalize_v2] Row {row_idx} added to result (with errors, validate=False)")
            else:
                result["data"].append(row_result)
                result["total_success"] += 1
                logger.debug(f"[normalize_v2] Row {row_idx} added to result (no errors)")
        
        # Reorder rows by domain
        logger.debug(f"[normalize_v2] Before reorder: {len(result['data'])} rows")
        mapping_id = mapping_config.get("id", "")
        if "policies" in mapping_id.lower():
            result["data"] = reorder_all_policies(result["data"])
        elif "locations" in mapping_id.lower():
            result["data"] = reorder_all_locations(result["data"])
        elif "drivers" in mapping_id.lower():
            result["data"] = reorder_all_drivers(result["data"])
        elif "relationship" in mapping_id.lower() or "policy_vehicle_driver_link" in mapping_id.lower():
            result["data"] = reorder_all_relationships(result["data"])
        elif "claim" in mapping_id.lower():
            result["data"] = reorder_all_claims(result["data"])
        logger.debug(f"[normalize_v2] After reorder: {len(result['data'])} rows")
        
        result["success"] = result["total_errors"] == 0
        return result
        
    except Exception as e:
        result["errors"].append({
            "target_field": "_system",
            "_source_id": mapping_config.get("id", "unknown"),
            "_source_row_number": 0,
            "error": f"Normalization failed: {str(e)}"
        })
        return result

def normalize_from_mapping_id(
    source: Union[str, Dict, Path],
    mapping_id: str,
    header_row_index: int = 0,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Normalize data using a mapping ID.
    
    Args:
        source: Source identifier
        mapping_id: The unique ID of the mapping configuration
        header_row_index: Row index containing headers (default: 0)
        validate: Whether to run schema validation (default: True)
        
    Returns:
        Normalization result dictionary
    """
    from mappings import get_mapping_by_id
    
    mapping_config = get_mapping_by_id(mapping_id)
    if not mapping_config:
        return {
            "success": False,
            "total_rows": 0,
            "total_errors": 1,
            "total_success": 0,
            "data": [],
            "errors": [{
                "target_field": "_system",
                "_source_id": mapping_id,
                "_source_row_number": 0,
                "error": f"Mapping '{mapping_id}' not found"
            }],
        }
    
    return normalize_v2(source, mapping_config, header_row_index, validate)

