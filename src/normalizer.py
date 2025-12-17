"""
Main dataset normalizer - Version 2

Supports new mapping structure with metadata, mappings array, and AI instructions.
Transform logic will be implemented in a future update.
"""

from typing import Any, Dict, List, Optional, Union
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
    normalize_values,
    drop_empty_rows,
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
                    # #region agent log
                    if row_result.get('vin') == 'ST420RJ98FDHKL4E':
                        import json
                        from datetime import datetime
                        with open('/Users/alexaherrera/Desktop/table_detector/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H5","location":"normalizer.py:198","message":"PDF Row 6: Warning generated","data":{"year":year,"year_int":year_int,"warnings":warnings},"timestamp":int(datetime.now().timestamp()*1000)}) + '\n')
                    # #endregion
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
    
    # Negative mileage warning: if mileage is not None and mileage < 0
    mileage = row_result.get("mileage")
    if mileage is not None:
        # Handle both int and string mileage values
        try:
            mileage_int = int(mileage) if isinstance(mileage, str) else mileage
            if mileage_int < 0:
                warnings.append("negative_mileage")
        except (ValueError, TypeError):
            # If mileage can't be converted to int, skip warning
            pass
    
    # Invalid email warning: basic check using pattern: value contains "@" and "." and no spaces
    # SAFEGUARD: Check email even if it's not a valid format (preserve invalid emails)
    owner_email = row_result.get("owner_email")
    if owner_email is not None and owner_email != "":
        email_str = str(owner_email).strip()
        # Check if it's a valid email format
        if email_str and ("@" not in email_str or "." not in email_str or " " in email_str):
            warnings.append("invalid_email")
            # #region agent log
            if row_result.get('vin') == 'ST420RJ98FDHKL4E':
                import json
                from datetime import datetime
                with open('/Users/alexaherrera/Desktop/table_detector/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H5","location":"normalizer.py:223","message":"PDF Row 6: Email warning generated","data":{"owner_email":owner_email,"email_str":email_str,"warnings":warnings},"timestamp":int(datetime.now().timestamp()*1000)}) + '\n')
            # #endregion
    
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
            logger.info(f"[DEBUG normalize_v2] Calling extract_from_source with extraction_source_type={extraction_source_type}, mapping_id={mapping_id}")
            logger.debug(f"[normalize_v2] Calling extract_from_source with extraction_source_type={extraction_source_type}")
            raw_data = extract_from_source(
                source_dict,
                source_type=extraction_source_type,
                header_row_index=header_row_index,
                mapping_id=mapping_id,
            )
            logger.info(f"[DEBUG normalize_v2] extract_from_source returned {len(raw_data) if raw_data else 0} rows")
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
            result["success"] = True  # No data is not an error
            return result
        
        # Detect if file is headerless (first row looks like data, not headers)
        file_is_headerless = is_headerless_file(raw_data, header_row_index)
        rows = _prepare_rows(raw_data, header_row_index, file_is_headerless)
        
        result["total_rows"] = len(rows)
        
        # Process each row
        for row_idx, row in enumerate(rows, 1):
            row_result = {
                "_source_id": mapping_id,
                "_source_row_number": row_idx,
                "_id": str(uuid.uuid4()),
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
                        # #region agent log
                        if row.get('vin') == 'ST420RJ98FDHKL4E' and target_field in ['year', 'make', 'model', 'color', 'owner_email', 'transmission']:
                            import json
                            from datetime import datetime
                            with open('/Users/alexaherrera/Desktop/table_detector/.cursor/debug.log', 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H3","location":"normalizer.py:492","message":"PDF Row 6: Direct key match","data":{"target_field":target_field,"value":value,"row_has_key":target_field in row,"row_keys":list(row.keys())[:10]},"timestamp":int(datetime.now().timestamp()*1000)}) + '\n')
                        # #endregion
                    
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
                    # TODO: Implement format specification logic
                    # This may be used for date formatting, number formatting, etc.
                    pass  # Format will be applied here once implemented
                
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
                # #region agent log - Enhanced debugging for PDF Row 6
                if row.get('vin') == 'ST420RJ98FDHKL4E' and target_field in ['year', 'make', 'model', 'color', 'owner_email', 'transmission']:
                    import json
                    from datetime import datetime
                    with open('/Users/alexaherrera/Desktop/table_detector/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H4","location":"normalizer.py:753","message":"PDF Row 6: After mapping assignment","data":{"target_field":target_field,"value":value,"value_type":type(value).__name__,"row_result_value":row_result.get(target_field),"row_result_value_type":type(row_result.get(target_field)).__name__ if row_result.get(target_field) is not None else None,"row_has_key":target_field in row,"source_field":mapping.get("source_field"),"has_ai_instruction":bool(mapping.get("ai_instruction"))},"timestamp":int(datetime.now().timestamp()*1000)}) + '\n')
                # #endregion
            
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
                    
                    # #region agent log
                    if row.get('vin') == 'ST420RJ98FDHKL4E' and matching_target in ['year', 'make', 'model', 'color', 'owner_email', 'transmission']:
                        import json
                        from datetime import datetime
                        with open('/Users/alexaherrera/Desktop/table_detector/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H4","location":"normalizer.py:799","message":"PDF Row 6: Safeguard preserved value","data":{"target_field":matching_target,"value":val,"current_value":current_value,"is_vision_extracted":is_vision_extracted},"timestamp":int(datetime.now().timestamp()*1000)}) + '\n')
                    # #endregion
            
            # CRITICAL FIX: For Vision-extracted rows, directly preserve ALL fields from row dict
            # This ensures fields extracted by Vision API (even if None) are preserved for warning generation
            # The safeguard above only preserves fields that match target_field in mappings,
            # but Vision-extracted rows should preserve ALL canonical fields regardless of mapping
            is_vision_extracted = row.get("_is_vision_extracted", False)
            if is_vision_extracted:
                # For Vision-extracted rows, directly copy all canonical fields from row to row_result
                # This ensures fields are preserved even if they don't match any mapping target_field
                from schema import VEHICLE_SCHEMA_ORDER
                for field_name in VEHICLE_SCHEMA_ORDER:
                    # Only set if not already set (mappings take priority)
                    if field_name not in row_result or row_result.get(field_name) is None:
                        row_value = row.get(field_name)
                        # Preserve even None values (for warning generation)
                        row_result[field_name] = row_value
                        if row_value is not None and row_value != "":
                            preserved_fields.append(field_name)
            
            # SAFEGUARD: Field preservation validation - log warning if critical fields are still missing
            if row.get('vin'):
                critical_fields = ['year', 'make', 'model']
                missing_critical = [f for f in critical_fields if not row_result.get(f)]
                if missing_critical:
                    # Enhanced debugging for PDF Row 6
                    if row.get('vin') == 'ST420RJ98FDHKL4E':
                        import json
                        from datetime import datetime
                        with open('/Users/alexaherrera/Desktop/table_detector/.cursor/debug.log', 'a') as f:
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H5","location":"normalizer.py:860","message":"PDF Row 6: Missing critical fields after preservation","data":{"missing_critical":missing_critical,"preserved_fields":preserved_fields,"row_keys":list(row.keys())[:15],"row_values":{k:row.get(k) for k in missing_critical if k in row},"row_result_values":{k:row_result.get(k) for k in missing_critical}},"timestamp":int(datetime.now().timestamp()*1000)}) + '\n')
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
            if "driver_id" in row_result:
                for key, val in row_result.items():
                    if isinstance(val, str):
                        val = val.strip()
                        # Convert empty strings to None
                        if val == "":
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
            
            # Correct VIN ONLY for handwritten OCR sources (NOT for Vision-extracted sources)
            # Vision-extracted VINs should be preserved exactly as extracted (no correction)
            if actual_source_type in {SourceType.PDF, SourceType.IMAGE} and row.get("_is_handwritten") and not row.get("_is_vision_extracted"):
                vin = row_result.get("vin")
                if vin:
                    corrected_vin = correct_vin_once(vin)
                    if corrected_vin != vin:
                        row_result["vin"] = corrected_vin
            
            # Add warnings for data quality issues
            row_result["_warnings"] = _generate_warnings(row_result)
            
            # Add row to results
            # When validate=False, include all rows even if they have errors
            if row_errors:
                result["errors"].extend(row_errors)
                result["total_errors"] += 1
                # Still add row to data when validate=False
                if not validate:
                    result["data"].append(row_result)
            else:
                result["data"].append(row_result)
                result["total_success"] += 1
        
        # Reorder rows by domain
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

