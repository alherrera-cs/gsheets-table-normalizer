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
# from transforms import apply_transform  # Transform logic will be implemented later
from schema import validate_schema, FieldType, ValidationError
from external_tables import (
    clean_header,
    rows2d_to_objects,
    normalize_values,
    drop_empty_rows,
)


class NormalizationError(Exception):
    """Raised when normalization fails."""
    pass


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
        
        # Prepare source for extraction
        if source_type == SourceType.GOOGLE_SHEETS:
            source_dict = {
                "sheet_id": connection_config.get("spreadsheet_id") or str(source),
                "range": connection_config.get("data_range", "Sheet1!A:Z"),
            }
        elif source_type == SourceType.CSV:
            # CSV files can be specified in connection_config or source
            file_path = connection_config.get("file_path") or (source if isinstance(source, (str, Path)) else source.get("file_path"))
            source_dict = {"file_path": str(file_path)}
        else:
            source_dict = source
        
        # Extract raw data
        try:
            raw_data = extract_from_source(
                source_dict,
                source_type=source_type,
                header_row_index=header_row_index,
            )
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
        
        # Convert 2D data to objects
        rows = rows2d_to_objects(raw_data, header_row_index=header_row_index)
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
                if source_field:
                    # Structured source - direct column lookup
                    # Clean the source field name to match cleaned headers
                    cleaned_source_field = clean_header(source_field)
                    value = row.get(cleaned_source_field)
                elif ai_instruction:
                    # AI-powered source - TODO: implement AI extraction
                    value = None
                    row_errors.append({
                        "target_field": target_field,
                        "_source_id": mapping_id,
                        "_source_row_number": row_idx,
                        "error": "AI/OCR extraction not yet implemented"
                    })
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
                
                # Apply transforms - transform logic will be implemented later
                if transform:
                    # TODO: Implement transform logic
                    pass  # Transform will be applied here once implemented
                
                if format_spec:
                    # TODO: Implement format specification logic
                    # This may be used for date formatting, number formatting, etc.
                    pass  # Format will be applied here once implemented
                
                # Type conversion
                if value is not None and value != "":
                    try:
                        if field_type == "integer":
                            value = int(float(str(value)))
                        elif field_type == "decimal":
                            value = float(str(value))
                        elif field_type == "date":
                            # Keep as string for now, date parsing can be added later
                            value = str(value).strip()
                    except (ValueError, TypeError):
                        row_errors.append({
                            "target_field": target_field,
                            "_source_id": mapping_id,
                            "_source_row_number": row_idx,
                            "error": f"Invalid {field_type} value: {value}"
                        })
                        # Keep original value (or None) and continue to add field
                        value = None
                
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
                    pass
                # If both are None/empty, keep the existing None
            
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
            
            # Add row to results
            if row_errors:
                result["errors"].extend(row_errors)
                result["total_errors"] += 1
            else:
                result["data"].append(row_result)
                result["total_success"] += 1
        
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

