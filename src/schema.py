"""
Schema validation utilities.

Validates normalized data against expected schema definitions.
"""

from typing import Any, Dict, List, Optional, Set
from enum import Enum


class ValidationError(Exception):
    """Raised when schema validation fails."""
    pass


class FieldType(Enum):
    """Supported field types for validation."""
    STRING = "string"
    INTEGER = "integer"
    DATE = "date"
    OPTIONAL_STRING = "optional_string"
    OPTIONAL_INTEGER = "optional_integer"


def validate_schema(
    rows: List[Dict[str, Any]],
    schema: Dict[str, FieldType],
    required_fields: Optional[Set[str]] = None,
) -> List[str]:
    """
    Validate rows against a schema definition.
    
    Args:
        rows: List of dictionaries to validate
        schema: Dictionary mapping field names to FieldType
        required_fields: Set of field names that must be present (default: all in schema)
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if required_fields is None:
        required_fields = set(schema.keys())
    
    for i, row in enumerate(rows, 1):
        row_errors = []
        
        # Check required fields
        for field in required_fields:
            if field not in row:
                row_errors.append(f"Row {i}: Missing required field '{field}'")
        
        # Validate field types
        for field, expected_type in schema.items():
            if field not in row:
                continue  # Already handled by required_fields check
            
            value = row[field]
            error = _validate_field_type(field, value, expected_type, i)
            if error:
                row_errors.append(error)
        
        errors.extend(row_errors)
    
    return errors


def _validate_field_type(
    field: str,
    value: Any,
    expected_type: FieldType,
    row_num: int,
) -> Optional[str]:
    """Validate a single field value against expected type."""
    
    # Handle None values
    if value is None:
        if expected_type in (FieldType.OPTIONAL_STRING, FieldType.OPTIONAL_INTEGER):
            return None  # None is valid for optional fields
        elif expected_type == FieldType.STRING:
            return f"Row {row_num}, field '{field}': Expected string, got None"
        elif expected_type == FieldType.INTEGER:
            return f"Row {row_num}, field '{field}': Expected integer, got None"
        elif expected_type == FieldType.DATE:
            return f"Row {row_num}, field '{field}': Expected date, got None"
    
    # Validate string types
    if expected_type in (FieldType.STRING, FieldType.OPTIONAL_STRING):
        if not isinstance(value, str):
            return f"Row {row_num}, field '{field}': Expected string, got {type(value).__name__}"
    
    # Validate integer types
    elif expected_type in (FieldType.INTEGER, FieldType.OPTIONAL_INTEGER):
        if not isinstance(value, int):
            return f"Row {row_num}, field '{field}': Expected integer, got {type(value).__name__}"
    
    # Validate date types (should be string in YYYY-MM-DD format)
    elif expected_type == FieldType.DATE:
        if not isinstance(value, str):
            return f"Row {row_num}, field '{field}': Expected date string, got {type(value).__name__}"
        # Basic date format check
        if len(value) != 10 or value[4] != '-' or value[7] != '-':
            return f"Row {row_num}, field '{field}': Invalid date format (expected YYYY-MM-DD), got '{value}'"
    
    return None


def get_default_vehicles_schema() -> Dict[str, FieldType]:
    """Get default schema for vehicles data."""
    return {
        "vin": FieldType.STRING,
        "vehicle_id": FieldType.OPTIONAL_STRING,
        "year": FieldType.INTEGER,
        "make": FieldType.STRING,
        "model": FieldType.STRING,
        "effective_date": FieldType.DATE,
        "notes": FieldType.OPTIONAL_STRING,
        "trim": FieldType.OPTIONAL_STRING,
        "weight": FieldType.OPTIONAL_INTEGER,
    }

