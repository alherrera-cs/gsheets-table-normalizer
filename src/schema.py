"""
Unified vehicle schema + ordering helpers for consistent test output.

Goals:
- Match the __schema__ ordering from truth files.
- Include all 15 canonical vehicle fields.
- Allow optional types (non-strict) so inference doesn't break.
- Provide reorder helpers used by tests for consistent output ordering.
"""

from enum import Enum
from typing import Any, Dict, List, Optional


class ValidationError(Exception):
    """Raised when schema validation fails."""
    pass


class FieldType(Enum):
    STRING = "string"
    OPTIONAL_STRING = "optional_string"
    INTEGER = "integer"
    OPTIONAL_INTEGER = "optional_integer"
    DATE = "date"
    OPTIONAL_DATE = "optional_date"


# 🚗 Canonical schema field order — exactly matches truth __schema__
VEHICLE_SCHEMA_ORDER = [
    "vin",
    "vehicle_id",
    "year",
    "make",
    "model",
    "effective_date",
    "notes",
    "color",
    "mileage",
    "trim",
    "body_style",
    "fuel_type",
    "transmission",
    "owner_email",
    "weight",
    "image_url",
    "description",
]


# 🚗 Field → type mapping (validation optional)
VEHICLE_SCHEMA_TYPES = {
    "vin": FieldType.OPTIONAL_STRING,
    "vehicle_id": FieldType.OPTIONAL_STRING,
    "year": FieldType.OPTIONAL_INTEGER,
    "make": FieldType.OPTIONAL_STRING,
    "model": FieldType.OPTIONAL_STRING,
    "effective_date": FieldType.OPTIONAL_DATE,
    "notes": FieldType.OPTIONAL_STRING,
    "color": FieldType.OPTIONAL_STRING,
    "mileage": FieldType.OPTIONAL_INTEGER,
    "trim": FieldType.OPTIONAL_STRING,
    "body_style": FieldType.OPTIONAL_STRING,
    "fuel_type": FieldType.OPTIONAL_STRING,
    "transmission": FieldType.OPTIONAL_STRING,
    "owner_email": FieldType.OPTIONAL_STRING,
    "weight": FieldType.OPTIONAL_INTEGER,
    "image_url": FieldType.OPTIONAL_STRING,
    "description": FieldType.OPTIONAL_STRING,
}


def reorder_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reorder row keys to match VEHICLE_SCHEMA_ORDER exactly.
    
    Missing fields become None; extra fields are appended at the bottom.
    """
    ordered = {field: row.get(field) for field in VEHICLE_SCHEMA_ORDER}
    
    # Preserve unexpected fields (debugging-friendly)
    for key in row:
        if key not in ordered:
            ordered[key] = row[key]
    
    return ordered


def reorder_all(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply reorder_fields() to every row."""
    return [reorder_fields(r) for r in rows]


def validate_value(field: str, value: Any, expected: FieldType) -> Optional[str]:
    """Soft validation (non-blocking)."""
    if value is None:
        return None
    
    if expected in (FieldType.STRING, FieldType.OPTIONAL_STRING):
        if not isinstance(value, str):
            return f"{field}: expected string, got {type(value).__name__}"
    
    if expected in (FieldType.INTEGER, FieldType.OPTIONAL_INTEGER):
        if not isinstance(value, int):
            return f"{field}: expected integer, got {type(value).__name__}"
    
    if expected in (FieldType.DATE, FieldType.OPTIONAL_DATE):
        if not isinstance(value, str) or len(value) != 10 or value[4] != "-" or value[7] != "-":
            return f"{field}: invalid date format '{value}'"
    
    return None


# 📋 Canonical schema field order for Policies
POLICY_SCHEMA_ORDER = [
    "policy_number",
    "insured_name",
    "effective_date",
    "expiration_date",
    "premium",
    "coverage_type",
    "vehicle_vin",
    "notes",
]


# 📋 Field → type mapping for Policies (validation optional)
POLICY_SCHEMA_TYPES = {
    "policy_number": FieldType.OPTIONAL_STRING,
    "insured_name": FieldType.OPTIONAL_STRING,
    "effective_date": FieldType.OPTIONAL_DATE,
    "expiration_date": FieldType.OPTIONAL_DATE,
    "premium": FieldType.OPTIONAL_STRING,  # Stored as string but validated as float
    "coverage_type": FieldType.OPTIONAL_STRING,
    "vehicle_vin": FieldType.OPTIONAL_STRING,
    "notes": FieldType.OPTIONAL_STRING,
}


def reorder_policy_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reorder row keys to match POLICY_SCHEMA_ORDER exactly.
    
    Missing fields become None; extra fields are appended at the bottom.
    """
    ordered = {field: row.get(field) for field in POLICY_SCHEMA_ORDER}
    
    # Preserve unexpected fields (debugging-friendly)
    for key in row:
        if key not in ordered:
            ordered[key] = row[key]
    
    return ordered


def reorder_all_policies(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply reorder_policy_fields() to every row."""
    return [reorder_policy_fields(r) for r in rows]


# 📍 Canonical schema field order for Locations
LOCATION_SCHEMA_ORDER = [
    "location_id",
    "insured_name",
    "address_line_1",
    "city",
    "state",
    "postal_code",
    "county",
    "territory_code",
    "protection_class",
    "latitude",
    "longitude",
    "notes",
]


# 📍 Field → type mapping for Locations (validation optional)
LOCATION_SCHEMA_TYPES = {
    "location_id": FieldType.OPTIONAL_STRING,
    "insured_name": FieldType.OPTIONAL_STRING,
    "address_line_1": FieldType.OPTIONAL_STRING,
    "city": FieldType.OPTIONAL_STRING,
    "state": FieldType.OPTIONAL_STRING,
    "postal_code": FieldType.OPTIONAL_STRING,
    "county": FieldType.OPTIONAL_STRING,
    "territory_code": FieldType.OPTIONAL_STRING,
    "protection_class": FieldType.OPTIONAL_INTEGER,
    "latitude": FieldType.OPTIONAL_STRING,  # Stored as string but validated as float
    "longitude": FieldType.OPTIONAL_STRING,  # Stored as string but validated as float
    "notes": FieldType.OPTIONAL_STRING,
}


def reorder_location_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reorder row keys to match LOCATION_SCHEMA_ORDER exactly.
    
    Missing fields become None; extra fields are appended at the bottom.
    """
    ordered = {field: row.get(field) for field in LOCATION_SCHEMA_ORDER}
    
    # Preserve unexpected fields (debugging-friendly)
    for key in row:
        if key not in ordered:
            ordered[key] = row[key]
    
    return ordered


def reorder_all_locations(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply reorder_location_fields() to every row."""
    return [reorder_location_fields(r) for r in rows]


# 👤 Canonical schema field order for Drivers
DRIVER_SCHEMA_ORDER = [
    "driver_id",
    "first_name",
    "last_name",
    "date_of_birth",
    "license_number",
    "license_state",
    "license_status",
    "years_experience",
    "violations_count",
    "training_completed",
    "notes",
]


# 👤 Field → type mapping for Drivers (validation optional)
DRIVER_SCHEMA_TYPES = {
    "driver_id": FieldType.OPTIONAL_STRING,
    "first_name": FieldType.OPTIONAL_STRING,
    "last_name": FieldType.OPTIONAL_STRING,
    "date_of_birth": FieldType.OPTIONAL_DATE,
    "license_number": FieldType.OPTIONAL_STRING,
    "license_state": FieldType.OPTIONAL_STRING,
    "license_status": FieldType.OPTIONAL_STRING,
    "years_experience": FieldType.OPTIONAL_INTEGER,
    "violations_count": FieldType.OPTIONAL_INTEGER,
    "training_completed": FieldType.OPTIONAL_STRING,
    "notes": FieldType.OPTIONAL_STRING,
}


def reorder_driver_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reorder row keys to match DRIVER_SCHEMA_ORDER exactly.
    
    Missing fields become None; extra fields are filtered out (only canonical schema fields kept).
    Metadata fields (_confidence, _warnings, _source, _flags, etc.) are preserved.
    """
    ordered = {field: row.get(field) for field in DRIVER_SCHEMA_ORDER}
    
    # Preserve metadata fields (not in DRIVER_SCHEMA_ORDER)
    metadata_fields = ["_confidence", "_warnings", "_source", "_flags", "_source_id", "_source_row_number", "_id"]
    for metadata_field in metadata_fields:
        if metadata_field in row:
            ordered[metadata_field] = row[metadata_field]
    
    # Ensure _warnings and _confidence always exist (even if empty or None)
    if "_warnings" not in ordered or ordered.get("_warnings") is None or not isinstance(ordered.get("_warnings"), list):
        ordered["_warnings"] = []
    if "_confidence" not in ordered or ordered.get("_confidence") is None or not isinstance(ordered.get("_confidence"), dict):
        ordered["_confidence"] = {}
    
    # Do NOT preserve unexpected non-canonical fields - only keep canonical DRIVER_SCHEMA_ORDER fields
    # This matches vehicle behavior where only canonical schema fields are kept
    
    return ordered


def reorder_all_drivers(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply reorder_driver_fields() to every row."""
    return [reorder_driver_fields(r) for r in rows]


# 🔗 Canonical schema field order for Relationships (Policy-Vehicle-Driver Links)
RELATIONSHIP_SCHEMA_ORDER = [
    "policy_number",
    "vehicle_vin",
    "driver_id",
    "relationship_type",
    "notes",
]


# 🔗 Field → type mapping for Relationships (validation optional)
RELATIONSHIP_SCHEMA_TYPES = {
    "policy_number": FieldType.OPTIONAL_STRING,
    "vehicle_vin": FieldType.OPTIONAL_STRING,
    "driver_id": FieldType.OPTIONAL_STRING,
    "relationship_type": FieldType.OPTIONAL_STRING,
    "notes": FieldType.OPTIONAL_STRING,
}


def reorder_relationship_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reorder row keys to match RELATIONSHIP_SCHEMA_ORDER exactly.
    
    Missing fields become None; extra fields are filtered out (only canonical schema fields kept).
    Metadata fields (_confidence, _warnings, _source, _flags, etc.) are preserved.
    """
    ordered = {field: row.get(field) for field in RELATIONSHIP_SCHEMA_ORDER}
    
    # Preserve metadata fields (not in RELATIONSHIP_SCHEMA_ORDER)
    metadata_fields = ["_confidence", "_warnings", "_source", "_flags", "_source_id", "_source_row_number", "_id"]
    for metadata_field in metadata_fields:
        if metadata_field in row:
            ordered[metadata_field] = row[metadata_field]
    
    # Do NOT preserve unexpected non-canonical fields - only keep canonical RELATIONSHIP_SCHEMA_ORDER fields
    # This matches driver behavior where only canonical schema fields are kept
    
    return ordered


def reorder_all_relationships(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply reorder_relationship_fields() to every row."""
    return [reorder_relationship_fields(r) for r in rows]


# 💼 Canonical schema field order for Claims
CLAIM_SCHEMA_ORDER = [
    "claim_number",
    "policy_number",
    "loss_date",
    "claim_type",
    "amount",
    "description",
    "status",
    "notes",
]


# 💼 Field → type mapping for Claims (validation optional)
CLAIM_SCHEMA_TYPES = {
    "claim_number": FieldType.OPTIONAL_STRING,
    "policy_number": FieldType.OPTIONAL_STRING,
    "loss_date": FieldType.OPTIONAL_DATE,
    "claim_type": FieldType.OPTIONAL_STRING,
    "amount": FieldType.OPTIONAL_STRING,  # Stored as string but validated as float
    "description": FieldType.OPTIONAL_STRING,
    "status": FieldType.OPTIONAL_STRING,
    "notes": FieldType.OPTIONAL_STRING,
}


def reorder_claim_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reorder row keys to match CLAIM_SCHEMA_ORDER exactly.
    
    Missing fields become None; extra fields are appended at the bottom.
    """
    ordered = {field: row.get(field) for field in CLAIM_SCHEMA_ORDER}
    
    # Preserve unexpected fields (debugging-friendly)
    for key in row:
        if key not in ordered:
            ordered[key] = row[key]
    
    return ordered


def reorder_all_claims(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply reorder_claim_fields() to every row."""
    return [reorder_claim_fields(r) for r in rows]
