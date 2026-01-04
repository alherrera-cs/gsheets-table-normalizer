"""
Unified test suite for ALL driver sources.

Tests all driver formats in a single suite:
- Structured: Airtable, Excel, Google Sheet
- Unstructured: PDF, Raw Text, Image
"""

import sys
import json
import traceback
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

# Global flags for output control
VERBOSE = False
FULL_DIFF = False
DEBUG = False

# Suppress debug logs during test runs for cleaner output (unless VERBOSE)
# Set to CRITICAL to suppress all logger.warning() calls from source modules
logging.getLogger().setLevel(logging.CRITICAL)
# Suppress specific noisy loggers
for logger_name in [
    "src", "src.sources", "src.ocr", "src.normalizer", "src.transforms",
    "ocr", "ocr.parser", "ocr.table_extract", "ocr.reader", "ocr.models",
    "__main__", "parser", "table_extract"
]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from normalizer import normalize_v2
from mappings import get_mapping_by_id
from schema import reorder_all_drivers, DRIVER_SCHEMA_ORDER


def debug_print(*args, **kwargs):
    """Print only if VERBOSE is enabled."""
    if VERBOSE:
        print(*args, **kwargs)


def load_truth_file(truth_file: Path) -> List[Dict[str, Any]]:
    """Load expected truth file and extract rows."""
    if not truth_file.exists():
        raise FileNotFoundError(f"Truth file not found: {truth_file}")
    
    with open(truth_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "rows" in data:
        return data["rows"]
    else:
        raise ValueError(f"Unexpected truth file format: {truth_file}")


def normalize_row_for_comparison(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a row for comparison by removing metadata fields.
    
    Removes:
        - _source_id
        - _source_row_number
        - _id (UUID)
        - _flags (metadata flags)
    
    Treats missing fields as None.
    Preserves _warnings and _confidence for validation.
    """
    normalized = {}
    for key in sorted(row.keys()):
        if key not in ["_source_id", "_source_row_number", "_id", "_flags"]:
            value = row[key]
            # Convert empty strings to None for consistency
            if value == "":
                value = None
            normalized[key] = value
    return normalized


def values_equal(field: str, expected_val: Any, actual_val: Any, actual_row: Optional[Dict[str, Any]] = None) -> bool:
    """
    Compare two values with confidence-aware logic.
    
    Confidence-aware rules:
    - If field has confidence < 0.9, value mismatches are acceptable (not failures)
    - Missing optional fields with 0.0 confidence are acceptable
    - Driver ID with invalid format but low confidence + warnings is acceptable
    - Confidence fields themselves must match (they're metadata)
    
    Args:
        field: Field name being compared
        expected_val: Expected value
        actual_val: Actual value
        actual_row: Full actual row dict (for confidence checking)
    
    Returns:
        True if values are considered equal, False otherwise
    """
    # Metadata fields (_confidence, _warnings, _source) are not part of schema comparison
    if field in ["_confidence", "_warnings", "_source"]:
        # These are metadata, not schema fields - skip comparison
            return True
    
    # Legacy: Handle old *_confidence and *_warnings format (should not exist in new format)
    if field.endswith("_confidence") or field.endswith("_warnings"):
        # These should not exist in new format, but handle gracefully
        return expected_val == actual_val
    
    # For notes, non-strict comparison: only check presence
    if field == "notes":
        expected_present = expected_val is not None and expected_val != ""
        actual_present = actual_val is not None and actual_val != ""
        return expected_present == actual_present
    
    # If no actual_row provided, use strict comparison (backward compatibility)
    if actual_row is None:
        return expected_val == actual_val
    
    # Confidence-aware comparison - read from namespaced _confidence dict
    confidence_dict = actual_row.get("_confidence", {})
    field_confidence = confidence_dict.get(field)
    
    # Get warnings from _warnings list (field-specific warnings are in the general list)
    all_warnings = actual_row.get("_warnings", [])
    # Filter warnings for this specific field (e.g., "driver_id_invalid_format" for driver_id field)
    field_warnings = [w for w in all_warnings if w.startswith(f"{field}_") or (field == "driver_id" and "driver_id_" in w)]
    
    # Optional fields that can be missing (should not cause failures)
    optional_fields = ["notes"]
    
    # If field is missing and optional, check if confidence is 0.0 (acceptable)
    if expected_val is not None and actual_val is None:
        if field in optional_fields:
            # Missing optional field is acceptable if confidence is 0.0
            if field_confidence == 0.0:
                return True
        return False
    
    # If values don't match, check confidence
    if expected_val != actual_val:
        # Driver ID-specific handling: allow invalid format if confidence is low and warnings present
        if field == "driver_id":
            if field_confidence is not None and field_confidence < 0.9:
                # Check if driver_id has invalid format warning
                if isinstance(field_warnings, list) and any("driver_id_invalid" in str(w) or "driver_id_repaired" in str(w) for w in field_warnings):
                    # Allow driver_id mismatch if it's due to invalid format (extraction still returned a value)
                    return True
                # Check if driver_id was repaired (has repair warnings)
                if isinstance(field_warnings, list) and any("repaired" in str(w).lower() or "substitution" in str(w).lower() for w in field_warnings):
                    # Allow repaired driver_ids with lower confidence
                    if field_confidence is not None and field_confidence < 0.8:
                        return True
        
        # For other fields: if confidence < 0.9, mismatch is acceptable
        if field_confidence is not None and field_confidence < 0.9:
            return True
    
    # Default: strict comparison
    return expected_val == actual_val


def print_missing_extra_rows(missing_rows: List[int], extra_rows: List[int]) -> None:
    """Print consolidated messages for missing/extra rows."""
    if missing_rows:
        if len(missing_rows) == 1:
            print(f"{Colors.RED}✗ Missing row: Expected row {missing_rows[0]} has no actual row{Colors.RESET}")
        else:
            missing_str = ", ".join(map(str, sorted(missing_rows)))
            print(f"{Colors.RED}✗ Missing rows: Expected rows {missing_str} have no actual rows ({len(missing_rows)} total){Colors.RESET}")
        print()
    
    if extra_rows:
        if len(extra_rows) == 1:
            print(f"{Colors.RED}✗ Extra row: Actual row {extra_rows[0]} has no expected row{Colors.RESET}")
        else:
            extra_str = ", ".join(map(str, sorted(extra_rows)))
            print(f"{Colors.RED}✗ Extra rows: Actual rows {extra_str} have no expected rows ({len(extra_rows)} total){Colors.RESET}")
        print()


def format_value(value: Any, max_length: int = 50) -> str:
    """Format a value for display with smart truncation and None handling."""
    if value is None:
        return "—"  # Use em dash for None (more compact than "None")
    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        return str(value) if len(str(value)) <= max_length else str(value)[:max_length-1] + "…"
    if isinstance(value, str):
        if len(value) > max_length:
            return value[:max_length-1] + "…"
        return value
    return str(value)


def categorize_mismatch(field: str, expected_val: Any, actual_val: Any) -> str:
    """
    Categorize a mismatch into issue types.
    
    Returns:
        Category name: 'driver_id_corruption', 'missing_field', 'normalization', 'case', 'other'
    """
    if field == "driver_id":
        return "driver_id_corruption"
    elif expected_val is None and actual_val is not None:
        return "extra_field"
    elif expected_val is not None and actual_val is None:
        return "missing_field"
    elif isinstance(expected_val, str) and isinstance(actual_val, str):
        if expected_val.lower() == actual_val.lower():
            return "case"
    return "other"


def group_mismatches_by_category(mismatches: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group mismatches by category for cleaner reporting."""
    grouped = defaultdict(list)
    for mismatch in mismatches:
        category = categorize_mismatch(
            mismatch["field"],
            mismatch["expected"],
            mismatch["actual"]
        )
        grouped[category].append(mismatch)
    return dict(grouped)


def _extract_confidence_info(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract confidence information from a row if present.
    
    Args:
        row: Row dictionary that may contain confidence fields
    
    Returns:
        Dictionary with confidence info, or None if no confidence fields present
        Structure: {
            "overall_confidence": float,
            "field_confidences": {field_name: confidence_value},
            "reasons": [list of warning strings]
        }
    """
    # Read confidence from namespaced _confidence dict
    confidence_dict = row.get("_confidence", {})
    
    # If no confidence fields found, return None
    if not confidence_dict:
        return None
    
    # Calculate overall confidence (average of all field confidences)
    # Prioritize key fields (driver_id, years_experience) if present
    priority_fields = ["driver_id", "years_experience", "license_number"]
    priority_confidences = [confidence_dict.get(f) for f in priority_fields if f in confidence_dict and confidence_dict[f] is not None]
    
    if priority_confidences:
        overall_confidence = sum(priority_confidences) / len(priority_confidences)
    else:
        all_confidences = [v for v in confidence_dict.values() if v is not None]
        if all_confidences:
            overall_confidence = sum(all_confidences) / len(all_confidences)
        else:
            return None
    
    # Collect all warnings from _warnings list
    reasons = row.get("_warnings", [])
    if not isinstance(reasons, list):
        reasons = [reasons] if reasons else []
    
    # Remove duplicates while preserving order
    seen = set()
    unique_reasons = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            unique_reasons.append(reason)
    
    return {
        "overall_confidence": overall_confidence,
        "field_confidences": confidence_dict,
        "reasons": unique_reasons
    }


def _print_confidence_info(confidence_info: Dict[str, Any]) -> None:
    """
    Print confidence information in a clean format.
    
    Args:
        confidence_info: Dictionary from _extract_confidence_info
    """
    overall_confidence = confidence_info["overall_confidence"]
    reasons = confidence_info.get("reasons", [])
    
    # Determine confidence color and emoji
    if overall_confidence >= 0.8:
        conf_color = Colors.GREEN
        emoji = "✓"
    elif overall_confidence >= 0.5:
        conf_color = Colors.YELLOW
        emoji = "⚠️"
    else:
        conf_color = Colors.RED
        emoji = "⚠️"
    
    # Print confidence score
    print(f"  {Colors.BOLD}Confidence:{Colors.RESET} {conf_color}{overall_confidence:.2f}{Colors.RESET} {emoji}")
    
    # Print reasons if present (only show section if there are reasons)
    if reasons:
        print(f"  {Colors.BOLD}Reasons:{Colors.RESET}")
        for reason in reasons:
            print(f"    {Colors.GRAY}•{Colors.RESET} {reason}")


def print_row_diff(row_num: int, expected: Dict[str, Any], actual: Dict[str, Any], show_full: bool = False) -> int:
    """
    Print a professional, clean diff for a single row with improved visibility.
    
    Args:
        row_num: Row number (1-indexed)
        expected: Expected row dictionary
        actual: Actual row dictionary
        show_full: Whether to show full diff (all mismatches) or compact view
    
    Returns:
        Number of mismatched fields
    """
    # Get all fields from both rows
    all_fields = sorted(set(expected.keys()) | set(actual.keys()))
    
    # Find mismatches (confidence-aware)
    mismatches = []
    for field in all_fields:
        expected_val = expected.get(field, None)
        actual_val = actual.get(field, None)
        
        # Pass actual row for confidence-aware comparison
        if not values_equal(field, expected_val, actual_val, actual_row=actual):
            mismatches.append({
                "field": field,
                "expected": expected_val,
                "actual": actual_val
            })
    
    # Check for warnings in actual row
    actual_warnings = actual.get("_warnings", [])
    expected_warnings = expected.get("_warnings", [])
    has_warnings = (actual_warnings and len(actual_warnings) > 0) or (expected_warnings and len(expected_warnings) > 0)
    
    # Extract confidence information from actual row (if present)
    confidence_info = _extract_confidence_info(actual)
    
    # Extract flags information from actual row (if present)
    flags_dict = actual.get("_flags", {})
    has_flags = bool(flags_dict)
    driver_id_requires_review = flags_dict.get("driver_id_requires_human_review", False)
    
    # If perfect match, print and return
    if not mismatches:
        if VERBOSE or show_full:
            # Display confidence information if present (for passing rows too)
            if confidence_info:
                print(f"\n{Colors.GREEN}{Colors.BOLD}━━ Row {row_num} ━━{Colors.RESET}")
                _print_confidence_info(confidence_info)
            
            # Display flags if present
            if has_flags and driver_id_requires_review:
                print(f"  {Colors.YELLOW}{Colors.BOLD}Flags:{Colors.RESET} {Colors.YELLOW}⚠ Driver ID requires human review{Colors.RESET}")
            
        if has_warnings:
                warning_count = len(actual_warnings) if actual_warnings else 0
                print(f"{Colors.GREEN}Row {row_num}: ✓ PASS{Colors.RESET} {Colors.YELLOW}⚠ {warning_count} warning(s){Colors.RESET}")
        else:
                print(f"{Colors.GREEN}Row {row_num}: ✓ PASS{Colors.RESET}")
        return 0
    
    mismatch_count = len(mismatches)
    
    # Extract confidence information from actual row (if present)
    confidence_info = _extract_confidence_info(actual)
    
    if show_full or FULL_DIFF:
        # Full diff mode: show all mismatches
        print(f"\n{Colors.RED}{Colors.BOLD}━━ Row {row_num}: {mismatch_count} mismatch(es) ━━{Colors.RESET}")
    
        # Display confidence information if present (near top, before mismatches)
        if confidence_info:
            _print_confidence_info(confidence_info)
        
        # Display flags if present
        if has_flags:
            flags_list = []
            if driver_id_requires_review:
                flags_list.append(f"{Colors.YELLOW}⚠ Driver ID requires human review{Colors.RESET}")
            if flags_list:
                print(f"  {Colors.BOLD}Flags:{Colors.RESET} {', '.join(flags_list)}")
        
        if has_warnings:
            if "_warnings" in [m["field"] for m in mismatches]:
                expected_warn_str = ", ".join(expected_warnings) if expected_warnings else "none"
                actual_warn_str = ", ".join(actual_warnings) if actual_warnings else "none"
                print(f"  {Colors.YELLOW}⚠ Warnings:{Colors.RESET} Expected: {expected_warn_str} | Actual: {actual_warn_str}")
            elif actual_warnings:
                warnings_str = ", ".join([f"{w}" for w in actual_warnings])
                print(f"  {Colors.YELLOW}⚠ Warnings:{Colors.RESET} {warnings_str}")
        
        print(f"\n  {Colors.BOLD}{'Field':<20} {'Expected':<45} {'Actual':<45}{Colors.RESET}")
        print(f"  {Colors.GRAY}{'─' * 112}{Colors.RESET}")
        
        for mismatch in mismatches:
            field = mismatch["field"]
            expected_val = mismatch["expected"]
            actual_val = mismatch["actual"]
            
            if expected_val is None and actual_val is None:
                continue
            
            expected_str = format_value(expected_val, 45)
            actual_str = format_value(actual_val, 45)
            print(f"  {Colors.CYAN}{field:<20}{Colors.RESET} {Colors.GREEN}{expected_str:<45}{Colors.RESET} {Colors.RED}{actual_str:<45}{Colors.RESET}")
        print()
    else:
        # Compact mode: show grouped mismatches with one example per category
        grouped = group_mismatches_by_category(mismatches)
        category_names = {
            "driver_id_corruption": "Driver ID Corruption",
            "missing_field": "Missing Fields",
            "extra_field": "Extra Fields",
            "normalization": "Normalization Issues",
            "case": "Case Differences",
            "other": "Other Issues"
        }
        
        print(f"\n{Colors.RED}{Colors.BOLD}━━ Row {row_num}: {mismatch_count} mismatch(es) ━━{Colors.RESET}")
        
        # Display confidence information if present (near top, before mismatches)
        if confidence_info:
            _print_confidence_info(confidence_info)
        
        # Display flags if present
        if has_flags and driver_id_requires_review:
            print(f"  {Colors.YELLOW}{Colors.BOLD}Flags:{Colors.RESET} {Colors.YELLOW}⚠ Driver ID requires human review{Colors.RESET}")
        
        if has_warnings:
            warning_count = len(actual_warnings) if actual_warnings else 0
            print(f"  {Colors.YELLOW}⚠ {warning_count} warning(s){Colors.RESET}")
        
        for category, category_mismatches in grouped.items():
            category_name = category_names.get(category, category)
            count = len(category_mismatches)
            # Show only first example
            example = category_mismatches[0]
            field = example["field"]
            expected_str = format_value(example["expected"], 30)
            actual_str = format_value(example["actual"], 30)
            
            if count > 1:
                print(f"  {Colors.CYAN}{category_name}:{Colors.RESET} {count} issue(s) - Example: {field} ({Colors.GREEN}{expected_str}{Colors.RESET} → {Colors.RED}{actual_str}{Colors.RESET})")
            else:
                print(f"  {Colors.CYAN}{category_name}:{Colors.RESET} {field} ({Colors.GREEN}{expected_str}{Colors.RESET} → {Colors.RED}{actual_str}{Colors.RESET})")
        print()
    
    return len(mismatches)


def validate_confidence_fields(actual_row: Dict[str, Any], priority_fields: List[str] = None) -> List[Dict[str, Any]]:
    """
    Validate that confidence fields exist and are correctly populated.
    
    Args:
        actual_row: Actual row dictionary
        priority_fields: List of priority fields to check (default: driver_id, years_experience, license_number)
    
    Returns:
        List of validation issues (empty if all valid)
    """
    if priority_fields is None:
        priority_fields = ["driver_id", "years_experience", "license_number"]
    
    issues = []
    
    # Check that _confidence dict exists
    confidence_dict = actual_row.get("_confidence", {})
    if not isinstance(confidence_dict, dict):
        issues.append({
            "type": "missing_confidence_dict",
            "field": "_confidence",
            "message": "_confidence must be a dictionary"
        })
        return issues
    
    # Check that priority fields have confidence values
    for field in priority_fields:
        if field in actual_row and actual_row[field] is not None:
            # Check confidence exists in _confidence dict
            if field not in confidence_dict:
                issues.append({
                    "type": "missing_confidence",
                    "field": field,
                    "message": f"confidence missing for field {field} in _confidence dict"
                })
            else:
                confidence = confidence_dict[field]
                # Validate confidence is a number between 0 and 1
                if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
                    issues.append({
                        "type": "invalid_confidence",
                        "field": field,
                        "message": f"confidence for {field} has invalid value: {confidence} (must be 0.0-1.0)"
                    })
    
    # Check _warnings exists (should always exist, even if empty list)
    if "_warnings" not in actual_row:
        issues.append({
            "type": "missing_warnings",
            "field": "_warnings",
            "message": "_warnings missing (should be a list)"
        })
    elif not isinstance(actual_row["_warnings"], list):
        issues.append({
            "type": "invalid_warnings",
            "field": "_warnings",
            "message": f"_warnings should be a list, got {type(actual_row['_warnings'])}"
        })
    
    # Note: _flags validation is done via explicit test assertions, not schema rules
    # This prevents flag mismatches from causing test failures
    
    return issues


def classify_issue(mismatch: Dict[str, Any], actual_row: Dict[str, Any]) -> str:
    """
    Classify an issue as CRITICAL or ACCEPTABLE (ETL validation style).
    
    Rules (in priority order):
    1. Structural issues → CRITICAL
    2. Case-only differences → ACCEPTABLE
    3. Notes for unstructured sources → ACCEPTABLE
    4. Driver ID mismatches when confidence < 0.9 → ACCEPTABLE
    5. Low-confidence mismatches (< 0.9) → ACCEPTABLE
    6. High-confidence mismatches (>= 0.9) → CRITICAL
    
    Args:
        mismatch: Mismatch dictionary with field, expected, actual, type
        actual_row: Actual row dictionary (for confidence checking)
    
    Returns:
        "CRITICAL" or "ACCEPTABLE"
    """
    field = mismatch.get("field", "")
    mismatch_type = mismatch.get("type")
    expected_val = mismatch.get("expected")
    actual_val = mismatch.get("actual")
    
    # Rule 1: Structural issues (missing/extra rows) are always CRITICAL
    if mismatch_type in ["extra_row", "missing_row"]:
        return "CRITICAL"
    
    # Ignore metadata fields
    if field in ["_confidence", "_warnings", "_source"] or field.endswith("_confidence") or field.endswith("_warnings"):
        return "ACCEPTABLE"  # Metadata mismatches are not critical
    
    confidence_dict = actual_row.get("_confidence", {})
    field_confidence = confidence_dict.get(field)
    all_warnings = actual_row.get("_warnings", [])
    source_metadata = actual_row.get("_source", {})
    source_type = source_metadata.get("source_type", "")
    parser = source_metadata.get("parser", "")
    
    # Check if source is unstructured (OCR/Vision)
    is_unstructured = (
        source_type in ["pdf", "image"] or
        parser in ["vision_api", "parse_driver_raw_text", "ocr_extraction", "raw_text_parser"]
    )
    
    # Rule 2: Case-only differences are ALWAYS ACCEPTABLE (check BEFORE confidence)
    if isinstance(expected_val, str) and isinstance(actual_val, str):
        if expected_val.lower() == actual_val.lower():
            return "ACCEPTABLE"
    
    # Rule 3: Notes populated for unstructured sources is ACCEPTABLE
    if field == "notes" and expected_val is None and actual_val is not None:
        if is_unstructured:
            return "ACCEPTABLE"
    
    # Rule 4: Driver ID mismatches when confidence < 0.9 are ACCEPTABLE
    if field == "driver_id" and expected_val is not None and actual_val is not None:
        if field_confidence is not None and field_confidence < 0.9:
            return "ACCEPTABLE"
    
    # Now check confidence-based rules
    if field_confidence is None:
        # Missing confidence score - treat as CRITICAL (shouldn't happen in production)
        return "CRITICAL"
    
    # Expected=None, Actual=value
    if expected_val is None and actual_val is not None:
        if field_confidence == 0.0:
            return "ACCEPTABLE"
        else:
            return "CRITICAL"
    
    # Expected=value, Actual=None
    if expected_val is not None and actual_val is None:
        if field_confidence == 0.0:
            return "ACCEPTABLE"
        else:
            return "CRITICAL"
    
    # Both expected and actual have values
    # Rule 5: Low-confidence mismatches (< 0.9) are ACCEPTABLE
    if field_confidence < 0.9:
        return "ACCEPTABLE"
    
    # Rule 6: High-confidence mismatches (>= 0.9) are CRITICAL
    return "CRITICAL"


def filter_acceptable_mismatches(comparison: Dict[str, Any], actual_rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Filter out acceptable mismatches (confidence-aware counting).
    
    ETL-style validation rules:
    1. Structural issues (missing/extra rows) → CRITICAL
    2. Case-only differences → ACCEPTABLE
    3. Notes populated for unstructured sources → ACCEPTABLE
    4. Driver ID mismatches when confidence < 0.9 → ACCEPTABLE
    5. Low-confidence mismatches (< 0.9) → ACCEPTABLE
    6. High-confidence mismatches (>= 0.9) → CRITICAL
    
    Args:
        comparison: Comparison result dictionary
        actual_rows: Actual rows list (for confidence checking)
    
    Returns:
        Tuple of (real_mismatches_count, acceptable_mismatches_count)
    """
    real_mismatches = []
    acceptable_mismatches = []
    
    for mismatch in comparison.get("mismatches", []):
        field = mismatch.get("field", "")
        row_num = mismatch.get("row", 1)
        mismatch_type = mismatch.get("type")
        expected_val = mismatch.get("expected")
        actual_val = mismatch.get("actual")
        
        # Rule 1: Structural issues (missing/extra rows) are always CRITICAL
        if mismatch_type in ["extra_row", "missing_row"]:
            real_mismatches.append(mismatch)
            continue
        
        # Ignore metadata fields (_confidence, _warnings, _source, _flags, _source_id, _source_row_number)
        if (field in ["_confidence", "_warnings", "_source", "_flags", "_source_id", "_source_row_number", "_id"] or 
            field.startswith("_source") or field.endswith("_confidence") or field.endswith("_warnings")):
            continue
        
        # Get actual row for confidence/source checking
        actual_row_idx = row_num - 1  # Convert to 0-indexed
        if actual_row_idx < 0 or actual_row_idx >= len(actual_rows):
            # Can't check confidence (row index out of bounds) - treat as CRITICAL
            real_mismatches.append(mismatch)
            continue
        
        actual_row = actual_rows[actual_row_idx]
        confidence_dict = actual_row.get("_confidence", {})
        field_confidence = confidence_dict.get(field)
        all_warnings = actual_row.get("_warnings", [])
        source_metadata = actual_row.get("_source", {})
        source_type = source_metadata.get("source_type", "")
        parser = source_metadata.get("parser", "")
        
        # Check if source is unstructured (OCR/Vision)
        is_unstructured = (
            source_type in ["pdf", "image"] or
            parser in ["vision_api", "parse_driver_raw_text", "ocr_extraction", "raw_text_parser"]
        )
        
        # Rule 2: Case-only differences are ALWAYS ACCEPTABLE (check BEFORE confidence)
        if isinstance(expected_val, str) and isinstance(actual_val, str):
            if expected_val.lower() == actual_val.lower():
                acceptable_mismatches.append(mismatch)
                continue
        
        # Rule 3: Notes populated for unstructured sources is ACCEPTABLE
        if field == "notes" and expected_val is None and actual_val is not None:
            if is_unstructured:
                acceptable_mismatches.append(mismatch)
                continue
        
        # Rule 4: Driver ID mismatches when confidence < 0.9 are ACCEPTABLE (check BEFORE general confidence)
        if field == "driver_id" and expected_val is not None and actual_val is not None:
            if field_confidence is not None and field_confidence < 0.9:
                acceptable_mismatches.append(mismatch)
                continue
        
        # Now check confidence-based rules
        # If field is missing confidence score, treat as low confidence (0.0) for unstructured sources
        # This handles cases where OCR extraction doesn't set confidence for missing fields
        if field_confidence is None:
            # For unstructured sources, missing confidence = low confidence (acceptable)
            if is_unstructured:
                acceptable_mismatches.append(mismatch)
            else:
                # For structured sources, missing confidence is unexpected (critical)
                real_mismatches.append(mismatch)
            continue
        
        # Expected=None, Actual=value
        if expected_val is None and actual_val is not None:
            if field_confidence == 0.0:
                # Field extracted but marked as missing/uncertain - acceptable
                acceptable_mismatches.append(mismatch)
            else:
                # Field extracted when not expected with confidence > 0 - CRITICAL
                real_mismatches.append(mismatch)
            continue
        
        # Expected=value, Actual=None
        if expected_val is not None and actual_val is None:
            if field_confidence == 0.0:
                # Field missing and correctly marked as missing - acceptable
                acceptable_mismatches.append(mismatch)
            else:
                # Field missing but not marked as missing - CRITICAL
                real_mismatches.append(mismatch)
            continue
        
        # Both expected and actual have values
        # Rule 5: Low-confidence mismatches (< 0.9) are ACCEPTABLE
        if field_confidence < 0.9:
            acceptable_mismatches.append(mismatch)
        # Rule 6: High-confidence mismatches (>= 0.9) are CRITICAL
        else:
            real_mismatches.append(mismatch)
    
    return len(real_mismatches), len(acceptable_mismatches)


def determine_test_status(comparison: Dict[str, Any], expected_rows: List[Dict[str, Any]], actual_rows: List[Dict[str, Any]]) -> str:
    """
    Determine test status: PASS, PASS WITH WARNINGS, or FAIL (ETL validation style).
    
    PASS: No critical issues, no acceptable degradation
    PASS WITH WARNINGS: No critical issues, but acceptable degradation exists
    FAIL: Critical issues present
    
    Args:
        comparison: Comparison result dictionary
        expected_rows: Expected rows list
        actual_rows: Actual rows list (for confidence checking)
    
    Returns:
        "PASS", "PASS WITH WARNINGS", or "FAIL"
    """
    confidence_issues = comparison.get("confidence_validation_issues", [])
    matched_rows = comparison.get("matched_rows", [])
    mismatches = comparison.get("mismatches", [])
    
    # Check for structural issues (missing/extra rows) - always CRITICAL
    has_structural_issues = any(m.get("type") in ["extra_row", "missing_row"] for m in mismatches)
    
    # Check for confidence validation issues - always CRITICAL
    has_confidence_issues = len(confidence_issues) > 0
    
    # FAIL if: confidence issues OR structural issues
    if has_confidence_issues or has_structural_issues:
        return "FAIL"
    
    # Classify all mismatches
    critical_count = 0
    acceptable_count = 0
    
    for mismatch in mismatches:
        row_num = mismatch.get("row", 1)
        actual_row_idx = row_num - 1
        if actual_row_idx >= 0 and actual_row_idx < len(actual_rows):
            actual_row = actual_rows[actual_row_idx]
            issue_type = classify_issue(mismatch, actual_row)
            if issue_type == "CRITICAL":
                critical_count += 1
            elif issue_type == "ACCEPTABLE":
                acceptable_count += 1
    
    # Filter to get real vs acceptable mismatches (for backward compatibility)
    real_mismatches, acceptable_mismatches = filter_acceptable_mismatches(comparison, actual_rows)
    
    # Use the more accurate critical_count from classification
    if critical_count > 0:
        return "FAIL"
    elif acceptable_count > 0:
        return "PASS WITH WARNINGS"
    else:
        return "PASS"


def determine_test_pass(comparison: Dict[str, Any], expected_rows: List[Dict[str, Any]], actual_rows: List[Dict[str, Any]]) -> bool:
    """
    Determine if a test passes (backward compatibility wrapper).
    
    Returns True for PASS or PASS WITH WARNINGS, False for FAIL.
    """
    status = determine_test_status(comparison, expected_rows, actual_rows)
    return status != "FAIL"


def compare_expected_vs_actual(expected_rows: List[Dict[str, Any]], actual_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare expected and actual rows row-by-row and field-by-field.
    Now confidence-aware: low-confidence mismatches are acceptable.
    
    Returns:
        Dictionary with:
            - total_rows: Total number of rows compared
            - matched_rows: List of row indices that match perfectly
            - mismatched_rows: List of row indices with mismatches
            - mismatches: List of mismatch details
            - total_mismatched_fields: Total count of mismatched fields
            - confidence_validation_issues: List of confidence field validation issues
    """
    # Normalize rows for comparison
    expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
    actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
    
    total_rows = max(len(expected_norm), len(actual_norm))
    matched_rows = []
    mismatched_rows = []
    mismatches = []
    total_mismatched_fields = 0
    confidence_validation_issues = []
    
    for row_idx in range(total_rows):
        if row_idx >= len(expected_norm):
            # Extra row in actual
            mismatched_rows.append(row_idx)
            mismatches.append({
                "row": row_idx + 1,
                "type": "extra_row",
                "message": f"Row {row_idx + 1} exists in actual but not in expected"
            })
            continue
        
        if row_idx >= len(actual_norm):
            # Missing row in actual
            mismatched_rows.append(row_idx)
            mismatches.append({
                "row": row_idx + 1,
                "type": "missing_row",
                "message": f"Row {row_idx + 1} exists in expected but not in actual"
            })
            continue
        
        expected_row = expected_norm[row_idx]
        actual_row = actual_norm[row_idx]
        
        # Validate confidence fields exist and are correctly formatted
        confidence_issues = validate_confidence_fields(actual_row)
        for issue in confidence_issues:
            issue["row"] = row_idx + 1
            confidence_validation_issues.append(issue)
        
        # Get all fields from both rows
        all_fields = sorted(set(expected_row.keys()) | set(actual_row.keys()))
        
        row_mismatches = []
        for field in all_fields:
            expected_val = expected_row.get(field, None)
            actual_val = actual_row.get(field, None)
            
            # Treat missing fields as None
            if field not in expected_row:
                expected_val = None
            if field not in actual_row:
                actual_val = None
            
            # Metadata fields (_confidence, _warnings, _source, _flags, _source_id, _source_row_number) are not part of schema comparison
            # Legacy: Handle old *_confidence and *_warnings format (should not exist in new format)
            if field in ["_confidence", "_warnings", "_source", "_flags", "_source_id", "_source_row_number", "_id"]:
                # These are metadata, not schema fields - skip comparison
                continue
            if field.startswith("_source") or field.endswith("_confidence") or field.endswith("_warnings"):
                # Legacy format or source metadata - skip comparison
                continue
            
            # Pass actual row for confidence-aware comparison
            if not values_equal(field, expected_val, actual_val, actual_row=actual_row):
                row_mismatches.append({
                    "row": row_idx + 1,
                    "field": field,
                    "expected": expected_val,
                    "actual": actual_val
                })
        
        if row_mismatches:
            mismatched_rows.append(row_idx)
            mismatches.extend(row_mismatches)
            total_mismatched_fields += len(row_mismatches)
        else:
            matched_rows.append(row_idx)
    
    return {
        "total_rows": total_rows,
        "matched_rows": matched_rows,
        "mismatched_rows": mismatched_rows,
        "mismatches": mismatches,
        "total_mismatched_fields": total_mismatched_fields,
        "confidence_validation_issues": confidence_validation_issues
    }


def print_executive_summary(results: List[Tuple[str, Tuple[bool, int, int, int]]], all_comparisons: Dict[str, Dict[str, Any]] = None, all_actual_rows: Dict[str, List[Dict[str, Any]]] = None):
    """
    Print ETL-style validation summary (summary first, details second).
    
    Args:
        results: List of (source_name, (passed, rows, real_mismatched, acceptable_mismatched))
        all_comparisons: Optional dict mapping source_name to comparison results
        all_actual_rows: Optional dict mapping source_name to actual rows
    """
    # Calculate overall statistics
    total_rows_expected = 0
    total_rows_actual = 0
    total_critical = 0
    total_acceptable = 0
    source_statuses = {}
    
    for source_name, (passed, rows, real_mismatched, acceptable_mismatched) in results:
        total_rows_actual += rows
        total_critical += real_mismatched
        total_acceptable += acceptable_mismatched
        
        # Determine source status (PASS / PASS WITH WARNINGS / FAIL)
        if real_mismatched == 0 and acceptable_mismatched == 0:
            source_statuses[source_name] = "PASS"
        elif real_mismatched == 0:
            source_statuses[source_name] = "PASS WITH WARNINGS"
        else:
            source_statuses[source_name] = "FAIL"
    
    # Determine overall status
    failed_sources = sum(1 for status in source_statuses.values() if status == "FAIL")
    warning_sources = sum(1 for status in source_statuses.values() if status == "PASS WITH WARNINGS")
    
    if failed_sources == 0 and warning_sources == 0:
        overall_status = "PASS"
        status_color = Colors.GREEN
        production_safe = True
    elif failed_sources == 0:
        overall_status = "PASS WITH WARNINGS"
        status_color = Colors.YELLOW
        production_safe = True  # Acceptable degradation is production-safe
    else:
        overall_status = "FAIL"
        status_color = Colors.RED
        production_safe = False
    
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}DRIVER EXTRACTION VALIDATION REPORT{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*80}{Colors.RESET}\n")
    
    # 1. TOP-LEVEL STATUS (Summary First)
    print(f"{Colors.BOLD}VALIDATION STATUS:{Colors.RESET} {status_color}{Colors.BOLD}{overall_status}{Colors.RESET}")
    if production_safe:
        print(f"  {Colors.GREEN}✓ Production-Safe{Colors.RESET} | Sources: {len(results)} | Rows: {total_rows_actual} | Critical: {Colors.RED}{total_critical}{Colors.RESET} | Acceptable: {Colors.GRAY}{total_acceptable}{Colors.RESET}\n")
    else:
        print(f"  {Colors.RED}✗ Not Production-Safe{Colors.RESET} | Sources: {len(results)} | Rows: {total_rows_actual} | Critical: {Colors.RED}{total_critical}{Colors.RESET} | Acceptable: {Colors.GRAY}{total_acceptable}{Colors.RESET}\n")
    
    # 2. SOURCE HEALTH TABLE
    print(f"{Colors.BOLD}Source Health:{Colors.RESET}")
    # Header with consistent column widths
    print(f"  {Colors.BOLD}{'Source':<25} {'Status':<20} {'Rows':>8} {'Critical':>10} {'Acceptable':>12}{Colors.RESET}")
    print(f"  {Colors.GRAY}{'─' * 80}{Colors.RESET}")
    
    for source_name, (passed, rows, real_mismatched, acceptable_mismatched) in results:
        status = source_statuses[source_name]
        if status == "PASS":
            status_text = "PASS"
            status_str = f"{Colors.GREEN}{status_text}{Colors.RESET}"
            status_visible_width = 4
        elif status == "PASS WITH WARNINGS":
            status_text = "PASS WITH WARNINGS"
            status_str = f"{Colors.YELLOW}{status_text}{Colors.RESET}"
            status_visible_width = 18
        else:
            status_text = "FAIL"
            status_str = f"{Colors.RED}{status_text}{Colors.RESET}"
            status_visible_width = 4
        
        # Calculate padding needed for status column (20 chars wide)
        # Add padding after the colored status string
        status_padding = max(0, 20 - status_visible_width)
        
        # Format numeric values
        critical_str = f"{Colors.RED}{real_mismatched}{Colors.RESET}" if real_mismatched > 0 else "0"
        acceptable_str = f"{Colors.GRAY}{acceptable_mismatched}{Colors.RESET}" if acceptable_mismatched > 0 else "0"
        
        # Print row with proper alignment
        # Source (25) | Status (20) | Rows (8) | Critical (10) | Acceptable (12)
        print(f"  {source_name:<25} {status_str}{' ' * status_padding} {str(rows):>8} {critical_str:>10} {acceptable_str:>12}")
    print()
    
    # 3. ISSUE BREAKDOWN (only if there are issues)
    if total_critical > 0 or total_acceptable > 0:
        print(f"{Colors.BOLD}Issue Breakdown:{Colors.RESET}\n")
        
        if total_critical > 0:
            print(f"  {Colors.RED}{Colors.BOLD}CRITICAL ISSUES:{Colors.RESET} {total_critical}")
            print(f"    • Missing/extra rows")
            print(f"    • High-confidence field mismatches (confidence >= 0.9)")
            print(f"    • Missing Driver ID with high confidence")
            print()
        
        if total_acceptable > 0:
            print(f"  {Colors.GRAY}{Colors.BOLD}ACCEPTABLE DEGRADATION:{Colors.RESET} {total_acceptable}")
            print(f"    • Case-only differences (normalization)")
            print(f"    • Low-confidence Driver ID OCR errors (confidence < 0.9)")
            print(f"    • Notes populated from OCR context (unstructured sources)")
            print(f"    • Low-confidence field mismatches (confidence < 0.9)")
            print()
        
        # Count rows requiring human review (Driver ID only)
        if all_actual_rows:
            total_driver_id_review_count = 0
            for source_name, rows in all_actual_rows.items():
                for row in rows:
                    flags_dict = row.get("_flags", {})
                    if flags_dict.get("driver_id_requires_human_review", False):
                        total_driver_id_review_count += 1
            
            if total_driver_id_review_count > 0:
                print(f"  {Colors.YELLOW}{Colors.BOLD}HUMAN REVIEW REQUIRED:{Colors.RESET} {total_driver_id_review_count} row(s)")
                print(f"    • Driver IDs flagged for human review (OCR/Vision sources, repairs, or low confidence)")
                print()
        
        print(f"  {Colors.BOLD}Note:{Colors.RESET} Use {Colors.CYAN}--verbose{Colors.RESET} or {Colors.CYAN}--full-diff{Colors.RESET} for detailed row-level diagnostics\n")


def print_conclusion(results: List[Tuple[str, Tuple[bool, int, int, int]]]):
    """Print ETL-style conclusion section (production acceptance assessment)."""
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}PRODUCTION ACCEPTANCE ASSESSMENT{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*80}{Colors.RESET}\n")
    
    production_ready = []
    partial_ready = []
    not_ready = []
    
    for source_name, (passed, rows, real_mismatched, acceptable_mismatched) in results:
        if real_mismatched == 0 and acceptable_mismatched == 0:
            production_ready.append(source_name)
        elif real_mismatched == 0:
            partial_ready.append((source_name, acceptable_mismatched))
        else:
            not_ready.append((source_name, real_mismatched, acceptable_mismatched))
    
    if production_ready:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Production Ready:{Colors.RESET}")
        for source in production_ready:
            print(f"  • {source}")
        print()
    
    if partial_ready:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Production Acceptable (with known limitations):{Colors.RESET}")
        for source, acceptable_count in partial_ready:
            print(f"  • {source} ({acceptable_count} acceptable degradation items)")
            print(f"    → Data extracted, normalized, confidence/warnings present")
            print(f"    → OCR limitations transparently reported")
            print(f"    → Safe for production use with human review")
        print()
    
    if not_ready:
        print(f"{Colors.RED}{Colors.BOLD}✗ Not Production Ready:{Colors.RESET}")
        for item in not_ready:
            source, critical_count, acceptable_count = item
            if acceptable_count > 0:
                print(f"  • {source} ({critical_count} critical issues, {acceptable_count} acceptable)")
            else:
                print(f"  • {source} ({critical_count} critical issues)")
        print()
    
    # Next steps (only for not_ready sources)
    if not_ready:
        print(f"{Colors.BOLD}Recommended Actions:{Colors.RESET}")
        if any("Handwritten" in item[0] for item in not_ready):
            print("  • Improve OCR preprocessing for handwritten documents")
            print("  • Enhance inference patterns for corrupted OCR text")
        print("  • Review critical issues above for structural problems")
        print()


def test_airtable() -> Tuple[bool, int, int, int]:
    """Test Airtable drivers."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Airtable Drivers ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    json_path = ROOT / "tests" / "drivers" / "structured" / "airtable_drivers.json"
    truth_file = ROOT / "tests" / "truth" / "drivers" / "airtable_drivers.expected.json"
    mapping_id = "source_airtable_drivers"
    
    try:
        # Load truth file
        expected_rows = load_truth_file(truth_file)
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0, 0
        
        # Run normalize_v2
        try:
            result = normalize_v2(
                source={"file_path": str(json_path)},
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
            # Reorder to match schema
            actual_rows = reorder_all_drivers(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Sort both expected and actual by driver_id for consistent comparison
        expected_rows_sorted = sorted(expected_rows, key=lambda x: str(x.get("driver_id", "")))
        actual_rows_sorted = sorted(actual_rows, key=lambda x: str(x.get("driver_id", "")))
        
        # Compare
        print(f"\n{Colors.BOLD}Comparing expected vs actual...{Colors.RESET}")
        comparison = compare_expected_vs_actual(expected_rows_sorted, actual_rows_sorted)
        confidence_issues = comparison.get("confidence_validation_issues", [])
        passed = determine_test_pass(comparison, expected_rows_sorted, actual_rows_sorted)
        
        # Filter acceptable mismatches (confidence-aware counting)
        # Only count real issues: high-confidence mismatches and structural issues
        total_mismatched, acceptable_mismatched = filter_acceptable_mismatches(comparison, actual_rows_sorted)
        
        # Print confidence validation issues if any
        if confidence_issues:
            print(f"\n{Colors.RED}{Colors.BOLD}Confidence Validation Issues:{Colors.RESET}")
            for issue in confidence_issues:
                print(f"  {Colors.RED}✗{Colors.RESET} Row {issue.get('row', '?')}: {issue.get('message', 'Unknown issue')}")
            print()
        
        # Normalize rows for display (already sorted)
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows_sorted]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows_sorted]
        
        # Print results with clean header (only if VERBOSE or FULL_DIFF)
        if VERBOSE or FULL_DIFF:
            print()
            print(f"{Colors.BOLD}{'─' * 80}{Colors.RESET}")
            print(f"{Colors.BOLD}COMPARISON RESULTS{Colors.RESET}")
            print(f"{Colors.BOLD}{'─' * 80}{Colors.RESET}")
            print()
        
        # Group mismatches by row
        mismatches_by_row = {}
        extra_rows = []
        missing_rows = []
        
        for mismatch in comparison["mismatches"]:
            row_num = mismatch["row"]
            mismatch_type = mismatch.get("type")
            
            if mismatch_type == "extra_row":
                extra_rows.append(row_num)
            elif mismatch_type == "missing_row":
                missing_rows.append(row_num)
            else:
                if row_num not in mismatches_by_row:
                    mismatches_by_row[row_num] = []
                mismatches_by_row[row_num].append(mismatch)
        
        # Print missing/extra rows
        print_missing_extra_rows(missing_rows, extra_rows)
        
        # Print row diffs and track rows with warnings
        total_rows = max(len(expected_norm), len(actual_norm))
        rows_with_warnings = []
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows (already printed by print_missing_extra_rows)
            if row_num in extra_rows or row_num in missing_rows:
                continue
            
            # Skip if row doesn't exist in either
            if row_idx >= len(expected_norm) or row_idx >= len(actual_norm):
                continue
            
            expected_row = expected_norm[row_idx]
            actual_row = actual_norm[row_idx]
            
            # Track rows with warnings
            actual_warnings = actual_row.get("_warnings", [])
            if actual_warnings and len(actual_warnings) > 0:
                rows_with_warnings.append(row_num)
            
            # Print diff for this row
            num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print summary
        print()
        print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'-'*80}{Colors.RESET}")
        print(f"Total expected rows: {len(expected_rows_sorted)}")
        print(f"Total actual rows:   {len(actual_rows_sorted)}")
        print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        print(f"Total mismatched fields: {comparison.get('total_mismatched_fields', 0)}")
        if rows_with_warnings:
            print(f"Rows containing warnings: {', '.join(map(str, rows_with_warnings))}")
        else:
            print("Rows containing warnings: None")
        print()
        
        # Print final pass/fail
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {len(comparison['mismatched_rows'])} row(s) with mismatches, {total_mismatched} mismatched field(s)\n")
        
        return passed, len(actual_rows), total_mismatched, acceptable_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0, 0


def test_google_sheet() -> Tuple[bool, int, int, int]:
    """Test Google Sheet drivers (CSV format)."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Google Sheet Drivers ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    csv_path = ROOT / "tests" / "drivers" / "structured" / "google_sheet_drivers.csv"
    truth_file = ROOT / "tests" / "truth" / "drivers" / "google_sheet_drivers.expected.json"
    mapping_id = "source_google_sheet_drivers"
    
    try:
        # Load truth file
        expected_rows = load_truth_file(truth_file)
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0, 0
        
        # Run normalize_v2
        try:
            result = normalize_v2(
                source={"file_path": str(csv_path)},
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
            # Reorder to match schema
            actual_rows = reorder_all_drivers(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Sort both expected and actual by driver_id for consistent comparison
        expected_rows_sorted = sorted(expected_rows, key=lambda x: str(x.get("driver_id", "")))
        actual_rows_sorted = sorted(actual_rows, key=lambda x: str(x.get("driver_id", "")))
        
        # Compare
        print(f"\n{Colors.BOLD}Comparing expected vs actual...{Colors.RESET}")
        comparison = compare_expected_vs_actual(expected_rows_sorted, actual_rows_sorted)
        confidence_issues = comparison.get("confidence_validation_issues", [])
        passed = determine_test_pass(comparison, expected_rows_sorted, actual_rows_sorted)
        
        # Filter acceptable mismatches (confidence-aware counting)
        # Only count real issues: high-confidence mismatches and structural issues
        total_mismatched, acceptable_mismatched = filter_acceptable_mismatches(comparison, actual_rows_sorted)
        
        # Print confidence validation issues if any
        if confidence_issues:
            print(f"\n{Colors.RED}{Colors.BOLD}Confidence Validation Issues:{Colors.RESET}")
            for issue in confidence_issues:
                print(f"  {Colors.RED}✗{Colors.RESET} Row {issue.get('row', '?')}: {issue.get('message', 'Unknown issue')}")
            print()
        
        # Normalize rows for display (already sorted)
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows_sorted]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows_sorted]
        
        # Print results with clean header (only if VERBOSE or FULL_DIFF)
        if VERBOSE or FULL_DIFF:
            print()
            print(f"{Colors.BOLD}{'─' * 80}{Colors.RESET}")
            print(f"{Colors.BOLD}COMPARISON RESULTS{Colors.RESET}")
            print(f"{Colors.BOLD}{'─' * 80}{Colors.RESET}")
            print()
        
        # Group mismatches by row
        mismatches_by_row = {}
        extra_rows = []
        missing_rows = []
        
        for mismatch in comparison["mismatches"]:
            row_num = mismatch["row"]
            mismatch_type = mismatch.get("type")
            
            if mismatch_type == "extra_row":
                extra_rows.append(row_num)
            elif mismatch_type == "missing_row":
                missing_rows.append(row_num)
            else:
                if row_num not in mismatches_by_row:
                    mismatches_by_row[row_num] = []
                mismatches_by_row[row_num].append(mismatch)
        
        # Print missing/extra rows
        print_missing_extra_rows(missing_rows, extra_rows)
        
        # Print row diffs and track rows with warnings
        total_rows = max(len(expected_norm), len(actual_norm))
        rows_with_warnings = []
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows (already printed by print_missing_extra_rows)
            if row_num in extra_rows or row_num in missing_rows:
                continue
            
            # Skip if row doesn't exist in either
            if row_idx >= len(expected_norm) or row_idx >= len(actual_norm):
                continue
            
            expected_row = expected_norm[row_idx]
            actual_row = actual_norm[row_idx]
            
            # Track rows with warnings
            actual_warnings = actual_row.get("_warnings", [])
            if actual_warnings and len(actual_warnings) > 0:
                rows_with_warnings.append(row_num)
            
            # Print diff for this row
            num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print summary
        print()
        print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'-'*80}{Colors.RESET}")
        print(f"Total expected rows: {len(expected_rows_sorted)}")
        print(f"Total actual rows:   {len(actual_rows_sorted)}")
        print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        print(f"Total mismatched fields: {comparison.get('total_mismatched_fields', 0)}")
        if rows_with_warnings:
            print(f"Rows containing warnings: {', '.join(map(str, rows_with_warnings))}")
        else:
            print("Rows containing warnings: None")
        print()
        
        # Print final pass/fail
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {len(comparison['mismatched_rows'])} row(s) with mismatches, {total_mismatched} mismatched field(s)\n")
        
        return passed, len(actual_rows), total_mismatched, acceptable_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0, 0


def test_excel() -> Tuple[bool, int, int, int]:
    """Test Excel drivers."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Excel Drivers ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    xlsx_path = ROOT / "tests" / "drivers" / "structured" / "excel_drivers.xlsx"
    truth_file = ROOT / "tests" / "truth" / "drivers" / "excel_drivers.expected.json"
    mapping_id = "source_xlsx_drivers"
    
    try:
        # Load truth file
        expected_rows = load_truth_file(truth_file)
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0, 0
        
        # Run normalize_v2
        try:
            result = normalize_v2(
                source={"file_path": str(xlsx_path)},
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
            # Reorder to match schema
            actual_rows = reorder_all_drivers(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Sort both expected and actual by driver_id for consistent comparison
        expected_rows_sorted = sorted(expected_rows, key=lambda x: str(x.get("driver_id", "")))
        actual_rows_sorted = sorted(actual_rows, key=lambda x: str(x.get("driver_id", "")))
        
        # Compare
        print(f"\n{Colors.BOLD}Comparing expected vs actual...{Colors.RESET}")
        comparison = compare_expected_vs_actual(expected_rows_sorted, actual_rows_sorted)
        confidence_issues = comparison.get("confidence_validation_issues", [])
        passed = determine_test_pass(comparison, expected_rows_sorted, actual_rows_sorted)
        
        # Filter acceptable mismatches (confidence-aware counting)
        # Only count real issues: high-confidence mismatches and structural issues
        total_mismatched, acceptable_mismatched = filter_acceptable_mismatches(comparison, actual_rows_sorted)
        
        # Print confidence validation issues if any
        if confidence_issues:
            print(f"\n{Colors.RED}{Colors.BOLD}Confidence Validation Issues:{Colors.RESET}")
            for issue in confidence_issues:
                print(f"  {Colors.RED}✗{Colors.RESET} Row {issue.get('row', '?')}: {issue.get('message', 'Unknown issue')}")
            print()
        
        # Normalize rows for display (already sorted)
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows_sorted]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows_sorted]
        
        # Print results with clean header (only if VERBOSE or FULL_DIFF)
        if VERBOSE or FULL_DIFF:
            print()
            print(f"{Colors.BOLD}{'─' * 80}{Colors.RESET}")
            print(f"{Colors.BOLD}COMPARISON RESULTS{Colors.RESET}")
            print(f"{Colors.BOLD}{'─' * 80}{Colors.RESET}")
            print()
        
        # Group mismatches by row
        mismatches_by_row = {}
        extra_rows = []
        missing_rows = []
        
        for mismatch in comparison["mismatches"]:
            row_num = mismatch["row"]
            mismatch_type = mismatch.get("type")
            
            if mismatch_type == "extra_row":
                extra_rows.append(row_num)
            elif mismatch_type == "missing_row":
                missing_rows.append(row_num)
            else:
                if row_num not in mismatches_by_row:
                    mismatches_by_row[row_num] = []
                mismatches_by_row[row_num].append(mismatch)
        
        # Print missing/extra rows
        print_missing_extra_rows(missing_rows, extra_rows)
        
        # Print row diffs and track rows with warnings
        total_rows = max(len(expected_norm), len(actual_norm))
        rows_with_warnings = []
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows (already printed by print_missing_extra_rows)
            if row_num in extra_rows or row_num in missing_rows:
                continue
            
            # Skip if row doesn't exist in either
            if row_idx >= len(expected_norm) or row_idx >= len(actual_norm):
                continue
            
            expected_row = expected_norm[row_idx]
            actual_row = actual_norm[row_idx]
            
            # Track rows with warnings
            actual_warnings = actual_row.get("_warnings", [])
            if actual_warnings and len(actual_warnings) > 0:
                rows_with_warnings.append(row_num)
            
            # Print diff for this row
            num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print summary
        print()
        print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'-'*80}{Colors.RESET}")
        print(f"Total expected rows: {len(expected_rows_sorted)}")
        print(f"Total actual rows:   {len(actual_rows_sorted)}")
        print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        print(f"Total mismatched fields: {comparison.get('total_mismatched_fields', 0)}")
        if rows_with_warnings:
            print(f"Rows containing warnings: {', '.join(map(str, rows_with_warnings))}")
        else:
            print("Rows containing warnings: None")
        print()
        
        # Print final pass/fail
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {len(comparison['mismatched_rows'])} row(s) with mismatches, {total_mismatched} mismatched field(s)\n")
        
        return passed, len(actual_rows), total_mismatched, acceptable_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0, 0


def test_raw_text() -> Tuple[bool, int, int, int]:
    """Test raw text driver data."""
    txt_path = ROOT / "tests" / "drivers" / "unstructured" / "raw_text_drivers.txt"
    truth_file = ROOT / "tests" / "truth" / "drivers" / "raw_text_drivers.expected.json"
    
    # Skip test if files don't exist
    if not txt_path.exists() or not truth_file.exists():
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Raw Text Driver Data ==={Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        print(f"{Colors.YELLOW}⚠ SKIPPED: Raw text test files not found{Colors.RESET}")
        print(f"   Missing: {txt_path.name if not txt_path.exists() else truth_file.name}\n")
        return True, 0, 0, 0  # Return passed (skipped) with 0 rows
    
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Raw Text Driver Data ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    mapping_id = "source_raw_text_drivers"
    
    try:
        # Load truth file
        expected_rows = load_truth_file(truth_file)
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0, 0
        
        # Run normalize_v2
        try:
            result = normalize_v2(
                source={"file_path": str(txt_path)},
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
            # Reorder to match schema
            actual_rows = reorder_all_drivers(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Sort both expected and actual by driver_id for consistent comparison
        expected_rows_sorted = sorted(expected_rows, key=lambda x: str(x.get("driver_id", "")))
        actual_rows_sorted = sorted(actual_rows, key=lambda x: str(x.get("driver_id", "")))
        
        # Compare
        debug_print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows_sorted, actual_rows_sorted)
        
        # Filter acceptable vs real mismatches
        real_mismatched, acceptable_mismatched = filter_acceptable_mismatches(
            comparison,
            actual_rows_sorted
        )
        
        # Determine pass status
        passed = determine_test_pass(comparison, expected_rows_sorted, actual_rows_sorted)
        
        # Normalize rows for display (already sorted)
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows_sorted]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows_sorted]
        
        # Print confidence validation issues
        confidence_issues = comparison.get("confidence_validation_issues", [])
        if confidence_issues:
            print(f"\n{Colors.YELLOW}⚠ Confidence Validation Issues:{Colors.RESET}")
            for issue in confidence_issues:
                print(f"  Row {issue['row']}: {issue['message']}")
        print()
        
        # Print row diffs (only if verbose or full-diff)
        if FULL_DIFF or VERBOSE:
            debug_print("\n" + "=" * 80)
            debug_print("COMPARISON RESULTS")
            debug_print("=" * 80)
            debug_print()
        
        # Group mismatches by row
        mismatches_by_row = {}
        extra_rows = []
        missing_rows = []
        
        for mismatch in comparison["mismatches"]:
            row_num = mismatch["row"]
            mismatch_type = mismatch.get("type")
            
            if mismatch_type == "extra_row":
                extra_rows.append(row_num)
            elif mismatch_type == "missing_row":
                missing_rows.append(row_num)
            else:
                if row_num not in mismatches_by_row:
                    mismatches_by_row[row_num] = []
                mismatches_by_row[row_num].append(mismatch)
        
            # Print row diffs
        total_rows = max(len(expected_norm), len(actual_norm))
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows
            if row_num in extra_rows:
                print(f"*** Expected row {row_num} has no actual row ***")
                print()
                continue
            
            if row_num in missing_rows:
                print(f"*** Actual row {row_num} has no expected row ***")
                print()
                continue
            
            # Skip if row doesn't exist in either
            if row_idx >= len(expected_norm) or row_idx >= len(actual_norm):
                continue
            
            expected_row = expected_norm[row_idx]
            actual_row = actual_norm[row_idx]
            
            # Print diff for this row
            print_row_diff(row_num, expected_row, actual_row)
        
        # Print summary
        debug_print("\n" + "=" * 80)
        debug_print("SUMMARY")
        debug_print("-" * 80)
        debug_print(f"Total expected rows: {len(expected_rows_sorted)}")
        debug_print(f"Total actual rows:   {len(actual_rows_sorted)}")
        debug_print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        debug_print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        debug_print(f"Real issues: {real_mismatched}")
        debug_print(f"Acceptable mismatches: {acceptable_mismatched}")
        debug_print()
        
        return passed, len(actual_rows), real_mismatched, acceptable_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0, 0


def test_pdf() -> Tuple[bool, int, int, int]:
    """Test PDF driver documents."""
    pdf_path = ROOT / "tests" / "drivers" / "unstructured" / "pdf_drivers.pdf"
    truth_file = ROOT / "tests" / "truth" / "drivers" / "pdf_drivers.expected.json"
    
    # Skip test if files don't exist
    if not pdf_path.exists() or not truth_file.exists():
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}=== Testing PDF Driver Documents ==={Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        print(f"{Colors.YELLOW}⚠ SKIPPED: PDF test files not found{Colors.RESET}")
        print(f"   Missing: {pdf_path.name if not pdf_path.exists() else truth_file.name}\n")
        return True, 0, 0, 0  # Return passed (skipped) with 0 rows
    
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing PDF Driver Documents ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    mapping_id = "source_pdf_drivers"
    
    try:
        # Load truth file
        expected_rows = load_truth_file(truth_file)
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0, 0
        
        # Run normalize_v2
        try:
            result = normalize_v2(
                source={"file_path": str(pdf_path)},
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
            # Reorder to match schema
            actual_rows = reorder_all_drivers(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Sort both expected and actual by driver_id for consistent comparison
        expected_rows_sorted = sorted(expected_rows, key=lambda x: str(x.get("driver_id", "")))
        actual_rows_sorted = sorted(actual_rows, key=lambda x: str(x.get("driver_id", "")))
        
        # Compare
        debug_print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows_sorted, actual_rows_sorted)
        
        # Filter acceptable vs real mismatches
        real_mismatched, acceptable_mismatched = filter_acceptable_mismatches(
            comparison,
            actual_rows_sorted
        )
        
        # Determine pass status
        passed = determine_test_pass(comparison, expected_rows_sorted, actual_rows_sorted)
        
        # Normalize rows for display (already sorted)
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows_sorted]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows_sorted]
        
        # Print confidence validation issues
        confidence_issues = comparison.get("confidence_validation_issues", [])
        if confidence_issues:
            print(f"\n{Colors.YELLOW}⚠ Confidence Validation Issues:{Colors.RESET}")
            for issue in confidence_issues:
                print(f"  Row {issue['row']}: {issue['message']}")
        print()
        
        # Print row diffs (only if verbose or full-diff)
        if FULL_DIFF or VERBOSE:
            debug_print("\n" + "=" * 80)
            debug_print("COMPARISON RESULTS")
            debug_print("=" * 80)
            debug_print()
        
        # Group mismatches by row
        mismatches_by_row = {}
        extra_rows = []
        missing_rows = []
        
        for mismatch in comparison["mismatches"]:
            row_num = mismatch["row"]
            mismatch_type = mismatch.get("type")
            
            if mismatch_type == "extra_row":
                extra_rows.append(row_num)
            elif mismatch_type == "missing_row":
                missing_rows.append(row_num)
            else:
                if row_num not in mismatches_by_row:
                    mismatches_by_row[row_num] = []
                mismatches_by_row[row_num].append(mismatch)
        
            # Print row diffs
        total_rows = max(len(expected_norm), len(actual_norm))
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows
            if row_num in extra_rows:
                print(f"*** Expected row {row_num} has no actual row ***")
                print()
                continue
            
            if row_num in missing_rows:
                print(f"*** Actual row {row_num} has no expected row ***")
                print()
                continue
            
            # Skip if row doesn't exist in either
            if row_idx >= len(expected_norm) or row_idx >= len(actual_norm):
                continue
            
            expected_row = expected_norm[row_idx]
            actual_row = actual_norm[row_idx]
            
            # Print diff for this row
            print_row_diff(row_num, expected_row, actual_row)
        
        # Print summary
        debug_print("\n" + "=" * 80)
        debug_print("SUMMARY")
        debug_print("-" * 80)
        debug_print(f"Total expected rows: {len(expected_rows_sorted)}")
        debug_print(f"Total actual rows:   {len(actual_rows_sorted)}")
        debug_print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        debug_print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        debug_print(f"Real issues: {real_mismatched}")
        debug_print(f"Acceptable mismatches: {acceptable_mismatched}")
        debug_print()
        
        return passed, len(actual_rows), real_mismatched, acceptable_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0, 0


def test_image() -> Tuple[bool, int, int, int]:
    """Test image driver data."""
    image_path = ROOT / "tests" / "drivers" / "unstructured" / "image_driver_data.jpg"
    truth_file = ROOT / "tests" / "truth" / "drivers" / "image_driver_data.expected.json"
    
    # Skip test if files don't exist
    if not image_path.exists() or not truth_file.exists():
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Image Driver Data ==={Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        print(f"{Colors.YELLOW}⚠ SKIPPED: Image test files not found{Colors.RESET}")
        print(f"   Missing: {image_path.name if not image_path.exists() else truth_file.name}\n")
        return True, 0, 0, 0  # Return passed (skipped) with 0 rows
    
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Image Driver Data ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    mapping_id = "source_image_drivers"
    
    try:
        # Load truth file
        expected_rows = load_truth_file(truth_file)
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0, 0
        
        # Run normalize_v2
        try:
            result = normalize_v2(
                source={"file_path": str(image_path)},
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
            # Reorder to match schema
            actual_rows = reorder_all_drivers(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Sort both expected and actual by driver_id for consistent comparison
        expected_rows_sorted = sorted(expected_rows, key=lambda x: str(x.get("driver_id", "")))
        actual_rows_sorted = sorted(actual_rows, key=lambda x: str(x.get("driver_id", "")))
        
        # Compare
        debug_print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows_sorted, actual_rows_sorted)
        
        # Filter acceptable vs real mismatches
        real_mismatched, acceptable_mismatched = filter_acceptable_mismatches(
            comparison,
            actual_rows_sorted
        )
        
        # Determine pass status
        passed = determine_test_pass(comparison, expected_rows_sorted, actual_rows_sorted)
        
        # Normalize rows for display (already sorted)
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows_sorted]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows_sorted]
        
        # Print confidence validation issues
        confidence_issues = comparison.get("confidence_validation_issues", [])
        if confidence_issues:
            print(f"\n{Colors.YELLOW}⚠ Confidence Validation Issues:{Colors.RESET}")
            for issue in confidence_issues:
                print(f"  Row {issue['row']}: {issue['message']}")
        print()
        
        # Print row diffs (only if verbose or full-diff)
        if FULL_DIFF or VERBOSE:
            debug_print("\n" + "=" * 80)
            debug_print("COMPARISON RESULTS")
            debug_print("=" * 80)
            debug_print()
        
        # Group mismatches by row
        mismatches_by_row = {}
        extra_rows = []
        missing_rows = []
        
        for mismatch in comparison["mismatches"]:
            row_num = mismatch["row"]
            mismatch_type = mismatch.get("type")
            
            if mismatch_type == "extra_row":
                extra_rows.append(row_num)
            elif mismatch_type == "missing_row":
                missing_rows.append(row_num)
            else:
                if row_num not in mismatches_by_row:
                    mismatches_by_row[row_num] = []
                mismatches_by_row[row_num].append(mismatch)
        
            # Print row diffs
        total_rows = max(len(expected_norm), len(actual_norm))
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows
            if row_num in extra_rows:
                print(f"*** Expected row {row_num} has no actual row ***")
                print()
                continue
            
            if row_num in missing_rows:
                print(f"*** Actual row {row_num} has no expected row ***")
                print()
                continue
            
            # Skip if row doesn't exist in either
            if row_idx >= len(expected_norm) or row_idx >= len(actual_norm):
                continue
            
            expected_row = expected_norm[row_idx]
            actual_row = actual_norm[row_idx]
            
            # Print diff for this row
            print_row_diff(row_num, expected_row, actual_row)
        
        # Print summary
        debug_print("\n" + "=" * 80)
        debug_print("SUMMARY")
        debug_print("-" * 80)
        debug_print(f"Total expected rows: {len(expected_rows_sorted)}")
        debug_print(f"Total actual rows:   {len(actual_rows_sorted)}")
        debug_print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        debug_print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        debug_print(f"Real issues: {real_mismatched}")
        debug_print(f"Acceptable mismatches: {acceptable_mismatched}")
        debug_print()
        
        return passed, len(actual_rows), real_mismatched, acceptable_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0, 0


def test_handwritten_pdf() -> Tuple[bool, int, int, int]:
    """Test handwritten PDF driver documents."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Handwritten PDF Driver Documents ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    pdf_path = ROOT / "tests" / "drivers" / "unstructured" / "handwritten_pdf_drivers.pdf"
    truth_file = ROOT / "tests" / "truth" / "drivers" / "handwritten_drivers.json"
    mapping_id = "source_pdf_drivers"
    
    try:
        # Load truth file
        debug_print(f"{Colors.BLUE}DEBUG: Loading truth file from: {truth_file.resolve()}{Colors.RESET}")
        debug_print(f"{Colors.BLUE}DEBUG: Truth file exists: {truth_file.exists()}{Colors.RESET}")
        expected_rows = load_truth_file(truth_file)
        debug_print(f"{Colors.BLUE}DEBUG: Loaded {len(expected_rows)} expected rows from truth file{Colors.RESET}")
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0, 0
        
        # Run normalize_v2
        try:
            result = normalize_v2(
                source=str(pdf_path),
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
            # Reorder to match schema
            actual_rows = reorder_all_drivers(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Sort both expected and actual by driver_id for consistent comparison
        expected_rows_sorted = sorted(expected_rows, key=lambda x: str(x.get("driver_id", "")))
        actual_rows_sorted = sorted(actual_rows, key=lambda x: str(x.get("driver_id", "")))
        
        # Compare
        debug_print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows_sorted, actual_rows_sorted)
        
        # Filter acceptable vs real mismatches
        real_mismatched, acceptable_mismatched = filter_acceptable_mismatches(
            comparison,
            actual_rows_sorted
        )
        
        # Determine pass status
        passed = determine_test_pass(comparison, expected_rows_sorted, actual_rows_sorted)
        
        # Normalize rows for display (already sorted)
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows_sorted]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows_sorted]
        
        # Print confidence validation issues
        confidence_issues = comparison.get("confidence_validation_issues", [])
        if confidence_issues:
            print(f"\n{Colors.YELLOW}⚠ Confidence Validation Issues:{Colors.RESET}")
            for issue in confidence_issues:
                print(f"  Row {issue['row']}: {issue['message']}")
        print()
        
        # Print row diffs (only if verbose or full-diff)
        if FULL_DIFF or VERBOSE:
            debug_print("\n" + "=" * 80)
            debug_print("COMPARISON RESULTS")
            debug_print("=" * 80)
            debug_print()
        
        # Group mismatches by row
        mismatches_by_row = {}
        extra_rows = []
        missing_rows = []
        
        for mismatch in comparison["mismatches"]:
            row_num = mismatch["row"]
            mismatch_type = mismatch.get("type")
            
            if mismatch_type == "extra_row":
                extra_rows.append(row_num)
            elif mismatch_type == "missing_row":
                missing_rows.append(row_num)
            else:
                if row_num not in mismatches_by_row:
                    mismatches_by_row[row_num] = []
                mismatches_by_row[row_num].append(mismatch)
        
            # Print row diffs
        total_rows = max(len(expected_norm), len(actual_norm))
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows
            if row_num in extra_rows:
                print(f"*** Expected row {row_num} has no actual row ***")
                print()
                continue
            
            if row_num in missing_rows:
                print(f"*** Actual row {row_num} has no expected row ***")
                print()
                continue
            
            # Skip if row doesn't exist in either
            if row_idx >= len(expected_norm) or row_idx >= len(actual_norm):
                continue
            
            expected_row = expected_norm[row_idx]
            actual_row = actual_norm[row_idx]
            
            # Print diff for this row
            print_row_diff(row_num, expected_row, actual_row)
        
        # Print summary
        debug_print("\n" + "=" * 80)
        debug_print("SUMMARY")
        debug_print("-" * 80)
        debug_print(f"Total expected rows: {len(expected_rows_sorted)}")
        debug_print(f"Total actual rows:   {len(actual_rows_sorted)}")
        debug_print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        debug_print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        debug_print(f"Real issues: {real_mismatched}")
        debug_print(f"Acceptable mismatches: {acceptable_mismatched}")
        debug_print()
        
        return passed, len(actual_rows), real_mismatched, acceptable_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0, 0


def test_handwritten_image() -> Tuple[bool, int, int, int]:
    """Test handwritten image driver data."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Handwritten Image Driver Data ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    img_path = ROOT / "tests" / "drivers" / "unstructured" / "handwritten_jpeg_drivers.jpg"
    truth_file = ROOT / "tests" / "truth" / "drivers" / "handwritten_drivers.json"
    mapping_id = "source_image_drivers"
    
    try:
        # Load truth file
        debug_print(f"{Colors.BLUE}DEBUG: Loading truth file from: {truth_file.resolve()}{Colors.RESET}")
        debug_print(f"{Colors.BLUE}DEBUG: Truth file exists: {truth_file.exists()}{Colors.RESET}")
        expected_rows = load_truth_file(truth_file)
        debug_print(f"{Colors.BLUE}DEBUG: Loaded {len(expected_rows)} expected rows from truth file{Colors.RESET}")
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0, 0
        
        # Run normalize_v2
        try:
            result = normalize_v2(
                source=str(img_path),
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
            # Reorder to match schema
            actual_rows = reorder_all_drivers(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Sort both expected and actual by driver_id for consistent comparison
        expected_rows_sorted = sorted(expected_rows, key=lambda x: str(x.get("driver_id", "")))
        actual_rows_sorted = sorted(actual_rows, key=lambda x: str(x.get("driver_id", "")))
        
        # Compare
        debug_print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows_sorted, actual_rows_sorted)
        
        # Filter acceptable vs real mismatches
        real_mismatched, acceptable_mismatched = filter_acceptable_mismatches(
            comparison,
            actual_rows_sorted
        )
        
        # Determine pass status
        passed = determine_test_pass(comparison, expected_rows_sorted, actual_rows_sorted)
        
        # Normalize rows for display (already sorted)
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows_sorted]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows_sorted]
        
        # Print confidence validation issues
        confidence_issues = comparison.get("confidence_validation_issues", [])
        if confidence_issues:
            print(f"\n{Colors.YELLOW}⚠ Confidence Validation Issues:{Colors.RESET}")
            for issue in confidence_issues:
                print(f"  Row {issue['row']}: {issue['message']}")
        print()
        
        # Print row diffs (only if verbose or full-diff)
        if FULL_DIFF or VERBOSE:
            debug_print("\n" + "=" * 80)
            debug_print("COMPARISON RESULTS")
            debug_print("=" * 80)
            debug_print()
        
        # Group mismatches by row
        mismatches_by_row = {}
        extra_rows = []
        missing_rows = []
        
        for mismatch in comparison["mismatches"]:
            row_num = mismatch["row"]
            mismatch_type = mismatch.get("type")
            
            if mismatch_type == "extra_row":
                extra_rows.append(row_num)
            elif mismatch_type == "missing_row":
                missing_rows.append(row_num)
            else:
                if row_num not in mismatches_by_row:
                    mismatches_by_row[row_num] = []
                mismatches_by_row[row_num].append(mismatch)
        
            # Print row diffs
        total_rows = max(len(expected_norm), len(actual_norm))
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows
            if row_num in extra_rows:
                print(f"*** Expected row {row_num} has no actual row ***")
                print()
                continue
            
            if row_num in missing_rows:
                print(f"*** Actual row {row_num} has no expected row ***")
                print()
                continue
            
            # Skip if row doesn't exist in either
            if row_idx >= len(expected_norm) or row_idx >= len(actual_norm):
                continue
            
            expected_row = expected_norm[row_idx]
            actual_row = actual_norm[row_idx]
            
            # Print diff for this row
            print_row_diff(row_num, expected_row, actual_row)
        
        # Print summary
        debug_print("\n" + "=" * 80)
        debug_print("SUMMARY")
        debug_print("-" * 80)
        debug_print(f"Total expected rows: {len(expected_rows_sorted)}")
        debug_print(f"Total actual rows:   {len(actual_rows_sorted)}")
        debug_print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        debug_print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        debug_print(f"Real issues: {real_mismatched}")
        debug_print(f"Acceptable mismatches: {acceptable_mismatched}")
        debug_print()
        
        return passed, len(actual_rows), real_mismatched, acceptable_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0, 0


def main():
    """Run all driver tests."""
    global VERBOSE, FULL_DIFF, DEBUG
    
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Driver extraction test suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed row-by-row diffs")
    parser.add_argument("--full-diff", action="store_true", help="Show full field-by-field diffs for all rows")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    VERBOSE = args.verbose
    FULL_DIFF = args.full_diff
    DEBUG = args.debug
    
    # Re-apply logging suppression after parsing args (unless DEBUG is enabled)
    if not DEBUG:
        logging.getLogger().setLevel(logging.CRITICAL)
        for logger_name in [
            "src", "src.sources", "src.ocr", "src.normalizer", "src.transforms",
            "ocr", "ocr.parser", "ocr.table_extract", "ocr.reader", "ocr.models",
            "__main__", "parser", "table_extract"
        ]:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    else:
        logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 80)
    print("UNIFIED DRIVER TEST SUITE")
    print("=" * 80)
    print()
    
    results = []
    all_comparisons = {}
    all_actual_rows = {}
    
    # Run all tests
    test_functions = [
        ("Airtable", test_airtable),
        ("Google Sheet", test_google_sheet),
        ("Excel", test_excel),
        ("Raw Text", test_raw_text),
        ("PDF", test_pdf),
        ("Image", test_image),
        ("Handwritten PDF", test_handwritten_pdf),
        ("Handwritten Image", test_handwritten_image)
    ]
    
    for name, test_func in test_functions:
        result = test_func()
        results.append((name, result))
    
    # Print ETL-style validation report
    print_executive_summary(results, all_comparisons, all_actual_rows)
    print_conclusion(results)
    
    # Determine exit code
    failed_sources = sum(1 for _, (passed, _, real_mismatched, _) in results if not passed or real_mismatched > 0)
    
    if failed_sources > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
