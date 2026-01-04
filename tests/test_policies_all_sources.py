"""
Unified test suite for ALL policy sources.

Tests all policy formats in a single suite:
- Structured: Airtable, Excel, Google Sheet
- Unstructured: PDF, Raw Text
"""

import sys
import json
import traceback
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Global flags for output control
VERBOSE = False
FULL_DIFF = False
DEBUG = False

# Suppress debug logs during test runs for cleaner output (unless VERBOSE)
logging.getLogger().setLevel(logging.WARNING)

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
from schema import reorder_all_policies, POLICY_SCHEMA_ORDER


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
    Normalize a row for comparison by removing metadata fields and extra fields.
    
    Removes:
        - _source_id
        - _source_row_number
        - _id (UUID)
        - Any fields not in POLICY_SCHEMA_ORDER (e.g., "year" from inference)
    
    Treats missing fields as None.
    """
    normalized = {}
    # Only include fields from the policy schema
    for field in POLICY_SCHEMA_ORDER + ["_warnings"]:
        value = row.get(field)
        # Convert empty strings to None for consistency
        if value == "":
            value = None
        normalized[field] = value
    return normalized


def values_equal(field: str, expected_val: Any, actual_val: Any) -> bool:
    """
    Compare two values, treating None and [] as equal for _warnings field.
    
    Args:
        field: Field name being compared
        expected_val: Expected value
        actual_val: Actual value
    
    Returns:
        True if values are considered equal, False otherwise
    """
    if field == "_warnings":
        # For _warnings, treat None and [] as equal
        if expected_val is None and actual_val == []:
            return True
        if expected_val == [] and actual_val is None:
            return True
    return expected_val == actual_val


def format_value(value: Any) -> str:
    """Format a value for display in the diff table - show FULL values, no truncation."""
    if value is None:
        return "None"
    if isinstance(value, str):
        return value  # Show full string, no truncation
    return str(value)


def print_row_diff(row_num: int, expected: Dict[str, Any], actual: Dict[str, Any]) -> int:
    """
    Print a compact table-style diff for a single row (matching individual test format exactly).
    
    Args:
        row_num: Row number (1-indexed)
        expected: Expected row dictionary
        actual: Actual row dictionary
    
    Returns:
        Number of mismatched fields
    """
    # Get all fields from both rows
    all_fields = sorted(set(expected.keys()) | set(actual.keys()))
    
    # Find mismatches
    mismatches = []
    for field in all_fields:
        expected_val = expected.get(field, None)
        actual_val = actual.get(field, None)
        
        if not values_equal(field, expected_val, actual_val):
            mismatches.append({
                "field": field,
                "expected": expected_val,
                "actual": actual_val
            })
    
    # Check for warnings in actual row
    actual_warnings = actual.get("_warnings", [])
    has_warnings = actual_warnings and len(actual_warnings) > 0
    
    # If perfect match, print and return
    if not mismatches:
        if has_warnings:
            print(f"{Colors.GREEN}Row {row_num}: ✓ PERFECT MATCH{Colors.RESET} {Colors.YELLOW}⚠️  (has {len(actual_warnings)} warning(s)){Colors.RESET}")
            # Show warnings even for perfect matches
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  WARNINGS:{Colors.RESET}")
            warning_desc = {
                "invalid_date_range": "Expiration date is before effective date",
                "negative_premium": "Premium is negative",
                "invalid_premium": "Premium format is invalid"
            }
            for warning in actual_warnings:
                desc = warning_desc.get(warning, warning)
                print(f"{Colors.YELLOW}   ⚠  {warning}{Colors.RESET} - {Colors.GRAY}{desc}{Colors.RESET}")
            print()
        else:
            print(f"{Colors.GREEN}Row {row_num}: ✓ PERFECT MATCH{Colors.RESET}")
        return 0
    
    # Add separator line for visual clarity
    print(f"{Colors.GRAY}{'-' * 82}{Colors.RESET}")
    
    # Determine primary mismatch type for header
    primary_mismatch = mismatches[0]["field"]
    print(f"{Colors.RED}{Colors.BOLD}{'=' * 30}{Colors.RESET}")
    print(f"{Colors.RED}{Colors.BOLD}ROW {row_num} — {primary_mismatch} mismatch{Colors.RESET}")
    print(f"{Colors.RED}{Colors.BOLD}{'=' * 30}{Colors.RESET}")
    print()
    
    # Print warnings section if actual row has warnings
    if has_warnings:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  WARNINGS ({len(actual_warnings)}):{Colors.RESET}")
        warning_desc = {
            "invalid_date_range": "Expiration date is before effective date",
            "negative_premium": "Premium is negative",
            "invalid_premium": "Premium format is invalid"
        }
        for i, warning in enumerate(actual_warnings, 1):
            desc = warning_desc.get(warning, warning)
            print(f"{Colors.YELLOW}   [{i}] {warning}{Colors.RESET} - {Colors.GRAY}{desc}{Colors.RESET}")
        print()
    
    print(f"{Colors.BOLD}{'FIELD':<18} {'EXPECTED':<32} {'ACTUAL':<32}{Colors.RESET}")
    print(f"{Colors.GRAY}{'-' * 82}{Colors.RESET}")
    
    # Print all fields (highlight mismatches) - show FULL values, no truncation
    for field in all_fields:
        expected_val = expected.get(field, None)
        actual_val = actual.get(field, None)
        
        # Format values - show FULL values, no truncation
        if field == "_warnings":
            if isinstance(expected_val, list):
                expected_str = str(expected_val) if expected_val else "[]"
            else:
                expected_str = format_value(expected_val)
            if isinstance(actual_val, list):
                actual_str = str(actual_val) if actual_val else "[]"
            else:
                actual_str = format_value(actual_val)
        else:
            expected_str = format_value(expected_val)
            actual_str = format_value(actual_val)
        
        # NO TRUNCATION - show full values (may wrap in terminal, but that's OK)
        # Highlight if mismatch
        if not values_equal(field, expected_val, actual_val):
            print(f"{Colors.CYAN}{field:<18}{Colors.RESET} {Colors.GREEN}{expected_str:<32}{Colors.RESET} {Colors.RED}{actual_str:<32}{Colors.RESET} {Colors.RED}{Colors.BOLD}⚠ MISMATCH{Colors.RESET}")
        else:
            print(f"{Colors.CYAN}{field:<18}{Colors.RESET} {Colors.WHITE}{expected_str:<32}{Colors.RESET} {Colors.WHITE}{actual_str:<32}{Colors.RESET}")
    
    print()
    return len(mismatches)


def compare_expected_vs_actual(expected_rows: List[Dict[str, Any]], actual_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare expected and actual rows row-by-row and field-by-field.
    
    Returns:
        Dictionary with:
            - total_rows: Total number of rows compared
            - matched_rows: List of row indices that match perfectly
            - mismatched_rows: List of row indices with mismatches
            - mismatches: List of mismatch details
            - total_mismatched_fields: Total count of mismatched fields
    """
    # Normalize rows for comparison
    expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
    actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
    
    total_rows = max(len(expected_norm), len(actual_norm))
    matched_rows = []
    mismatched_rows = []
    mismatches = []
    total_mismatched_fields = 0
    
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
            
            if not values_equal(field, expected_val, actual_val):
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
        "total_mismatched_fields": total_mismatched_fields
    }


def filter_acceptable_mismatches(comparison: Dict[str, Any], actual_rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Filter out acceptable mismatches (confidence-aware counting).
    
    ETL-style validation rules:
    1. Structural issues (missing/extra rows) → CRITICAL
    2. Case-only differences → ACCEPTABLE
    3. Notes populated for unstructured sources → ACCEPTABLE
    4. Policy Number/VIN mismatches when confidence < 0.9 → ACCEPTABLE (OCR errors)
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
        
        # Ignore metadata fields
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
        source_metadata = actual_row.get("_source", {})
        source_type = source_metadata.get("source_type", "")
        parser = source_metadata.get("parser", "")
        
        # Check if source is unstructured (OCR/Vision)
        is_unstructured = (
            source_type in ["pdf", "image"] or
            parser in ["vision_api", "ocr_extraction", "raw_text_parser"]
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
        
        # Rule 4: Key field mismatches (policy_number, vehicle_vin) when confidence < 0.9 are ACCEPTABLE (OCR errors)
        if field in ["policy_number", "vehicle_vin"] and expected_val is not None and actual_val is not None:
            if field_confidence is not None and field_confidence < 0.9:
                acceptable_mismatches.append(mismatch)
                continue
        
        # Now check confidence-based rules
        # If field is missing confidence score:
        # - For unstructured sources, treat as low confidence (acceptable - OCR errors)
        # - For structured sources, treat as CRITICAL (shouldn't happen)
        if field_confidence is None:
            if is_unstructured:
                # Unstructured source without confidence = likely OCR error, treat as acceptable
                acceptable_mismatches.append(mismatch)
            else:
                # Structured source should have confidence - treat as CRITICAL
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


def test_airtable() -> Tuple[bool, int, int]:
    """Test Airtable policies."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Airtable Policies ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    json_path = ROOT / "tests" / "policies" / "structured" / "airtable_policies.json"
    truth_file = ROOT / "tests" / "truth" / "policies" / "airtable_policies.expected.json"
    mapping_id = "source_airtable_policies"
    
    try:
        # Load truth file
        expected_rows = load_truth_file(truth_file)
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0
        
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
            actual_rows = reorder_all_policies(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # DEBUG: Print raw rows only if DEBUG is enabled
        if DEBUG:
            debug_print(f"\n{Colors.BOLD}{Colors.YELLOW}=== RAW NORMALIZED ROWS ==={Colors.RESET}")
            debug_print(json.dumps(actual_rows, indent=2))
            debug_print(f"\n{Colors.BOLD}{Colors.YELLOW}=== EXPECTED TRUTH ROWS ==={Colors.RESET}")
            debug_print(json.dumps(expected_rows, indent=2))
            debug_print()
        
        # Compare
        debug_print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        real_mismatched, acceptable_mismatched = filter_acceptable_mismatches(comparison, actual_rows)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and real_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results (only if verbose or full-diff)
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
        
        # Print row diffs and track rows with warnings
        total_rows = max(len(expected_norm), len(actual_norm))
        rows_with_warnings = []
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows
            if row_num in extra_rows:
                if VERBOSE or FULL_DIFF:
                    print(f"*** Expected row {row_num} has no actual row ***")
                    print()
                continue
            
            if row_num in missing_rows:
                if VERBOSE or FULL_DIFF:
                    print(f"*** Actual row {row_num} has no expected row ***")
                    print()
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
            
            # Print diff for this row (only if verbose)
            if VERBOSE or FULL_DIFF:
                num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            else:
                # Just count mismatches without printing
                all_fields = sorted(set(expected_row.keys()) | set(actual_row.keys()))
                num_mismatches = sum(1 for field in all_fields 
                                    if not values_equal(field, expected_row.get(field), actual_row.get(field)))
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print summary (only if verbose)
        debug_print("\n" + "=" * 80)
        debug_print("SUMMARY")
        debug_print("-" * 80)
        debug_print(f"Total expected rows: {len(expected_rows)}")
        debug_print(f"Total actual rows:   {len(actual_rows)}")
        debug_print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        debug_print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        debug_print(f"Critical mismatched fields: {real_mismatched}")
        if acceptable_mismatched > 0:
            debug_print(f"Acceptable mismatched fields (low confidence): {acceptable_mismatched}")
        if rows_with_warnings:
            debug_print(f"Rows containing warnings: {', '.join(map(str, rows_with_warnings))}")
        debug_print()
        
        # Print final pass/fail
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {len(comparison['mismatched_rows'])} row(s) with mismatches, {real_mismatched} critical mismatched field(s)\n")
        
        return passed, len(actual_rows), real_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_google_sheet() -> Tuple[bool, int, int]:
    """Test Google Sheet policies."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Google Sheet Policies ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    csv_path = ROOT / "tests" / "policies" / "structured" / "google_sheet_policies.csv"
    truth_file = ROOT / "tests" / "truth" / "policies" / "google_sheet_policies.expected.json"
    mapping_id = "source_google_sheet_policies"
    
    try:
        # Load truth file
        expected_rows = load_truth_file(truth_file)
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0
        
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
            actual_rows = reorder_all_policies(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Compare
        debug_print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        real_mismatched, acceptable_mismatched = filter_acceptable_mismatches(comparison, actual_rows)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and real_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results (only if verbose or full-diff)
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
        
        # Print row diffs and track rows with warnings
        total_rows = max(len(expected_norm), len(actual_norm))
        rows_with_warnings = []
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows
            if row_num in extra_rows:
                if VERBOSE or FULL_DIFF:
                    print(f"*** Expected row {row_num} has no actual row ***")
                    print()
                continue
            
            if row_num in missing_rows:
                if VERBOSE or FULL_DIFF:
                    print(f"*** Actual row {row_num} has no expected row ***")
                    print()
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
            
            # Print diff for this row (only if verbose)
            if VERBOSE or FULL_DIFF:
                num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            else:
                # Just count mismatches without printing
                all_fields = sorted(set(expected_row.keys()) | set(actual_row.keys()))
                num_mismatches = sum(1 for field in all_fields 
                                    if not values_equal(field, expected_row.get(field), actual_row.get(field)))
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print summary (only if verbose)
        debug_print("\n" + "=" * 80)
        debug_print("SUMMARY")
        debug_print("-" * 80)
        debug_print(f"Total expected rows: {len(expected_rows)}")
        debug_print(f"Total actual rows:   {len(actual_rows)}")
        debug_print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        debug_print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        debug_print(f"Critical mismatched fields: {real_mismatched}")
        if acceptable_mismatched > 0:
            debug_print(f"Acceptable mismatched fields (low confidence): {acceptable_mismatched}")
        if rows_with_warnings:
            debug_print(f"Rows containing warnings: {', '.join(map(str, rows_with_warnings))}")
        debug_print()
        
        # Print final pass/fail
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {len(comparison['mismatched_rows'])} row(s) with mismatches, {real_mismatched} critical mismatched field(s)\n")
        
        return passed, len(actual_rows), real_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_excel() -> Tuple[bool, int, int]:
    """Test Excel policies."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Excel Policies ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    xlsx_path = ROOT / "tests" / "policies" / "structured" / "excel_policies.xlsx"
    truth_file = ROOT / "tests" / "truth" / "policies" / "excel_policies.expected.json"
    mapping_id = "source_xlsx_policies"
    
    try:
        # Load truth file
        expected_rows = load_truth_file(truth_file)
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0
        
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
            actual_rows = reorder_all_policies(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Compare
        debug_print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        real_mismatched, acceptable_mismatched = filter_acceptable_mismatches(comparison, actual_rows)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and real_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results (only if verbose or full-diff)
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
        
        # Print row diffs and track rows with warnings
        total_rows = max(len(expected_norm), len(actual_norm))
        rows_with_warnings = []
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows
            if row_num in extra_rows:
                if VERBOSE or FULL_DIFF:
                    print(f"*** Expected row {row_num} has no actual row ***")
                    print()
                continue
            
            if row_num in missing_rows:
                if VERBOSE or FULL_DIFF:
                    print(f"*** Actual row {row_num} has no expected row ***")
                    print()
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
            
            # Print diff for this row (only if verbose)
            if VERBOSE or FULL_DIFF:
                num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            else:
                # Just count mismatches without printing
                all_fields = sorted(set(expected_row.keys()) | set(actual_row.keys()))
                num_mismatches = sum(1 for field in all_fields 
                                    if not values_equal(field, expected_row.get(field), actual_row.get(field)))
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print summary (only if verbose)
        debug_print("\n" + "=" * 80)
        debug_print("SUMMARY")
        debug_print("-" * 80)
        debug_print(f"Total expected rows: {len(expected_rows)}")
        debug_print(f"Total actual rows:   {len(actual_rows)}")
        debug_print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        debug_print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        debug_print(f"Critical mismatched fields: {real_mismatched}")
        if acceptable_mismatched > 0:
            debug_print(f"Acceptable mismatched fields (low confidence): {acceptable_mismatched}")
        if rows_with_warnings:
            debug_print(f"Rows containing warnings: {', '.join(map(str, rows_with_warnings))}")
        debug_print()
        
        # Print final pass/fail
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {len(comparison['mismatched_rows'])} row(s) with mismatches, {real_mismatched} critical mismatched field(s)\n")
        
        return passed, len(actual_rows), real_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_raw_text() -> Tuple[bool, int, int]:
    """Test raw text policy data."""
    txt_path = ROOT / "tests" / "policies" / "unstructured" / "raw_text_policies.txt"
    truth_file = ROOT / "tests" / "truth" / "policies" / "raw_text_policies.expected.json"
    
    # Skip test if files don't exist
    if not txt_path.exists() or not truth_file.exists():
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Raw Text Policy Data ==={Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        print(f"{Colors.YELLOW}⚠ SKIPPED: Raw text test files not found{Colors.RESET}")
        print(f"   Missing: {txt_path.name if not txt_path.exists() else truth_file.name}\n")
        return True, 0, 0  # Return passed (skipped) with 0 rows
    
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Raw Text Policy Data ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    mapping_id = "source_raw_text_policies"
    
    try:
        # Load truth file
        expected_rows = load_truth_file(truth_file)
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0
        
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
            actual_rows = reorder_all_policies(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Compare
        debug_print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        real_mismatched, acceptable_mismatched = filter_acceptable_mismatches(comparison, actual_rows)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and real_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results (only if verbose or full-diff)
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
        
        # Print row diffs and track rows with warnings
        total_rows = max(len(expected_norm), len(actual_norm))
        rows_with_warnings = []
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows
            if row_num in extra_rows:
                if VERBOSE or FULL_DIFF:
                    print(f"*** Expected row {row_num} has no actual row ***")
                    print()
                continue
            
            if row_num in missing_rows:
                if VERBOSE or FULL_DIFF:
                    print(f"*** Actual row {row_num} has no expected row ***")
                    print()
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
            
            # Print diff for this row (only if verbose)
            if VERBOSE or FULL_DIFF:
                num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            else:
                # Just count mismatches without printing
                all_fields = sorted(set(expected_row.keys()) | set(actual_row.keys()))
                num_mismatches = sum(1 for field in all_fields 
                                    if not values_equal(field, expected_row.get(field), actual_row.get(field)))
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print summary (only if verbose)
        debug_print("\n" + "=" * 80)
        debug_print("SUMMARY")
        debug_print("-" * 80)
        debug_print(f"Total expected rows: {len(expected_rows)}")
        debug_print(f"Total actual rows:   {len(actual_rows)}")
        debug_print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        debug_print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        debug_print(f"Critical mismatched fields: {real_mismatched}")
        if acceptable_mismatched > 0:
            debug_print(f"Acceptable mismatched fields (low confidence): {acceptable_mismatched}")
        if rows_with_warnings:
            debug_print(f"Rows containing warnings: {', '.join(map(str, rows_with_warnings))}")
        debug_print()
        
        # Print final pass/fail
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {len(comparison['mismatched_rows'])} row(s) with mismatches, {real_mismatched} critical mismatched field(s)\n")
        
        return passed, len(actual_rows), real_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_pdf() -> Tuple[bool, int, int]:
    """Test PDF policy documents."""
    pdf_path = ROOT / "tests" / "policies" / "unstructured" / "pdf_policies.pdf"
    truth_file = ROOT / "tests" / "truth" / "policies" / "pdf_policies.expected.json"
    
    # Skip test if files don't exist
    if not pdf_path.exists() or not truth_file.exists():
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}=== Testing PDF Policy Documents ==={Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        print(f"{Colors.YELLOW}⚠ SKIPPED: PDF test files not found{Colors.RESET}")
        missing = pdf_path.name if not pdf_path.exists() else truth_file.name
        print(f"   Missing: {missing}\n")
        return True, 0, 0  # Return passed (skipped) with 0 rows
    
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing PDF Policy Documents ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    # Use raw_text_policies mapping for PDF (PDFs are unstructured like raw text)
    mapping_id = "source_raw_text_policies"
    
    try:
        # Load truth file
        expected_rows = load_truth_file(truth_file)
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0
        
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
            actual_rows = reorder_all_policies(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # DEBUG: Print actual rows if DEBUG is enabled
        if DEBUG:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== ACTUAL ROWS (before sorting) ==={Colors.RESET}")
            for i, row in enumerate(actual_rows):
                pn = row.get('policy_number', 'MISSING')
                print(f"Row {i+1}: policy_number={pn}, notes={row.get('notes', 'None')[:50] if row.get('notes') else 'None'}")
            print()
        
        # Sort both expected and actual by policy_number for consistent comparison
        expected_rows_sorted = sorted(expected_rows, key=lambda x: str(x.get("policy_number", "")))
        actual_rows_sorted = sorted(actual_rows, key=lambda x: str(x.get("policy_number", "")))
        
        # DEBUG: Print sorted rows if DEBUG is enabled
        if DEBUG:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== EXPECTED ROWS (after sorting) ==={Colors.RESET}")
            for i, row in enumerate(expected_rows_sorted):
                pn = row.get('policy_number', 'MISSING')
                print(f"Row {i+1}: policy_number={pn}")
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== ACTUAL ROWS (after sorting) ==={Colors.RESET}")
            for i, row in enumerate(actual_rows_sorted):
                pn = row.get('policy_number', 'MISSING')
                print(f"Row {i+1}: policy_number={pn}, notes={row.get('notes', 'None')[:50] if row.get('notes') else 'None'}")
            print(f"Expected count: {len(expected_rows_sorted)}, Actual count: {len(actual_rows_sorted)}")
            print()
        
        # Compare
        debug_print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows_sorted, actual_rows_sorted)
        real_mismatched, acceptable_mismatched = filter_acceptable_mismatches(comparison, actual_rows_sorted)
        passed = len(comparison["matched_rows"]) == len(expected_rows_sorted) and real_mismatched == 0
        
        # Normalize rows for display (already sorted)
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows_sorted]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows_sorted]
        
        # Print results (only if verbose or full-diff)
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
        
        # Print row diffs and track rows with warnings
        total_rows = max(len(expected_norm), len(actual_norm))
        rows_with_warnings = []
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Handle extra/missing rows
            if row_num in extra_rows:
                if VERBOSE or FULL_DIFF:
                    print(f"*** Expected row {row_num} has no actual row ***")
                    print()
                continue
            
            if row_num in missing_rows:
                if VERBOSE or FULL_DIFF:
                    print(f"*** Actual row {row_num} has no expected row ***")
                    print()
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
            
            # Print diff for this row (only if verbose)
            if VERBOSE or FULL_DIFF:
                num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            else:
                # Just count mismatches without printing
                all_fields = sorted(set(expected_row.keys()) | set(actual_row.keys()))
                num_mismatches = sum(1 for field in all_fields 
                                    if not values_equal(field, expected_row.get(field), actual_row.get(field)))
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print summary (only if verbose)
        debug_print("\n" + "=" * 80)
        debug_print("SUMMARY")
        debug_print("-" * 80)
        debug_print(f"Total expected rows: {len(expected_rows_sorted)}")
        debug_print(f"Total actual rows:   {len(actual_rows_sorted)}")
        debug_print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        debug_print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        debug_print(f"Critical mismatched fields: {real_mismatched}")
        if acceptable_mismatched > 0:
            debug_print(f"Acceptable mismatched fields (low confidence): {acceptable_mismatched}")
        if rows_with_warnings:
            debug_print(f"Rows containing warnings: {', '.join(map(str, rows_with_warnings))}")
        debug_print()
        
        # Print final pass/fail
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {len(comparison['mismatched_rows'])} row(s) with mismatches, {real_mismatched} critical mismatched field(s)\n")
        
        return passed, len(actual_rows), real_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def main():
    """Run all policy tests."""
    global VERBOSE, FULL_DIFF, DEBUG
    
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Policy extraction test suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed row-by-row diffs")
    parser.add_argument("--full-diff", action="store_true", help="Show full field-by-field diffs for all rows")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    VERBOSE = args.verbose
    FULL_DIFF = args.full_diff
    DEBUG = args.debug
    
    if DEBUG:
        logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 80)
    print("UNIFIED POLICY TEST SUITE")
    print("=" * 80)
    print()
    
    results = []
    
    # Run all tests
    results.append(("Airtable", test_airtable()))
    results.append(("Google Sheet", test_google_sheet()))
    results.append(("Excel", test_excel()))
    results.append(("Raw Text", test_raw_text()))
    results.append(("PDF", test_pdf()))
    
    # Print final summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    total_tests = len(results)
    passed_tests = sum(1 for _, (p, _, _) in results if p)
    failed_tests = total_tests - passed_tests
    total_mismatched_fields = sum(m for _, (_, _, m) in results)
    
    print(f"Total tests:        {total_tests}")
    print(f"Passed:             {passed_tests}")
    print(f"Failed:             {failed_tests}")
    print(f"Total mismatched fields: {total_mismatched_fields}")
    print()
    print("Per-test results:")
    print()
    
    for name, (passed, rows, mismatched) in results:
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {name:<20} {status}  ({rows} rows, {mismatched} mismatched fields)")
    
    print()
    
    # Exit with error code if any tests failed
    if failed_tests > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
