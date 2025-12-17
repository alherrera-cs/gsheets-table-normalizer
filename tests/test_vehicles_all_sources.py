"""
Unified test suite for ALL vehicle sources.

Tests all vehicle formats in a single suite:
- Structured: Airtable, Excel, Google Sheet
- Unstructured: PDF, Raw Text, Image Metadata
"""

# DEBUG MODE TOGGLE - Set to True to see raw JSON dumps
DEBUG = False

import sys
import json
import traceback
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Suppress debug logs during test runs for cleaner output
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
    
    Treats missing fields as None.
    """
    normalized = {}
    for key in sorted(row.keys()):
        if key not in ["_source_id", "_source_row_number", "_id"]:
            value = row[key]
            # Convert empty strings to None for consistency
            if value == "":
                value = None
            normalized[key] = value
    return normalized


def values_equal(field: str, expected_val: Any, actual_val: Any) -> bool:
    """
    Compare two values, treating None and [] as equal for _warnings field.
    For notes field, only check presence (non-strict comparison).
    
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
    elif field == "notes":
        # For notes, non-strict comparison: only check presence
        # Both must be present (non-null) or both must be absent
        expected_present = expected_val is not None and expected_val != ""
        actual_present = actual_val is not None and actual_val != ""
        return expected_present == actual_present
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


def print_row_diff(row_num: int, expected: Dict[str, Any], actual: Dict[str, Any]) -> int:
    """
    Print a professional, clean diff for a single row with improved visibility.
    
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
    expected_warnings = expected.get("_warnings", [])
    has_warnings = (actual_warnings and len(actual_warnings) > 0) or (expected_warnings and len(expected_warnings) > 0)
    
    # If perfect match, print and return
    if not mismatches:
        if has_warnings:
            warning_count = len(actual_warnings) if actual_warnings else 0
            print(f"{Colors.GREEN}Row {row_num}: ✓ PASS{Colors.RESET} {Colors.YELLOW}⚠ {warning_count} warning(s){Colors.RESET}")
        else:
            print(f"{Colors.GREEN}Row {row_num}: ✓ PASS{Colors.RESET}")
        return 0
    
    # Professional header with clear grouping
    mismatch_count = len(mismatches)
    print(f"\n{Colors.RED}{Colors.BOLD}━━ Row {row_num}: {mismatch_count} mismatch(es) ━━{Colors.RESET}")
    
    # Group warnings separately for better visibility
    if has_warnings:
        warning_desc = {
            "invalid_year": "Year outside range (1990-2035)",
            "negative_mileage": "Negative mileage",
            "invalid_email": "Invalid email format",
            "unknown_transmission": "Unrecognized transmission",
            "unknown_fuel_type": "Unrecognized fuel type",
            "unknown_body_style": "Unrecognized body style"
        }
        
        # Show expected vs actual warnings if they differ
        if "_warnings" in [m["field"] for m in mismatches]:
            expected_warn_str = ", ".join(expected_warnings) if expected_warnings else "none"
            actual_warn_str = ", ".join(actual_warnings) if actual_warnings else "none"
            print(f"  {Colors.YELLOW}⚠ Warnings:{Colors.RESET} Expected: {expected_warn_str} | Actual: {actual_warn_str}")
        elif actual_warnings:
            warnings_str = ", ".join([f"{w}" for w in actual_warnings])
            print(f"  {Colors.YELLOW}⚠ Warnings:{Colors.RESET} {warnings_str}")
    
    # Clean table format with better alignment
    print(f"\n  {Colors.BOLD}{'Field':<20} {'Expected':<45} {'Actual':<45}{Colors.RESET}")
    print(f"  {Colors.GRAY}{'─' * 112}{Colors.RESET}")
    
    # Print only mismatched fields (skip None/None pairs)
    for mismatch in mismatches:
        field = mismatch["field"]
        expected_val = mismatch["expected"]
        actual_val = mismatch["actual"]
        
        # Skip if both are None (already handled by mismatch detection, but double-check)
        if expected_val is None and actual_val is None:
            continue
        
        # Smart formatting based on field type
        if field == "_warnings":
            expected_str = str(expected_val) if expected_val else "[]"
            actual_str = str(actual_val) if actual_val else "[]"
            max_len = 45
        elif field == "notes":
            max_len = 45  # Truncate notes to fit column
        else:
            max_len = 45
        
        expected_str = format_value(expected_val, max_len)
        actual_str = format_value(actual_val, max_len)
        
        # Color code: green for expected, red for actual mismatch
        print(f"  {Colors.CYAN}{field:<20}{Colors.RESET} {Colors.GREEN}{expected_str:<45}{Colors.RESET} {Colors.RED}{actual_str:<45}{Colors.RESET}")
    
    print()  # Spacing after row
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


def test_airtable() -> Tuple[bool, int, int]:
    """Test Airtable fleet vehicles."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Airtable Fleet Vehicles ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    json_path = ROOT / "tests" / "vehicles" / "structured" / "airtable_fleet_vehicles.json"
    truth_file = ROOT / "tests" / "truth" / "vehicles" / "airtable_fleet_vehicles.expected.json"
    mapping_id = "source_airtable_vehicles"
    
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
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # DEBUG: Print raw rows only if DEBUG is enabled
        if DEBUG:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== RAW NORMALIZED ROWS ==={Colors.RESET}")
            print(json.dumps(actual_rows, indent=2))
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== EXPECTED TRUTH ROWS ==={Colors.RESET}")
            print(json.dumps(expected_rows, indent=2))
            print()
        
        # Compare
        print(f"Comparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and total_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results with clean header
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
        
        # Consolidate missing/extra row messages
        print_missing_extra_rows(missing_rows, extra_rows)
        
        # Print row diffs and track rows with warnings
        total_rows = max(len(expected_norm), len(actual_norm))
        rows_with_warnings = []
        for row_idx in range(total_rows):
            row_num = row_idx + 1
            
            # Skip if row is missing/extra (already reported above)
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
        
        # Print professional summary
        print()
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        
        # Use a clean table format for summary
        total_expected = len(expected_rows)
        total_actual = len(actual_rows)
        perfect_matches = len(comparison['matched_rows'])
        mismatched_rows = len(comparison['mismatched_rows'])
        total_mismatched_fields = comparison.get('total_mismatched_fields', 0)
        
        print(f"  {Colors.BOLD}Rows:{Colors.RESET}        Expected: {total_expected:<4} | Actual: {total_actual:<4}")
        print(f"  {Colors.BOLD}Matches:{Colors.RESET}     Perfect: {Colors.GREEN}{perfect_matches:<4}{Colors.RESET} | Mismatched: {Colors.RED}{mismatched_rows:<4}{Colors.RESET}")
        print(f"  {Colors.BOLD}Fields:{Colors.RESET}      Total mismatches: {Colors.RED}{total_mismatched_fields}{Colors.RESET}")
        
        if rows_with_warnings:
            warning_rows_str = ", ".join(map(str, sorted(rows_with_warnings)))
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   Rows: {Colors.YELLOW}{warning_rows_str}{Colors.RESET}")
        else:
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   None")
        
        print()
        
        # Print final pass/fail with clear visual indicator
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {mismatched_rows} row(s) with mismatches, {total_mismatched_fields} field(s)\n")
        
        return passed, len(actual_rows), total_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_excel() -> Tuple[bool, int, int]:
    """Test Excel vehicle inventory."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Excel Vehicle Inventory ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    xlsx_path = ROOT / "tests" / "vehicles" / "structured" / "excel_vehicle_export.xlsx"
    truth_file = ROOT / "tests" / "truth" / "vehicles" / "excel_vehicle_export.expected.json"
    mapping_id = "source_xlsx_vehicles"
    
    try:
        # Load truth file
        print(f"{Colors.BLUE}DEBUG: Loading truth file from: {truth_file.resolve()}{Colors.RESET}")
        print(f"{Colors.BLUE}DEBUG: Truth file exists: {truth_file.exists()}{Colors.RESET}")
        expected_rows = load_truth_file(truth_file)
        print(f"{Colors.BLUE}DEBUG: Loaded {len(expected_rows)} expected rows from truth file{Colors.RESET}")
        
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
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # DEBUG: Print raw rows
        if DEBUG:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== RAW NORMALIZED ROWS ==={Colors.RESET}")
            print(json.dumps(actual_rows, indent=2))
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== EXPECTED TRUTH ROWS ==={Colors.RESET}")
            print(json.dumps(expected_rows, indent=2))
            print()
        
        # Compare
        print(f"\n{Colors.BOLD}Comparing expected vs actual...{Colors.RESET}")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and total_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results
        print()
        print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}COMPARISON RESULTS{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
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
                print(f"{Colors.RED}*** Expected row {row_num} has no actual row ***{Colors.RESET}")
                print()
                continue
            
            if row_num in missing_rows:
                print(f"{Colors.RED}*** Actual row {row_num} has no expected row ***{Colors.RESET}")
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
        print(f"Total expected rows: {len(expected_rows)}")
        print(f"Total actual rows:   {len(actual_rows)}")
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
        
        return passed, len(actual_rows), total_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_google_sheet() -> Tuple[bool, int, int]:
    """Test Google Sheet vehicle inventory."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Google Sheet Vehicle Inventory ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    csv_path = ROOT / "tests" / "vehicles" / "structured" / "google_sheet_vehicle_inventory.csv"
    truth_file = ROOT / "tests" / "truth" / "vehicles" / "google_sheet_vehicle_inventory.expected.json"
    mapping_id = "source_google_sheet_vehicles"
    
    try:
        # Load truth file
        print(f"{Colors.BLUE}DEBUG: Loading truth file from: {truth_file.resolve()}{Colors.RESET}")
        print(f"{Colors.BLUE}DEBUG: Truth file exists: {truth_file.exists()}{Colors.RESET}")
        expected_rows = load_truth_file(truth_file)
        print(f"{Colors.BLUE}DEBUG: Loaded {len(expected_rows)} expected rows from truth file{Colors.RESET}")
        
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
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # DEBUG: Print raw rows
        if DEBUG:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== RAW NORMALIZED ROWS ==={Colors.RESET}")
            print(json.dumps(actual_rows, indent=2))
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== EXPECTED TRUTH ROWS ==={Colors.RESET}")
            print(json.dumps(expected_rows, indent=2))
            print()
        
        # Compare
        print(f"\n{Colors.BOLD}Comparing expected vs actual...{Colors.RESET}")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and total_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results
        print()
        print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}COMPARISON RESULTS{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
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
                print(f"{Colors.RED}*** Expected row {row_num} has no actual row ***{Colors.RESET}")
                print()
                continue
            
            if row_num in missing_rows:
                print(f"{Colors.RED}*** Actual row {row_num} has no expected row ***{Colors.RESET}")
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
        print(f"Total expected rows: {len(expected_rows)}")
        print(f"Total actual rows:   {len(actual_rows)}")
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
        
        return passed, len(actual_rows), total_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_pdf() -> Tuple[bool, int, int]:
    """Test PDF vehicle documents."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing PDF Vehicle Documents ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    pdf_path = ROOT / "tests" / "vehicles" / "unstructured" / "pdf_vehicle_documents.pdf"
    truth_file = ROOT / "tests" / "truth" / "vehicles" / "pdf_vehicle_documents.expected.json"
    mapping_id = "source_pdf_vehicles"
    
    try:
        # Load truth file
        print(f"{Colors.BLUE}DEBUG: Loading truth file from: {truth_file.resolve()}{Colors.RESET}")
        print(f"{Colors.BLUE}DEBUG: Truth file exists: {truth_file.exists()}{Colors.RESET}")
        expected_rows = load_truth_file(truth_file)
        print(f"{Colors.BLUE}DEBUG: Loaded {len(expected_rows)} expected rows from truth file{Colors.RESET}")
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0
        
        # Run normalize_v2 (DO NOT mock Vision/OCR - call as-is)
        try:
            result = normalize_v2(
                source=str(pdf_path),
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Check for zero rows
        if len(actual_rows) == 0:
            if DEBUG:
                print(f"{Colors.YELLOW}DEBUG: normalize_v2 returned 0 rows. Something is wrong.{Colors.RESET}")
        
        # DEBUG: Print raw rows only if DEBUG is enabled
        if DEBUG:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== RAW NORMALIZED ROWS ==={Colors.RESET}")
            print(json.dumps(actual_rows, indent=2))
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== EXPECTED TRUTH ROWS ==={Colors.RESET}")
            print(json.dumps(expected_rows, indent=2))
            print()
        
        # Compare
        print(f"Comparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and total_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results with clean header
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
            
            # Track rows with warnings
            actual_warnings = actual_row.get("_warnings", [])
            if actual_warnings and len(actual_warnings) > 0:
                rows_with_warnings.append(row_num)
            
            # Print diff for this row
            num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print professional summary
        print()
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        
        # Use a clean table format for summary
        total_expected = len(expected_rows)
        total_actual = len(actual_rows)
        perfect_matches = len(comparison['matched_rows'])
        mismatched_rows = len(comparison['mismatched_rows'])
        total_mismatched_fields = comparison.get('total_mismatched_fields', 0)
        
        print(f"  {Colors.BOLD}Rows:{Colors.RESET}        Expected: {total_expected:<4} | Actual: {total_actual:<4}")
        print(f"  {Colors.BOLD}Matches:{Colors.RESET}     Perfect: {Colors.GREEN}{perfect_matches:<4}{Colors.RESET} | Mismatched: {Colors.RED}{mismatched_rows:<4}{Colors.RESET}")
        print(f"  {Colors.BOLD}Fields:{Colors.RESET}      Total mismatches: {Colors.RED}{total_mismatched_fields}{Colors.RESET}")
        
        if rows_with_warnings:
            warning_rows_str = ", ".join(map(str, sorted(rows_with_warnings)))
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   Rows: {Colors.YELLOW}{warning_rows_str}{Colors.RESET}")
        else:
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   None")
        
        print()
        
        # Print final pass/fail with clear visual indicator
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {mismatched_rows} row(s) with mismatches, {total_mismatched_fields} field(s)\n")
        
        return passed, len(actual_rows), total_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_raw_text() -> Tuple[bool, int, int]:
    """Test raw text vehicle data."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Raw Text Vehicle Data ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    txt_path = ROOT / "tests" / "vehicles" / "unstructured" / "raw_text_vehicle_data.txt"
    truth_file = ROOT / "tests" / "truth" / "vehicles" / "raw_text_vehicle_data.expected.json"
    mapping_id = "source_raw_text_vehicles"
    
    try:
        # Load truth file
        print(f"{Colors.BLUE}DEBUG: Loading truth file from: {truth_file.resolve()}{Colors.RESET}")
        print(f"{Colors.BLUE}DEBUG: Truth file exists: {truth_file.exists()}{Colors.RESET}")
        expected_rows = load_truth_file(truth_file)
        print(f"{Colors.BLUE}DEBUG: Loaded {len(expected_rows)} expected rows from truth file{Colors.RESET}")
        
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
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Check for zero rows
        if len(actual_rows) == 0:
            if DEBUG:
                print(f"{Colors.YELLOW}DEBUG: normalize_v2 returned 0 rows. Something is wrong.{Colors.RESET}")
        
        # DEBUG: Print raw rows only if DEBUG is enabled
        if DEBUG:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== RAW NORMALIZED ROWS ==={Colors.RESET}")
            print(json.dumps(actual_rows, indent=2))
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== EXPECTED TRUTH ROWS ==={Colors.RESET}")
            print(json.dumps(expected_rows, indent=2))
            print()
        
        # Compare
        print(f"Comparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and total_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results with clean header
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
            
            # Track rows with warnings
            actual_warnings = actual_row.get("_warnings", [])
            if actual_warnings and len(actual_warnings) > 0:
                rows_with_warnings.append(row_num)
            
            # Print diff for this row
            num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print professional summary
        print()
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        
        # Use a clean table format for summary
        total_expected = len(expected_rows)
        total_actual = len(actual_rows)
        perfect_matches = len(comparison['matched_rows'])
        mismatched_rows = len(comparison['mismatched_rows'])
        total_mismatched_fields = comparison.get('total_mismatched_fields', 0)
        
        print(f"  {Colors.BOLD}Rows:{Colors.RESET}        Expected: {total_expected:<4} | Actual: {total_actual:<4}")
        print(f"  {Colors.BOLD}Matches:{Colors.RESET}     Perfect: {Colors.GREEN}{perfect_matches:<4}{Colors.RESET} | Mismatched: {Colors.RED}{mismatched_rows:<4}{Colors.RESET}")
        print(f"  {Colors.BOLD}Fields:{Colors.RESET}      Total mismatches: {Colors.RED}{total_mismatched_fields}{Colors.RESET}")
        
        if rows_with_warnings:
            warning_rows_str = ", ".join(map(str, sorted(rows_with_warnings)))
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   Rows: {Colors.YELLOW}{warning_rows_str}{Colors.RESET}")
        else:
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   None")
        
        print()
        
        # Print final pass/fail with clear visual indicator
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {mismatched_rows} row(s) with mismatches, {total_mismatched_fields} field(s)\n")
        
        return passed, len(actual_rows), total_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_pdf2() -> Tuple[bool, int, int]:
    """Test PDF2 vehicle documents."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing PDF2 Vehicle Documents ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    pdf_path = ROOT / "tests" / "vehicles" / "unstructured" / "pdf2_vehicles.pdf"
    truth_file = ROOT / "tests" / "truth" / "vehicles" / "pdf2_vehicles.expected.json"
    mapping_id = "source_pdf_vehicles"
    
    try:
        # Load truth file
        print(f"{Colors.BLUE}DEBUG: Loading truth file from: {truth_file.resolve()}{Colors.RESET}")
        print(f"{Colors.BLUE}DEBUG: Truth file exists: {truth_file.exists()}{Colors.RESET}")
        expected_rows = load_truth_file(truth_file)
        print(f"{Colors.BLUE}DEBUG: Loaded {len(expected_rows)} expected rows from truth file{Colors.RESET}")
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0
        
        # Run normalize_v2 (DO NOT mock Vision/OCR - call as-is)
        try:
            result = normalize_v2(
                source=str(pdf_path),
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Check for zero rows
        if len(actual_rows) == 0:
            if DEBUG:
                print(f"{Colors.YELLOW}DEBUG: normalize_v2 returned 0 rows. Something is wrong.{Colors.RESET}")
        
        # DEBUG: Print raw rows only if DEBUG is enabled
        if DEBUG:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== RAW NORMALIZED ROWS ==={Colors.RESET}")
            print(json.dumps(actual_rows, indent=2))
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== EXPECTED TRUTH ROWS ==={Colors.RESET}")
            print(json.dumps(expected_rows, indent=2))
            print()
        
        # Compare
        print(f"Comparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and total_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results with clean header
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
            
            # Track rows with warnings
            actual_warnings = actual_row.get("_warnings", [])
            if actual_warnings and len(actual_warnings) > 0:
                rows_with_warnings.append(row_num)
            
            # Print diff for this row
            num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print professional summary
        print()
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        
        # Use a clean table format for summary
        total_expected = len(expected_rows)
        total_actual = len(actual_rows)
        perfect_matches = len(comparison['matched_rows'])
        mismatched_rows = len(comparison['mismatched_rows'])
        total_mismatched_fields = comparison.get('total_mismatched_fields', 0)
        
        print(f"  {Colors.BOLD}Rows:{Colors.RESET}        Expected: {total_expected:<4} | Actual: {total_actual:<4}")
        print(f"  {Colors.BOLD}Matches:{Colors.RESET}     Perfect: {Colors.GREEN}{perfect_matches:<4}{Colors.RESET} | Mismatched: {Colors.RED}{mismatched_rows:<4}{Colors.RESET}")
        print(f"  {Colors.BOLD}Fields:{Colors.RESET}      Total mismatches: {Colors.RED}{total_mismatched_fields}{Colors.RESET}")
        
        if rows_with_warnings:
            warning_rows_str = ", ".join(map(str, sorted(rows_with_warnings)))
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   Rows: {Colors.YELLOW}{warning_rows_str}{Colors.RESET}")
        else:
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   None")
        
        print()
        
        # Print final pass/fail with clear visual indicator
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {mismatched_rows} row(s) with mismatches, {total_mismatched_fields} field(s)\n")
        
        return passed, len(actual_rows), total_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_png() -> Tuple[bool, int, int]:
    """Test PNG image vehicle data."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing PNG Image Vehicle Data ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    png_path = ROOT / "tests" / "vehicles" / "unstructured" / "vehicles_png.png"
    truth_file = ROOT / "tests" / "truth" / "vehicles" / "vehicles_png.expected.json"
    mapping_id = "source_image_vehicles"
    
    try:
        # Load truth file
        print(f"{Colors.BLUE}DEBUG: Loading truth file from: {truth_file.resolve()}{Colors.RESET}")
        print(f"{Colors.BLUE}DEBUG: Truth file exists: {truth_file.exists()}{Colors.RESET}")
        expected_rows = load_truth_file(truth_file)
        print(f"{Colors.BLUE}DEBUG: Loaded {len(expected_rows)} expected rows from truth file{Colors.RESET}")
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0
        
        # Run normalize_v2
        try:
            result = normalize_v2(
                source=str(png_path),
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Check for zero rows
        if len(actual_rows) == 0:
            if DEBUG:
                print(f"{Colors.YELLOW}DEBUG: normalize_v2 returned 0 rows. Something is wrong.{Colors.RESET}")
        
        # DEBUG: Print raw rows only if DEBUG is enabled
        if DEBUG:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== RAW NORMALIZED ROWS ==={Colors.RESET}")
            print(json.dumps(actual_rows, indent=2))
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== EXPECTED TRUTH ROWS ==={Colors.RESET}")
            print(json.dumps(expected_rows, indent=2))
            print()
        
        # Compare
        print(f"Comparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and total_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results with clean header
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
            
            # Track rows with warnings
            actual_warnings = actual_row.get("_warnings", [])
            if actual_warnings and len(actual_warnings) > 0:
                rows_with_warnings.append(row_num)
            
            # Print diff for this row
            num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print professional summary
        print()
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        
        # Use a clean table format for summary
        total_expected = len(expected_rows)
        total_actual = len(actual_rows)
        perfect_matches = len(comparison['matched_rows'])
        mismatched_rows = len(comparison['mismatched_rows'])
        total_mismatched_fields = comparison.get('total_mismatched_fields', 0)
        
        print(f"  {Colors.BOLD}Rows:{Colors.RESET}        Expected: {total_expected:<4} | Actual: {total_actual:<4}")
        print(f"  {Colors.BOLD}Matches:{Colors.RESET}     Perfect: {Colors.GREEN}{perfect_matches:<4}{Colors.RESET} | Mismatched: {Colors.RED}{mismatched_rows:<4}{Colors.RESET}")
        print(f"  {Colors.BOLD}Fields:{Colors.RESET}      Total mismatches: {Colors.RED}{total_mismatched_fields}{Colors.RESET}")
        
        if rows_with_warnings:
            warning_rows_str = ", ".join(map(str, sorted(rows_with_warnings)))
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   Rows: {Colors.YELLOW}{warning_rows_str}{Colors.RESET}")
        else:
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   None")
        
        print()
        
        # Print final pass/fail with clear visual indicator
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {mismatched_rows} row(s) with mismatches, {total_mismatched_fields} field(s)\n")
        
        return passed, len(actual_rows), total_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_handwritten_pdf() -> Tuple[bool, int, int]:
    """Test handwritten PDF vehicle documents."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Handwritten PDF Vehicle Documents ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    pdf_path = ROOT / "tests" / "vehicles" / "unstructured" / "handwritten_pdf_vehicles.pdf"
    truth_file = ROOT / "tests" / "truth" / "vehicles" / "handwritten_vehicles.json"
    mapping_id = "source_pdf_vehicles"
    
    try:
        # Load truth file
        print(f"{Colors.BLUE}DEBUG: Loading truth file from: {truth_file.resolve()}{Colors.RESET}")
        print(f"{Colors.BLUE}DEBUG: Truth file exists: {truth_file.exists()}{Colors.RESET}")
        expected_rows = load_truth_file(truth_file)
        print(f"{Colors.BLUE}DEBUG: Loaded {len(expected_rows)} expected rows from truth file{Colors.RESET}")
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0
        
        # Run normalize_v2
        try:
            result = normalize_v2(
                source=str(pdf_path),
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Compare
        print(f"Comparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and total_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results with clean header
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
            
            # Track rows with warnings
            actual_warnings = actual_row.get("_warnings", [])
            if actual_warnings and len(actual_warnings) > 0:
                rows_with_warnings.append(row_num)
            
            # Print diff for this row
            num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print professional summary
        print()
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        
        # Use a clean table format for summary
        total_expected = len(expected_rows)
        total_actual = len(actual_rows)
        perfect_matches = len(comparison['matched_rows'])
        mismatched_rows = len(comparison['mismatched_rows'])
        total_mismatched_fields = comparison.get('total_mismatched_fields', 0)
        
        print(f"  {Colors.BOLD}Rows:{Colors.RESET}        Expected: {total_expected:<4} | Actual: {total_actual:<4}")
        print(f"  {Colors.BOLD}Matches:{Colors.RESET}     Perfect: {Colors.GREEN}{perfect_matches:<4}{Colors.RESET} | Mismatched: {Colors.RED}{mismatched_rows:<4}{Colors.RESET}")
        print(f"  {Colors.BOLD}Fields:{Colors.RESET}      Total mismatches: {Colors.RED}{total_mismatched_fields}{Colors.RESET}")
        
        if rows_with_warnings:
            warning_rows_str = ", ".join(map(str, sorted(rows_with_warnings)))
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   Rows: {Colors.YELLOW}{warning_rows_str}{Colors.RESET}")
        else:
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   None")
        
        print()
        
        # Print final pass/fail with clear visual indicator
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {mismatched_rows} row(s) with mismatches, {total_mismatched_fields} field(s)\n")
        
        return passed, len(actual_rows), total_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_handwritten_image() -> Tuple[bool, int, int]:
    """Test handwritten image vehicle data."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Handwritten Image Vehicle Data ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    img_path = ROOT / "tests" / "vehicles" / "unstructured" / "handwritten_image_vehicles.jpg"
    truth_file = ROOT / "tests" / "truth" / "vehicles" / "handwritten_vehicles.json"
    mapping_id = "source_image_vehicles"
    
    try:
        # Load truth file
        print(f"{Colors.BLUE}DEBUG: Loading truth file from: {truth_file.resolve()}{Colors.RESET}")
        print(f"{Colors.BLUE}DEBUG: Truth file exists: {truth_file.exists()}{Colors.RESET}")
        expected_rows = load_truth_file(truth_file)
        print(f"{Colors.BLUE}DEBUG: Loaded {len(expected_rows)} expected rows from truth file{Colors.RESET}")
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0
        
        # Run normalize_v2
        try:
            result = normalize_v2(
                source=str(img_path),
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Compare
        print(f"Comparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and total_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results with clean header
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
            
            # Track rows with warnings
            actual_warnings = actual_row.get("_warnings", [])
            if actual_warnings and len(actual_warnings) > 0:
                rows_with_warnings.append(row_num)
            
            # Print diff for this row
            num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print professional summary
        print()
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        
        # Use a clean table format for summary
        total_expected = len(expected_rows)
        total_actual = len(actual_rows)
        perfect_matches = len(comparison['matched_rows'])
        mismatched_rows = len(comparison['mismatched_rows'])
        total_mismatched_fields = comparison.get('total_mismatched_fields', 0)
        
        print(f"  {Colors.BOLD}Rows:{Colors.RESET}        Expected: {total_expected:<4} | Actual: {total_actual:<4}")
        print(f"  {Colors.BOLD}Matches:{Colors.RESET}     Perfect: {Colors.GREEN}{perfect_matches:<4}{Colors.RESET} | Mismatched: {Colors.RED}{mismatched_rows:<4}{Colors.RESET}")
        print(f"  {Colors.BOLD}Fields:{Colors.RESET}      Total mismatches: {Colors.RED}{total_mismatched_fields}{Colors.RESET}")
        
        if rows_with_warnings:
            warning_rows_str = ", ".join(map(str, sorted(rows_with_warnings)))
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   Rows: {Colors.YELLOW}{warning_rows_str}{Colors.RESET}")
        else:
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   None")
        
        print()
        
        # Print final pass/fail with clear visual indicator
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {mismatched_rows} row(s) with mismatches, {total_mismatched_fields} field(s)\n")
        
        return passed, len(actual_rows), total_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def test_image() -> Tuple[bool, int, int]:
    """Test image metadata vehicle data."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Image Metadata Vehicle Data ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    json_path = ROOT / "tests" / "vehicles" / "unstructured" / "image_metadata_sample.json"
    truth_file = ROOT / "tests" / "truth" / "vehicles" / "image_vehicle_data.expected.json"
    mapping_id = "source_image_metadata_json_vehicles"
    
    try:
        # Load truth file
        print(f"{Colors.BLUE}DEBUG: Loading truth file from: {truth_file.resolve()}{Colors.RESET}")
        print(f"{Colors.BLUE}DEBUG: Truth file exists: {truth_file.exists()}{Colors.RESET}")
        expected_rows = load_truth_file(truth_file)
        print(f"{Colors.BLUE}DEBUG: Loaded {len(expected_rows)} expected rows from truth file{Colors.RESET}")
        
        # Get mapping
        mapping_config = get_mapping_by_id(mapping_id)
        if not mapping_config:
            print(f"{Colors.RED}✗ Mapping not found: {mapping_id}{Colors.RESET}\n")
            return False, 0, 0
        
        # Run normalize_v2
        # Load the single file image_metadata_sample.json (not multiple images)
        try:
            result = normalize_v2(
                source={"file_path": str(json_path)},
                mapping_config=mapping_config,
                header_row_index=0,
                validate=False
            )
            actual_rows = result.get("data", [])
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Check for zero rows
        if len(actual_rows) == 0:
            if DEBUG:
                print(f"{Colors.YELLOW}DEBUG: normalize_v2 returned 0 rows. Something is wrong.{Colors.RESET}")
        
        # DEBUG: Print raw rows only if DEBUG is enabled
        if DEBUG:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== RAW NORMALIZED ROWS ==={Colors.RESET}")
            print(json.dumps(actual_rows, indent=2))
            print(f"\n{Colors.BOLD}{Colors.YELLOW}=== EXPECTED TRUTH ROWS ==={Colors.RESET}")
            print(json.dumps(expected_rows, indent=2))
            print()
        
        # Compare
        print(f"Comparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and total_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results with clean header
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
            
            # Track rows with warnings
            actual_warnings = actual_row.get("_warnings", [])
            if actual_warnings and len(actual_warnings) > 0:
                rows_with_warnings.append(row_num)
            
            # Print diff for this row
            num_mismatches = print_row_diff(row_num, expected_row, actual_row)
            
            # If perfect match, already printed by print_row_diff
            if num_mismatches == 0:
                continue
        
        # Print professional summary
        print()
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        
        # Use a clean table format for summary
        total_expected = len(expected_rows)
        total_actual = len(actual_rows)
        perfect_matches = len(comparison['matched_rows'])
        mismatched_rows = len(comparison['mismatched_rows'])
        total_mismatched_fields = comparison.get('total_mismatched_fields', 0)
        
        print(f"  {Colors.BOLD}Rows:{Colors.RESET}        Expected: {total_expected:<4} | Actual: {total_actual:<4}")
        print(f"  {Colors.BOLD}Matches:{Colors.RESET}     Perfect: {Colors.GREEN}{perfect_matches:<4}{Colors.RESET} | Mismatched: {Colors.RED}{mismatched_rows:<4}{Colors.RESET}")
        print(f"  {Colors.BOLD}Fields:{Colors.RESET}      Total mismatches: {Colors.RED}{total_mismatched_fields}{Colors.RESET}")
        
        if rows_with_warnings:
            warning_rows_str = ", ".join(map(str, sorted(rows_with_warnings)))
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   Rows: {Colors.YELLOW}{warning_rows_str}{Colors.RESET}")
        else:
            print(f"  {Colors.BOLD}Warnings:{Colors.RESET}   None")
        
        print()
        
        # Print final pass/fail with clear visual indicator
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ PASS{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ FAIL{Colors.RESET} - {mismatched_rows} row(s) with mismatches, {total_mismatched_fields} field(s)\n")
        
        return passed, len(actual_rows), total_mismatched
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {e}{Colors.RESET}\n")
        traceback.print_exc()
        return False, 0, 0


def main():
    """Run all vehicle tests and print summary."""
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}UNIFIED VEHICLE TEST SUITE{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*80}{Colors.RESET}\n")
    
    # Run all tests
    results = []
    
    # Structured inputs
    results.append(("Airtable", test_airtable()))
    results.append(("Excel", test_excel()))
    results.append(("Google Sheet", test_google_sheet()))
    
    # Unstructured inputs
    results.append(("PDF", test_pdf()))
    results.append(("PDF2", test_pdf2()))
    results.append(("Handwritten PDF", test_handwritten_pdf()))
    results.append(("Raw Text", test_raw_text()))
    results.append(("PNG Image", test_png()))
    results.append(("Handwritten Image", test_handwritten_image()))
    results.append(("Image Metadata", test_image()))
    
    # Calculate summary
    total_tests = len(results)
    passed_tests = sum(1 for _, (passed, _, _) in results if passed)
    failed_tests = total_tests - passed_tests
    total_mismatched_fields = sum(mismatched for _, (_, _, mismatched) in results)
    
    # Print summary
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*80}{Colors.RESET}\n")
    
    print(f"Total tests:        {total_tests}")
    print(f"{Colors.GREEN}Passed:             {passed_tests}{Colors.RESET}")
    if failed_tests > 0:
        print(f"{Colors.RED}Failed:             {failed_tests}{Colors.RESET}")
    else:
        print(f"Failed:             {failed_tests}")
    print(f"Total mismatched fields: {total_mismatched_fields}\n")
    
    # Print per-test results
    print(f"{Colors.BOLD}Per-test results:{Colors.RESET}\n")
    for source_name, (passed, rows, mismatched) in results:
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {source_name:<20} {status}  ({rows} rows, {mismatched} mismatched fields)")
    
    print()
    
    # Exit with error code if any mismatches
    if failed_tests > 0 or total_mismatched_fields > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
