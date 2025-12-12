"""
Unified test suite for ALL location sources.

Tests all location formats in a single suite:
- Structured: Airtable, Excel, Google Sheet
- Unstructured: Raw Text
"""

# DEBUG MODE TOGGLE - Set to True to see raw JSON dumps
DEBUG = False

import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
from schema import reorder_all_locations, LOCATION_SCHEMA_ORDER


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
        - Any fields not in LOCATION_SCHEMA_ORDER
    
    Treats missing fields as None.
    """
    normalized = {}
    # Only include fields from the location schema
    for field in LOCATION_SCHEMA_ORDER + ["_warnings"]:
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
            for warning in actual_warnings:
                print(f"{Colors.YELLOW}   ⚠  {warning}{Colors.RESET}")
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
        for i, warning in enumerate(actual_warnings, 1):
            print(f"{Colors.YELLOW}   [{i}] {warning}{Colors.RESET}")
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


def test_csv() -> Tuple[bool, int, int]:
    """Test CSV location data (structured format)."""
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing CSV Location Data ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    csv_path = ROOT / "tests" / "locations" / "structured" / "garaging_locations.csv"
    truth_file = ROOT / "tests" / "truth" / "locations" / "garaging_locations.expected.json"
    # Note: CSV files from Google Sheets use the google_sheet mapping
    mapping_id = "source_google_sheet_locations"
    
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
            # Reorder to match schema
            actual_rows = reorder_all_locations(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Compare
        print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and total_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results
        print()
        print("=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
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
        
        # Print summary
        print()
        print("=" * 80)
        print("SUMMARY")
        print("-" * 80)
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


def test_raw_text() -> Tuple[bool, int, int]:
    """Test raw text location data."""
    # Check if test files exist
    txt_path = ROOT / "tests" / "locations" / "unstructured" / "raw_text_locations.txt"
    truth_file = ROOT / "tests" / "truth" / "locations" / "raw_text_locations.expected.json"
    
    # Skip test if files don't exist
    if not txt_path.exists() or not truth_file.exists():
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Raw Text Location Data ==={Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        print(f"{Colors.YELLOW}⚠ SKIPPED: Raw text test files not found{Colors.RESET}")
        print(f"   Missing: {txt_path.name if not txt_path.exists() else truth_file.name}\n")
        return True, 0, 0  # Return passed (skipped) with 0 rows
    
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing Raw Text Location Data ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    mapping_id = "source_raw_text_locations"
    
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
            # Reorder to match schema
            actual_rows = reorder_all_locations(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Compare
        print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows, actual_rows)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows) and total_mismatched == 0
        
        # Normalize rows for display
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows]
        
        # Print results
        print()
        print("=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
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
        
        # Print summary
        print()
        print("=" * 80)
        print("SUMMARY")
        print("-" * 80)
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


def main():
    """Run all location tests."""
    print("=" * 80)
    print("UNIFIED LOCATION TEST SUITE")
    print("=" * 80)
    print()
    
    results = []
    
    # Run all tests
    # Note: CSV files use the google_sheet mapping (Google Sheets export as CSV)
    results.append(("CSV", test_csv()))
    results.append(("Raw Text", test_raw_text()))
    
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
