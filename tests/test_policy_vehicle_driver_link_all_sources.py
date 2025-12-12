"""
Unified test suite for ALL policy-vehicle-driver link sources.

Tests all relationship formats in a single suite:
- Structured: CSV, Excel, Google Sheet
- Unstructured: PDF, Raw Text, Image (stubs for now)
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
from schema import reorder_all_relationships, RELATIONSHIP_SCHEMA_ORDER


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
        - Any fields not in RELATIONSHIP_SCHEMA_ORDER
    
    Treats missing fields as None.
    """
    normalized = {}
    # Only include fields from the relationship schema
    for field in RELATIONSHIP_SCHEMA_ORDER + ["_warnings"]:
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
    """Format a value for display, handling None and long strings."""
    if value is None:
        return "None"
    if isinstance(value, str) and len(value) > 50:
        return value[:47] + "..."
    return str(value)


def print_row_diff(row_num: int, expected: Dict[str, Any], actual: Dict[str, Any], mismatches: List[Dict[str, Any]]):
    """Print a formatted diff for a single row."""
    print(f"{Colors.GRAY}{'-'*80}{Colors.RESET}")
    if mismatches:
        mismatch_fields = [m.get("field", "unknown") for m in mismatches]
        print(f"{Colors.RED}{Colors.BOLD}{'='*30}")
        print(f"ROW {row_num} — {', '.join(mismatch_fields)} mismatch")
        print(f"{'='*30}{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}{Colors.BOLD}Row {row_num}: ✓ PERFECT MATCH{Colors.RESET}")
        return
    
    # Print warnings if present
    if actual.get("_warnings"):
        print(f"\n{Colors.YELLOW}Warnings:{Colors.RESET}")
        for warning in actual["_warnings"]:
            print(f"  {Colors.YELLOW}- {warning}{Colors.RESET}")
        print()
    
    # Print field comparison table
    print(f"{Colors.BOLD}FIELD              EXPECTED                         ACTUAL                          {Colors.RESET}")
    print(f"{Colors.GRAY}{'-'*80}{Colors.RESET}")
    
    all_fields = set(expected.keys()) | set(actual.keys())
    for field in sorted(all_fields):
        if field in ["_source_id", "_source_row_number", "_id"]:
            continue
        
        expected_val = expected.get(field)
        actual_val = actual.get(field)
        is_match = values_equal(field, expected_val, actual_val)
        
        expected_str = format_value(expected_val)
        actual_str = format_value(actual_val)
        
        if not is_match:
            print(f"{Colors.CYAN}{field:<18}{Colors.RESET} {Colors.WHITE}{expected_str:<32}{Colors.RESET} {Colors.WHITE}{actual_str:<32}{Colors.RESET} {Colors.RED}{Colors.BOLD}⚠ MISMATCH{Colors.RESET}")
        else:
            print(f"{Colors.CYAN}{field:<18}{Colors.RESET} {Colors.WHITE}{expected_str:<32}{Colors.RESET} {Colors.WHITE}{actual_str:<32}{Colors.RESET}")


def compare_expected_vs_actual(expected_rows: List[Dict[str, Any]], actual_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare expected vs actual rows row-by-row and field-by-field.
    
    Returns:
        Dictionary with:
        - matched_rows: List of row numbers that match perfectly
        - mismatched_rows: List of row numbers with mismatches
        - total_mismatched_fields: Total count of mismatched fields
    """
    matched_rows = []
    mismatched_rows = []
    total_mismatched_fields = 0
    
    max_rows = max(len(expected_rows), len(actual_rows))
    
    for i in range(max_rows):
        if i >= len(expected_rows):
            print(f"{Colors.RED}*** Actual row {i+1} has no expected row ***{Colors.RESET}")
            mismatched_rows.append(i+1)
            continue
        if i >= len(actual_rows):
            print(f"{Colors.RED}*** Expected row {i+1} has no actual row ***{Colors.RESET}")
            mismatched_rows.append(i+1)
            continue
        
        expected = normalize_row_for_comparison(expected_rows[i])
        actual = normalize_row_for_comparison(actual_rows[i])
        
        # Find mismatches
        row_mismatches = []
        all_fields = set(expected.keys()) | set(actual.keys())
        
        for field in all_fields:
            if field in ["_source_id", "_source_row_number", "_id"]:
                continue
            
            expected_val = expected.get(field)
            actual_val = actual.get(field)
            
            if not values_equal(field, expected_val, actual_val):
                row_mismatches.append({
                    "field": field,
                    "expected": expected_val,
                    "actual": actual_val
                })
                total_mismatched_fields += 1
        
        if row_mismatches:
            mismatched_rows.append(i+1)
            print_row_diff(i+1, expected, actual, row_mismatches)
        else:
            matched_rows.append(i+1)
            if actual.get("_warnings"):
                print(f"{Colors.GREEN}Row {i+1}: ✓ PERFECT MATCH (with warnings){Colors.RESET}")
            else:
                print(f"{Colors.GREEN}Row {i+1}: ✓ PERFECT MATCH{Colors.RESET}")
    
    return {
        "matched_rows": matched_rows,
        "mismatched_rows": mismatched_rows,
        "total_mismatched_fields": total_mismatched_fields
    }


def test_csv() -> Tuple[bool, int, int]:
    """Test CSV relationships."""
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing CSV Relationships ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    csv_path = ROOT / "tests" / "relationships" / "structured" / "policy_vehicle_driver_link.csv"
    truth_file = ROOT / "tests" / "truth" / "relationships" / "policy_vehicle_driver_link.expected.json"
    mapping_id = "source_policy_vehicle_driver_link"
    
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
        except Exception as e:
            print(f"{Colors.RED}✗ Normalization error: {e}{Colors.RESET}\n")
            traceback.print_exc()
            return False, 0, 0
        
        actual_rows = result.get("data", [])
        actual_rows = reorder_all_relationships(actual_rows)
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Sort both expected and actual by composite key (policy_number, vehicle_vin, driver_id) for consistent comparison
        expected_rows_sorted = sorted(expected_rows, key=lambda x: (
            str(x.get("policy_number", "")),
            str(x.get("vehicle_vin", "")),
            str(x.get("driver_id", ""))
        ))
        actual_rows_sorted = sorted(actual_rows, key=lambda x: (
            str(x.get("policy_number", "")),
            str(x.get("vehicle_vin", "")),
            str(x.get("driver_id", ""))
        ))
        
        # Compare
        print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows_sorted, actual_rows_sorted)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows_sorted) and total_mismatched == 0
        
        # Normalize rows for display (already sorted)
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows_sorted]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows_sorted]
        
        # Summary
        rows_with_warnings = [
            i+1 for i, row in enumerate(actual_norm) 
            if row.get("_warnings")
        ]
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}SUMMARY{Colors.RESET}")
        print(f"{Colors.GRAY}{'-'*80}{Colors.RESET}")
        print(f"Total expected rows: {len(expected_rows_sorted)}")
        print(f"Total actual rows:   {len(actual_rows_sorted)}")
        print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        print(f"Total mismatched fields: {total_mismatched}")
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
    """Test PDF relationship documents."""
    pdf_path = ROOT / "tests" / "relationships" / " unstructured" / "pdf_relationships.pdf"
    truth_file = ROOT / "tests" / "truth" / "relationships" / "pdf_relationships.expected.json"
    
    # Skip test if files don't exist
    if not pdf_path.exists() or not truth_file.exists():
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}=== Testing PDF Relationship Documents ==={Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        print(f"{Colors.YELLOW}⚠ SKIPPED: PDF test files not found{Colors.RESET}")
        missing = pdf_path.name if not pdf_path.exists() else truth_file.name
        print(f"   Missing: {missing}")
        print(f"   PDF path: {pdf_path}")
        print(f"   Truth path: {truth_file}\n")
        return True, 0, 0  # Return passed (skipped) with 0 rows
    
    # Initialize passed at the top
    passed = True
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing PDF Relationship Documents ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    mapping_id = "source_pdf_policy_vehicle_driver_link"
    
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
            actual_rows = reorder_all_relationships(actual_rows)
        except Exception as e:
            print(f"{Colors.RED}ERROR during normalize_v2: {e}{Colors.RESET}")
            traceback.print_exc()
            return False, 0, 0
        
        print(f"{Colors.GREEN}✓{Colors.RESET} Rows extracted: {len(actual_rows)}")
        
        # Sort both expected and actual by composite key (policy_number, vehicle_vin, driver_id) for consistent comparison
        expected_rows_sorted = sorted(expected_rows, key=lambda x: (
            str(x.get("policy_number", "")),
            str(x.get("vehicle_vin", "")),
            str(x.get("driver_id", ""))
        ))
        actual_rows_sorted = sorted(actual_rows, key=lambda x: (
            str(x.get("policy_number", "")),
            str(x.get("vehicle_vin", "")),
            str(x.get("driver_id", ""))
        ))
        
        # Compare
        print(f"\nComparing expected vs actual...")
        comparison = compare_expected_vs_actual(expected_rows_sorted, actual_rows_sorted)
        total_mismatched = comparison.get("total_mismatched_fields", 0)
        passed = len(comparison["matched_rows"]) == len(expected_rows_sorted) and total_mismatched == 0
        
        # Normalize rows for display (already sorted)
        expected_norm = [normalize_row_for_comparison(row) for row in expected_rows_sorted]
        actual_norm = [normalize_row_for_comparison(row) for row in actual_rows_sorted]
        
        # Summary
        rows_with_warnings = [
            i+1 for i, row in enumerate(actual_norm) 
            if row.get("_warnings")
        ]
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}SUMMARY{Colors.RESET}")
        print(f"{Colors.GRAY}{'-'*80}{Colors.RESET}")
        print(f"Total expected rows: {len(expected_rows_sorted)}")
        print(f"Total actual rows:   {len(actual_rows_sorted)}")
        print(f"Perfect matches:     {len(comparison['matched_rows'])}")
        print(f"Rows with mismatches: {len(comparison['mismatched_rows'])}")
        print(f"Total mismatched fields: {total_mismatched}")
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
    """Run all relationship tests."""
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}UNIFIED RELATIONSHIP TEST SUITE{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    results = []
    
    # Test CSV
    passed, rows, mismatched = test_csv()
    results.append(("CSV", passed, rows, mismatched))
    
    # Test PDF
    passed, rows, mismatched = test_pdf()
    results.append(("PDF", passed, rows, mismatched))
    
    # Summary
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, p, _, _ in results if p)
    failed_tests = total_tests - passed_tests
    total_mismatched = sum(m for _, _, _, m in results)
    
    print(f"Total tests:        {total_tests}")
    print(f"Passed:             {passed_tests}")
    print(f"Failed:             {failed_tests}")
    print(f"Total mismatched fields: {total_mismatched}\n")
    print("Per-test results:\n")
    
    for name, passed, rows, mismatched in results:
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {name:<20} {status}  ({rows} rows, {mismatched} mismatched fields)")


if __name__ == "__main__":
    main()
