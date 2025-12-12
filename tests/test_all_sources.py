"""
Unified test runner for ALL domains and sources.

Runs all domain test suites in order:
1. Policies
2. Drivers
3. Vehicles
4. Claims
5. Relationships

Reuses existing per-domain test logic without modification.

================================================================================
TEST STATUS SUMMARY (as of latest run)
================================================================================

DOMAINS PASSING FULLY:
- Policies: All structured sources (Airtable, Google Sheet, Excel, Raw Text, PDF)
- Drivers: All structured sources (Airtable, Google Sheet, Excel, Raw Text, PDF, Image)
- Vehicles: All structured sources (Airtable, Excel, Google Sheet, PDF, Raw Text, Image)
- Claims: Google Sheet passes fully
- Relationships: CSV passes fully

PDF EXTRACTION ISSUES:
- Claims PDF: Missing 1 of 3 expected rows (C003 claim not extracted)
  - Issue: Block splitting may not detect unstructured claims without explicit headers
  - Recommendation: Improve narrative pattern detection for claims without "Claim Number:" headers

- Relationships PDF: Missing 1 of 3 expected rows (P002 relationship not extracted)
  - Issue: Structured pattern requires all 3 fields on one line; multi-line formats not handled
  - Recommendation: Make relationship block splitting more flexible to handle fields on separate lines

- OCR Character Errors: VIN misreads (e.g., "3R5UAL4YUKPYGF1GZ" → "3R5JL4AYUKPYGF1GZ")
  - Issue: OCR accuracy limitations with similar characters (U vs J, A vs L)
  - Recommendation: Add fuzzy matching or validation for critical fields like VINs

NEXT RECOMMENDED IMPROVEMENTS:
1. Flexible Block Splitting: Enhance relationship/claim parsers to handle multi-line field formats
2. OCR Post-Processing: Add fuzzy matching for VINs and other critical fields with known OCR error patterns
3. Better Delimiter Detection: Improve detection of block boundaries in PDFs with varied layouts
4. Fallback Strategies: Add more fallback patterns for missing fields, especially for unstructured claims

================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# ANSI color codes
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


def run_domain_tests(domain_name: str, test_module, test_functions: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Run all test functions for a domain and return aggregated results.
    
    Args:
        domain_name: Human-readable domain name (e.g., "Policies")
        test_module: Imported test module
        test_functions: List of (source_name, function_name) tuples
    
    Returns:
        Dictionary with aggregated results
    """
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}=== Testing {domain_name} ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    results = []
    for source_name, func_name in test_functions:
        try:
            test_func = getattr(test_module, func_name)
            passed, rows, mismatched = test_func()
            results.append({
                "source": source_name,
                "passed": passed,
                "rows": rows,
                "mismatched_fields": mismatched
            })
        except Exception as e:
            print(f"{Colors.RED}✗ ERROR running {source_name}: {e}{Colors.RESET}\n")
            import traceback
            traceback.print_exc()
            results.append({
                "source": source_name,
                "passed": False,
                "rows": 0,
                "mismatched_fields": 0,
                "error": str(e)
            })
    
    # Calculate domain summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["passed"])
    failed_tests = total_tests - passed_tests
    total_rows = sum(r["rows"] for r in results)
    total_mismatched = sum(r["mismatched_fields"] for r in results)
    
    # Print domain summary
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{domain_name.upper()} SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    print(f"Total tests:        {total_tests}")
    print(f"{Colors.GREEN}Passed:             {passed_tests}{Colors.RESET}")
    if failed_tests > 0:
        print(f"{Colors.RED}Failed:             {failed_tests}{Colors.RESET}")
    else:
        print(f"Failed:             {failed_tests}")
    print(f"Total rows extracted: {total_rows}")
    print(f"Total mismatched fields: {total_mismatched}\n")
    print("Per-source results:\n")
    
    for r in results:
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if r["passed"] else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {r['source']:<25} {status}  ({r['rows']} rows, {r['mismatched_fields']} mismatched fields)")
        if "error" in r:
            print(f"    {Colors.RED}Error: {r['error']}{Colors.RESET}")
    
    return {
        "domain": domain_name,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "total_rows": total_rows,
        "total_mismatched_fields": total_mismatched,
        "results": results
    }


def main():
    """Run all domain test suites in order."""
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}UNIFIED TEST SUITE - ALL DOMAINS{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    
    domain_results = []
    
    # Add tests directory to path for imports
    TESTS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(TESTS_DIR))
    
    # 1. POLICIES
    try:
        import test_policies_all_sources as policies_module
        policies_result = run_domain_tests(
            "Policies",
            policies_module,
            [
                ("Airtable", "test_airtable"),
                ("Google Sheet", "test_google_sheet"),
                ("Excel", "test_excel"),
                ("Raw Text", "test_raw_text"),
                ("PDF", "test_pdf"),
            ]
        )
        domain_results.append(policies_result)
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR importing policies tests: {e}{Colors.RESET}\n")
        import traceback
        traceback.print_exc()
    
    # 2. DRIVERS
    try:
        import test_drivers_all_sources as drivers_module
        drivers_result = run_domain_tests(
            "Drivers",
            drivers_module,
            [
                ("Airtable", "test_airtable"),
                ("Google Sheet", "test_google_sheet"),
                ("Excel", "test_excel"),
                ("Raw Text", "test_raw_text"),
                ("PDF", "test_pdf"),
                ("Image", "test_image"),
            ]
        )
        domain_results.append(drivers_result)
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR importing drivers tests: {e}{Colors.RESET}\n")
        import traceback
        traceback.print_exc()
    
    # 3. VEHICLES
    try:
        import test_vehicles_all_sources as vehicles_module
        vehicles_result = run_domain_tests(
            "Vehicles",
            vehicles_module,
            [
                ("Airtable", "test_airtable"),
                ("Excel", "test_excel"),
                ("Google Sheet", "test_google_sheet"),
                ("PDF", "test_pdf"),
                ("Raw Text", "test_raw_text"),
                ("Image Metadata", "test_image"),
            ]
        )
        domain_results.append(vehicles_result)
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR importing vehicles tests: {e}{Colors.RESET}\n")
        import traceback
        traceback.print_exc()
    
    # 4. CLAIMS
    try:
        import test_claims_all_sources as claims_module
        claims_result = run_domain_tests(
            "Claims",
            claims_module,
            [
                ("Google Sheet", "test_google_sheet"),
                ("PDF", "test_pdf"),
            ]
        )
        domain_results.append(claims_result)
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR importing claims tests: {e}{Colors.RESET}\n")
        import traceback
        traceback.print_exc()
    
    # 5. RELATIONSHIPS
    try:
        import test_policy_vehicle_driver_link_all_sources as relationships_module
        relationships_result = run_domain_tests(
            "Relationships",
            relationships_module,
            [
                ("CSV", "test_csv"),
                ("PDF", "test_pdf"),
            ]
        )
        domain_results.append(relationships_result)
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR importing relationships tests: {e}{Colors.RESET}\n")
        import traceback
        traceback.print_exc()
    
    # GLOBAL SUMMARY
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}GLOBAL SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    total_domains = len(domain_results)
    total_tests = sum(r["total_tests"] for r in domain_results)
    total_passed = sum(r["passed_tests"] for r in domain_results)
    total_failed = sum(r["failed_tests"] for r in domain_results)
    total_rows = sum(r["total_rows"] for r in domain_results)
    total_mismatched = sum(r["total_mismatched_fields"] for r in domain_results)
    
    print(f"Total domains tested: {total_domains}")
    print(f"Total tests run:      {total_tests}")
    print(f"{Colors.GREEN}Total passed:        {total_passed}{Colors.RESET}")
    if total_failed > 0:
        print(f"{Colors.RED}Total failed:        {total_failed}{Colors.RESET}")
    else:
        print(f"Total failed:        {total_failed}")
    print(f"Total rows extracted: {total_rows}")
    print(f"Total mismatched fields: {total_mismatched}\n")
    
    print("Per-domain summary:\n")
    for r in domain_results:
        domain_status = f"{Colors.GREEN}✓{Colors.RESET}" if r["failed_tests"] == 0 else f"{Colors.RED}✗{Colors.RESET}"
        print(f"  {domain_status} {r['domain']:<15} {r['passed_tests']}/{r['total_tests']} tests passed, {r['total_mismatched_fields']} mismatched fields")
    
    print()
    
    # Exit with error code if any tests failed
    if total_failed > 0 or total_mismatched > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
