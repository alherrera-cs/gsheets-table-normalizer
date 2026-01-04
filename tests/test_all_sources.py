"""
Unified test runner for ALL domains and sources.

Production-ready test suite that runs all domain tests with:
- Logging suppression (clean output by default)
- Verbosity control (--verbose, --full-diff, --debug)
- Executive summary format (for drivers/vehicles)
- Simple summary format (for other domains)
- Confidence-aware mismatch classification

Runs all domain test suites in order:
1. Policies
2. Drivers
3. Vehicles
4. Claims
5. Relationships
6. Locations
"""

import sys
import logging
import argparse
import io
import contextlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Suppress logging by default (will be re-enabled if --verbose or --debug)
# This must happen BEFORE importing any modules that might log
# Set root logger to CRITICAL to suppress all logging by default
logging.getLogger().setLevel(logging.CRITICAL)
# Suppress specific noisy loggers (use exact module names)
for logger_name in [
    "src", "src.sources", "src.ocr", "src.normalizer", "src.transforms",
    "ocr", "ocr.parser", "ocr.table_extract", "ocr.reader", "ocr.models",
    "__main__", "parser", "table_extract"
]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Global flags for output control
VERBOSE = False
FULL_DIFF = False
DEBUG = False

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


def print_unified_executive_summary(all_domain_results: List[Dict[str, Any]]) -> None:
    """
    Print unified executive summary across all domains.
    
    Args:
        all_domain_results: List of domain result dictionaries
    """
    # Aggregate statistics
    total_domains = len(all_domain_results)
    total_tests = sum(r["total_tests"] for r in all_domain_results)
    total_passed = sum(r["passed_tests"] for r in all_domain_results)
    total_failed = sum(r["failed_tests"] for r in all_domain_results)
    total_rows = sum(r["total_rows"] for r in all_domain_results)
    total_critical = sum(r.get("total_critical", 0) for r in all_domain_results)
    total_acceptable = sum(r.get("total_acceptable", 0) for r in all_domain_results)
    
    # Determine overall status
    if total_failed == 0 and total_critical == 0:
        overall_status = "PASS"
        status_color = Colors.GREEN
        production_safe = True
    elif total_failed == 0:
        overall_status = "PASS WITH WARNINGS"
        status_color = Colors.YELLOW
        production_safe = True
    else:
        overall_status = "FAIL"
        status_color = Colors.RED
        production_safe = False
    
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}UNIFIED VALIDATION REPORT - ALL DOMAINS{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*80}{Colors.RESET}\n")
    
    # Top-level status
    print(f"{Colors.BOLD}VALIDATION STATUS:{Colors.RESET} {status_color}{Colors.BOLD}{overall_status}{Colors.RESET}")
    if production_safe:
        print(f"  {Colors.GREEN}✓ Production-Safe{Colors.RESET} | Domains: {total_domains} | Tests: {total_tests} | Rows: {total_rows} | Critical: {Colors.RED}{total_critical}{Colors.RESET} | Acceptable: {Colors.GRAY}{total_acceptable}{Colors.RESET}\n")
    else:
        print(f"  {Colors.RED}✗ Not Production-Safe{Colors.RESET} | Domains: {total_domains} | Tests: {total_tests} | Rows: {total_rows} | Critical: {Colors.RED}{total_critical}{Colors.RESET} | Acceptable: {Colors.GRAY}{total_acceptable}{Colors.RESET}\n")
    
    # Domain health table
    print(f"{Colors.BOLD}Domain Health:{Colors.RESET}")
    print(f"  {Colors.BOLD}{'Domain':<20} {'Status':<20} {'Tests':>8} {'Passed':>8} {'Rows':>8} {'Critical':>10} {'Acceptable':>12}{Colors.RESET}")
    print(f"  {Colors.GRAY}{'─' * 100}{Colors.RESET}")
    
    for r in all_domain_results:
        domain = r["domain"]
        passed_tests = r["passed_tests"]
        total_tests_domain = r["total_tests"]
        total_rows_domain = r["total_rows"]
        critical = r.get("total_critical", 0)
        acceptable = r.get("total_acceptable", 0)
        
        # Determine domain status
        if r["failed_tests"] == 0 and critical == 0:
            status_text = "PASS"
            status_str = f"{Colors.GREEN}{status_text}{Colors.RESET}"
            status_visible_width = 4
        elif r["failed_tests"] == 0:
            status_text = "PASS WITH WARNINGS"
            status_str = f"{Colors.YELLOW}{status_text}{Colors.RESET}"
            status_visible_width = 18
        else:
            status_text = "FAIL"
            status_str = f"{Colors.RED}{status_text}{Colors.RESET}"
            status_visible_width = 4
        
        status_padding = max(0, 20 - status_visible_width)
        critical_str = f"{Colors.RED}{critical}{Colors.RESET}" if critical > 0 else "0"
        acceptable_str = f"{Colors.GRAY}{acceptable}{Colors.RESET}" if acceptable > 0 else "0"
        
        print(f"  {domain:<20} {status_str}{' ' * status_padding} {total_tests_domain:>8} {passed_tests:>8} {total_rows_domain:>8} {critical_str:>10} {acceptable_str:>12}")
    print()
    
    # Issue breakdown
    if total_critical > 0 or total_acceptable > 0:
        print(f"{Colors.BOLD}Issue Breakdown:{Colors.RESET}\n")
        
        if total_critical > 0:
            print(f"  {Colors.RED}{Colors.BOLD}CRITICAL ISSUES:{Colors.RESET} {total_critical}")
            print(f"    • Missing/extra rows")
            print(f"    • High-confidence field mismatches (confidence >= 0.9)")
            print(f"    • Missing critical fields (VIN, Driver ID, etc.)")
            print()
        
        if total_acceptable > 0:
            print(f"  {Colors.GRAY}{Colors.BOLD}ACCEPTABLE DEGRADATION:{Colors.RESET} {total_acceptable}")
            print(f"    • Case-only differences (normalization)")
            print(f"    • Low-confidence OCR errors (confidence < 0.9)")
            print(f"    • Notes populated from OCR context (unstructured sources)")
            print()
        
        print(f"  {Colors.BOLD}Note:{Colors.RESET} Use {Colors.CYAN}--verbose{Colors.RESET} or {Colors.CYAN}--full-diff{Colors.RESET} for detailed row-level diagnostics\n")


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
    if not VERBOSE:
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}Testing {domain_name}...{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    results = []
    for source_name, func_name in test_functions:
        try:
            test_func = getattr(test_module, func_name)
            
            # Suppress individual test function output when not verbose
            if VERBOSE or FULL_DIFF:
                # Show all output when verbose
                result = test_func()
            else:
                # Suppress stdout from individual test functions for clean output
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    result = test_func()
                # Only show minimal status
                if not result[0]:  # If test failed
                    print(f"  {Colors.RED}✗{Colors.RESET} {source_name}")
                else:
                    print(f"  {Colors.GREEN}✓{Colors.RESET} {source_name} ({result[1]} rows)")
            
            # Handle both return formats:
            # - 3 values: (passed, rows, mismatched) for policies/claims/relationships/locations
            # - 4 values: (passed, rows, real_mismatched, acceptable_mismatched) for drivers/vehicles
            if len(result) == 4:
                passed, rows, real_mismatched, acceptable_mismatched = result
                mismatched = real_mismatched
            else:
                passed, rows, mismatched = result
                real_mismatched = mismatched
                acceptable_mismatched = 0
            
            results.append({
                "source": source_name,
                "passed": passed,
                "rows": rows,
                "mismatched_fields": mismatched,
                "real_mismatched": real_mismatched,
                "acceptable_mismatched": acceptable_mismatched
            })
        except Exception as e:
            if VERBOSE:
                print(f"{Colors.RED}✗ ERROR running {source_name}: {e}{Colors.RESET}\n")
                import traceback
                traceback.print_exc()
            else:
                print(f"  {Colors.RED}✗{Colors.RESET} {source_name} - ERROR: {e}")
            results.append({
                "source": source_name,
                "passed": False,
                "rows": 0,
                "mismatched_fields": 0,
                "real_mismatched": 0,
                "acceptable_mismatched": 0,
                "error": str(e)
            })
    
    # Calculate domain summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["passed"])
    failed_tests = total_tests - passed_tests
    total_rows = sum(r["rows"] for r in results)
    total_mismatched = sum(r["mismatched_fields"] for r in results)
    total_critical = sum(r["real_mismatched"] for r in results)
    total_acceptable = sum(r["acceptable_mismatched"] for r in results)
    
    if VERBOSE or FULL_DIFF:
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
        print(f"Total mismatched fields: {total_mismatched}")
        if total_critical > 0 or total_acceptable > 0:
            print(f"Critical mismatches: {total_critical}")
            print(f"Acceptable mismatches: {total_acceptable}")
        print()
        print("Per-source results:\n")
        
        for r in results:
            status = f"{Colors.GREEN}PASS{Colors.RESET}" if r["passed"] else f"{Colors.RED}FAIL{Colors.RESET}"
            if total_critical > 0 or total_acceptable > 0:
                print(f"  {r['source']:<25} {status}  ({r['rows']} rows, {r['real_mismatched']} critical, {r['acceptable_mismatched']} acceptable)")
            else:
                print(f"  {r['source']:<25} {status}  ({r['rows']} rows, {r['mismatched_fields']} mismatched fields)")
            if "error" in r:
                print(f"    {Colors.RED}Error: {r['error']}{Colors.RESET}")
        print()
    
    return {
        "domain": domain_name,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "total_rows": total_rows,
        "total_mismatched_fields": total_mismatched,
        "total_critical": total_critical,
        "total_acceptable": total_acceptable,
        "results": results
    }


def main():
    """Run all domain test suites in order."""
    global VERBOSE, FULL_DIFF, DEBUG
    
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Unified test suite for all domains")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output including per-domain summaries")
    parser.add_argument("--full-diff", action="store_true", help="Show full field-by-field diffs for all rows")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    VERBOSE = args.verbose
    FULL_DIFF = args.full_diff
    DEBUG = args.debug
    
    # Re-apply logging suppression after parsing args (in case modules reset it)
    if not (VERBOSE or DEBUG):
        # Suppress ALL logging by default for clean output
        logging.getLogger().setLevel(logging.CRITICAL)
        # Suppress specific noisy loggers (use exact module names)
        for logger_name in [
            "src", "src.sources", "src.ocr", "src.normalizer", "src.transforms",
            "ocr", "ocr.parser", "ocr.table_extract", "ocr.reader", "ocr.models",
            "__main__", "parser", "table_extract"
        ]:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    
    if not VERBOSE:
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}UNIFIED TEST SUITE - ALL DOMAINS{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*80}{Colors.RESET}\n")
        print(f"{Colors.GRAY}Running all domain tests... (use --verbose for detailed output){Colors.RESET}\n")
    
    domain_results = []
    
    # Add tests directory to path for imports
    TESTS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(TESTS_DIR))
    
    # 1. POLICIES
    try:
        import test_policies_all_sources as policies_module
        # Set verbosity flags in the module if it supports them
        if hasattr(policies_module, 'VERBOSE'):
            policies_module.VERBOSE = VERBOSE
        if hasattr(policies_module, 'FULL_DIFF'):
            policies_module.FULL_DIFF = FULL_DIFF
        if hasattr(policies_module, 'DEBUG'):
            policies_module.DEBUG = DEBUG
        
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
        if VERBOSE:
            import traceback
            traceback.print_exc()
    
    # 2. DRIVERS
    try:
        import test_drivers_all_sources as drivers_module
        if hasattr(drivers_module, 'VERBOSE'):
            drivers_module.VERBOSE = VERBOSE
        if hasattr(drivers_module, 'FULL_DIFF'):
            drivers_module.FULL_DIFF = FULL_DIFF
        if hasattr(drivers_module, 'DEBUG'):
            drivers_module.DEBUG = DEBUG
        
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
                ("Handwritten PDF", "test_handwritten_pdf"),
                ("Handwritten Image", "test_handwritten_image"),
            ]
        )
        domain_results.append(drivers_result)
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR importing drivers tests: {e}{Colors.RESET}\n")
        if VERBOSE:
            import traceback
            traceback.print_exc()
    
    # 3. VEHICLES
    try:
        import test_vehicles_all_sources as vehicles_module
        if hasattr(vehicles_module, 'VERBOSE'):
            vehicles_module.VERBOSE = VERBOSE
        if hasattr(vehicles_module, 'FULL_DIFF'):
            vehicles_module.FULL_DIFF = FULL_DIFF
        if hasattr(vehicles_module, 'DEBUG'):
            vehicles_module.DEBUG = DEBUG
        
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
        if VERBOSE:
            import traceback
            traceback.print_exc()
    
    # 4. CLAIMS
    try:
        import test_claims_all_sources as claims_module
        if hasattr(claims_module, 'VERBOSE'):
            claims_module.VERBOSE = VERBOSE
        if hasattr(claims_module, 'FULL_DIFF'):
            claims_module.FULL_DIFF = FULL_DIFF
        if hasattr(claims_module, 'DEBUG'):
            claims_module.DEBUG = DEBUG
        
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
        if VERBOSE:
            import traceback
            traceback.print_exc()
    
    # 5. RELATIONSHIPS
    try:
        import test_policy_vehicle_driver_link_all_sources as relationships_module
        if hasattr(relationships_module, 'VERBOSE'):
            relationships_module.VERBOSE = VERBOSE
        if hasattr(relationships_module, 'FULL_DIFF'):
            relationships_module.FULL_DIFF = FULL_DIFF
        if hasattr(relationships_module, 'DEBUG'):
            relationships_module.DEBUG = DEBUG
        
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
        if VERBOSE:
            import traceback
            traceback.print_exc()
    
    # 6. LOCATIONS
    try:
        import test_locations_all_sources as locations_module
        if hasattr(locations_module, 'VERBOSE'):
            locations_module.VERBOSE = VERBOSE
        if hasattr(locations_module, 'FULL_DIFF'):
            locations_module.FULL_DIFF = FULL_DIFF
        if hasattr(locations_module, 'DEBUG'):
            locations_module.DEBUG = DEBUG
        
        locations_result = run_domain_tests(
            "Locations",
            locations_module,
            [
                ("CSV", "test_csv"),
                ("PDF", "test_pdf"),
            ]
        )
        domain_results.append(locations_result)
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR importing locations tests: {e}{Colors.RESET}\n")
        if VERBOSE:
            import traceback
            traceback.print_exc()
    
    # Print unified executive summary
    print_unified_executive_summary(domain_results)
    
    # Determine exit code
    total_failed = sum(r["failed_tests"] for r in domain_results)
    total_critical = sum(r.get("total_critical", 0) for r in domain_results)
    
    if total_failed > 0 or total_critical > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
