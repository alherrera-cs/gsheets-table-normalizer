"""
DEPRECATED: This file has been superseded by test_all_sources.py

This older unified test runner is kept for reference but should not be used.
The new test_all_sources.py provides:
- Better structured results capture
- Per-domain and global summaries
- Consistent test ordering (policies, drivers, vehicles, claims, relationships)
- More detailed reporting

Please use tests/test_all_sources.py instead.

Original description:
Unified test runner for all domain test suites.

Runs all domain-level tests:
- test_drivers_all_sources
- test_vehicles_all_sources
- test_policies_all_sources
- test_locations_all_sources
- test_policy_vehicle_driver_link_all_sources
- test_claims_all_sources
"""

import sys
from pathlib import Path

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
    CYAN = '\033[96m'


def run_domain_test(module_name: str, test_name: str) -> tuple:
    """Run a domain test module and return (passed, rows, mismatched)."""
    try:
        module = __import__(f"tests.{module_name}", fromlist=[test_name])
        test_func = getattr(module, test_name)
        return test_func()
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR running {module_name}.{test_name}: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return (False, 0, 0)


def main():
    """Run all domain test suites."""
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}UNIFIED DOMAIN TEST RUNNER{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    # Import and run each domain test suite
    domain_tests = [
        ("test_drivers_all_sources", "main"),
        ("test_vehicles_all_sources", "main"),
        ("test_policies_all_sources", "main"),
        ("test_locations_all_sources", "main"),
        ("test_policy_vehicle_driver_link_all_sources", "main"),
        ("test_claims_all_sources", "main"),
    ]
    
    results = []
    for module_name, test_name in domain_tests:
        print(f"\n{Colors.BOLD}Running {module_name}...{Colors.RESET}\n")
        try:
            module = __import__(f"tests.{module_name}", fromlist=[test_name])
            test_func = getattr(module, test_name)
            test_func()
            # Note: Individual test suites print their own summaries
            results.append((module_name, True))
        except Exception as e:
            print(f"{Colors.RED}✗ ERROR in {module_name}: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            results.append((module_name, False))
    
    # Final summary
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}FINAL SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Total domain test suites: {total}")
    print(f"{Colors.GREEN}Successfully completed: {passed}{Colors.RESET}")
    if passed < total:
        print(f"{Colors.RED}Failed: {total - passed}{Colors.RESET}")
    
    print("\nDomain test suites:")
    for module_name, success in results:
        status = f"{Colors.GREEN}✓{Colors.RESET}" if success else f"{Colors.RED}✗{Colors.RESET}"
        print(f"  {status} {module_name}")
    
    print()
    
    # Exit with error code if any failed
    if passed < total:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
