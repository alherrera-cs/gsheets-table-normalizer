#!/usr/bin/env python3
"""
Test script for new mapping structure.

Tests all dataset mappings (drivers, policies, locations, relationships, claims)
using sample CSV files from the tests directory.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from normalizer import normalize_v2
from mappings import (
    VEHICLES_MAPPINGS,
    DRIVERS_MAPPINGS,
    POLICIES_MAPPINGS,
    LOCATIONS_MAPPINGS,
    RELATIONSHIPS_MAPPINGS,
    CLAIMS_MAPPINGS,
    get_mapping_by_id,
)

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    DIM = '\033[2m'


def colorize(text: str, color: str) -> str:
    """Apply color to text if terminal supports it."""
    try:
        return f"{color}{text}{Colors.RESET}" if sys.stdout.isatty() else text
    except:
        return text


# Test configurations - mapping files to mapping IDs
# Tests all three structured file formats: CSV, Excel (XLSX), and Airtable (JSON)
TEST_CONFIGS = [
    # Vehicles - all three formats
    {
        "name": "Vehicles - CSV",
        "file_path": "vehicles/structured/google_sheet_vehicle_inventory.csv",
        "mapping_id": "source_google_sheet_vehicles",
    },
    {
        "name": "Vehicles - Excel",
        "file_path": "vehicles/structured/excel_vehicle_export.xlsx",
        "mapping_id": "source_google_sheet_vehicles",
    },
    {
        "name": "Vehicles - Airtable",
        "file_path": "vehicles/structured/airtable_fleet_vehicles.json",
        "mapping_id": "source_google_sheet_vehicles",
    },
    # Drivers - all three formats
    {
        "name": "Drivers - CSV",
        "file_path": "drivers/structured/google_sheet_drivers.csv",
        "mapping_id": "source_google_sheet_drivers",
    },
    {
        "name": "Drivers - Excel",
        "file_path": "drivers/structured/excel_drivers.xlsx",
        "mapping_id": "source_google_sheet_drivers",
    },
    {
        "name": "Drivers - Airtable",
        "file_path": "drivers/structured/airtable_drivers.json",
        "mapping_id": "source_google_sheet_drivers",
    },
    # Policies - all three formats
    {
        "name": "Policies - CSV",
        "file_path": "policies/structured/google_sheet_policies.csv",
        "mapping_id": "source_google_sheet_policies",
    },
    {
        "name": "Policies - Excel",
        "file_path": "policies/structured/excel_policies.xlsx",
        "mapping_id": "source_google_sheet_policies",
    },
    {
        "name": "Policies - Airtable",
        "file_path": "policies/structured/airtable_policies.json",
        "mapping_id": "source_google_sheet_policies",
    },
    # Locations - CSV only (no other formats available)
    {
        "name": "Locations - CSV",
        "file_path": "locations/structured/garaging_locations.csv",
        "mapping_id": "source_google_sheet_locations",
    },
    # Relationships - CSV only
    {
        "name": "Relationships - CSV",
        "file_path": "relationships/structured/policy_vehicle_driver_link.csv",
        "mapping_id": "source_google_sheet_relationships",
    },
    # Claims - CSV only
    {
        "name": "Claims - CSV",
        "file_path": "claims/structured/google_sheet_claims.csv",
        "mapping_id": "source_google_sheet_claims",
    },
]


def test_mapping(config: Dict[str, str]) -> Dict[str, Any]:
    """
    Test a single mapping configuration.
    
    Args:
        config: Test configuration with name, csv_file, and mapping_id
        
    Returns:
        Test result dictionary
    """
    test_name = config["name"]
    csv_file = ROOT / "tests" / config["csv_file"]
    mapping_id = config["mapping_id"]
    
    print(f"\n{colorize('─' * 80, Colors.BOLD)}")
    print(f"{colorize(f'Testing: {test_name}', Colors.BOLD + Colors.CYAN)}")
    print(f"{colorize('─' * 80, Colors.BOLD)}")
    print(f"CSV File: {csv_file.name}")
    print(f"Mapping ID: {mapping_id}")
    
    # Get mapping configuration
    mapping_config = get_mapping_by_id(mapping_id)
    if not mapping_config:
        print(f"{colorize('❌ ERROR: Mapping not found!', Colors.RED)}")
        return {
            "name": test_name,
            "success": False,
            "error": f"Mapping '{mapping_id}' not found",
        }
    
    # Update connection_config to point to CSV file
    mapping_config["metadata"]["connection_config"]["file_path"] = str(csv_file)
    mapping_config["metadata"]["source_type"] = "csv"  # Override for CSV testing
    
    # Prepare source
    source = {"file_path": str(csv_file)}
    
    try:
        # Run normalization
        result = normalize_v2(
            source=source,
            mapping_config=mapping_config,
            header_row_index=0,
            validate=True,
        )
        
        # Print results
        print(f"\n{colorize('Results:', Colors.BOLD)}")
        print(f"  Success: {colorize('✓', Colors.GREEN) if result['success'] else colorize('✗', Colors.RED)}")
        print(f"  Total Rows: {result['total_rows']}")
        print(f"  Successful: {colorize(result['total_success'], Colors.GREEN)}")
        print(f"  Errors: {colorize(result['total_errors'], Colors.RED) if result['total_errors'] > 0 else colorize('0', Colors.GREEN)}")
        
        # Show sample data
        if result['data']:
            print(f"\n{colorize('Sample Normalized Data (first row):', Colors.BOLD)}")
            sample = result['data'][0]
            for key, value in sample.items():
                print(f"  {key}: {value}")
        
        # Show errors if any
        if result['errors']:
            print(f"\n{colorize('Validation Errors:', Colors.RED + Colors.BOLD)}")
            # Group errors by row for better readability
            errors_by_row = {}
            for error in result['errors']:
                row_num = error.get('_source_row_number', '?')
                if row_num not in errors_by_row:
                    errors_by_row[row_num] = []
                errors_by_row[row_num].append(error)
            
            for row_num in sorted(errors_by_row.keys(), key=lambda x: x if isinstance(x, int) else 999):
                errors = errors_by_row[row_num]
                print(f"  {colorize(f'Row {row_num}:', Colors.YELLOW)}")
                for error in errors[:3]:  # Show first 3 errors per row
                    field = error.get('target_field', 'unknown field')
                    msg = error.get('error', 'Unknown error')
                    print(f"    • {field}: {msg}")
                if len(errors) > 3:
                    print(f"    ... and {len(errors) - 3} more errors for this row")
            
            if len(result['errors']) > len(errors_by_row) * 3:
                total_shown = sum(min(3, len(errors_by_row[r])) for r in errors_by_row)
                print(f"  {colorize(f'... and {len(result["errors"]) - total_shown} more errors total', Colors.DIM)}")
        
        return {
            "name": test_name,
            "success": result['success'],
            "total_rows": result['total_rows'],
            "total_success": result['total_success'],
            "total_errors": result['total_errors'],
            "errors": result['errors'],
            "data": result.get('data', []),  # Include normalized data for relevance analysis
            "mapping_id": mapping_id,  # Include mapping_id for reference
            "csv_file": str(csv_file),  # Include CSV file path for reference
        }
        
    except Exception as e:
        print(f"{colorize(f'❌ EXCEPTION: {str(e)}', Colors.RED)}")
        import traceback
        traceback.print_exc()
        return {
            "name": test_name,
            "success": False,
            "error": str(e),
        }


def main():
    """Run all mapping tests."""
    print(f"\n{colorize('═' * 80, Colors.BOLD)}")
    print(f"{colorize('MAPPING TESTS', Colors.BOLD + Colors.CYAN)}")
    print(f"{colorize('═' * 80, Colors.BOLD)}")
    print(f"Testing {len(TEST_CONFIGS)} mapping configurations...\n")
    
    results = []
    for config in TEST_CONFIGS:
        result = test_mapping(config)
        results.append(result)
    
    # Summary
    print(f"\n{colorize('═' * 80, Colors.BOLD)}")
    print(f"{colorize('SUMMARY', Colors.BOLD + Colors.CYAN)}")
    print(f"{colorize('═' * 80, Colors.BOLD)}")
    
    passed = sum(1 for r in results if r.get('success', False))
    failed = len(results) - passed
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"{colorize(f'Passed: {passed}', Colors.GREEN)}")
    print(f"{colorize(f'Failed: {failed}', Colors.RED if failed > 0 else Colors.GREEN)}")
    
    if failed > 0:
        print(f"\n{colorize('Failed Tests:', Colors.RED + Colors.BOLD)}")
        for result in results:
            if not result.get('success', False):
                print(f"  - {result['name']}")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
    
    # Exit with error code if any tests failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run mapping tests')
    parser.add_argument('--visual', '-v', action='store_true', 
                       help='Generate visual HTML report after tests')
    args = parser.parse_args()
    
    main()
    
    # Generate visual report if requested
    if args.visual:
        print("\n" + "=" * 80)
        print("Generating visual analysis report...")
        print("=" * 80)
        try:
            from test_visual_analysis import generate_html_report
            from datetime import datetime
            from pathlib import Path
            
            # Re-run tests to get results (or you could pass results from main)
            # For now, we'll just note that --visual flag was used
            print("\n💡 Tip: Run 'python tests/test_visual_analysis.py' for full visual analysis")
        except ImportError:
            print("⚠️  Visual analysis module not available")


