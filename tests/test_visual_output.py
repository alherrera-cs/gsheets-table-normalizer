#!/usr/bin/env python3
"""
Visual output test for normalization pipeline - tests all file formats.

Prints normalized output for visual verification:
- Dataset name and file format
- Success/error counts
- Fields found
- Missing required fields (if any)
- Sample rows (if verbose)
- Missing header variants
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, Set, List, Optional
from collections import defaultdict

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from normalizer import normalize_v2
from mappings import get_mapping_by_id
from sources import detect_source_type, SourceType, extract_from_source
from external_tables import rows2d_to_objects, clean_header

# Import TEST_CONFIGS
try:
    from test_mappings import TEST_CONFIGS
except ImportError:
    # Fallback if test_mappings not available
    TEST_CONFIGS = []

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    DIM = '\033[2m'


def colorize(text: str, color: str) -> str:
    """Apply color to text if terminal supports it."""
    try:
        return f"{color}{text}{Colors.RESET}" if sys.stdout.isatty() else text
    except:
        return text


def get_schema_from_data(data: list) -> Set[str]:
    """Extract all field names from normalized data (excluding system fields)."""
    schema = set()
    for row in data:
        for key in row.keys():
            if not key.startswith("_") and key != "errors":
                schema.add(key)
    return schema


def get_source_headers(file_path: Path, source_type: SourceType) -> List[str]:
    """Extract headers from source file."""
    try:
        if source_type == SourceType.CSV:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                headers = [clean_header(h) for h in next(reader)]
                return headers
        elif source_type == SourceType.AIRTABLE:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("records"):
                    headers = [clean_header(h) for h in data['records'][0]['fields'].keys()]
                    return headers
        elif source_type == SourceType.XLSX:
            # Try to get headers from Excel
            try:
                from openpyxl import load_workbook
                wb = load_workbook(file_path, data_only=True)
                row = [cell.value for cell in wb.active[1]]
                headers = [clean_header(str(h)) for h in row if h]
                return headers
            except ImportError:
                # Fallback: assume same as CSV (for testing)
                return []
    except Exception as e:
        return []
    return []


def get_mapping_source_fields(mapping_config: Dict) -> Dict[str, List[str]]:
    """Get all source field variants for each target field."""
    variants = defaultdict(list)
    for mapping in mapping_config.get("mappings", []):
        target = mapping.get("target_field")
        source = mapping.get("source_field", "")
        if source:
            variants[target].append(clean_header(source))
    return dict(variants)


def find_missing_variants(
    source_headers: List[str],
    mapping_variants: Dict[str, List[str]],
    target_field: str
) -> List[str]:
    """Find source headers that could map to target_field but aren't in mappings."""
    # This is a simple heuristic - could be improved
    target_lower = target_field.lower()
    missing = []
    for header in source_headers:
        header_lower = header.lower()
        # Check if header contains target field name but isn't mapped
        if target_lower in header_lower or header_lower in target_lower:
            if header not in mapping_variants.get(target_field, []):
                missing.append(header)
    return missing


def format_row(row: Dict[str, Any], verbose: bool = False) -> str:
    """Format a row for pretty printing."""
    display_fields = {k: v for k, v in row.items() 
                     if not k.startswith("_") and k != "errors"}
    
    if not display_fields:
        return colorize("  (no normalized fields)", Colors.DIM)
    
    if not verbose:
        # Compact format - just show field count
        non_empty = sum(1 for v in display_fields.values() if v is not None and v != "")
        return f"  {non_empty}/{len(display_fields)} fields populated"
    
    # Verbose format - show all fields
    lines = []
    for key, value in sorted(display_fields.items()):
        key_colored = colorize(key, Colors.CYAN)
        if value is None:
            value_str = colorize("None", Colors.DIM)
            lines.append(f"    {key_colored}: {value_str}")
        elif isinstance(value, str) and len(value) > 50:
            lines.append(f"    {key_colored}: {value[:47]}...")
        else:
            lines.append(f"    {key_colored}: {value}")
    
    return "\n".join(lines)


def print_dataset_output(config: Dict[str, str], verbose: bool = False) -> None:
    """Process one dataset and print visual output."""
    test_name = config["name"]
    file_path = config.get("file_path") or config.get("csv_file")
    test_file = ROOT / "tests" / file_path
    mapping_id = config["mapping_id"]
    
    # Detect file format
    detected_type = detect_source_type(test_file)
    format_name = {
        SourceType.CSV: "CSV",
        SourceType.XLSX: "Excel",
        SourceType.AIRTABLE: "Airtable",
        SourceType.GOOGLE_SHEETS: "Google Sheets",
    }.get(detected_type, "Unknown")
    
    print(f"\n{colorize('=' * 80, Colors.BOLD)}")
    print(f"{colorize('DATASET:', Colors.BOLD)} {colorize(test_name, Colors.BOLD + Colors.CYAN)}")
    print(f"{colorize('FORMAT:', Colors.BOLD)} {colorize(format_name, Colors.YELLOW)}")
    print(f"{colorize('=' * 80, Colors.BOLD)}")
    
    # Get mapping configuration
    mapping_config = get_mapping_by_id(mapping_id)
    if not mapping_config:
        print(f"{colorize('❌ ERROR:', Colors.RED + Colors.BOLD)} Mapping '{mapping_id}' not found!")
        return
    
    # Update connection_config
    mapping_config["metadata"]["connection_config"]["file_path"] = str(test_file)
    
    # Map SourceType to string
    source_type_map = {
        SourceType.CSV: "csv",
        SourceType.XLSX: "xlsx_file",
        SourceType.AIRTABLE: "airtable",
        SourceType.GOOGLE_SHEETS: "google_sheet",
    }
    mapping_config["metadata"]["source_type"] = source_type_map.get(detected_type, "csv")
    
    # Prepare source
    source = {"file_path": str(test_file)}
    
    try:
        # Run normalization
        result = normalize_v2(
            source=source,
            mapping_config=mapping_config,
            header_row_index=0,
            validate=True,
        )
        
        # Print summary
        print(f"\n{colorize('Summary:', Colors.BOLD)}")
        print(f"  Total rows: {result['total_rows']}")
        success_count = result['total_success']
        error_count = result['total_errors']
        if success_count > 0:
            print(f"  {colorize('Successful:', Colors.GREEN)} {colorize(str(success_count), Colors.GREEN + Colors.BOLD)}")
        if error_count > 0:
            print(f"  {colorize('Errors:', Colors.RED)} {colorize(str(error_count), Colors.RED + Colors.BOLD)}")
        else:
            print(f"  {colorize('Errors:', Colors.GREEN)} {error_count}")
        
        # Get source headers for variant analysis
        source_headers = get_source_headers(test_file, detected_type)
        mapping_variants = get_mapping_source_fields(mapping_config)
        
        # Print fields found
        data = result.get("data", [])
        if data:
            all_fields = get_schema_from_data(data)
            print(f"\n{colorize('Fields Found:', Colors.BOLD)} {len(all_fields)}")
            
            if verbose:
                for field in sorted(all_fields):
                    print(f"  - {colorize(field, Colors.CYAN)}")
            
            # Check required fields
            required_fields = {}
            for mapping in mapping_config.get("mappings", []):
                if mapping.get("required") and mapping.get("target_field") not in required_fields:
                    required_fields[mapping.get("target_field")] = False
            
            missing_required = []
            for field in required_fields.keys():
                found = any(row.get(field) for row in data if row.get(field))
                required_fields[field] = found
                if not found:
                    missing_required.append(field)
            
            if missing_required:
                print(f"\n{colorize('⚠️  Missing Required Fields:', Colors.RED + Colors.BOLD)}")
                for field in missing_required:
                    print(f"  - {colorize(field, Colors.RED)}")
                    # Show missing variants
                    if source_headers:
                        missing_vars = find_missing_variants(source_headers, mapping_variants, field)
                        if missing_vars:
                            print(f"    Potential unmapped headers: {', '.join(missing_vars)}")
            else:
                print(f"  {colorize('✓ All required fields found', Colors.GREEN)}")
        else:
            print(f"\n  {colorize('No successful rows to display', Colors.DIM)}")
        
        # Print sample rows (if verbose or if there are successful rows)
        if data and (verbose or len(data) <= 3):
            print(f"\n{colorize('Sample Rows:', Colors.BOLD)}")
            for i, row in enumerate(data[:2], 1):
                print(f"\n  {colorize(f'Row {i}:', Colors.YELLOW + Colors.BOLD)}")
                print(format_row(row, verbose=verbose))
        
        # Print validation errors (compact)
        errors = result.get("errors", [])
        if errors:
            print(f"\n{colorize('Validation Errors:', Colors.RED + Colors.BOLD)} ({len(errors)} total)")
            error_types = defaultdict(int)
            for error in errors:
                error_msg = error.get("error", "Unknown")
                error_types[error_msg] += 1
            
            for err_msg, count in sorted(error_types.items(), key=lambda x: -x[1])[:5]:
                print(f"  - {err_msg} ({count}x)")
            if len(error_types) > 5:
                print(f"  {colorize(f'... and {len(error_types) - 5} more error types', Colors.DIM)}")
        else:
            print(f"\n  {colorize('No validation errors', Colors.GREEN)}")
        
        # Print schema
        if data:
            schema = get_schema_from_data(data)
            if schema:
                print(f"\n{colorize('Schema:', Colors.BOLD)} {len(schema)} fields")
                if verbose:
                    for field in sorted(schema):
                        print(f"  - {colorize(field, Colors.BLUE)}")
        
    except Exception as e:
        print(f"\n{colorize('❌ EXCEPTION:', Colors.RED + Colors.BOLD)} {colorize(str(e), Colors.RED)}")
        if verbose:
            import traceback
            traceback.print_exc()


def main():
    """Run visual output test for all datasets."""
    import argparse
    parser = argparse.ArgumentParser(description="Test normalization pipeline across all file formats")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    print(colorize("=" * 80, Colors.BOLD))
    print(colorize("NORMALIZATION VISUAL OUTPUT TEST", Colors.BOLD + Colors.CYAN))
    print(colorize("=" * 80, Colors.BOLD))
    print(f"Testing all file formats (CSV, Excel, Airtable)...\n")
    
    for config in TEST_CONFIGS:
        print_dataset_output(config, verbose=args.verbose)
    
    print(f"\n{colorize('=' * 80, Colors.BOLD)}")
    print(colorize("TEST COMPLETE", Colors.BOLD + Colors.GREEN))
    print(f"{colorize('=' * 80, Colors.BOLD)}\n")


if __name__ == "__main__":
    main()
