"""
Google Sheets Normalization — Test Runner
Compares actual normalized output vs expected truth JSON files.
"""

import sys
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from external_tables import fetch_from_google_sheets, list_sheets


# ANSI color codes for terminal output
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


# ------------------------------------------------------------
# Dataset definitions
# ------------------------------------------------------------
DATASETS = [
    {
        "name": "datasetA_sheet1",
        "mapping_name": "vehicles_basic",
        "sheet_id": "1rzGjivKD0-FoaXdDzQGeFtPrGRH3ALA7zECfYYgwlc8",
        "range_": "Sheet1!A:Z",
    },
    {
        "name": "datasetA_sheet2",
        "mapping_name": "vehicles_basic",
        "sheet_id": "1rzGjivKD0-FoaXdDzQGeFtPrGRH3ALA7zECfYYgwlc8",
        "range_": "Sheet2!A:Z",
    },
    {
        "name": "datasetA_sheet3",
        "mapping_name": "vehicles_basic",
        "sheet_id": "1rzGjivKD0-FoaXdDzQGeFtPrGRH3ALA7zECfYYgwlc8",
        "range_": "Sheet3!A:Z",
    },
    {
        "name": "datasetA_sheet4",
        "mapping_name": "vehicles_basic",
        "sheet_id": "1rzGjivKD0-FoaXdDzQGeFtPrGRH3ALA7zECfYYgwlc8",
        "range_": "Sheet4!A:Z",
    },
]


# ------------------------------------------------------------
# Helpers for expected-truth testing
# ------------------------------------------------------------
EXPECTED_DIR = Path(__file__).parent / "expected"


def load_expected(dataset_name: str):
    """
    Load expected truth JSON for the given dataset.
    """
    path = EXPECTED_DIR / f"{dataset_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Expected truth file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def compare(expected: List[Dict], actual: List[Dict]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Compare two lists of dicts and return list of differences plus statistics.
    
    Returns:
        Tuple of (differences list, statistics dict)
    """
    diffs = []
    stats = {
        "total_rows": len(expected),
        "matching_rows": 0,
        "mismatched_rows": 0,
        "missing_fields": 0,
        "extra_fields": 0,
        "value_mismatches": 0,
    }

    if len(expected) != len(actual):
        diffs.append(
            colorize(f"Row count mismatch: expected {len(expected)}, got {len(actual)}", Colors.RED)
        )
        stats["mismatched_rows"] = abs(len(expected) - len(actual))

    min_len = min(len(expected), len(actual))
    for i in range(min_len):
        exp_row = expected[i]
        act_row = actual[i]
        row_diffs = []
        
        all_keys = sorted(set(exp_row.keys()) | set(act_row.keys()))
        row_match = True
        
        for key in all_keys:
            exp_val = exp_row.get(key)
            act_val = act_row.get(key)
            
            if key not in exp_row:
                stats["extra_fields"] += 1
                row_diffs.append(f"  {colorize('+', Colors.YELLOW)} Extra field '{key}': {act_val!r}")
                row_match = False
            elif key not in act_row:
                stats["missing_fields"] += 1
                row_diffs.append(f"  {colorize('-', Colors.YELLOW)} Missing field '{key}': expected {exp_val!r}")
                row_match = False
            elif exp_val != act_val:
                stats["value_mismatches"] += 1
                row_diffs.append(
                    f"  {colorize('≠', Colors.RED)} Field '{key}':\n"
                    f"      Expected: {colorize(repr(exp_val), Colors.GREEN)}\n"
                    f"      Got:       {colorize(repr(act_val), Colors.RED)}"
                )
                row_match = False
        
        if row_diffs:
            diffs.append(f"\n{colorize(f'Row {i+1}:', Colors.BOLD)}")
            diffs.extend(row_diffs)
            stats["mismatched_rows"] += 1
        else:
            stats["matching_rows"] += 1
    
    return diffs, stats


# ------------------------------------------------------------
# Test logic
# ------------------------------------------------------------
def test_dataset(dataset: Dict) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Test a single dataset.
    
    Returns:
        Tuple of (passed: bool, duration: float, stats: dict)
    """
    test_name = dataset['name']
    start_time = time.time()
    
    print(f"\n{colorize('─' * 80, Colors.DIM)}")
    print(f"{colorize('TEST:', Colors.BOLD)} {colorize(test_name, Colors.CYAN)}")
    print(f"{colorize('─' * 80, Colors.DIM)}")
    
    # Load expected truth
    try:
        expected = load_expected(test_name)
        print(f"  {colorize('✓', Colors.GREEN)} Loaded expected data: {len(expected)} rows")
    except FileNotFoundError as e:
        print(f"  {colorize('✗', Colors.RED)} {e}")
        return False, 0, {}

    # Fetch actual data
    try:
        print(f"  {colorize('→', Colors.BLUE)} Fetching from Google Sheets...", end=" ", flush=True)
        actual = fetch_from_google_sheets(
            dataset["mapping_name"],
            dataset["sheet_id"],
            dataset["range_"],
        )
        print(f"{colorize('✓', Colors.GREEN)} Fetched {len(actual)} rows")
    except Exception as e:
        print(f"\n  {colorize('✗', Colors.RED)} ERROR fetching data: {e}")
        return False, time.time() - start_time, {}

    # Compare
    diffs, stats = compare(expected, actual)
    duration = time.time() - start_time

    # Print results
    if not diffs:
        print(f"  {colorize('✓ PASS', Colors.GREEN + Colors.BOLD)} — Output matches expected truth")
        print(f"  {colorize('  Stats:', Colors.DIM)} {stats['matching_rows']}/{stats['total_rows']} rows match")
    else:
        print(f"  {colorize('✗ FAIL', Colors.RED + Colors.BOLD)} — {len([d for d in diffs if d.startswith('\n')])} row(s) with differences")
        print(f"\n  {colorize('Differences:', Colors.BOLD)}")
        for d in diffs:
            print(d)
        
        print(f"\n  {colorize('Summary:', Colors.BOLD)}")
        print(f"    • Matching rows:     {colorize(str(stats['matching_rows']), Colors.GREEN)}/{stats['total_rows']}")
        print(f"    • Mismatched rows:   {colorize(str(stats['mismatched_rows']), Colors.RED)}")
        print(f"    • Value mismatches:  {colorize(str(stats['value_mismatches']), Colors.RED)}")
        if stats['missing_fields'] > 0:
            print(f"    • Missing fields:     {colorize(str(stats['missing_fields']), Colors.YELLOW)}")
        if stats['extra_fields'] > 0:
            print(f"    • Extra fields:        {colorize(str(stats['extra_fields']), Colors.YELLOW)}")
    
    print(f"  {colorize(f'Duration: {duration:.3f}s', Colors.DIM)}")
    
    return len(diffs) == 0, duration, stats


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    suite_start = time.time()
    
    print(f"\n{colorize('═' * 80, Colors.BOLD)}")
    print(f"{colorize('GOOGLE SHEETS NORMALIZATION TEST SUITE', Colors.BOLD + Colors.CYAN)}")
    print(f"{colorize('═' * 80, Colors.BOLD)}")
    print(f"Running {len(DATASETS)} test(s)...\n")
    
    results = []
    for dataset in DATASETS:
        passed, duration, stats = test_dataset(dataset)
        results.append({
            "name": dataset["name"],
            "passed": passed,
            "duration": duration,
            "stats": stats,
        })
    
    suite_duration = time.time() - suite_start
    
    # Print summary
    print(f"\n{colorize('═' * 80, Colors.BOLD)}")
    print(f"{colorize('TEST SUMMARY', Colors.BOLD + Colors.CYAN)}")
    print(f"{colorize('═' * 80, Colors.BOLD)}")
    
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = len(results) - passed_count
    total_duration = sum(r["duration"] for r in results)
    
    # Summary table
    print(f"\n{colorize('Results:', Colors.BOLD)}")
    for r in results:
        status = colorize("PASS", Colors.GREEN + Colors.BOLD) if r["passed"] else colorize("FAIL", Colors.RED + Colors.BOLD)
        duration_str = f"{r['duration']:.3f}s"
        print(f"  {status:6}  {r['name']:25}  {colorize(duration_str, Colors.DIM)}")
    
    # Overall summary
    print(f"\n{colorize('Overall:', Colors.BOLD)}")
    print(f"  Total tests:  {len(results)}")
    print(f"  {colorize('Passed:', Colors.GREEN)}      {colorize(str(passed_count), Colors.GREEN + Colors.BOLD)}")
    if failed_count > 0:
        print(f"  {colorize('Failed:', Colors.RED)}      {colorize(str(failed_count), Colors.RED + Colors.BOLD)}")
    print(f"  Duration:     {colorize(f'{total_duration:.3f}s', Colors.DIM)}")
    
    # Final status
    print(f"\n{colorize('─' * 80, Colors.DIM)}")
    if failed_count == 0:
        print(f"{colorize('✓ ALL TESTS PASSED', Colors.GREEN + Colors.BOLD)}")
    else:
        print(f"{colorize('✗ SOME TESTS FAILED', Colors.RED + Colors.BOLD)}")
    print(f"{colorize('─' * 80, Colors.DIM)}\n")
    
    sys.exit(0 if failed_count == 0 else 1)