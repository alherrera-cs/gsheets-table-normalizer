# Table Data Normalizer

A Python library for normalizing structured and unstructured table data across multiple file formats (CSV, Excel, Airtable JSON, Google Sheets, PDF, Raw Text, Images) into clean, validated, canonical schemas.

## What It Does

The normalizer:

- **Loads multiple file formats** with unified loaders (CSV, Excel, Airtable JSON, Google Sheets, PDF, Raw Text, Images)
- **Applies schema mappings** to convert inconsistent header names into canonical field names
- **Extracts unstructured data** from PDFs and raw text using OCR and pattern-based parsing
- **Validates data** using configurable validation rules (patterns, min/max, required fields, etc.)
- **Produces clean normalized rows** with consistent field names and data types
- **Includes comprehensive test suite** for verifying normalization across all domains and major source types

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT SOURCES                                │
├─────────────────────────────────────────────────────────────────┤
│  Structured:                    Unstructured:                   │
│  • CSV                         • PDF (OCR)                     │
│  • Excel (XLSX)                • Raw Text                      │
│  • Airtable JSON               • Images (OCR)                  │
│  • Google Sheets               • Image Metadata (JSON)         │
└────────────────────┬──────────────────────────┬─────────────────┘
                     │                          │
                     ▼                          ▼
         ┌──────────────────────┐   ┌──────────────────────┐
         │  Structured Loader   │   │   OCR Extraction      │
         │  (external_tables)   │   │   (ocr/reader.py)    │
         └──────────┬───────────┘   └──────────┬───────────┘
                    │                          │
                    │                          ▼
                    │              ┌──────────────────────┐
                    │              │  Raw Text Output     │
                    │              └──────────┬───────────┘
                    │                          │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Domain Detection     │
                    │  (sources.py)        │
                    │  • Policies          │
                    │  • Drivers           │
                    │  • Vehicles          │
                    │  • Claims            │
                    │  • Relationships     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  4-Step Parser      │
                    │  (sources.py)       │
                    │  1. Normalize       │
                    │  2. Split Blocks    │
                    │  3. Extract Fields  │
                    │  4. Fallback Search│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Schema Mapping      │
                    │  (mappings/*.py)     │
                    │  • Header variants   │
                    │  • Field mapping     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Validation         │
                    │  (schema.py)        │
                    │  • Type conversion  │
                    │  • Pattern checks   │
                    │  • Required fields  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Normalized Rows     │
                    │  (canonical schema)  │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Test Comparison     │
                    │  (tests/*.py)        │
                    │  vs. Truth Files     │
                    └──────────────────────┘
```

## Supported Sources

### Structured Sources
- **CSV**: Standard comma-separated values files
- **Excel (XLSX)**: Requires `openpyxl` package
- **Airtable JSON**: Exported Airtable records format
- **Google Sheets**: Requires Google API credentials (service account JSON)

### Unstructured Sources
- **PDF**: OCR text extraction using Google Vision API or fallback OCR
- **Raw Text**: Direct text parsing with pattern-based extraction
- **Images**: OCR text extraction (PNG, JPG, JPEG) using Tesseract → EasyOCR → Vision API fallback chain
- **Image Metadata**: JSON metadata files from image processing

## The 4-Step Parsing Contract

All unstructured data parsers (`parse_*_raw_text`) follow a consistent 4-step contract to ensure robust, maintainable extraction:

### Step 1: Normalize OCR Text (Lightly)
- Collapse multiple spaces to single space
- Normalize label punctuation variants:
  - `"Claim #"` → `"Claim Number:"`
  - `"Claim ID"` → `"Claim Number:"`
  - `"Notes -"` → `"Notes:"`
- Normalize spacing around colons: `"Notes : value"` → `"Notes: value"`
- **Do NOT** lowercase the entire text (preserve original casing)

### Step 2: Split into Blocks Using Strong Delimiter
- **Claims**: Split on `"Claim ID:"`, `"Claim Number:"`, `"Claim #:"`, or narrative patterns
- **Relationships**: Split on `"Relationship ID:"`, `"RELATIONSHIP RECORD"`, or structured patterns
- **Drivers**: Split on driver name patterns or structured field headers
- Each block represents one entity (claim, relationship, driver, etc.)

### Step 3: Extract Structured Fields from Each Block (Immutable Once Set)
- Use OCR-friendly regex patterns (handle missing colons, extra spaces, line breaks)
- Track extracted fields in a `structured_fields` set
- Once a field is set via structured extraction, it cannot be overwritten
- Extract all canonical schema fields from the block text

### Step 4: Fallback Search in Full Text
- Only search for fields that are:
  - Missing (not found in block text)
  - Not already in `structured_fields` set
- Search the original `full_text` (not just the block)
- **Never overwrite** fields that were already set in Step 3

This contract ensures:
- **Robustness**: Handles OCR errors and format variations
- **Immutability**: Prevents accidental overwrites of correctly extracted fields
- **Maintainability**: Clear separation of concerns, easy to debug
- **Future-proof**: New patterns can be added without breaking existing logic

## Installation

### Requirements

- **Python 3.8+**
- Required packages:
  ```bash
  pip install openpyxl  # Required for Excel file support
  pip install -r requirements.txt
  ```

### Optional Dependencies

For Google Sheets support:
```bash
pip install google-auth google-api-python-client
```

For PDF and Image OCR support:
```bash
# Google Vision API (recommended for best accuracy)
# Requires GOOGLE_APPLICATION_CREDENTIALS environment variable

# Or fallback OCR libraries
pip install pytesseract pdf2image  # For PDFs
pip install easyocr  # Optional, recommended for image OCR (better for handwritten text)
```

## Quick Start

### Basic Usage

```python
from pathlib import Path
from src.normalizer import normalize_v2
from src.mappings import get_mapping_by_id

# Load a mapping configuration
mapping_config = get_mapping_by_id("source_google_sheet_vehicles")

# Normalize a CSV file
source = {"file_path": "tests/vehicles/structured/google_sheet_vehicle_inventory.csv"}
result = normalize_v2(
    source=source,
    mapping_config=mapping_config,
    header_row_index=0,
    validate=True,
)

# Access normalized data
for row in result["data"]:
    print(row)  # Clean, normalized row with canonical field names
```

### Example Output

**Input (CSV with inconsistent headers):**
```csv
VIN Number,Make,Model,Year,Exterior Color
1HGBH41JXMN109186,Honda,Civic,2020,Blue
```

**Output (normalized):**
```python
{
    "vin": "1HGBH41JXMN109186",
    "make": "Honda",
    "model": "Civic",
    "year": 2020,
    "color": "Blue",
    "_source_id": "source_google_sheet_vehicles",
    "_source_row_number": 1
}
```

## Running Tests

### Unified Test Suite (Recommended)

Run all domain tests in a single command:

```bash
python tests/test_all_sources.py
```

This runs tests for all domains in order:
1. Policies
2. Drivers
3. Vehicles
4. Claims
5. Relationships

The unified test suite provides:
- Per-domain summaries
- Global summary across all domains
- Detailed pass/fail reporting
- Mismatched field counts

### Individual Domain Tests

Run tests for a specific domain:

```bash
# Policies
python tests/test_policies_all_sources.py

# Drivers
python tests/test_drivers_all_sources.py

# Vehicles
python tests/test_vehicles_all_sources.py

# Claims
python tests/test_claims_all_sources.py

# Relationships
python tests/test_policy_vehicle_driver_link_all_sources.py
```

### Test Structure

Tests are organized by domain type:

- `tests/policies/` - Insurance policies
- `tests/drivers/` - Driver information
- `tests/vehicles/` - Vehicle inventory data
- `tests/claims/` - Insurance claims
- `tests/relationships/` - Policy-vehicle-driver links
- `tests/locations/` - Garaging locations

Each domain includes:
- `structured/` - CSV, Excel, Airtable JSON files
- `unstructured/` - Raw text, PDF, image files

### Expected Truth Files

Expected truth files are stored in `tests/truth/[domain]/` and define the expected output for each source:

```json
{
  "__schema__": [
    "field1",
    "field2",
    ...
  ],
  "rows": [
    {
      "field1": "value1",
      "field2": "value2",
      "_warnings": []
    },
    ...
  ]
}
```

**Important**: Truth files are manually verified and should NOT be auto-generated. They serve as the ground truth for test comparisons.

## Project Structure

```
gsheets-table-normalizer/
├── src/
│   ├── normalizer.py          # Main normalization entry point
│   ├── sources.py             # Source loaders + domain parsers
│   ├── schema.py              # Schema definitions + validation
│   ├── transforms.py          # Field transformation logic
│   ├── external_tables.py     # Structured file loaders
│   ├── ocr/
│   │   ├── reader.py          # OCR text extraction
│   │   ├── parser.py          # OCR text parsing utilities
│   │   └── table_extract.py   # Table extraction from images
│   └── mappings/
│       ├── vehicles_mappings.py
│       ├── drivers_mappings.py
│       ├── policies_mappings.py
│       ├── locations_mappings.py
│       ├── relationships_mappings.py
│       └── claims_mappings.py
│
├── tests/
│   ├── test_all_sources.py    # Unified test runner (use this!)
│   ├── test_policies_all_sources.py
│   ├── test_drivers_all_sources.py
│   ├── test_vehicles_all_sources.py
│   ├── test_claims_all_sources.py
│   ├── test_policy_vehicle_driver_link_all_sources.py
│   ├── test_all_domains.py    # DEPRECATED - use test_all_sources.py
│   ├── policies/
│   ├── drivers/
│   ├── vehicles/
│   ├── claims/
│   ├── relationships/
│   ├── locations/
│   └── truth/                 # Expected truth files
│
└── README.md
```

## How It Works

### Normalization Pipeline

```
                 ┌────────────────────────────┐
                 │     Source File (any)      │
                 │                            │
                 │  CSV / Excel / Airtable    │
                 │  Google Sheets / PDF /    │
                 │  Raw Text / Image         │
                 └─────────────┬──────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │   1. Detect Format     │
                  │   sources.py           │
                  └─────────┬──────────────┘
                            │
                            ▼
           ┌──────────────────────────────────────┐
           │ 2. Load Data                        │
           │    Structured: external_tables.py   │
           │    Unstructured: ocr/reader.py      │
           └─────────────────┬────────────────────┘
                             │
                             ▼
         ┌──────────────────────────────────────────┐
         │ 3. Domain Detection                      │
         │    sources.py                            │
         │  - Policies, Drivers, Vehicles, etc.    │
         └──────────────────┬───────────────────────┘
                            │
                            ▼
       ┌──────────────────────────────────────────────┐
       │ 4. Parse (if unstructured)                   │
       │    sources.py - parse_*_raw_text()          │
       │  - Follow 4-step contract                    │
       └───────────────────┬──────────────────────────┘
                           │
                           ▼
       ┌──────────────────────────────────────────────┐
       │ 5. Apply Mappings                           │
       │    mappings/*.py                            │
       │  - Try header variants                      │
       │  - Fill canonical schema fields            │
       │  - "First variant wins" logic                │
       └───────────────────┬──────────────────────────┘
                           │
                           ▼
       ┌──────────────────────────────────────────────┐
       │ 6. Normalize + Validate                      │
       │    schema.py + external_tables.py           │
       │  - Type conversion                           │
       │  - Regex pattern checks                      │
       │  - Required field validation                 │
       │  - Min/max length checks                     │
       └───────────────────┬──────────────────────────┘
                           │
                           ▼
           ┌────────────────────────────────────────┐
           │ 7. Output                               │
           │   - result["data"]  (clean rows)        │
           │   - result["errors"] (validation errs)  │
           │   - result["total_success"]              │
           │   - result["total_errors"]               │
           └────────────────────────────────────────┘
```

### Detailed Steps

#### 1. Format Detection

The normalizer automatically detects file format:
- `.csv` → CSV loader
- `.xlsx` → Excel loader (requires `openpyxl`)
- `.json` → Airtable JSON loader (checks for "records" structure)
- `.pdf` → PDF OCR extraction
- `.txt` → Raw text parsing
- `.png`, `.jpg`, `.jpeg` → Image OCR extraction
- Google Sheets → Google Sheets API (optional, requires credentials)

#### 2. Domain Detection

For unstructured sources, the normalizer detects the domain based on:
- Mapping ID patterns (e.g., `source_pdf_claims` → Claims domain)
- Content analysis (keywords, field patterns)
- Routing to appropriate parser: `parse_claim_raw_text()`, `parse_driver_raw_text()`, etc.

#### 3. Field Mapping

Mappings support multiple source field variants for each canonical field:

```python
{
    "target_field": "vin",
    "source_field": ["VIN Number", "VIN", "vin_number"],  # Multiple variants
    "type": "string",
    "required": True,
    "validation": {
        "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
        "min_length": 17,
        "max_length": 17
    }
}
```

The normalizer uses "first variant wins" logic - it tries each variant until it finds a non-empty value.

#### 4. Validation

Each field can have validation rules:
- **pattern**: Regex pattern matching
- **min/max**: Numeric range validation
- **min_length/max_length**: String length validation
- **enum**: Allowed value list
- **required**: Field must be present and non-empty

#### 5. Type Conversion

Fields are automatically converted to their specified types:
- `"string"` → String (with whitespace trimming)
- `"integer"` → Integer
- `"float"` → Float
- `"date"` → Date object
- `"boolean"` → Boolean

## Mapping Configuration

Mappings are defined in `src/mappings/*.py` files. Each mapping includes:

```python
{
    "id": "source_google_sheet_vehicles",
    "metadata": {
        "source_name": "Vehicle Inventory",
        "source_type": "csv",  # or "xlsx_file", "airtable", "google_sheet", "pdf", "raw_text"
        "connection_config": {
            "file_path": "path/to/file.csv"
        }
    },
    "mappings": [
        {
            "target_field": "vin",
            "source_field": ["VIN Number", "VIN"],
            "type": "string",
            "required": True,
            "validation": {...}
        },
        ...
    ]
}
```

## Known Limitations

### OCR / PDF Extraction

- **OCR Accuracy**: PDF text extraction may have character recognition errors (e.g., "UAL" → "JLA" in VINs)
- **Layout Sensitivity**: Complex PDF layouts may not split correctly into blocks
- **Missing Rows**: Some PDFs may not extract all expected rows if:
  - Block delimiters are not detected
  - Fields are on separate lines and patterns don't match
  - OCR quality is poor

### Current PDF Issues

Based on test results:
- **Relationships PDF**: Missing 1 of 3 expected rows (P002 relationship not extracted)
- **Claims PDF**: Missing 1 of 3 expected rows (C003 claim not extracted)
- **VIN OCR Errors**: Some VINs misread by OCR (e.g., `3R5UAL4YUKPYGF1GZ` → `3R5JL4AYUKPYGF1GZ`)

### Recommended Improvements

1. **Flexible Block Splitting**: Make relationship/claim block splitting more flexible to handle multi-line formats
2. **OCR Post-Processing**: Add fuzzy matching or validation for VINs and other critical fields
3. **Better Delimiter Detection**: Improve detection of block boundaries in PDFs with varied layouts
4. **Fallback Strategies**: Add more fallback patterns for missing fields

## Error Handling

The normalizer returns detailed error information:

```python
result = normalize_v2(...)

# Check overall success
if result["success"]:
    print(f"✓ Processed {result['total_success']} rows successfully")
else:
    print(f"✗ {result['total_errors']} rows had errors")

# Access successful rows
for row in result["data"]:
    # Process normalized row
    pass

# Access errors
for error in result["errors"]:
    print(f"Row {error['_source_row_number']}: {error['error']}")
```

## Contributing

When adding new mappings:

1. Create or update mapping file in `src/mappings/`
2. Add test data files to `tests/[domain]/structured/` or `tests/[domain]/unstructured/`
3. Create expected truth file in `tests/truth/[domain]/`
4. Add test functions to `tests/test_[domain]_all_sources.py`
5. Run `python tests/test_all_sources.py` to verify

When adding new domain parsers:

1. Follow the 4-step parsing contract
2. Implement `parse_[domain]_raw_text()` in `src/sources.py`
3. Add domain detection logic in `_extract_from_ocr_source()`
4. Create test functions and truth files
5. Update this README
