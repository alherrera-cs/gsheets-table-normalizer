# Table Data Normalizer

A Python library for normalizing structured table data across multiple file formats (CSV, Excel, Airtable JSON, and optionally Google Sheets) into a clean, validated, canonical schema.

## What It Does

The normalizer:

- **Loads multiple file formats** with unified loaders (CSV, Excel XLSX, Airtable JSON, Google Sheets)
- **Applies schema mappings** to convert inconsistent header names into canonical field names
- **Validates data** using configurable validation rules (patterns, min/max, required fields, etc.)
- **Produces clean normalized rows** with consistent field names and data types
- **Includes a visualization test suite** for verifying normalization across all formats

## Installation

### Requirements

- **Python 3.8+**
- Required packages:
  ```bash
  pip install openpyxl  # Required for Excel file support
  ```

### Optional Dependencies

For Google Sheets support:
```bash
pip install google-auth google-api-python-client
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

### Visual Output Test

Run the comprehensive test suite that tests all file formats:

```bash
# Basic output (compact)
python tests/test_visual_output.py

# Verbose output (shows all fields)
python tests/test_visual_output.py --verbose
```

The test suite:
- Tests CSV, Excel, and Airtable formats for each dataset
- Shows success/error counts per format
- Displays fields found and missing required fields
- Identifies unmapped header variants
- Shows validation errors

### Test Structure

Tests are organized by dataset type:
- `tests/vehicles/` - Vehicle inventory data
- `tests/drivers/` - Driver information
- `tests/policies/` - Insurance policies
- `tests/locations/` - Garaging locations
- `tests/relationships/` - Policy-vehicle-driver links
- `tests/claims/` - Insurance claims

Each dataset includes:
- `structured/` - CSV, Excel, Airtable JSON files
- `unstructured/` - Raw text, PDF, image files (for future OCR support)

## Project Structure

```
gsheets-table-normalizer/
├── src/
│   ├── normalizer.py
│   ├── sources.py
│   ├── schema.py
│   ├── transforms.py
│   ├── external_tables.py
│   └── mappings/
│       ├── vehicles_mappings.py
│       ├── drivers_mappings.py
│       ├── policies_mappings.py
│       ├── locations_mappings.py
│       ├── relationships_mappings.py
│       └── claims_mappings.py
│
├── tests/
│   ├── test_visual_output.py
│   ├── test_mappings.py
│   ├── vehicles/
│   │   ├── structured/
│   │   └── unstructured/
│   ├── drivers/
│   │   ├── structured/
│   │   └── unstructured/
│   ├── policies/
│   │   ├── structured/
│   │   └── unstructured/
│   ├── locations/
│   │   ├── structured/
│   │   └── unstructured/
│   ├── relationships/
│   │   ├── structured/
│   │   └── unstructured/
│   └── claims/
│       ├── structured/
│       └── unstructured/
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
                 │  Google Sheets (optional)  │
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
           │ 2. Clean Headers                     │
           │    external_tables.clean_header()    │
           │  - lowercase                          │
           │  - spaces → underscores              │
           └─────────────────┬────────────────────┘
                             │
                             ▼
         ┌──────────────────────────────────────────┐
         │ 3. Apply Mappings                        │
         │    mappings/*.py                         │
         │  - Try header variants                   │
         │  - Fill canonical schema fields          │
         │  - "First variant wins" logic             │
         └──────────────────┬───────────────────────┘
                            │
                            ▼
       ┌──────────────────────────────────────────────┐
       │ 4. Normalize + Validate                      │
       │    schema.py + external_tables.py            │
       │  - Type conversion                           │
       │  - Regex pattern checks                      │
       │  - Required field validation                 │
       │  - Min/max length checks                     │
       └───────────────────┬──────────────────────────┘
                           │
                           ▼
           ┌────────────────────────────────────────┐
           │ 5. Output                               │
           │   - result["data"]  (clean rows)        │
           │   - result["errors"] (validation errs)   │
           │   - result["total_success"]              │
           │   - result["total_errors"]                │
           └────────────────────────────────────────┘
```

### Detailed Steps

The normalizer automatically detects file format:
- `.csv` → CSV loader
- `.xlsx` → Excel loader (requires `openpyxl`)
- `.json` → Airtable JSON loader (checks for "records" structure)
- Google Sheets → Google Sheets API (optional, requires credentials)

### 2. Header Cleaning

All headers are normalized:
- Lowercase
- Spaces → underscores
- Special characters removed

Example: `"VIN Number"` → `"vin_number"`

### 3. Field Mapping

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

### 4. Validation

Each field can have validation rules:
- **pattern**: Regex pattern matching
- **min/max**: Numeric range validation
- **min_length/max_length**: String length validation
- **enum**: Allowed value list
- **required**: Field must be present and non-empty

### 5. Type Conversion

Fields are automatically converted to their specified types:
- `"string"` → String (with whitespace trimming)
- `"integer"` → Integer
- `"float"` → Float
- `"date"` → Date object
- `"boolean"` → Boolean

## Supported File Formats

### CSV
- Standard comma-separated values
- Headers in first row
- UTF-8 encoding

### Excel (XLSX)
- Requires `openpyxl` package
- Reads first sheet by default
- Headers in first row

### Airtable JSON
- Exported Airtable records format
- Structure: `{"records": [{"fields": {...}}, ...]}`
- All fields from "fields" dict are extracted

### Google Sheets (Optional)
- Requires Google API credentials
- Service account JSON file
- Read-only access to specified ranges

## Mapping Configuration

Mappings are defined in `src/mappings/*.py` files. Each mapping includes:

```python
{
    "id": "source_google_sheet_vehicles",
    "metadata": {
        "source_name": "Vehicle Inventory",
        "source_type": "csv",  # or "xlsx_file", "airtable", "google_sheet"
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

## What's Not Implemented

### OCR / Unstructured Data
- PDF text extraction
- Image OCR (text extraction from images)
- Raw text parsing

These features are planned for future implementation. The current system focuses on structured data formats (CSV, Excel, Airtable).

### Transform Logic
- Field value transformations (uppercase, lowercase, capitalize, etc.)
- Transform logic will be implemented in a future update

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
2. Add test data files to `tests/[dataset]/structured/`
3. Update `tests/test_mappings.py` with new `TEST_CONFIGS` entry
4. Run `python tests/test_visual_output.py` to verify

## License

[Add your license here]
