# gsheets-table-normalizer: Technical Documentation

## 1. Overview

### High-Level Purpose

The `gsheets-table-normalizer` is a production-ready ETL (Extract, Transform, Load) system that normalizes structured and unstructured table data across multiple file formats into clean, validated, canonical schemas. It serves as a unified data ingestion pipeline for insurance-related datasets including policies, drivers, vehicles, claims, relationships, and locations.

### Problem It Solves

The system addresses several critical data integration challenges:

1. **Schema Inconsistency**: Different sources use different field names (e.g., "VIN" vs "Vehicle ID" vs "vin")
2. **Format Diversity**: Data arrives in multiple formats (CSV, Excel, Google Sheets, PDF, images, raw text)
3. **OCR Quality**: Unstructured sources (PDFs, handwritten documents) require OCR extraction with varying quality
4. **Data Quality**: Missing fields, invalid values, and OCR errors must be transparently reported
5. **Production Safety**: The system must never hallucinate data or hide extraction failures

### What "Production-Ready" Means

In this context, "production-ready" means:

- **Transparent Degradation**: The system clearly reports when extraction fails or produces low-confidence results
- **No Data Hallucination**: Missing fields are left as `None` rather than inferred or guessed
- **Confidence Scoring**: Every extracted field has a confidence score (0.0-1.0) indicating extraction reliability
- **Warning System**: Data quality issues are explicitly flagged in `_warnings` arrays
- **Validation Logic**: Tests distinguish between critical failures (blocking) and acceptable degradation (non-blocking)
- **Deterministic Output**: Same input always produces same output (no randomness in extraction)

The system prioritizes **accuracy over completeness**: it's better to have a missing field than an incorrect field.

---

## 2. Dataset Types

### Policies

**What it represents**: Insurance policy records with coverage details, dates, premiums, and vehicle associations.

**Supported sources**:

- Structured: CSV, Google Sheets, Excel, Airtable JSON
- Unstructured: PDF (typed), Raw Text

**Critical fields**: `policy_number` (primary identifier)

**Optional fields**: All other fields (`insured_name`, `effective_date`, `expiration_date`, `premium`, `coverage_type`, `vehicle_vin`, `notes`)

**Example normalized output**:

```json
{
  "policy_number": "P001",
  "insured_name": "John Doe",
  "effective_date": "2024-01-01",
  "expiration_date": "2025-01-01",
  "premium": "1200.00",
  "coverage_type": "Comprehensive",
  "vehicle_vin": "1HGBH41JXMN109186",
  "notes": null,
  "_confidence": {
    "policy_number": 1.0,
    "insured_name": 0.9,
    "effective_date": 1.0
  },
  "_warnings": []
}
```

**Schema order**: `POLICY_SCHEMA_ORDER` in `src/schema.py` (lines 115-124)

---

### Drivers

**What it represents**: Driver records with identification, license information, experience, and training status.

**Supported sources**:

- Structured: CSV, Google Sheets, Excel, Airtable JSON
- Unstructured: PDF (typed), PDF (handwritten), Raw Text, Images (handwritten)

**Critical fields**: `driver_id` (primary identifier)

**Optional fields**: All other fields (`first_name`, `last_name`, `date_of_birth`, `license_number`, `license_state`, `license_status`, `years_experience`, `violations_count`, `training_completed`, `notes`)

**Example normalized output**:

```json
{
  "driver_id": "D001",
  "first_name": "John",
  "last_name": "Doe",
  "date_of_birth": "1985-03-15",
  "license_number": "D1234567",
  "license_state": "CA",
  "license_status": "Valid",
  "years_experience": 18,
  "violations_count": 0,
  "training_completed": "Yes",
  "notes": "Example OCR test.",
  "_confidence": {
    "driver_id": 1.0,
    "first_name": 0.9,
    "date_of_birth": 0.8
  },
  "_warnings": ["extracted_from_ocr"]
}
```

**Schema order**: `DRIVER_SCHEMA_ORDER` in `src/schema.py` (lines 217-229)

---

### Vehicles

**What it represents**: Vehicle records with VIN, make, model, year, and physical characteristics.

**Supported sources**:

- Structured: CSV, Google Sheets, Excel, Airtable JSON
- Unstructured: PDF (typed), PDF (handwritten), Raw Text, Images (handwritten)

**Critical fields**: `vin` (primary identifier, validated for format)

**Optional fields**: All other fields (`vehicle_id`, `year`, `make`, `model`, `effective_date`, `notes`, `color`, `mileage`, `trim`, `body_style`, `fuel_type`, `transmission`, `owner_email`, `weight`, `image_url`, `description`)

**Example normalized output**:

```json
{
  "vin": "1HGBH41JXMN109186",
  "vehicle_id": null,
  "year": 2024,
  "make": "Toyota",
  "model": "Camry",
  "effective_date": null,
  "notes": "Blue sedan, low mileage",
  "color": "Blue",
  "mileage": 12345,
  "trim": "LE",
  "body_style": "sedan",
  "fuel_type": "gas",
  "transmission": "automatic",
  "owner_email": "owner@example.com",
  "weight": null,
  "image_url": null,
  "description": null,
  "_confidence": {
    "vin": 1.0,
    "make": 0.9,
    "year": 0.8
  },
  "_warnings": ["extracted_from_ocr"]
}
```

**Schema order**: `VEHICLE_SCHEMA_ORDER` in `src/schema.py` (lines 30-48)

---

### Claims

**What it represents**: Insurance claim records with claim numbers, dates, amounts, and associated entities.

**Supported sources**:

- Structured: CSV, Google Sheets, Excel
- Unstructured: PDF (typed), Raw Text

**Critical fields**: `claim_number` (primary identifier)

**Optional fields**: All other fields (`claim_date`, `claim_amount`, `claim_type`, `status`, `vehicle_vin`, `driver_id`, `policy_number`, `notes`)

**Example normalized output**:

```json
{
  "claim_number": "C001",
  "claim_date": "2024-01-15",
  "claim_amount": "5000.00",
  "claim_type": "Collision",
  "status": "Open",
  "vehicle_vin": "1HGBH41JXMN109186",
  "driver_id": "D001",
  "policy_number": "P001",
  "notes": null,
  "_confidence": {
    "claim_number": 1.0,
    "claim_date": 0.9
  },
  "_warnings": []
}
```

**Schema order**: `CLAIM_SCHEMA_ORDER` in `src/schema.py` (lines 371-381)

**Note**: Claims do NOT receive inferred warnings (`extracted_from_ocr`, `extracted_from_vision`). Only curated warnings from truth files are preserved.

---

### Relationships

**What it represents**: Link records connecting policies, vehicles, and drivers (many-to-many relationships).

**Supported sources**:

- Structured: CSV
- Unstructured: PDF (typed)

**Critical fields**: `relationship_id` (primary identifier)

**Optional fields**: `policy_number`, `vehicle_vin`, `driver_id`, `notes`

**Example normalized output**:

```json
{
  "relationship_id": "R001",
  "policy_number": "P001",
  "vehicle_vin": "1HGBH41JXMN109186",
  "driver_id": "D001",
  "notes": null,
  "_confidence": {
    "relationship_id": 1.0,
    "policy_number": 0.9,
    "vehicle_vin": 0.8
  },
  "_warnings": []
}
```

**Schema order**: `RELATIONSHIP_SCHEMA_ORDER` in `src/schema.py` (lines 383-390)

**Note**: Relationships do NOT receive inferred warnings. Only curated warnings from truth files are preserved.

---

### Locations

**What it represents**: Physical location records (garaging addresses) with geographic and administrative data.

**Supported sources**:

- Structured: CSV, Google Sheets, Excel
- Unstructured: PDF (typed), Raw Text

**Critical fields**: `location_id` (primary identifier)

**Optional fields**: All other fields (`insured_name`, `address_line_1`, `city`, `state`, `postal_code`, `county`, `territory_code`, `protection_class`, `latitude`, `longitude`, `notes`)

**Example normalized output**:

```json
{
  "location_id": "L001",
  "insured_name": "John Doe",
  "address_line_1": "123 Main St",
  "city": "San Francisco",
  "state": "CA",
  "postal_code": "94102",
  "county": "San Francisco",
  "territory_code": "CA-01",
  "protection_class": 3,
  "latitude": "37.7749",
  "longitude": "-122.4194",
  "notes": null,
  "_confidence": {
    "location_id": 1.0,
    "address_line_1": 0.9
  },
  "_warnings": []
}
```

**Schema order**: `LOCATION_SCHEMA_ORDER` in `src/schema.py` (lines 162-175)

**Note**: Locations do NOT receive inferred warnings. Only curated warnings from truth files are preserved.

---

## 3. Extraction Pipeline

### High-Level Flow

```
Raw Input → Source Detection → Extraction → Parsing → Normalization → Validation → Output
```

### Detailed Pipeline Stages

#### Stage 1: Source Detection (`src/sources.py`)

The system detects the source type and domain:

1. **File Extension Detection**: `.csv`, `.xlsx`, `.pdf`, `.png`, `.jpg`, `.txt`
2. **Domain Detection**: Based on mapping configuration (`domain` field in mapping metadata)
3. **Source Type Classification**: `structured` (CSV, Excel, Sheets, Airtable) vs `unstructured` (PDF, Image, Raw Text)

**Key functions**:

- `_extract_from_structured_source()`: Handles CSV, Excel, Sheets, Airtable
- `_extract_from_ocr_source()`: Handles PDF, Images, Raw Text

#### Stage 2: OCR Extraction (`src/ocr/reader.py`)

For unstructured sources, OCR engines extract text:

**OCR Engine Priority** (for handwritten sources):

1. **EasyOCR** (primary for handwritten): Better accuracy for handwritten text
2. **Tesseract** (fallback): Multiple PSM modes (6, 11, 12, 13) tried sequentially
3. **Vision API** (optional): Used for PDF table extraction if available

**For typed PDFs**:

1. **Vision API** (primary): Structured table extraction with high accuracy
2. **Tesseract OCR** (fallback): If Vision API unavailable or fails

**Image Preprocessing** (`src/ocr/reader.py`):

- Denoising (Gaussian blur)
- Thresholding (adaptive threshold)
- Morphological operations (dilation/erosion)
- DPI upscaling (300 DPI minimum)

**Key functions**:

- `extract_text_from_image()`: Handles PNG, JPG, JPEG
- `extract_text_from_pdf()`: Handles PDF (converts pages to images)
- `_extract_with_easyocr()`: EasyOCR extraction
- `_extract_with_tesseract()`: Tesseract extraction with PSM fallbacks

#### Stage 3: Domain-Specific Parsing (`src/sources.py`)

Each domain has a dedicated parser following a unified 4-step contract:

**4-Step Parsing Contract**:

1. **PASS 1: Normalize OCR Text**

   - Collapse multiple spaces to single space
   - Normalize label variants (`"Claim #"` → `"Claim Number:"`)
   - Normalize spacing around colons
   - Preserve original casing (do NOT lowercase)
2. **PASS 2: Split into Blocks**

   - Use strong delimiters (e.g., `"Driver ID:"`, `"Claim Number:"`, `"VIN:"`)
   - Each block represents one entity (driver, claim, vehicle, etc.)
   - For typed PDFs: Require hard anchors (e.g., `"Driver ID:"` or `"DRIVER DETAIL SHEET"`) to prevent over-extraction
3. **PASS 3: Extract Structured Fields**

   - Use OCR-friendly regex patterns (handle missing colons, extra spaces, line breaks)
   - Track extracted fields in `structured_fields` set (immutable once set)
   - Extract all canonical schema fields from block text
4. **PASS 4: Fallback Search**

   - Only search for fields that are missing and not in `structured_fields`
   - Search the original `full_text` (not just the block)
   - Never overwrite fields already in `structured_fields`

**Parser functions**:

- `parse_driver_raw_text()`: Lines 2765-3517 in `src/sources.py`
- `parse_vehicle_raw_text()`: Lines 2103-2200 in `src/sources.py`
- `parse_policy_raw_text()`: Lines 2202-2631 in `src/sources.py`
- `parse_claim_raw_text()`: Lines 3700-3950 in `src/sources.py`
- `parse_relationship_raw_text()`: Lines 3952-4305 in `src/sources.py`
- `parse_locations_raw_text()`: Lines 2633-2763 in `src/sources.py`

**Helper functions** (extracted for maintainability):

- `split_by_driver_id()`, `split_by_claim_number()`, `split_by_policy_number()`, etc.
- `extract_driver_fields_from_block()`, `extract_claim_fields_from_block()`, etc.

#### Stage 4: Schema Mapping (`src/mappings/*.py`)

Field mappings convert source field names to canonical schema names:

**Mapping structure**:

```python
{
    "id": "source_pdf_drivers",
    "metadata": {
        "source_name": "PDF Drivers",
        "source_type": "pdf",
        "domain": "driver"
    },
    "mappings": [
        {
            "source_field": "Driver ID",
            "target_field": "driver_id",
            "transform": null
        }
    ]
}
```

**Mapping files**:

- `src/mappings/drivers_mappings.py`
- `src/mappings/vehicles_mappings.py`
- `src/mappings/policies_mappings.py`
- `src/mappings/claims_mappings.py`
- `src/mappings/relationships_mappings.py`
- `src/mappings/locations_mappings.py`

#### Stage 5: Field Transforms (`src/transforms.py`)

Field transforms are applied after raw value extraction but before type conversion. Transforms allow complex field manipulations using a string-based syntax.

**Transform Functions** (29 available):

**String Operations**:
- `capitalize(field)`: Capitalize first letter
- `uppercase(field)`: Convert to uppercase
- `lowercase(field)`: Convert to lowercase
- `prepend(field, prefix)`: Add prefix to value
- `append(field, suffix)`: Add suffix to value
- `join([field1, field2], separator)`: Join multiple fields
- `slice(field, start, end)`: Extract substring
- `regex(field, pattern, group)`: Extract using regex

**Type Conversions**:
- `date(field, format)`: Parse date string
- `number(field)`: Convert to number
- `currency(field)`: Parse currency value
- `phone(field)`: Format phone number
- `boolean(field)`: Convert to boolean

**Array Operations**:
- `split(field, delimiter)`: Split string into array
- `index(array, index)`: Get array element
- `first(array)`: Get first element
- `last(array)`: Get last element
- `length(array)`: Get array length
- `sum(array)`: Sum numeric array
- `count(array)`: Count array elements
- `filter(array, condition)`: Filter array elements
- `flatten(array)`: Flatten nested arrays
- `arrayfrom(field1, field2, ...)`: Create array from fields

**Conditional Operations**:
- `if(condition, true_value, false_value)`: Conditional value
- `checkif(field, value)`: Check if field equals value

**Domain-Specific**:
- `standardize_fuel_type(field)`: Standardize fuel type values
- `combine_image_metadata_notes(...)`: Combine image metadata into notes

**Transform Application** (`src/normalizer.py`, lines 1351-1410):

1. Transforms are applied AFTER raw value extraction
2. Transforms run BEFORE type conversion
3. Row-based transforms (like `combine_image_metadata_notes`) can access entire row context
4. Transform errors are captured and reported in `row_errors`

**Example Transform Usage**:

```python
{
    "target_field": "full_name",
    "source_field": ["First Name", "Last Name"],
    "transform": "join([firstName, lastName], ' ')"
}
```

**Key functions**:

- `apply_transform()`: Main transform application (`src/transforms.py:15`)
- `parse_and_apply_transform()`: Transform parsing and execution (`src/transforms.py:37`)
- `apply_transforms()`: Batch transform application (`src/transforms.py:1097`)

#### Stage 6: Normalization (`src/normalizer.py`)

The `normalize_v2()` function performs:

1. **Field Mapping**: Apply mappings from source fields to canonical fields
2. **Field Transforms**: Apply transforms to mapped values (if specified)
3. **Type Conversion**: Convert strings to integers, dates, etc.
4. **Confidence Calculation**: Calculate confidence scores for each field
5. **Warning Generation**: Generate data quality warnings
6. **Schema Reordering**: Reorder fields to match canonical schema order

**Key functions**:

- `normalize_v2()`: Main normalization function (lines 1065-2598)
- `calculate_field_confidence()`: Confidence scoring (lines 839-903)
- `_generate_warnings()`: Warning generation (lines 965-1063)

#### Stage 7: Validation (`src/schema.py`)

Schema validation ensures:

- Field types match expected types (string, integer, date)
- Field ordering matches canonical schema
- Metadata fields (`_confidence`, `_warnings`, `_source`) are preserved

**Reordering functions**:

- `reorder_driver_fields()`: Lines 248-275 in `src/schema.py`
- `reorder_vehicle_fields()`: Lines 73-86 in `src/schema.py`
- `reorder_policy_fields()`: Lines 140-153 in `src/schema.py`
- Similar functions for other domains

### Differences: Structured vs Unstructured Sources

| Aspect                       | Structured Sources              | Unstructured Sources                              |
| ---------------------------- | ------------------------------- | ------------------------------------------------- |
| **Input**              | CSV, Excel, Sheets, Airtable    | PDF, Images, Raw Text                             |
| **Extraction**         | Direct field access             | OCR text extraction                               |
| **Confidence**         | 1.0 (perfect) or 0.33 (default) | 0.9 (Vision) or 0.3-0.8 (OCR)                     |
| **Warnings**           | None (unless validation fails)  | `extracted_from_ocr`, `extracted_from_vision` |
| **Field Completeness** | Usually complete                | Often incomplete (OCR limitations)                |
| **Error Handling**     | Type conversion errors          | OCR parsing errors, missing fields                |

### Where Extraction Can Intentionally Fail

Extraction can intentionally fail (without being an error) in these scenarios:

1. **Missing Fields in OCR Text**: If a field is not present in the OCR text, it's left as `None` with confidence `0.0`
2. **Low-Confidence OCR**: Fields extracted with low confidence (< 0.9) are marked but not rejected
3. **Handwritten Documents**: Handwritten text often produces incomplete extraction (expected degradation)
4. **Corrupted OCR**: Severely corrupted OCR text may produce 0 rows (acceptable if source is known to be problematic)

**These are NOT errors** because:

- The system transparently reports confidence scores
- Warnings indicate extraction source (OCR/Vision)
- Tests classify these as "acceptable degradation" not "critical failures"

---

## 4. Confidence System

### How Confidence is Initialized

Confidence scores are calculated in `calculate_field_confidence()` (`src/normalizer.py`, lines 839-903).

**Initial confidence**: `1.0` (perfect)

**Reduction factors**:

1. **OCR/Vision Extraction**: `-0.1` (confidence becomes `0.9`)

   - Applied when `is_ocr_extracted=True` or `is_vision_extracted=True`
   - Only adds `extracted_from_ocr`/`extracted_from_vision` warnings if `allow_inferred_warnings=True` (vehicles and drivers only)
2. **Field Repairs**:

   - Minor repair: `-0.2` (confidence becomes `0.8`)
   - Moderate repair: `-0.25` (confidence becomes `0.75`)
   - Severe repair: `-0.3` (confidence becomes `0.7`)
3. **Character Substitutions** (OCR ambiguity): `-0.2 * min(char_substitutions, 3)`

   - Example: 2 substitutions → `-0.4` (confidence becomes `0.6`)
4. **Missing Fields**: Confidence set to `0.0` (explicitly missing)

**Default confidence for structured sources**: `0.33` (if field exists but no explicit confidence calculated)

### How Confidence is Reduced

Confidence is reduced through:

1. **Extraction Source**: OCR/Vision extraction reduces confidence by `0.1`
2. **Repair Operations**: Field repairs reduce confidence based on severity
3. **OCR Errors**: Character substitutions reduce confidence
4. **Missing Fields**: Explicitly set to `0.0`

**Confidence ranges**:

- `1.0`: Perfect (structured source, no repairs)
- `0.9`: High (Vision API extraction)
- `0.7-0.8`: Medium (OCR with minor repairs)
- `0.3-0.6`: Low (OCR with multiple repairs/substitutions)
- `0.0`: Missing (field not found)

### Low Confidence vs Missing Confidence

**Low Confidence** (`0.1` - `0.8`):

- Field was extracted but with uncertainty
- OCR errors, repairs, or substitutions occurred
- Field value is present but may be inaccurate
- **Test classification**: ACCEPTABLE (if < 0.9)

**Missing Confidence** (`0.0` or `None`):

- `0.0`: Field explicitly marked as missing (expected)
- `None`: Confidence not calculated (should not happen in production)
- Field value is `None`
- **Test classification**:
  - `0.0`: ACCEPTABLE (explicitly missing)
  - `None`: CRITICAL (should not happen)

### How Confidence Affects Validation

Confidence affects validation in tests (`tests/test_drivers_all_sources.py`, `tests/test_vehicles_all_sources.py`):

**Rule**: Low-confidence mismatches (< 0.9) are classified as **ACCEPTABLE**, not CRITICAL.

**Example**:

- Expected: `"John Doe"`
- Actual: `"JOHN DOE"` (case difference)
- Confidence: `0.8` (OCR extraction)
- **Classification**: ACCEPTABLE (case-only difference + low confidence)

**Confidence does NOT affect extraction**: The system extracts fields regardless of confidence. Confidence only affects how mismatches are classified in tests.

---

## 5. Warnings System

### What Warnings Represent

Warnings are data quality indicators stored in the `_warnings` array of each row. They indicate:

1. **Extraction Source**: How the field was extracted (OCR, Vision API)
2. **Data Quality Issues**: Invalid values, missing required fields
3. **Repair Operations**: Fields that required repair
4. **Validation Failures**: Values that don't match expected patterns

**Warnings are NOT errors**: They are informational flags that help downstream systems make decisions about data quality.

### Common Warning Types

#### Extraction Warnings (Inferred - Vehicles/Drivers Only)

- `extracted_from_ocr`: Field extracted via OCR (Tesseract/EasyOCR)
- `extracted_from_vision`: Field extracted via Vision API

**Note**: These warnings are only added for vehicles and drivers. Policies, claims, relationships, and locations do NOT receive inferred warnings (they use curated warnings from truth files).

#### Inference Warnings (Vehicles/Drivers Only)

- `vin_missing`: VIN is missing (vehicles only)
- `vin_requires_human_review`: VIN format is suspicious (vehicles only)
- `driver_id_missing`: Driver ID is missing (drivers only)

#### Validation Warnings (All Domains)

- `invalid_year`: Year is outside reasonable range (1990-2035)
- `invalid_date_of_birth`: Date of birth year is outside reasonable range (1900-2010)
- `invalid_date_range`: Expiration date < effective date (policies)
- `negative_premium`: Premium is negative (policies)
- `invalid_email`: Email format is invalid
- `unknown_transmission`: Transmission value not in allowed list
- `unknown_fuel_type`: Fuel type value not in allowed list
- `unknown_body_style`: Body style value not in allowed list

#### Repair Warnings (All Domains)

- `field_repaired_minor`: Field required minor repair
- `field_repaired_moderate`: Field required moderate repair
- `field_repaired_severe`: Field required severe repair
- `ocr_char_substitutions_N`: N character substitutions made during OCR

### Why Warnings are Acceptable

Warnings are acceptable because:

1. **Transparency**: They explicitly report data quality issues
2. **Non-Blocking**: They don't prevent data from being processed
3. **Human Review**: They flag records that may need human review
4. **Downstream Decisions**: Downstream systems can use warnings to make decisions (e.g., "low confidence records need review")

**Example**: A driver record with `extracted_from_ocr` warning is still valid data; it just indicates the source was OCR and may have lower accuracy.

### Warning Generation Logic

Warnings are generated in two places:

1. **`calculate_field_confidence()`** (`src/normalizer.py`, lines 839-903):

   - Adds `extracted_from_ocr`/`extracted_from_vision` (if `allow_inferred_warnings=True`)
   - Adds repair warnings (`field_repaired_*`, `ocr_char_substitutions_*`)
2. **`_generate_warnings()`** (`src/normalizer.py`, lines 965-1063):

   - Adds validation warnings (`invalid_year`, `invalid_email`, etc.)
   - Adds domain-specific warnings (`vin_missing`, `driver_id_missing`, etc.)

**Warning merging**: Warnings from both sources are merged (not overwritten) in `normalize_v2()` (lines 2351-2356).

---

## 6. Validation Logic

### Difference Between Critical Issues and Acceptable Degradation

**Critical Issues** (blocking):

- Structural issues (missing/extra rows)
- High-confidence mismatches (>= 0.9) where expected ≠ actual
- Missing confidence scores (`None`) - should not happen in production

**Acceptable Degradation** (non-blocking):

- Case-only differences (e.g., `"John"` vs `"JOHN"`)
- Low-confidence mismatches (< 0.9) - OCR errors
- Missing fields with confidence `0.0` (explicitly missing)
- Notes populated for unstructured sources (when expected is `None`)

### How PASS / PASS WITH WARNINGS / FAIL is Determined

**PASS**:

- All rows match expected (perfect matches)
- No critical mismatches
- No acceptable mismatches (or acceptable mismatches are allowed)

**PASS WITH WARNINGS**:

- All rows match expected (perfect matches) OR
- Some rows have mismatches, but all are classified as ACCEPTABLE
- No critical mismatches
- Production-safe (data is usable with known limitations)

**FAIL**:

- Critical mismatches present
- Structural issues (missing/extra rows)
- High-confidence mismatches (>= 0.9)

**Test logic** (`tests/test_drivers_all_sources.py`, `filter_acceptable_mismatches()`, lines 642-760):

```python
def filter_acceptable_mismatches(comparison, actual_rows):
    real_mismatches = []  # Critical
    acceptable_mismatches = []  # Acceptable
  
    for mismatch in comparison["mismatches"]:
        # Rule 1: Structural issues → CRITICAL
        if mismatch["type"] in ["extra_row", "missing_row"]:
            real_mismatches.append(mismatch)
            continue
      
        # Rule 2: Case-only differences → ACCEPTABLE
        if expected.lower() == actual.lower():
            acceptable_mismatches.append(mismatch)
            continue
      
        # Rule 3: Low-confidence (< 0.9) → ACCEPTABLE
        if field_confidence < 0.9:
            acceptable_mismatches.append(mismatch)
            continue
      
        # Rule 4: High-confidence (>= 0.9) → CRITICAL
        real_mismatches.append(mismatch)
  
    return len(real_mismatches), len(acceptable_mismatches)
```

### Why Handwritten OCR Often Passes with Warnings

Handwritten OCR often passes with warnings because:

1. **Low Confidence**: Handwritten text produces low-confidence extractions (< 0.9)
2. **Missing Fields**: Many fields are missing (confidence `0.0`)
3. **OCR Errors**: Character substitutions are common
4. **Test Classification**: Low-confidence mismatches are classified as ACCEPTABLE

**Example**:

- Expected: `"John Doe"`, `"1985-03-15"`, `"D1234567"`
- Actual: `"John Doe"`, `None`, `"D123456"` (missing DOB, truncated license)
- Confidence: `0.3` (low)
- **Result**: PASS WITH WARNINGS (acceptable degradation)

---

## 7. Technologies Used

### OCR Libraries

#### Google Vision API (`openai` library)

**Location**: `src/sources.py`, `_extract_table_with_vision_api()` (lines 4307-4662)

**Usage**:

- Primary extraction for typed PDFs
- Structured table extraction with high accuracy
- Requires `OPENAI_API_KEY` environment variable

**When used**:

- PDF extraction (typed documents)
- Fallback for handwritten documents (if available)

**Dependencies**: `openai>=1.37.0` (in `requirements.txt`)

#### EasyOCR

**Location**: `src/ocr/reader.py`, `_extract_with_easyocr()` (lines 120-180)

**Usage**:

- Primary OCR engine for handwritten images
- Better accuracy for handwritten text than Tesseract
- Supports multiple languages

**When used**:

- Handwritten images (primary)
- Handwritten PDFs (primary)

**Dependencies**: `easyocr` (in `requirements.txt`)

#### Tesseract OCR (`pytesseract`)

**Location**: `src/ocr/reader.py`, `_extract_with_tesseract()` (lines 182-250)

**Usage**:

- Fallback OCR engine
- Multiple PSM (Page Segmentation Mode) fallbacks: 6, 11, 12, 13
- Tries original image if preprocessing fails

**When used**:

- Fallback for handwritten images (if EasyOCR fails)
- Fallback for typed PDFs (if Vision API unavailable)
- Raw text extraction (if needed)

**Dependencies**: `pytesseract` (in `requirements.txt`)

**PSM Modes**:

- PSM 6: Uniform block of text
- PSM 11: Sparse text
- PSM 12: Sparse text with OSD
- PSM 13: Raw line (no OSD)

### PDF Parsing

**Location**: `src/ocr/reader.py`, `extract_text_from_pdf()` (lines 210-350)

**Libraries**:

- `pdf2image`: Converts PDF pages to images
- `PIL` (Pillow): Image processing

**Process**:

1. Convert PDF pages to images (300 DPI minimum)
2. Extract text from each image using OCR engines
3. Combine text from all pages

**Dependencies**: `pdf2image` (in `requirements.txt`)

### Image Handling

**Location**: `src/ocr/reader.py`, `extract_text_from_image()` (lines 52-207)

**Libraries**:

- `opencv-python-headless`: Image preprocessing (denoising, thresholding, morphological operations)
- `numpy`: Array operations
- `PIL` (Pillow): Image loading/saving

**Preprocessing steps**:

1. **Denoising**: Gaussian blur to reduce noise
2. **Thresholding**: Adaptive threshold to convert to binary
3. **Morphological Operations**: Dilation/erosion to clean up text
4. **DPI Upscaling**: Ensure 300 DPI minimum for OCR accuracy

**Dependencies**:

- `opencv-python-headless>=4.8.0`
- `numpy>=1.26.0`

### File Paths

**OCR Module Structure**:

```
src/ocr/
├── __init__.py          # Module initialization
├── models.py            # Data models (TextBlock, OCRMetadata, etc.)
├── parser.py            # Text block parsing and filtering
├── reader.py            # OCR extraction (EasyOCR, Tesseract, Vision API)
└── table_extract.py     # Table extraction from text blocks
```

**Key Functions**:

- `src/ocr/reader.py`:

  - `extract_text_from_image()`: Image OCR extraction
  - `extract_text_from_pdf()`: PDF OCR extraction
  - `_extract_with_easyocr()`: EasyOCR wrapper
  - `_extract_with_tesseract()`: Tesseract wrapper
- `src/ocr/parser.py`:

  - `parse_text_blocks()`: Parse and filter text blocks
- `src/ocr/table_extract.py`:

  - `extract_tables_from_blocks()`: Extract structured tables from text blocks

---

## 8. Testing

### Quick Start

Run all tests for all domains:

```bash
python tests/test_all_sources.py
```

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

# Locations
python tests/test_locations_all_sources.py

# Relationships
python tests/test_policy_vehicle_driver_link_all_sources.py
```

### Test Output Modes

**Default Mode** (minimal output):
```bash
python tests/test_all_sources.py
```
- Shows domain status (PASS/FAIL)
- Row counts and mismatch counts
- Clean, production-friendly summary

**Verbose Mode** (detailed comparisons):
```bash
python tests/test_all_sources.py --verbose
```
- Shows row-by-row comparisons
- Displays all mismatches
- Includes detailed summary statistics

**Full-Diff Mode** (complete field-by-field diffs):
```bash
python tests/test_all_sources.py --full-diff
```
- Shows complete field-by-field comparisons for all rows
- Useful for debugging extraction issues
- Includes all mismatch details

**Debug Mode** (maximum verbosity):
```bash
python tests/test_all_sources.py --debug
```
- Enables debug logging
- Shows OCR extraction details
- Shows parsing details
- Most verbose output

### Understanding Test Results

#### Test Status: PASS vs FAIL

**PASS**: 
- All rows match expected (perfect matches)
- No critical mismatches
- Production-safe (data is usable)

**PASS WITH WARNINGS**:
- All rows match expected OR
- Some rows have mismatches, but all are classified as ACCEPTABLE
- No critical mismatches
- Production-safe (data is usable with known limitations)

**FAIL**:
- Critical mismatches present
- Structural issues (missing/extra rows)
- High-confidence mismatches (>= 0.9) where expected ≠ actual

#### Mismatch Classification

Tests use confidence-aware validation to distinguish between critical issues and acceptable degradation:

**CRITICAL Mismatches** (cause FAIL):
- Structural issues (missing/extra rows)
- High-confidence mismatches (>= 0.9) where expected ≠ actual
- Missing confidence scores (`None`) - should not happen in production

**ACCEPTABLE Mismatches** (do not cause FAIL):
- Case-only differences (e.g., `"John"` vs `"JOHN"`)
- Low-confidence mismatches (< 0.9) - OCR errors
- Missing fields with confidence `0.0` (explicitly missing)
- Notes populated for unstructured sources (when expected is `None`)

#### Example Test Output

```
================================================================================
UNIFIED POLICY TEST SUITE
================================================================================

=== Testing PDF Policy Documents ===

✓ Rows extracted: 2

────────────────────────────────────────────────────────────────────────────────
COMPARISON RESULTS
────────────────────────────────────────────────────────────────────────────────

Row 1: ✓ PERFECT MATCH
Row 2: ✓ PERFECT MATCH
✓ PASS

================================================================================
SUMMARY
================================================================================

Total tests:        5
Passed:             5
Failed:             0
Total mismatched fields: 0

Per-test results:

  Airtable             PASS  (3 rows, 0 mismatched fields)
  Google Sheet         PASS  (3 rows, 0 mismatched fields)
  Excel                PASS  (3 rows, 0 mismatched fields)
  Raw Text             PASS  (3 rows, 0 mismatched fields)
  PDF                  PASS  (2 rows, 0 mismatched fields)
```

### Test Architecture

#### Individual Domain Test Suites

Each domain has its own test suite:

- `tests/test_policies_all_sources.py`: Policy tests (confidence-aware)
- `tests/test_drivers_all_sources.py`: Driver tests (confidence-aware)
- `tests/test_vehicles_all_sources.py`: Vehicle tests (confidence-aware)
- `tests/test_claims_all_sources.py`: Claim tests
- `tests/test_locations_all_sources.py`: Location tests
- `tests/test_policy_vehicle_driver_link_all_sources.py`: Relationship tests

**Test Structure**:

1. Load expected truth file (`tests/truth/{domain}/{source}.expected.json`)
2. Run `normalize_v2()` on source file
3. Compare expected vs actual rows
4. Classify mismatches (CRITICAL vs ACCEPTABLE) using `filter_acceptable_mismatches()`
5. Report results

#### Unified Test Suite

**Location**: `tests/test_all_sources.py`

**Purpose**: Run all domain tests in a single execution with unified output.

**Features**:

- Logging suppression (clean output by default)
- Verbosity control (`--verbose`, `--full-diff`, `--debug`)
- Executive summary format (for drivers/vehicles)
- Simple summary format (for other domains)
- Confidence-aware mismatch classification

**Execution**:

```bash
python tests/test_all_sources.py [--verbose] [--full-diff] [--debug]
```

**Output**:

- Domain health table (status, rows, critical, acceptable)
- Issue breakdown (critical vs acceptable)
- Production acceptance assessment

### How Output Suppression Works

**Logging Suppression** (`tests/test_all_sources.py`, lines 28-38):

```python
# Suppress logging by default
logging.getLogger().setLevel(logging.CRITICAL)

# Suppress specific noisy loggers
for logger_name in [
    "src", "src.sources", "src.ocr", "src.normalizer", "src.transforms",
    "ocr", "ocr.parser", "ocr.table_extract", "ocr.reader", "ocr.models"
]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
```

**Output Suppression** (`tests/test_all_sources.py`, `run_domain_tests()`, lines 150-170):

```python
def run_domain_tests(domain_name, test_module, test_functions):
    # Suppress individual test output when not verbose
    if not VERBOSE:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            results = _run_tests_and_collect_results(test_module, test_functions)
    else:
        results = _run_tests_and_collect_results(test_module, test_functions)
```

**Debug Print Control** (individual test files):

```python
def debug_print(*args, **kwargs):
    """Print only if DEBUG flag is set."""
    if DEBUG:
        print(*args, **kwargs)
```

### How Full-Diff and Verbose Modes Work

**Verbose Mode** (`--verbose`):

- Shows detailed row-level comparisons
- Shows "COMPARISON RESULTS" section
- Shows individual row diffs
- Shows "SUMMARY" section
- Enables logging output

**Full-Diff Mode** (`--full-diff`):

- Same as verbose mode
- Shows all mismatches (including acceptable ones)
- Useful for debugging extraction issues

**Debug Mode** (`--debug`):

- Enables `debug_print()` statements
- Shows OCR extraction details
- Shows parsing details
- Most verbose output

**Default Mode** (no flags):

- Minimal output (status, row counts, critical/acceptable counts)
- No detailed comparisons
- No logging output
- Production-friendly summary

---

## 9. Design Decisions

### Why the System Avoids Hallucinating Data

**Principle**: Missing fields are preferable to incorrect fields.

**Rationale**:

1. **Downstream Safety**: Incorrect data can cause downstream systems to make wrong decisions
2. **Transparency**: Missing fields are explicitly reported (confidence `0.0`)
3. **Human Review**: Missing fields can be filled by humans; incorrect fields require correction
4. **Audit Trail**: Missing fields leave a clear audit trail; incorrect fields may go unnoticed

**Example**:

- **Bad**: System infers `"John Doe"` as driver name from context → incorrect if actual name is `"Jane Smith"`
- **Good**: System leaves `first_name=None` with confidence `0.0` → human can fill in correct value

### Why Missing Fields are Preferable to Incorrect Fields

**Missing Fields**:

- Explicitly reported (`None` with confidence `0.0`)
- Can be filled by humans or downstream systems
- Clear audit trail (field was not found)
- No false positives

**Incorrect Fields**:

- May go unnoticed (no explicit flag)
- Can cause downstream errors
- Require correction (more work than filling missing)
- False positives (system appears to work but produces wrong data)

**Example**:

- **Missing**: `date_of_birth=None` → Human fills in `"1985-03-15"` → Correct
- **Incorrect**: `date_of_birth="1985-03-15"` (inferred from context, but actual is `"1987-07-22"`) → System uses wrong date → Error

### How the System Supports Human Review

The system supports human review through:

1. **Confidence Scores**: Low-confidence fields (< 0.9) flag records for review
2. **Warnings**: `_warnings` array explicitly lists data quality issues
3. **Missing Fields**: Fields with confidence `0.0` are explicitly missing
4. **Transparent Reporting**: Tests classify issues as CRITICAL (needs review) vs ACCEPTABLE (known limitations)

**Human Review Workflow**:

1. System extracts data with confidence scores and warnings
2. Downstream system filters records with low confidence or warnings
3. Human reviewers check flagged records
4. Human reviewers fill missing fields or correct incorrect fields
5. Corrected records are re-processed

**Example Output for Human Review**:

```json
{
  "driver_id": "D001",
  "first_name": "John",
  "last_name": null,  // Missing - needs human review
  "date_of_birth": "1985-03-15",
  "_confidence": {
    "driver_id": 1.0,
    "first_name": 0.8,  // Low confidence - needs review
    "last_name": 0.0,   // Missing - needs human input
    "date_of_birth": 0.9
  },
  "_warnings": ["extracted_from_ocr"]  // Flagged for review
}
```

---

## 10. Future Improvements

### Where Extraction Can Be Improved Safely

**Safe Improvements** (do not change existing behavior):

1. **OCR Pattern Refinement**: Improve regex patterns for field extraction (e.g., handle more label variants)
2. **Image Preprocessing**: Enhance preprocessing for better OCR accuracy (e.g., better denoising, thresholding)
3. **Fallback Logic**: Add more fallback patterns for missing fields (without changing existing extraction)
4. **Confidence Calibration**: Fine-tune confidence reduction factors based on real-world accuracy data

**Location**: `src/sources.py`, parser functions (`parse_*_raw_text()`)

### Which OCR Fields Could Be Relaxed

**Current Strict Fields** (could be relaxed):

1. **Driver ID**: Currently requires exact format (e.g., `"D001"`). Could relax to handle variations (e.g., `"Driver 001"`, `"D-001"`).
2. **VIN**: Currently requires 17 characters. Could relax to handle OCR errors (16-18 characters).
3. **Date Formats**: Currently requires `YYYY-MM-DD`. Could relax to handle variations (e.g., `"03/15/1985"`, `"March 15, 1985"`).

**Location**: `src/sources.py`, `extract_*_fields_from_block()` functions

**Note**: Relaxing fields should be done carefully to avoid false positives. Always test with real-world data.

### How Confidence Thresholds Could Evolve

**Current Thresholds**:

- High confidence: `>= 0.9` (CRITICAL if mismatch)
- Low confidence: `< 0.9` (ACCEPTABLE if mismatch)

**Potential Evolution**:

1. **Domain-Specific Thresholds**: Different thresholds for different domains (e.g., policies: `>= 0.95`, drivers: `>= 0.9`)
2. **Field-Specific Thresholds**: Different thresholds for different fields (e.g., `vin`: `>= 0.95`, `notes`: `>= 0.7`)
3. **Source-Specific Thresholds**: Different thresholds for different sources (e.g., structured: `>= 0.95`, OCR: `>= 0.8`)

**Location**: `tests/test_*_all_sources.py`, `filter_acceptable_mismatches()` functions

**Calibration Process**:

1. Collect real-world accuracy data
2. Measure false positive rate (incorrect fields marked as high confidence)
3. Measure false negative rate (correct fields marked as low confidence)
4. Adjust thresholds to balance accuracy and completeness

---

## Appendix: File Structure and Code Locations

### Directory Structure

```
gsheets-table-normalizer/
├── src/                        # Source code
│   ├── normalizer.py          # Main normalization entry point
│   ├── schema.py              # Schema definitions and field reordering
│   ├── sources.py             # Source extraction and domain parsers
│   ├── transforms.py          # Field transformation functions
│   ├── external_tables.py     # Structured file loaders (CSV, Excel, etc.)
│   ├── mappings.py            # Mapping registry and lookup
│   ├── mappings/              # Field mapping configurations
│   │   ├── drivers_mappings.py
│   │   ├── vehicles_mappings.py
│   │   ├── policies_mappings.py
│   │   ├── claims_mappings.py
│   │   ├── relationships_mappings.py
│   │   └── locations_mappings.py
│   └── ocr/                   # OCR extraction module
│       ├── __init__.py
│       ├── reader.py          # OCR engines (EasyOCR, Tesseract, Vision API)
│       ├── parser.py          # Text block parsing and filtering
│       ├── table_extract.py   # Table extraction from text blocks
│       └── models.py          # Data models (TextBlock, OCRMetadata)
├── tests/                      # Test suite
│   ├── test_all_sources.py    # Unified test runner
│   ├── test_policies_all_sources.py
│   ├── test_drivers_all_sources.py
│   ├── test_vehicles_all_sources.py
│   ├── test_claims_all_sources.py
│   ├── test_locations_all_sources.py
│   ├── test_policy_vehicle_driver_link_all_sources.py
│   ├── policies/              # Policy test data
│   │   ├── structured/        # CSV, Excel, Airtable, Google Sheets
│   │   └── unstructured/     # PDF, Raw Text
│   ├── drivers/               # Driver test data
│   ├── vehicles/              # Vehicle test data
│   ├── claims/                # Claim test data
│   ├── relationships/         # Relationship test data
│   ├── locations/             # Location test data
│   └── truth/                 # Expected truth files
│       ├── policies/
│       ├── drivers/
│       ├── vehicles/
│       ├── claims/
│       ├── relationships/
│       └── locations/
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation
```

### Key File Locations

#### Core Normalization (`src/normalizer.py`)

- **`normalize_v2()`** (line 1065): Main entry point for normalization
- **`calculate_field_confidence()`** (line 839): Confidence score calculation
- **`_generate_warnings()`** (line 965): Warning generation logic
- **Transform application** (lines 1351-1410): Field transform execution

#### Source Extraction (`src/sources.py`)

- **`_extract_from_structured_source()`**: CSV, Excel, Sheets, Airtable extraction
- **`_extract_from_ocr_source()`**: PDF, Image, Raw Text extraction
- **`_extract_table_with_vision_api()`** (line 4307): Vision API table extraction
- **Domain parsers**:
  - `parse_driver_raw_text()` (line 2765)
  - `parse_vehicle_raw_text()` (line 2103)
  - `parse_policy_raw_text()` (line 2417)
  - `parse_claim_raw_text()` (line 3700)
  - `parse_relationship_raw_text()` (line 4033)
  - `parse_locations_raw_text()` (line 2633)

#### Field Transforms (`src/transforms.py`)

- **`apply_transform()`** (line 15): Main transform application
- **`parse_and_apply_transform()`** (line 37): Transform parsing and execution
- **Transform functions** (lines 236-869): All 29 transform implementations
- **`apply_transforms()`** (line 1097): Batch transform application

#### Schema Management (`src/schema.py`)

- **Schema definitions**: `POLICY_SCHEMA_ORDER`, `DRIVER_SCHEMA_ORDER`, etc.
- **Reordering functions**:
  - `reorder_vehicle_fields()` (line 73)
  - `reorder_policy_fields()` (line 140)
  - `reorder_driver_fields()` (line 248)
  - Similar functions for other domains

#### OCR Extraction (`src/ocr/reader.py`)

- **`extract_text_from_image()`** (line 52): Image OCR extraction
- **`extract_text_from_pdf()`** (line 210): PDF OCR extraction
- **`_extract_with_easyocr()`** (line 120): EasyOCR wrapper
- **`_extract_with_tesseract()`** (line 182): Tesseract wrapper

#### OCR Processing (`src/ocr/parser.py`)

- **`parse_text_blocks()`**: Text block parsing and filtering
- Block filtering and validation logic

#### Table Extraction (`src/ocr/table_extract.py`)

- **`extract_tables_from_blocks()`**: Table extraction from text blocks
- Table detection and row extraction

#### Mapping Configurations (`src/mappings/`)

- **`drivers_mappings.py`**: Driver field mappings
- **`vehicles_mappings.py`**: Vehicle field mappings
- **`policies_mappings.py`**: Policy field mappings
- **`claims_mappings.py`**: Claim field mappings
- **`relationships_mappings.py`**: Relationship field mappings
- **`locations_mappings.py`**: Location field mappings

#### Test Suites (`tests/`)

- **`test_all_sources.py`**: Unified test runner
- **`test_policies_all_sources.py`**: Policy tests (confidence-aware)
- **`test_drivers_all_sources.py`**: Driver tests (confidence-aware)
- **`test_vehicles_all_sources.py`**: Vehicle tests (confidence-aware)
- **`test_claims_all_sources.py`**: Claim tests
- **`test_locations_all_sources.py`**: Location tests
- **`test_policy_vehicle_driver_link_all_sources.py`**: Relationship tests

#### Test Data (`tests/{domain}/`)

- **`structured/`**: CSV, Excel, Airtable JSON, Google Sheets test files
- **`unstructured/`**: PDF, Raw Text, Image test files

#### Truth Files (`tests/truth/{domain}/`)

- **`{source}.expected.json`**: Expected normalized output for each source
- Manually verified ground truth for test comparisons

---

## Appendix: Key Functions Reference

### Normalization (`src/normalizer.py`)

- **`normalize_v2()`** (line 1065): Main normalization entry point
- **`calculate_field_confidence()`** (line 839): Confidence score calculation
- **`_generate_warnings()`** (line 965): Warning generation logic

### Source Extraction (`src/sources.py`)

- **`_extract_from_structured_source()`**: Structured file extraction (CSV, Excel, Sheets, Airtable)
- **`_extract_from_ocr_source()`**: Unstructured file extraction (PDF, Image, Raw Text)
- **`_extract_table_with_vision_api()`** (line 4307): Vision API table extraction
- **Domain parsers**:
  - **`parse_driver_raw_text()`** (line 2765): Driver parsing from unstructured text
  - **`parse_vehicle_raw_text()`** (line 2103): Vehicle parsing from unstructured text
  - **`parse_policy_raw_text()`** (line 2417): Policy parsing from unstructured text
  - **`parse_claim_raw_text()`** (line 3700): Claim parsing from unstructured text
  - **`parse_relationship_raw_text()`** (line 4033): Relationship parsing from unstructured text
  - **`parse_locations_raw_text()`** (line 2633): Location parsing from unstructured text

### Field Transforms (`src/transforms.py`)

- **`apply_transform()`** (line 15): Main transform application function
- **`parse_and_apply_transform()`** (line 37): Transform parsing and execution
- **`apply_transforms()`** (line 1097): Batch transform application
- **Transform implementations** (lines 236-869): All 29 transform functions

### OCR Extraction (`src/ocr/reader.py`)

- **`extract_text_from_image()`** (line 52): Image OCR extraction (PNG, JPG, JPEG)
- **`extract_text_from_pdf()`** (line 210): PDF OCR extraction
- **`_extract_with_easyocr()`** (line 120): EasyOCR wrapper
- **`_extract_with_tesseract()`** (line 182): Tesseract wrapper with PSM fallbacks

### OCR Processing (`src/ocr/parser.py`)

- **`parse_text_blocks()`**: Text block parsing and filtering

### Table Extraction (`src/ocr/table_extract.py`)

- **`extract_tables_from_blocks()`**: Table extraction from text blocks

### Schema Management (`src/schema.py`)

- **`reorder_vehicle_fields()`** (line 73): Vehicle field reordering
- **`reorder_policy_fields()`** (line 140): Policy field reordering
- **`reorder_driver_fields()`** (line 248): Driver field reordering
- Similar reordering functions for other domains

### Test Utilities (`tests/test_*_all_sources.py`)

- **`filter_acceptable_mismatches()`**: Confidence-aware mismatch classification
- **`compare_expected_vs_actual()`**: Row-by-row comparison logic
- **`determine_test_status()`**: PASS/FAIL determination (drivers/vehicles)
