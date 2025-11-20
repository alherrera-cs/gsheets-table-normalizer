# Google Sheets Table Normalizer

Small Python module for turning **Google Sheets tables** into a **normalized list of dictionaries** that can be used by other systems (e.g. a proposal builder).

## How the Normalizer Works

The program takes a messy Google Sheet and transforms it into clean, structured, validated JSON. The pipeline has 8 stages:

1. **Fetch Raw Sheet Values**  
   Pulls the entire range using the Google Sheets API as a 2D array of raw strings.

2. **Clean Headers (`clean_header`)**  
   - lowercase  
   - strip whitespace  
   - replace spaces with underscores  
   
   Example: `"Vehicle ID #"` → `"vehicle_id_#"`

3. **Convert 2D Rows → Objects (`rows2d_to_objects`)**  
   Turns each row into a `dict` where the cleaned headers become keys.

4. **Apply Mapping (`apply_mapping`)**  
   This is where the magic happens.  
   All header variants get mapped into a **canonical schema**:
   
   ```
   vin, vehicle_id, year, make, model, effective_date, notes, trim, weight
   ```
   
   Examples:  
   - `"unit_number"` → `"vehicle_id"`  
   - `"start_dt"` → `"effective_date"`  
   - `"vin_#"` → `"vin"`  
   - `"g_v_w"` → `"weight"`

5. **Normalize Values (`normalize_values`)**  
   - Convert numeric fields like year/weight into ints  
   - Convert empty/None notes into `""`  
   - Strip whitespace from all strings  
   - Ensure consistency across all sheets

6. **Drop Empty Rows (`drop_empty_rows`)**  
   Removes accidental blank rows or Google Sheets auto-expanded space.

7. **Return Clean Structured Output**  
   Output is stable, predictable, and matches the `expected/` truth directory.

8. **Testing**  
   `tests/run_gsheets_tests.py` ensures:  
   - Every sheet matches canonical schema  
   - Normalization rules applied consistently  
   - Mapping works across sheets with inconsistent headers

Example output:

```python
[
  {"vin": "12345", "year": 2020, "make": "Ford", "model": "F-150", "notes": ""},
  {"vin": "67890", "year": 2021, "make": "Honda", "model": "Civic", "notes": "Lease"},
]


           ┌──────────────────────────────┐
           │  Google Sheet (raw data)     │
           │  e.g.                        │
           │  VIN | Model Year | Make ... │
           └───────────────┬──────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  1. Fetch via Sheets API                            │
│     • Uses service account JSON                     │
│     • Read-only scope                               │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  2. rows2d_to_objects()                             │
│     Converts 2D list → list of dicts:               │
│     [                                              │
│       {"vin": "12345", "model_year": "2020", ...}, │
│       {"vin": "67890", "model_year": "2021", ...}, │
│     ]                                              │
│     • Normalizes headers (lowercase, underscores)   │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  3. apply_mapping()                                 │
│     Mapping profile applied:                        │
│       "model_year" → "year"                         │
│       "vehicle_id" → "vin"                          │
│       "make" stays "make"                           │
│     Output becomes:                                 │
│     [                                              │
│       {"vin": "12345", "year": "2020", ...},        │
│       {"vin": "67890", "year": "2021", ...},        │
│     ]                                              │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  4. normalize_values()                              │
│     • Convert year/weight to integers               │
│     • Normalize empty notes to ""                   │
│     • Strip whitespace from strings                │
│     Output becomes:                                 │
│     [                                              │
│       {"vin": "12345", "year": 2020, ...},          │
│       {"vin": "67890", "year": 2021, ...},          │
│     ]                                              │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  5. drop_empty_rows()                               │
│     Removes blank or partially empty rows           │
│     Ensures clean, usable data                      │
└─────────────────────────────────────────────────────┘
                           │
                           ▼
           ┌─────────────────────────────────────────────────────┐
│  FINAL OUTPUT                                        │
│  Clean, normalized, ready-to-use rows                │
│  e.g.:                                               │
│  [                                                   │
│    {"vin": "12345", "year": 2020,                   │
│     "make": "Ford", "model": "F-150", "notes": ""}, │
│    {"vin": "67890", "year": 2021,                   │
│     "make": "Honda", "model": "Civic", "notes": "Lease"}, │
│  ]                                                   │
└─────────────────────────────────────────────────────┘

### Why Some Tests Failed Earlier

Earlier failures were caused by:

- Mismatched data types (e.g., year and weight as strings vs integers)
- Messy input sheets with inconsistent header names
- Expected truth files not matching the new normalization rules

All tests now pass after aligning expected outputs with normalized behavior.