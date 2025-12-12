# Expected Truth Files

These are **manually verified** expected truth files for the normalization pipeline tests.

## ⚠️ IMPORTANT

**DO NOT regenerate these files automatically.** They are manually created and verified to ensure accuracy.

## File Structure

Each truth file is a JSON object with a schema definition and rows array:
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

The `__schema__` array defines the canonical field order for the domain.
Each row includes all schema fields (missing fields are `null`) plus `_warnings` array.

## Row Counts

Expected row counts per domain:

- **Drivers**: google_sheet=3, airtable=3, excel=0, raw_text=2
- **Policies**: google_sheet=3, airtable=3, excel=0, raw_text=3
- **Claims**: google_sheet=2
- **Locations**: garaging_locations=2
- **Relationships**: policy_vehicle_driver_link=3
- **Vehicles**: 
  - airtable_fleet_vehicles=8
  - google_sheet_vehicle_inventory=8
  - excel_vehicle_export=4
  - pdf_vehicle_documents=4
  - raw_text_vehicle_data=6

## Notes

- Truth files include ALL rows from source data, even those with validation errors
- Unmapped fields are set to `null`
- Empty strings are normalized to `null` in comparisons
- These files represent the expected output after extraction + mapping (NOT after full normalization with validation filtering)
