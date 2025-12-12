"""
Policy data mappings from multiple data sources to standardized schema
Each source represents a different vendor, system, or data provider

New simplified schema with 8 fields:
- policy_number
- insured_name
- effective_date
- expiration_date
- premium
- coverage_type
- vehicle_vin
- notes
"""

POLICIES_MAPPINGS = [
    {
        "id": "source_airtable_policies",
        "metadata": {
            "source_name": "Airtable Policies Table",
            "source_type": "airtable",
            "connection_type": "rest_api",
            "notes": "Policy data from Airtable export",
            "connection_config": {
                "base_id": "UNKNOWN_BASE_ID",
                "table_name": "Policies",
                "view_name": "All Policies"
            }
        },
        "mappings": [
            {
                "target_field": "policy_number",
                "source_field": "Policy #",
                "type": "string",
                "required": True
            },
            {
                "target_field": "insured_name",
                "source_field": "Named Insured",
                "type": "string",
                "required": False
            },
            {
                "target_field": "effective_date",
                "source_field": "Effective",
                "type": "date",
                "required": False
            },
            {
                "target_field": "expiration_date",
                "source_field": "Expiration",
                "type": "date",
                "required": False
            },
            {
                "target_field": "premium",
                "source_field": "Premium",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "coverage_type",
                "source_field": "Product",
                "type": "string",
                "required": False
            },
            {
                "target_field": "vehicle_vin",
                "source_field": "",
                "type": "string",
                "required": False
            },
            {
                "target_field": "notes",
                "source_field": "Cancel Reason",
                "type": "string",
                "required": False
            }
        ]
    },
    {
        "id": "source_google_sheet_policies",
        "metadata": {
            "source_name": "Google Sheets Policies",
            "source_type": "google_sheet",
            "connection_type": "google_api",
            "notes": "Policy data from Google Sheets, manually maintained",
            "connection_config": {
                "spreadsheet_id": "",
                "sheet_name": "Policies",
                "data_range": "A1:Z1000"
            }
        },
        "mappings": [
            {
                "target_field": "policy_number",
                "source_field": "Policy Number",
                "type": "string",
                "required": True
            },
            {
                "target_field": "insured_name",
                "source_field": "Named Insured",
                "type": "string",
                "required": False
            },
            {
                "target_field": "effective_date",
                "source_field": "Effective Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "expiration_date",
                "source_field": "Expiration Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "premium",
                "source_field": "Policy Premium",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "coverage_type",
                "source_field": "Product",
                "type": "string",
                "required": False
            },
            {
                "target_field": "vehicle_vin",
                "source_field": "",
                "type": "string",
                "required": False
            },
            {
                "target_field": "notes",
                "source_field": "Cancel Reason",
                "type": "string",
                "required": False
            }
        ]
    },
    {
        "id": "source_xlsx_policies",
        "metadata": {
            "source_name": "Excel Policies Export",
            "source_type": "xlsx_file",
            "connection_type": "file_upload",
            "notes": "Policy data from Excel export file",
            "connection_config": {
                "download_link": "",
                "sheet_name": "Policies",
                "data_range": "A1:Z1000"
            }
        },
        "mappings": [
            {
                "target_field": "policy_number",
                "source_field": "Policy Number",
                "type": "string",
                "required": True
            },
            {
                "target_field": "insured_name",
                "source_field": "Named Insured",
                "type": "string",
                "required": False
            },
            {
                "target_field": "effective_date",
                "source_field": "Effective Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "expiration_date",
                "source_field": "Expiration Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "premium",
                "source_field": "Policy Premium",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "coverage_type",
                "source_field": "Product",
                "type": "string",
                "required": False
            },
            {
                "target_field": "vehicle_vin",
                "source_field": "",
                "type": "string",
                "required": False
            },
            {
                "target_field": "notes",
                "source_field": "Cancel Reason",
                "type": "string",
                "required": False
            }
        ]
    },
    {
        "id": "source_raw_text_policies",
        "metadata": {
            "source_name": "Raw Text Policy Data",
            "source_type": "raw_text",
            "connection_type": "text_input",
            "notes": "Highly variable format, requires NLP and AI interpretation to extract structured data",
            "connection_config": {
                "text_content": "",
                "download_link": ""
            }
        },
        "mappings": [
            {
                "target_field": "policy_number",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the policy number (e.g., P001, P002, P_BAD1) from the text."
            },
            {
                "target_field": "insured_name",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the insured name from the policy description."
            },
            {
                "target_field": "effective_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the effective date if mentioned in the text."
            },
            {
                "target_field": "expiration_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the expiration date if mentioned in the text."
            },
            {
                "target_field": "premium",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the premium amount if mentioned in the text."
            },
            {
                "target_field": "coverage_type",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the coverage type (e.g., 'personal auto', 'high-risk auto') from the text."
            },
            {
                "target_field": "vehicle_vin",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract vehicle VIN if mentioned in the policy description."
            },
            {
                "target_field": "notes",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract any additional notes or cancellation information from the text."
            }
        ]
    }
]
