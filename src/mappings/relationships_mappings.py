"""
Relationship/Link data mappings (policy-vehicle-driver links) from multiple data sources to standardized schema
Single mapping reused by all source types (CSV, Google Sheets, Excel, etc.)
"""

RELATIONSHIPS_MAPPINGS = [
    {
        "id": "source_policy_vehicle_driver_link",
        "metadata": {
            "source_name": "Policy-Vehicle-Driver Link",
            "source_type": "csv",
            "connection_type": "file_upload",
            "notes": "Relationship/link data connecting policies, vehicles, and drivers. Reused by all source types (CSV, Google Sheets, Excel, etc.)",
            "connection_config": {}
        },
        "mappings": [
            {
                "target_field": "policy_number",
                "source_field": "Policy Number",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                }
            },
            {
                "target_field": "vehicle_vin",
                "source_field": "VIN",
                "type": "string",
                "required": True,
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                }
            },
            {
                "target_field": "driver_id",
                "source_field": "Driver ID",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                }
            },
            {
                "target_field": "relationship_type",
                "source_field": "Usage",
                "type": "string",
                "required": False
            },
            {
                "target_field": "notes",
                "source_field": "",
                "type": "string",
                "required": False
            }
        ]
    },
    {
        "id": "source_pdf_policy_vehicle_driver_link",
        "metadata": {
            "source_name": "PDF Policy-Vehicle-Driver Link Documents",
            "source_type": "pdf",
            "connection_type": "file_upload",
            "notes": "Requires OCR and AI extraction, layout and format varies significantly",
            "connection_config": {}
        },
        "mappings": [
            {
                "target_field": "policy_number",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the policy number (e.g., P001, P_BAD1) from the text."
            },
            {
                "target_field": "vehicle_vin",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the vehicle VIN (17 characters) from the text."
            },
            {
                "target_field": "driver_id",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the driver ID (e.g., D001, DRV001) from the text."
            },
            {
                "target_field": "relationship_type",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the relationship type or usage (e.g., 'Primary driver', 'Secondary driver', 'Owner', 'Operator') from the text."
            },
            {
                "target_field": "notes",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract any additional notes or information about the relationship from the text."
            }
        ]
    }
]
