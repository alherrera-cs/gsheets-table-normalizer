"""
Claims data mappings from multiple data sources to standardized schema
Single mapping reused by all source types (CSV, Google Sheets, Excel, etc.)
"""

CLAIMS_MAPPINGS = [
    {
        "id": "source_claims",
        "metadata": {
            "source_name": "Claims",
            "source_type": "csv",
            "connection_type": "file_upload",
            "notes": "Claims data. Reused by all source types (CSV, Google Sheets, Excel, etc.)",
            "connection_config": {}
        },
        "mappings": [
            {
                "target_field": "claim_number",
                "source_field": "Claim Number",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                }
            },
            {
                "target_field": "policy_number",
                "source_field": "Policy Number",
                "type": "string",
                "required": False
            },
            {
                "target_field": "loss_date",
                "source_field": "Loss Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "claim_type",
                "source_field": "Cause of Loss",
                "type": "string",
                "required": False
            },
            {
                "target_field": "amount",
                "source_field": "Total Incurred",
                "type": "decimal",
                "required": False,
                "validation": {
                    "min": 0
                }
            },
            {
                "target_field": "description",
                "source_field": "Loss Description",
                "type": "string",
                "required": False
            },
            {
                "target_field": "status",
                "source_field": "Claim Status",
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
        "id": "source_pdf_claims",
        "metadata": {
            "source_name": "PDF Claim Documents",
            "source_type": "pdf",
            "connection_type": "file_upload",
            "notes": "Requires OCR and AI extraction, layout and format varies significantly",
            "connection_config": {}
        },
        "mappings": [
            {
                "target_field": "claim_number",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the claim number (e.g., C001, CLM001, CLM_BAD1) from the text."
            },
            {
                "target_field": "policy_number",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the policy number associated with this claim from the text."
            },
            {
                "target_field": "loss_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the loss date if mentioned in the text."
            },
            {
                "target_field": "claim_type",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the claim type or cause of loss (e.g., 'Collision', 'Theft', 'Fire') from the text."
            },
            {
                "target_field": "amount",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the claim amount or total incurred if mentioned in the text."
            },
            {
                "target_field": "description",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the claim description or loss description from the text."
            },
            {
                "target_field": "status",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the claim status (e.g., 'Open', 'Closed', 'Pending') from the text."
            },
            {
                "target_field": "notes",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract any additional notes or information from the text."
            }
        ]
    }
]
