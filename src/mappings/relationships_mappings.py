"""
Relationship/Link data mappings (policy-vehicle-driver links) from multiple data sources to standardized schema
Each source represents a different vendor, system, or data provider
"""

RELATIONSHIPS_MAPPINGS = [
    {
        "id": "source_google_sheet_relationships",
        "metadata": {
            "source_name": "Google Sheets Policy-Vehicle-Driver Links",
            "source_type": "google_sheet",
            "connection_type": "google_api",
            "notes": "Relationship/link data connecting policies, vehicles, and drivers from Google Sheets",
            "connection_config": {
                "spreadsheet_id": "",
                "sheet_name": "Policy Vehicle Driver Link",
                "data_range": "A1:Z1000"
            }
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
                "target_field": "vin",
                "source_field": "VIN",
                "type": "string",
                "required": True,
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                }
                # "transform": "uppercase"  # Transform logic will be implemented later
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
                "target_field": "location_id",
                "source_field": "Location ID",
                "type": "string",
                "required": False
            },
            {
                "target_field": "usage",
                "source_field": "Usage",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Commute", "Pleasure", "Business", "Farm", "commute", "pleasure", "business", "farm", ""]
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "annual_mileage",
                "source_field": "Annual Mileage",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 100000
                }
            }
        ]
    },
    {
        "id": "source_airtable_relationships",
        "metadata": {
            "source_name": "Airtable Relationships Table",
            "source_type": "airtable",
            "connection_type": "rest_api",
            "notes": "Relationship/link data from Airtable export",
            "connection_config": {
                "base_id": "UNKNOWN_BASE_ID",
                "table_name": "Policy Vehicle Driver Links",
                "view_name": "All Links"
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
                "target_field": "vin",
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
                "required": True
            },
            {
                "target_field": "location_id",
                "source_field": "Location ID",
                "type": "string",
                "required": False
            },
            {
                "target_field": "usage",
                "source_field": "Usage",
                "type": "string",
                "required": False
            },
            {
                "target_field": "annual_mileage",
                "source_field": "Annual Mileage",
                "type": "integer",
                "required": False
            }
        ]
    },
    {
        "id": "source_xlsx_relationships",
        "metadata": {
            "source_name": "Excel Relationships Export",
            "source_type": "xlsx_file",
            "connection_type": "file_upload",
            "notes": "Relationship data from Excel export file",
            "connection_config": {
                "download_link": "",
                "sheet_name": "Relationships",
                "data_range": "A1:Z1000"
            }
        },
        "mappings": [
            {
                "target_field": "policy_number",
                "source_field": "POLICY_NUMBER",
                "type": "string",
                "required": True
            },
            {
                "target_field": "vin",
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
                "source_field": "DRIVER_ID",
                "type": "string",
                "required": True
            },
            {
                "target_field": "location_id",
                "source_field": "LOCATION_ID",
                "type": "string",
                "required": False
            },
            {
                "target_field": "usage",
                "source_field": "USAGE",
                "type": "string",
                "required": False
            },
            {
                "target_field": "annual_mileage",
                "source_field": "ANNUAL_MILEAGE",
                "type": "integer",
                "required": False
            }
        ]
    },
    {
        "id": "source_pdf_relationships",
        "metadata": {
            "source_name": "PDF Relationship Documents",
            "source_type": "pdf",
            "connection_type": "file_upload",
            "notes": "Requires OCR and AI extraction, layout and format varies significantly",
            "connection_config": {
                "download_link": "",
                "pages": []
            }
        },
        "mappings": [
            {
                "target_field": "policy_number",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the policy number from the document. Look for labels like 'Policy Number', 'Policy #', 'Policy ID', or policy identifiers in relationship or link information sections."
            },
            {
                "target_field": "vin",
                "source_field": "",
                "type": "string",
                "required": True,
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                },
                "ai_instruction": "Extract the Vehicle Identification Number (VIN) from the document. Look for a 17-character alphanumeric code, often labeled as 'VIN', 'VIN Number', 'Vehicle Identification Number', or found in vehicle relationship information. The VIN excludes the letters I, O, and Q."
            },
            {
                "target_field": "driver_id",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the driver ID from the document. Look for labels like 'Driver ID', 'Driver Number', 'Driver', or driver identifiers in relationship or link information sections."
            },
            {
                "target_field": "location_id",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the location ID from the document. Look for labels like 'Location ID', 'Location Number', 'Location', or location identifiers in relationship information sections."
            },
            {
                "target_field": "usage",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Commute", "Pleasure", "Business", "Farm", "commute", "pleasure", "business", "farm", ""]
                },
                "ai_instruction": "Extract the vehicle usage type from the document. Look for labels like 'Usage', 'Vehicle Usage', 'Use', or usage descriptions. Common values include Commute, Pleasure, Business, or Farm."
            },
            {
                "target_field": "annual_mileage",
                "source_field": "",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 100000
                },
                "ai_instruction": "Extract the annual mileage from the document. Look for labels like 'Annual Mileage', 'Mileage', 'Annual Miles', or mileage values (numeric, may include commas). Remove commas and return only the numeric value."
            }
        ]
    },
    {
        "id": "source_raw_text_relationships",
        "metadata": {
            "source_name": "Raw Text Relationship Data",
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
                "ai_instruction": "Search the text for a policy number. It may be mentioned explicitly with keywords like 'Policy Number:', 'Policy #:', 'Policy:', or may appear as a unique identifier associated with policy information."
            },
            {
                "target_field": "vin",
                "source_field": "",
                "type": "string",
                "required": True,
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                },
                "ai_instruction": "Search the text for a 17-character Vehicle Identification Number (VIN). It may be mentioned explicitly with keywords like 'VIN:', 'VIN number:', 'Vehicle ID:', or may appear as a standalone 17-character alphanumeric string. Remember VINs do not contain the letters I, O, or Q."
            },
            {
                "target_field": "driver_id",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Search the text for a driver ID. It may be mentioned explicitly with keywords like 'Driver ID:', 'Driver Number:', 'Driver:', or may appear as a unique identifier associated with driver information."
            },
            {
                "target_field": "usage",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Commute", "Pleasure", "Business", "Farm", "commute", "pleasure", "business", "farm", ""]
                },
                "ai_instruction": "Identify the vehicle usage type from the text. Look for usage descriptions mentioned with keywords like 'Usage:', 'Vehicle Usage:', 'Use:', or usage types like Commute, Pleasure, Business, or Farm in relationship descriptions."
            },
            {
                "target_field": "annual_mileage",
                "source_field": "",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 100000
                },
                "ai_instruction": "Extract the annual mileage from the text. Look for mileage values mentioned with keywords like 'Annual Mileage:', 'Mileage:', 'Annual Miles:', or numbers followed by 'miles' or 'mi'. Remove commas and units, return only the numeric value."
            }
        ]
    },
    {
        "id": "source_image_relationships",
        "metadata": {
            "source_name": "Relationship Image Analysis",
            "source_type": "image",
            "connection_type": "file_upload",
            "notes": "Requires OCR and computer vision, accuracy depends heavily on image quality and lighting",
            "connection_config": {
                "download_link": "",
                "accepted_formats": ["jpg", "jpeg", "png", "heic", "webp"]
            }
        },
        "mappings": [
            {
                "target_field": "policy_number",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Analyze the image for a policy number. The policy number may appear: on policy documents, on relationship/link forms, on declarations pages, or in any policy information visible in the image. Use OCR to extract the policy identifier."
            },
            {
                "target_field": "vin",
                "source_field": "",
                "type": "string",
                "required": True,
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                },
                "ai_instruction": "Analyze the image for a Vehicle Identification Number (VIN). The VIN is a 17-character code that may appear: on vehicle documents, on registration cards, on relationship forms, or in any vehicle information visible in the image. Use OCR to extract the 17-character alphanumeric sequence, excluding letters I, O, and Q."
            },
            {
                "target_field": "driver_id",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Analyze the image for a driver ID. The driver ID may appear: on driver's license documents, on relationship/link forms, on policy documents, or in any driver information visible in the image. Use OCR to extract the driver identifier."
            },
            {
                "target_field": "usage",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Commute", "Pleasure", "Business", "Farm", "commute", "pleasure", "business", "farm", ""]
                },
                "ai_instruction": "Determine the vehicle usage type from the image. Look for: 1) Usage fields on relationship or link forms. 2) Usage descriptions on policy documents. 3) Usage indicators labeled as 'Usage', 'Vehicle Usage', or 'Use'. Common values include Commute, Pleasure, Business, or Farm."
            },
            {
                "target_field": "annual_mileage",
                "source_field": "",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 100000
                },
                "ai_instruction": "Extract the annual mileage from the image. Look for: 1) Mileage fields on relationship or link forms. 2) Annual mileage on policy documents. 3) Mileage values labeled as 'Annual Mileage', 'Mileage', or 'Annual Miles'. Extract only the numeric value, removing commas and units."
            }
        ]
    }
]
