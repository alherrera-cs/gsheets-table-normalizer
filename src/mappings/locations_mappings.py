"""
Location/Garaging location data mappings from multiple data sources to standardized schema
Each source represents a different vendor, system, or data provider
"""

LOCATIONS_MAPPINGS = [
    {
        "id": "source_google_sheet_locations",
        "metadata": {
            "source_name": "Google Sheets Garaging Locations",
            "source_type": "google_sheet",
            "connection_type": "google_api",
            "notes": "Garaging location data from Google Sheets",
            "connection_config": {
                "spreadsheet_id": "",
                "sheet_name": "Garaging Locations",
                "data_range": "A1:Z1000"
            }
        },
        "mappings": [
            {
                "target_field": "location_id",
                "source_field": "Location ID",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                }
            },
            {
                "target_field": "named_insured",
                "source_field": "Named Insured",
                "type": "string",
                "required": False
            },
            {
                "target_field": "address1",
                "source_field": "Address Line 1",
                "type": "string",
                "required": False
            },
            {
                "target_field": "city",
                "source_field": "City",
                "type": "string",
                "required": False
            },
            {
                "target_field": "state",
                "source_field": "State",
                "type": "string",
                "required": False,
                "validation": {
                    "min_length": 2,
                    "max_length": 2
                }
                # "transform": "uppercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "zip",
                "source_field": "Postal Code",
                "type": "string",
                "required": False
            },
            {
                "target_field": "county",
                "source_field": "County",
                "type": "string",
                "required": False
            },
            {
                "target_field": "territory_code",
                "source_field": "Territory Code",
                "type": "string",
                "required": False
            },
            {
                "target_field": "protection_class",
                "source_field": "Protection Class",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 1,
                    "max": 10
                }
            },
            {
                "target_field": "latitude",
                "source_field": "Latitude",
                "type": "decimal",
                "required": False,
                "validation": {
                    "min": -90,
                    "max": 90
                }
            },
            {
                "target_field": "longitude",
                "source_field": "Longitude",
                "type": "decimal",
                "required": False,
                "validation": {
                    "min": -180,
                    "max": 180
                }
            }
        ]
    },
    {
        "id": "source_airtable_locations",
        "metadata": {
            "source_name": "Airtable Locations Table",
            "source_type": "airtable",
            "connection_type": "rest_api",
            "notes": "Location data from Airtable export",
            "connection_config": {
                "base_id": "UNKNOWN_BASE_ID",
                "table_name": "Garaging Locations",
                "view_name": "All Locations"
            }
        },
        "mappings": [
            {
                "target_field": "location_id",
                "source_field": "Location ID",
                "type": "string",
                "required": True
            },
            {
                "target_field": "named_insured",
                "source_field": "Named Insured",
                "type": "string",
                "required": False
            },
            {
                "target_field": "address1",
                "source_field": "Address Line 1",
                "type": "string",
                "required": False
            },
            {
                "target_field": "city",
                "source_field": "City",
                "type": "string",
                "required": False
            },
            {
                "target_field": "state",
                "source_field": "State",
                "type": "string",
                "required": False
            },
            {
                "target_field": "zip",
                "source_field": "Postal Code",
                "type": "string",
                "required": False
            },
            {
                "target_field": "county",
                "source_field": "County",
                "type": "string",
                "required": False
            },
            {
                "target_field": "territory_code",
                "source_field": "Territory Code",
                "type": "string",
                "required": False
            },
            {
                "target_field": "protection_class",
                "source_field": "Protection Class",
                "type": "integer",
                "required": False
            },
            {
                "target_field": "latitude",
                "source_field": "Latitude",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "longitude",
                "source_field": "Longitude",
                "type": "decimal",
                "required": False
            }
        ]
    },
    {
        "id": "source_xlsx_locations",
        "metadata": {
            "source_name": "Excel Locations Export",
            "source_type": "xlsx_file",
            "connection_type": "file_upload",
            "notes": "Location data from Excel export file",
            "connection_config": {
                "download_link": "",
                "sheet_name": "Locations",
                "data_range": "A1:Z1000"
            }
        },
        "mappings": [
            {
                "target_field": "location_id",
                "source_field": "LOCATION_ID",
                "type": "string",
                "required": True
            },
            {
                "target_field": "named_insured",
                "source_field": "NAMED_INSURED",
                "type": "string",
                "required": False
            },
            {
                "target_field": "address1",
                "source_field": "ADDRESS_LINE_1",
                "type": "string",
                "required": False
            },
            {
                "target_field": "city",
                "source_field": "CITY",
                "type": "string",
                "required": False
            },
            {
                "target_field": "state",
                "source_field": "STATE",
                "type": "string",
                "required": False
            },
            {
                "target_field": "zip",
                "source_field": "POSTAL_CODE",
                "type": "string",
                "required": False
            },
            {
                "target_field": "county",
                "source_field": "COUNTY",
                "type": "string",
                "required": False
            },
            {
                "target_field": "territory_code",
                "source_field": "TERRITORY_CODE",
                "type": "string",
                "required": False
            },
            {
                "target_field": "protection_class",
                "source_field": "PROTECTION_CLASS",
                "type": "integer",
                "required": False
            },
            {
                "target_field": "latitude",
                "source_field": "LATITUDE",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "longitude",
                "source_field": "LONGITUDE",
                "type": "decimal",
                "required": False
            }
        ]
    },
    {
        "id": "source_pdf_locations",
        "metadata": {
            "source_name": "PDF Location Documents",
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
                "target_field": "location_id",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the location ID or location identifier from the document. Look for labels like 'Location ID', 'ID', 'Location Number', or unique identifiers associated with location information."
            },
            {
                "target_field": "address1",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the street address (first line) from the document. Look for labels like 'Address', 'Address Line 1', 'Street Address', 'Address 1', or street addresses in location information sections."
            },
            {
                "target_field": "city",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the city from the document. Look for labels like 'City', or city names in address sections."
            },
            {
                "target_field": "state",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the state from the document. Look for labels like 'State', or state abbreviations (2 letters) in address sections."
            },
            {
                "target_field": "zip",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the ZIP or postal code from the document. Look for labels like 'ZIP', 'Postal Code', 'ZIP Code', or postal codes (5 or 9 digits) in address sections."
            },
            {
                "target_field": "county",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the county from the document. Look for labels like 'County', or county names in location information sections."
            },
            {
                "target_field": "latitude",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the latitude coordinate from the document. Look for labels like 'Latitude', 'Lat', or latitude values (decimal numbers between -90 and 90) in location information."
            },
            {
                "target_field": "longitude",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the longitude coordinate from the document. Look for labels like 'Longitude', 'Lon', 'Lng', or longitude values (decimal numbers between -180 and 180) in location information."
            }
        ]
    },
    {
        "id": "source_raw_text_locations",
        "metadata": {
            "source_name": "Raw Text Location Data",
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
                "target_field": "location_id",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Search the text for a location ID or location identifier. It may be mentioned explicitly with keywords like 'Location ID:', 'ID:', 'Location Number:', or may appear as a unique identifier associated with location information."
            },
            {
                "target_field": "address1",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the street address from the text. Look for addresses mentioned with keywords like 'Address:', 'Street:', or street addresses in location descriptions."
            },
            {
                "target_field": "city",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the city from the text. Look for city names mentioned with keywords like 'City:', or city names in address descriptions."
            },
            {
                "target_field": "state",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the state from the text. Look for state abbreviations (2 letters like 'CA', 'TX') or state names mentioned with keywords like 'State:', or in address descriptions."
            },
            {
                "target_field": "zip",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the ZIP or postal code from the text. Look for postal codes mentioned with keywords like 'ZIP:', 'Postal Code:', or postal codes (5 or 9 digits) in address descriptions."
            }
        ]
    },
    {
        "id": "source_image_locations",
        "metadata": {
            "source_name": "Location Image Analysis",
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
                "target_field": "location_id",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Analyze the image for a location ID or location identifier. The ID may appear: on location documents, on registration forms, on policy documents, or in any location information visible in the image. Use OCR to extract the identifier."
            },
            {
                "target_field": "address1",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the street address from the image. Look for: 1) Addresses on location documents or registration forms. 2) Street addresses on policy documents. 3) Addresses in location information sections visible in the image."
            },
            {
                "target_field": "city",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the city from the image. Look for: 1) City names on location documents. 2) City information in address sections. 3) City names in location information visible in the image."
            },
            {
                "target_field": "state",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the state from the image. Look for: 1) State abbreviations or names on location documents. 2) State information in address sections. 3) State codes (2 letters) in location information visible in the image."
            },
            {
                "target_field": "zip",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the ZIP or postal code from the image. Look for: 1) Postal codes on location documents. 2) ZIP codes in address sections. 3) Postal codes (5 or 9 digits) in location information visible in the image."
            }
        ]
    }
]
