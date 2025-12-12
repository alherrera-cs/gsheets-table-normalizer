"""
Driver data mappings from multiple data sources to standardized schema
Each source represents a different vendor, system, or data provider
"""

DRIVERS_MAPPINGS = [
    {
        "id": "source_google_sheet_drivers",
        "metadata": {
            "source_name": "Google Sheets Drivers",
            "source_type": "google_sheet",
            "connection_type": "google_api",
            "notes": "Driver data from Google Sheets, manually maintained",
            "connection_config": {
                "spreadsheet_id": "",
                "sheet_name": "Drivers",
                "data_range": "A1:Z1000"
            }
        },
        "mappings": [
            {
                "target_field": "driver_id",
                "source_field": "Driver ID",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                }
                # "transform": "uppercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "policy_number",
                "source_field": "Policy Number",
                "type": "string",
                "required": False
            },
            {
                "target_field": "first_name",
                "source_field": "First Name",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 100
                }
                # "transform": "capitalize"  # Transform logic will be implemented later
            },
            {
                "target_field": "last_name",
                "source_field": "Last Name",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 100
                }
                # "transform": "capitalize"  # Transform logic will be implemented later
            },
            # Date of Birth - multiple variants (CSV: "Date of Birth", Airtable: "DOB")
            {
                "target_field": "date_of_birth",
                "source_field": "Date of Birth",
                "type": "date",
                "required": False
            },
            {
                "target_field": "date_of_birth",
                "source_field": "DOB",
                "type": "date",
                "required": False
            },
            {
                "target_field": "gender",
                "source_field": "Gender",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["M", "F", "Male", "Female", "Other", "Unknown", ""]
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "marital_status",
                "source_field": "Marital Status",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["single", "married", "divorced", "widowed", "Single", "Married", "Divorced", "Widowed", ""]
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            # License Number - multiple variants (CSV: "License Number", Airtable: "License #")
            {
                "target_field": "license_number",
                "source_field": "License Number",
                "type": "string",
                "required": False
            },
            {
                "target_field": "license_number",
                "source_field": "License #",
                "type": "string",
                "required": False
            },
            {
                "target_field": "license_state",
                "source_field": "License State",
                "type": "string",
                "required": False,
                "validation": {
                    "min_length": 2,
                    "max_length": 2
                }
                # "transform": "uppercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "license_status",
                "source_field": "License Status",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Valid", "Suspended", "Revoked", "Expired", "valid", "suspended", "revoked", "expired", ""]
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "years_experience",
                "source_field": "Years Licensed",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 70
                }
            },
            {
                "target_field": "email",
                "source_field": "Email",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "phone",
                "source_field": "Phone",
                "type": "string",
                "required": False
            },
            # Address1 - multiple variants (CSV: "Address Line 1", Airtable: "Address 1")
            {
                "target_field": "address1",
                "source_field": "Address Line 1",
                "type": "string",
                "required": False
            },
            {
                "target_field": "address1",
                "source_field": "Address 1",
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
            # ZIP - multiple variants (CSV: "Postal Code", Airtable: "ZIP")
            {
                "target_field": "zip",
                "source_field": "Postal Code",
                "type": "string",
                "required": False
            },
            {
                "target_field": "zip",
                "source_field": "ZIP",
                "type": "string",
                "required": False
            },
            # Primary Driver - multiple variants (CSV: "Primary Driver?", Airtable: "Primary Driver")
            {
                "target_field": "primary_driver",
                "source_field": "Primary Driver?",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Yes", "No", "yes", "no", "True", "False", "true", "false", "Maybe", ""]
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "primary_driver",
                "source_field": "Primary Driver",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Yes", "No", "yes", "no", "True", "False", "true", "false", "Maybe", ""]
                }
            },
            {
                "target_field": "occupation",
                "source_field": "Occupation",
                "type": "string",
                "required": False
            },
            # Credit Band - multiple variants (CSV: "Credit Score Band", Airtable: "Credit Band")
            {
                "target_field": "credit_band",
                "source_field": "Credit Score Band",
                "type": "string",
                "required": False
            },
            {
                "target_field": "credit_band",
                "source_field": "Credit Band",
                "type": "string",
                "required": False
            }
        ]
    },
    {
        "id": "source_airtable_drivers",
        "metadata": {
            "source_name": "Airtable Drivers Table",
            "source_type": "airtable",
            "connection_type": "rest_api",
            "notes": "Driver data from Airtable export",
            "connection_config": {
                "base_id": "UNKNOWN_BASE_ID",
                "table_name": "Drivers",
                "view_name": "All Drivers"
            }
        },
        "mappings": [
            {
                "target_field": "driver_id",
                "source_field": "Driver ID",
                "type": "string",
                "required": True
            },
            {
                "target_field": "policy_number",
                "source_field": "Policy Number",
                "type": "string",
                "required": False
            },
            {
                "target_field": "first_name",
                "source_field": "First Name",
                "type": "string",
                "required": True
                # "transform": "capitalize"  # Transform logic will be implemented later
            },
            {
                "target_field": "last_name",
                "source_field": "Last Name",
                "type": "string",
                "required": True
                # "transform": "capitalize"  # Transform logic will be implemented later
            },
            {
                "target_field": "date_of_birth",
                "source_field": "DOB",
                "type": "date",
                "required": False
            },
            {
                "target_field": "gender",
                "source_field": "Gender",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["M", "F", "Male", "Female", "Other", "Unknown", ""]
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "marital_status",
                "source_field": "Marital Status",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["single", "married", "divorced", "widowed", "Single", "Married", "Divorced", "Widowed", ""]
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "license_number",
                "source_field": "License #",
                "type": "string",
                "required": False
            },
            {
                "target_field": "license_state",
                "source_field": "License State",
                "type": "string",
                "required": False
                # "transform": "uppercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "license_status",
                "source_field": "License Status",
                "type": "string",
                "required": False
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "years_experience",
                "source_field": "Years Licensed",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 70
                }
            },
            {
                "target_field": "email",
                "source_field": "Email",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "phone",
                "source_field": "Phone",
                "type": "string",
                "required": False
            },
            {
                "target_field": "address1",
                "source_field": "Address 1",
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
                # "transform": "uppercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "zip",
                "source_field": "ZIP",
                "type": "string",
                "required": False
            },
            {
                "target_field": "primary_driver",
                "source_field": "Primary Driver",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["yes", "no", "Yes", "No", "true", "false", "True", "False", "Maybe", ""]
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "occupation",
                "source_field": "Occupation",
                "type": "string",
                "required": False
            },
            {
                "target_field": "credit_band",
                "source_field": "Credit Band",
                "type": "string",
                "required": False
            }
        ]
    },
    {
        "id": "source_xlsx_drivers",
        "metadata": {
            "source_name": "Excel Drivers Export",
            "source_type": "xlsx_file",
            "connection_type": "file_upload",
            "notes": "Driver data from Excel export file",
            "connection_config": {
                "download_link": "",
                "sheet_name": "Drivers",
                "data_range": "A1:Z1000"
            }
        },
        "mappings": [
            {
                "target_field": "driver_id",
                "source_field": "DRIVER_ID",
                "type": "string",
                "required": True
            },
            {
                "target_field": "policy_number",
                "source_field": "POLICY_NUMBER",
                "type": "string",
                "required": False
            },
            {
                "target_field": "first_name",
                "source_field": "FIRST_NAME",
                "type": "string",
                "required": True
            },
            {
                "target_field": "last_name",
                "source_field": "LAST_NAME",
                "type": "string",
                "required": True
            },
            {
                "target_field": "date_of_birth",
                "source_field": "DATE_OF_BIRTH",
                "type": "date",
                "required": False
            },
            {
                "target_field": "gender",
                "source_field": "GENDER",
                "type": "string",
                "required": False
            },
            {
                "target_field": "marital_status",
                "source_field": "MARITAL_STATUS",
                "type": "string",
                "required": False
            },
            {
                "target_field": "license_number",
                "source_field": "LICENSE_NUMBER",
                "type": "string",
                "required": False
            },
            {
                "target_field": "license_state",
                "source_field": "LICENSE_STATE",
                "type": "string",
                "required": False
            },
            {
                "target_field": "license_status",
                "source_field": "LICENSE_STATUS",
                "type": "string",
                "required": False
            },
            {
                "target_field": "years_experience",
                "source_field": "YEARS_LICENSED",
                "type": "integer",
                "required": False
            },
            {
                "target_field": "email",
                "source_field": "EMAIL",
                "type": "email",
                "required": False,
                "flag": ["pii"]
            },
            {
                "target_field": "phone",
                "source_field": "PHONE",
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
                "target_field": "primary_driver",
                "source_field": "PRIMARY_DRIVER",
                "type": "string",
                "required": False
            },
            {
                "target_field": "occupation",
                "source_field": "OCCUPATION",
                "type": "string",
                "required": False
            },
            {
                "target_field": "credit_band",
                "source_field": "CREDIT_SCORE_BAND",
                "type": "string",
                "required": False
            }
        ]
    },
    {
        "id": "source_pdf_drivers",
        "metadata": {
            "source_name": "PDF Driver Documents",
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
                "target_field": "driver_id",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the driver ID or driver identifier from the document. Look for labels like 'Driver ID', 'ID', 'Driver Number', 'License Holder ID', or unique identifiers associated with driver information."
            },
            {
                "target_field": "policy_number",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the policy number associated with this driver. Look for labels like 'Policy Number', 'Policy #', 'Policy ID', 'Policy', or policy identifiers near driver information."
            },
            {
                "target_field": "first_name",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the driver's first name from the document. Look for labels like 'First Name', 'Given Name', 'First', 'FName', or names at the beginning of driver information sections."
            },
            {
                "target_field": "last_name",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the driver's last name from the document. Look for labels like 'Last Name', 'Surname', 'Family Name', 'LName', or names following the first name."
            },
            {
                "target_field": "date_of_birth",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the driver's date of birth from the document. Look for labels like 'Date of Birth', 'DOB', 'Birth Date', 'Born', or dates in MM/DD/YYYY, YYYY-MM-DD, or other common date formats near personal information."
            },
            {
                "target_field": "gender",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["M", "F", "Male", "Female", "Other", "Unknown", ""]
                },
                "ai_instruction": "Extract the driver's gender from the document. Look for labels like 'Gender', 'Sex', or gender indicators. Common values include M, F, Male, Female, Other, or Unknown."
            },
            {
                "target_field": "marital_status",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["single", "married", "divorced", "widowed", "Single", "Married", "Divorced", "Widowed", ""]
                },
                "ai_instruction": "Extract the driver's marital status from the document. Look for labels like 'Marital Status', 'Marriage Status', or status indicators. Common values include Single, Married, Divorced, or Widowed."
            },
            {
                "target_field": "license_number",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's license number from the document. Look for labels like 'License', 'License Number', 'DL Number', 'Driver License', 'License #', or license number patterns (alphanumeric codes)."
            },
            {
                "target_field": "license_state",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the state that issued the driver's license. Look for labels like 'License State', 'State', 'DL State', 'Issuing State', or state abbreviations (2 letters) near license number information."
            },
            {
                "target_field": "license_status",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's license status from the document. Look for labels like 'License Status', 'Status', 'DL Status', or status indicators. Common values include Valid, Suspended, Revoked, or Expired."
            },
            {
                "target_field": "years_experience",
                "source_field": "",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 70
                },
                "ai_instruction": "Extract the number of years the driver has been licensed. Look for labels like 'Years Licensed', 'Years with License', 'License Years', or numeric values representing years of driving experience."
            },
            {
                "target_field": "email",
                "source_field": "",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                "ai_instruction": "Extract the driver's email address from the document. Look for labels like 'Email', 'Email Address', 'Contact Email', 'E-mail', or email addresses near contact information. Return the email in lowercase format."
            },
            {
                "target_field": "phone",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's phone number from the document. Look for labels like 'Phone', 'Phone Number', 'Contact', 'Telephone', 'Cell', or phone number patterns (XXX-XXX-XXXX, (XXX) XXX-XXXX, etc.)."
            },
            {
                "target_field": "address1",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's street address (first line) from the document. Look for labels like 'Address', 'Address Line 1', 'Street Address', 'Address 1', or street addresses near location information."
            },
            {
                "target_field": "city",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's city from the document. Look for labels like 'City', or city names in address sections."
            },
            {
                "target_field": "state",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's state from the document. Look for labels like 'State', or state abbreviations (2 letters) in address sections."
            },
            {
                "target_field": "zip",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's ZIP or postal code from the document. Look for labels like 'ZIP', 'Postal Code', 'ZIP Code', or postal codes (5 or 9 digits) in address sections."
            },
            {
                "target_field": "primary_driver",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["yes", "no", "Yes", "No", "true", "false", "True", "False", "Maybe", ""]
                },
                "ai_instruction": "Determine if this driver is the primary driver. Look for labels like 'Primary Driver', 'Primary', 'Main Driver', or indicators like 'Yes', 'No', 'True', 'False'. If not explicitly stated, infer from context (e.g., first listed driver, policyholder)."
            },
            {
                "target_field": "occupation",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's occupation from the document. Look for labels like 'Occupation', 'Job', 'Employment', 'Profession', or occupation descriptions near driver information."
            },
            {
                "target_field": "credit_band",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's credit score band or credit rating from the document. Look for labels like 'Credit Band', 'Credit Score', 'Credit Rating', 'Credit Range', or credit score ranges (e.g., '700-749', '<600')."
            }
        ]
    },
    {
        "id": "source_raw_text_drivers",
        "metadata": {
            "source_name": "Raw Text Driver Data",
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
                "target_field": "driver_id",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Search the text for a driver ID or driver identifier. It may be mentioned explicitly with keywords like 'Driver ID:', 'ID:', 'Driver Number:', or may appear as a unique identifier associated with driver information."
            },
            {
                "target_field": "first_name",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Identify the driver's first name from the text. Look for names following keywords like 'First Name:', 'Given Name:', or names at the beginning of driver descriptions. The first name typically appears before the last name."
            },
            {
                "target_field": "last_name",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Identify the driver's last name from the text. Look for names following keywords like 'Last Name:', 'Surname:', or names following the first name in driver descriptions."
            },
            {
                "target_field": "date_of_birth",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the driver's date of birth from the text. Look for dates mentioned with keywords like 'DOB', 'Date of Birth', 'Born', 'Birth Date', or dates in formats like '1985-03-15', '03/15/1985', or 'March 15, 1985' near driver information."
            },
            {
                "target_field": "license_number",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's license number from the text. Look for identifiers mentioned with keywords like 'License:', 'License Number:', 'DL Number:', 'License #:', or alphanumeric license number patterns."
            },
            {
                "target_field": "license_state",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the state that issued the driver's license from the text. Look for state abbreviations (2 letters like 'CA', 'TX') or state names mentioned with keywords like 'License State:', 'State:', or near license number information."
            },
            {
                "target_field": "email",
                "source_field": "",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                "ai_instruction": "Extract the driver's email address from the text. Look for email addresses mentioned with keywords like 'Email:', 'Contact:', 'E-mail:', or standalone email patterns (text@domain.com). Return the email in lowercase format."
            },
            {
                "target_field": "phone",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's phone number from the text. Look for phone numbers mentioned with keywords like 'Phone:', 'Contact:', 'Telephone:', or phone number patterns (XXX-XXX-XXXX, (XXX) XXX-XXXX, etc.)."
            },
            {
                "target_field": "occupation",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's occupation from the text. Look for job titles or professions mentioned with keywords like 'Occupation:', 'Job:', 'Employment:', 'Profession:', or occupation descriptions in driver information."
            },
            {
                "target_field": "notes",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract any additional notes or descriptive information about the driver from the text. This may include occupation, status descriptions, or other relevant details."
            }
        ]
    },
    {
        "id": "source_image_drivers",
        "metadata": {
            "source_name": "Driver Image Analysis",
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
                "target_field": "driver_id",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Analyze the image for a driver ID or driver identifier. The ID may appear: on driver's license documents, on registration forms, on policy documents, or in any driver information visible in the image. Use OCR to extract the identifier."
            },
            {
                "target_field": "first_name",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the driver's first name from the image. Look for: 1) Names on driver's license documents. 2) First names on registration or policy forms. 3) Names in document headers or driver information sections. The first name typically appears before the last name."
            },
            {
                "target_field": "last_name",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the driver's last name from the image. Look for: 1) Surnames on driver's license documents. 2) Last names on registration or policy forms. 3) Names following first names in driver information sections."
            },
            {
                "target_field": "date_of_birth",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the driver's date of birth from the image. Look for: 1) DOB fields on driver's license documents. 2) Birth dates on registration or policy forms. 3) Dates labeled as 'Date of Birth', 'DOB', or 'Born' in document sections."
            },
            {
                "target_field": "license_number",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's license number from the image. Look for: 1) License numbers on driver's license documents (often prominently displayed). 2) License numbers on registration or policy forms. 3) License identifiers labeled as 'License', 'License Number', 'DL Number', or 'License #'."
            },
            {
                "target_field": "license_state",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the state that issued the driver's license from the image. Look for: 1) State abbreviations or names on driver's license documents. 2) State information on registration or policy forms. 3) State codes (2 letters) near license number information."
            },
            {
                "target_field": "email",
                "source_field": "",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                "ai_instruction": "Extract the driver's email address from the image. Look for: 1) Email addresses on registration documents or policy forms. 2) Contact email fields on any forms visible in the image. 3) Email addresses near contact information. Use OCR to extract the email address and return it in lowercase format."
            },
            {
                "target_field": "phone",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the driver's phone number from the image. Look for: 1) Phone numbers on registration documents or policy forms. 2) Contact phone fields on any forms visible in the image. 3) Phone numbers near contact information. Use OCR to extract phone number patterns."
            }
        ]
    }
]
