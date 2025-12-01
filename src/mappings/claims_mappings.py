"""
Claims data mappings from multiple data sources to standardized schema
Each source represents a different vendor, system, or data provider
"""

CLAIMS_MAPPINGS = [
    {
        "id": "source_google_sheet_claims",
        "metadata": {
            "source_name": "Google Sheets Claims",
            "source_type": "google_sheet",
            "connection_type": "google_api",
            "notes": "Claims data from Google Sheets, manually maintained",
            "connection_config": {
                "spreadsheet_id": "",
                "sheet_name": "Claims",
                "data_range": "A1:Z1000"
            }
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
                "target_field": "report_date",
                "source_field": "Report Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "cause_of_loss",
                "source_field": "Cause of Loss",
                "type": "string",
                "required": False
            },
            {
                "target_field": "loss_description",
                "source_field": "Loss Description",
                "type": "string",
                "required": False
            },
            {
                "target_field": "claim_status",
                "source_field": "Claim Status",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Open", "Closed", "Pending", "Denied", "open", "closed", "pending", "denied", ""]
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "open_closed",
                "source_field": "Open/Closed",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Open", "Closed", "open", "closed", ""]
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "total_incurred",
                "source_field": "Total Incurred",
                "type": "decimal",
                "required": False,
                "validation": {
                    "min": 0
                }
            },
            {
                "target_field": "paid",
                "source_field": "Paid",
                "type": "decimal",
                "required": False,
                "validation": {
                    "min": 0
                }
            },
            {
                "target_field": "reserved",
                "source_field": "Reserved",
                "type": "decimal",
                "required": False,
                "validation": {
                    "min": 0
                }
            },
            {
                "target_field": "adjuster_name",
                "source_field": "Adjuster Name",
                "type": "string",
                "required": False
            },
            {
                "target_field": "adjuster_email",
                "source_field": "Adjuster Email",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            },
            {
                "target_field": "litigation_flag",
                "source_field": "Litigation Flag",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Yes", "No", "yes", "no", "True", "False", "true", "false", ""]
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
            }
        ]
    },
    {
        "id": "source_airtable_claims",
        "metadata": {
            "source_name": "Airtable Claims Table",
            "source_type": "airtable",
            "connection_type": "rest_api",
            "notes": "Claims data from Airtable export",
            "connection_config": {
                "base_id": "UNKNOWN_BASE_ID",
                "table_name": "Claims",
                "view_name": "All Claims"
            }
        },
        "mappings": [
            {
                "target_field": "claim_number",
                "source_field": "Claim Number",
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
                "target_field": "loss_date",
                "source_field": "Loss Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "report_date",
                "source_field": "Report Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "cause_of_loss",
                "source_field": "Cause of Loss",
                "type": "string",
                "required": False
            },
            {
                "target_field": "loss_description",
                "source_field": "Loss Description",
                "type": "string",
                "required": False
            },
            {
                "target_field": "claim_status",
                "source_field": "Claim Status",
                "type": "string",
                "required": False
            },
            {
                "target_field": "open_closed",
                "source_field": "Open/Closed",
                "type": "string",
                "required": False
            },
            {
                "target_field": "total_incurred",
                "source_field": "Total Incurred",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "paid",
                "source_field": "Paid",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "reserved",
                "source_field": "Reserved",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "adjuster_name",
                "source_field": "Adjuster Name",
                "type": "string",
                "required": False
            },
            {
                "target_field": "adjuster_email",
                "source_field": "Adjuster Email",
                "type": "email",
                "required": False,
                "flag": ["pii"]
            },
            {
                "target_field": "litigation_flag",
                "source_field": "Litigation Flag",
                "type": "string",
                "required": False
            }
        ]
    },
    {
        "id": "source_xlsx_claims",
        "metadata": {
            "source_name": "Excel Claims Export",
            "source_type": "xlsx_file",
            "connection_type": "file_upload",
            "notes": "Claims data from Excel export file",
            "connection_config": {
                "download_link": "",
                "sheet_name": "Claims",
                "data_range": "A1:Z1000"
            }
        },
        "mappings": [
            {
                "target_field": "claim_number",
                "source_field": "CLAIM_NUMBER",
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
                "target_field": "loss_date",
                "source_field": "LOSS_DATE",
                "type": "date",
                "required": False
            },
            {
                "target_field": "report_date",
                "source_field": "REPORT_DATE",
                "type": "date",
                "required": False
            },
            {
                "target_field": "cause_of_loss",
                "source_field": "CAUSE_OF_LOSS",
                "type": "string",
                "required": False
            },
            {
                "target_field": "loss_description",
                "source_field": "LOSS_DESCRIPTION",
                "type": "string",
                "required": False
            },
            {
                "target_field": "claim_status",
                "source_field": "CLAIM_STATUS",
                "type": "string",
                "required": False
            },
            {
                "target_field": "open_closed",
                "source_field": "OPEN_CLOSED",
                "type": "string",
                "required": False
            },
            {
                "target_field": "total_incurred",
                "source_field": "TOTAL_INCURRED",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "paid",
                "source_field": "PAID",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "reserved",
                "source_field": "RESERVED",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "adjuster_name",
                "source_field": "ADJUSTER_NAME",
                "type": "string",
                "required": False
            },
            {
                "target_field": "adjuster_email",
                "source_field": "ADJUSTER_EMAIL",
                "type": "email",
                "required": False,
                "flag": ["pii"]
            },
            {
                "target_field": "litigation_flag",
                "source_field": "LITIGATION_FLAG",
                "type": "string",
                "required": False
            }
        ]
    },
    {
        "id": "source_pdf_claims",
        "metadata": {
            "source_name": "PDF Claims Documents",
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
                "target_field": "claim_number",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Extract the claim number from the document. Look for labels like 'Claim Number', 'Claim #', 'Claim ID', 'Claim', or claim identifiers typically displayed prominently on claim documents, loss reports, or claim headers."
            },
            {
                "target_field": "policy_number",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the policy number associated with this claim from the document. Look for labels like 'Policy Number', 'Policy #', 'Policy ID', 'Policy', or policy identifiers near claim information."
            },
            {
                "target_field": "loss_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the loss date from the document. Look for labels like 'Loss Date', 'Date of Loss', 'Loss Occurred', 'Incident Date', or dates representing when the loss or incident occurred."
            },
            {
                "target_field": "report_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the report date from the document. Look for labels like 'Report Date', 'Date Reported', 'Reported On', 'Filing Date', or dates representing when the claim was reported."
            },
            {
                "target_field": "cause_of_loss",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the cause of loss from the document. Look for labels like 'Cause of Loss', 'Loss Cause', 'Cause', 'Type of Loss', or descriptions of what caused the loss (e.g., 'Rear-end collision', 'Single vehicle rollover', 'Theft', 'Fire')."
            },
            {
                "target_field": "loss_description",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the loss description from the document. Look for labels like 'Loss Description', 'Description', 'Loss Details', 'Narrative', or detailed descriptions of the loss or incident."
            },
            {
                "target_field": "claim_status",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Open", "Closed", "Pending", "Denied", "open", "closed", "pending", "denied", ""]
                },
                "ai_instruction": "Extract the claim status from the document. Look for labels like 'Claim Status', 'Status', 'Current Status', or status indicators. Common values include Open, Closed, Pending, or Denied."
            },
            {
                "target_field": "total_incurred",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the total incurred amount from the document. Look for labels like 'Total Incurred', 'Incurred', 'Total Loss', or total claim amounts (numeric values, may include currency symbols). Remove currency symbols and return only the numeric value."
            },
            {
                "target_field": "paid",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the paid amount from the document. Look for labels like 'Paid', 'Amount Paid', 'Paid to Date', or paid claim amounts (numeric values, may include currency symbols). Remove currency symbols and return only the numeric value."
            },
            {
                "target_field": "reserved",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the reserved amount from the document. Look for labels like 'Reserved', 'Reserve', 'Outstanding Reserve', or reserved claim amounts (numeric values, may include currency symbols). Remove currency symbols and return only the numeric value."
            },
            {
                "target_field": "adjuster_name",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the adjuster name from the document. Look for labels like 'Adjuster Name', 'Adjuster', 'Assigned Adjuster', 'Claims Adjuster', or adjuster names in claim information sections."
            },
            {
                "target_field": "adjuster_email",
                "source_field": "",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                "ai_instruction": "Extract the adjuster's email address from the document. Look for labels like 'Adjuster Email', 'Email', 'Contact Email', or email addresses near adjuster information. Return the email in lowercase format."
            },
            {
                "target_field": "litigation_flag",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Yes", "No", "yes", "no", "True", "False", "true", "false", ""]
                },
                "ai_instruction": "Determine if litigation is involved in this claim. Look for labels like 'Litigation Flag', 'Litigation', 'In Litigation', 'Legal Action', or indicators like 'Yes', 'No', 'True', 'False'. If not explicitly stated, infer from context (e.g., mentions of lawsuits, legal proceedings)."
            }
        ]
    },
    {
        "id": "source_raw_text_claims",
        "metadata": {
            "source_name": "Raw Text Claims Data",
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
                "target_field": "claim_number",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Search the text for a claim number. It may be mentioned explicitly with keywords like 'Claim Number:', 'Claim #:', 'Claim:', 'Claim ID:', or may appear as a unique identifier associated with claim information."
            },
            {
                "target_field": "policy_number",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Search the text for a policy number associated with this claim. It may be mentioned explicitly with keywords like 'Policy Number:', 'Policy #:', 'Policy:', or policy identifiers near claim information."
            },
            {
                "target_field": "loss_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the loss date from the text. Look for dates mentioned with keywords like 'Loss Date:', 'Date of Loss:', 'Loss Occurred:', or dates representing when the loss or incident occurred."
            },
            {
                "target_field": "report_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the report date from the text. Look for dates mentioned with keywords like 'Report Date:', 'Date Reported:', 'Reported On:', or dates representing when the claim was reported."
            },
            {
                "target_field": "cause_of_loss",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the cause of loss from the text. Look for descriptions mentioned with keywords like 'Cause of Loss:', 'Loss Cause:', 'Cause:', or descriptions of what caused the loss (e.g., 'collision', 'rollover', 'theft', 'fire')."
            },
            {
                "target_field": "loss_description",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the loss description from the text. Look for descriptions mentioned with keywords like 'Loss Description:', 'Description:', 'Loss Details:', or detailed narrative descriptions of the loss or incident."
            },
            {
                "target_field": "claim_status",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Open", "Closed", "Pending", "Denied", "open", "closed", "pending", "denied", ""]
                },
                "ai_instruction": "Identify the claim status from the text. Look for status indicators mentioned with keywords like 'Status:', 'Claim Status:', or status descriptions. Common values include Open, Closed, Pending, or Denied."
            },
            {
                "target_field": "total_incurred",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the total incurred amount from the text. Look for amounts mentioned with keywords like 'Total Incurred:', 'Incurred:', or total claim amounts (numeric values, may include currency symbols or commas). Remove currency symbols and commas, return only the numeric value."
            },
            {
                "target_field": "paid",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the paid amount from the text. Look for amounts mentioned with keywords like 'Paid:', 'Amount Paid:', or paid claim amounts (numeric values, may include currency symbols or commas). Remove currency symbols and commas, return only the numeric value."
            },
            {
                "target_field": "reserved",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the reserved amount from the text. Look for amounts mentioned with keywords like 'Reserved:', 'Reserve:', or reserved claim amounts (numeric values, may include currency symbols or commas). Remove currency symbols and commas, return only the numeric value."
            },
            {
                "target_field": "adjuster_name",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the adjuster name from the text. Look for names mentioned with keywords like 'Adjuster Name:', 'Adjuster:', 'Assigned Adjuster:', or adjuster names in claim information."
            },
            {
                "target_field": "adjuster_email",
                "source_field": "",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                "ai_instruction": "Extract the adjuster's email address from the text. Look for email addresses mentioned with keywords like 'Adjuster Email:', 'Email:', or email addresses near adjuster information. Return the email in lowercase format."
            },
            {
                "target_field": "litigation_flag",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Yes", "No", "yes", "no", "True", "False", "true", "false", ""]
                },
                "ai_instruction": "Determine if litigation is involved in this claim. Look for indicators mentioned with keywords like 'Litigation Flag:', 'Litigation:', 'In Litigation:', or mentions of lawsuits, legal action, or legal proceedings. Return 'Yes' if litigation is mentioned, 'No' otherwise."
            }
        ]
    },
    {
        "id": "source_image_claims",
        "metadata": {
            "source_name": "Claims Image Analysis",
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
                "target_field": "claim_number",
                "source_field": "",
                "type": "string",
                "required": True,
                "ai_instruction": "Analyze the image for a claim number. The claim number may appear: on claim documents, on loss reports, on claim forms, or in any claim information visible in the image. Use OCR to extract the claim identifier, which is typically displayed prominently."
            },
            {
                "target_field": "policy_number",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the policy number associated with this claim from the image. Look for: 1) Policy numbers on claim documents or loss reports. 2) Policy identifiers on claim forms. 3) Policy numbers in claim information sections visible in the image."
            },
            {
                "target_field": "loss_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the loss date from the image. Look for: 1) Loss date fields on claim documents or loss reports. 2) Dates labeled as 'Loss Date', 'Date of Loss', or 'Loss Occurred' on claim forms. 3) Dates in loss information sections visible in the image."
            },
            {
                "target_field": "report_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the report date from the image. Look for: 1) Report date fields on claim documents. 2) Dates labeled as 'Report Date', 'Date Reported', or 'Reported On' on claim forms. 3) Dates in report information sections visible in the image."
            },
            {
                "target_field": "cause_of_loss",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the cause of loss from the image. Look for: 1) Cause of loss fields on claim documents or loss reports. 2) Descriptions labeled as 'Cause of Loss', 'Loss Cause', or 'Cause' on claim forms. 3) Cause descriptions in loss information sections visible in the image."
            },
            {
                "target_field": "loss_description",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the loss description from the image. Look for: 1) Description fields on claim documents or loss reports. 2) Descriptions labeled as 'Loss Description', 'Description', or 'Narrative' on claim forms. 3) Detailed descriptions in loss information sections visible in the image."
            },
            {
                "target_field": "claim_status",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Open", "Closed", "Pending", "Denied", "open", "closed", "pending", "denied", ""]
                },
                "ai_instruction": "Determine the claim status from the image. Look for: 1) Status fields on claim documents. 2) Status indicators like 'Open', 'Closed', 'Pending', 'Denied' on claim forms. 3) Status information in claim headers or summary sections visible in the image."
            },
            {
                "target_field": "total_incurred",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the total incurred amount from the image. Look for: 1) Total incurred fields on claim documents. 2) Amounts labeled as 'Total Incurred', 'Incurred', or 'Total Loss' on claim forms. 3) Total amounts in financial sections visible in the image. Remove currency symbols and return only the numeric value."
            },
            {
                "target_field": "paid",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the paid amount from the image. Look for: 1) Paid amount fields on claim documents. 2) Amounts labeled as 'Paid', 'Amount Paid', or 'Paid to Date' on claim forms. 3) Paid amounts in financial sections visible in the image. Remove currency symbols and return only the numeric value."
            },
            {
                "target_field": "reserved",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the reserved amount from the image. Look for: 1) Reserved amount fields on claim documents. 2) Amounts labeled as 'Reserved', 'Reserve', or 'Outstanding Reserve' on claim forms. 3) Reserved amounts in financial sections visible in the image. Remove currency symbols and return only the numeric value."
            },
            {
                "target_field": "adjuster_name",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the adjuster name from the image. Look for: 1) Adjuster name fields on claim documents. 2) Names labeled as 'Adjuster Name', 'Adjuster', or 'Assigned Adjuster' on claim forms. 3) Adjuster names in claim information sections visible in the image."
            },
            {
                "target_field": "adjuster_email",
                "source_field": "",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                "ai_instruction": "Extract the adjuster's email address from the image. Look for: 1) Email addresses on claim documents or claim forms. 2) Email fields labeled as 'Adjuster Email', 'Email', or 'Contact Email' on claim forms. 3) Email addresses near adjuster information. Use OCR to extract the email address and return it in lowercase format."
            },
            {
                "target_field": "litigation_flag",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Yes", "No", "yes", "no", "True", "False", "true", "false", ""]
                },
                "ai_instruction": "Determine if litigation is involved in this claim from the image. Look for: 1) Litigation flag fields on claim documents. 2) Indicators labeled as 'Litigation Flag', 'Litigation', or 'In Litigation' on claim forms. 3) Mentions of lawsuits or legal action in claim information. Return 'Yes' if litigation is mentioned, 'No' otherwise."
            }
        ]
    }
]
