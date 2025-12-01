"""
Policy data mappings from multiple data sources to standardized schema
Each source represents a different vendor, system, or data provider
"""

POLICIES_MAPPINGS = [
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
            # Policy Number - multiple variants (CSV: "Policy Number", Airtable: "Policy #")
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
                "target_field": "policy_number",
                "source_field": "Policy #",
                "type": "string",
                "required": False,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                }
            },
            {
                "target_field": "carrier",
                "source_field": "Carrier",
                "type": "string",
                "required": False
            },
            {
                "target_field": "product",
                "source_field": "Product",
                "type": "string",
                "required": False
            },
            # Line of Business - multiple variants (CSV: "Line of Business", Airtable: "LOB")
            {
                "target_field": "line_of_business",
                "source_field": "Line of Business",
                "type": "string",
                "required": False
            },
            {
                "target_field": "line_of_business",
                "source_field": "LOB",
                "type": "string",
                "required": False
            },
            {
                "target_field": "named_insured",
                "source_field": "Named Insured",
                "type": "string",
                "required": False
            },
            # Effective Date - multiple variants (CSV: "Effective Date", Airtable: "Effective")
            {
                "target_field": "effective_date",
                "source_field": "Effective Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "effective_date",
                "source_field": "Effective",
                "type": "date",
                "required": False
            },
            # Expiration Date - multiple variants (CSV: "Expiration Date", Airtable: "Expiration")
            {
                "target_field": "expiration_date",
                "source_field": "Expiration Date",
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
                "target_field": "status",
                "source_field": "Status",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Active", "Cancelled", "Expired", "Quoted", "Pending", "active", "cancelled", "expired", "quoted", "pending", ""]
                }
                # "transform": "lowercase"  # Transform logic will be implemented later
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
            # Term Months - multiple variants (CSV: "Term (months)", Airtable: "Term Months")
            {
                "target_field": "term_months",
                "source_field": "Term (months)",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 36
                }
            },
            {
                "target_field": "term_months",
                "source_field": "Term Months",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 36
                }
            },
            {
                "target_field": "billing_plan",
                "source_field": "Billing Plan",
                "type": "string",
                "required": False
            },
            {
                "target_field": "payment_method",
                "source_field": "Payment Method",
                "type": "string",
                "required": False
            },
            # Policy Premium - multiple variants (CSV: "Policy Premium", Airtable: "Premium")
            {
                "target_field": "policy_premium",
                "source_field": "Policy Premium",
                "type": "decimal",
                "required": False,
                "validation": {
                    "min": 0
                }
            },
            {
                "target_field": "policy_premium",
                "source_field": "Premium",
                "type": "decimal",
                "required": False,
                "validation": {
                    "min": 0
                }
            },
            {
                "target_field": "fees",
                "source_field": "Fees",
                "type": "decimal",
                "required": False
            },
            # Total Written Premium - multiple variants (CSV: "Total Written Premium", Airtable: "Total Written")
            {
                "target_field": "total_written_premium",
                "source_field": "Total Written Premium",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "total_written_premium",
                "source_field": "Total Written",
                "type": "decimal",
                "required": False
            },
            # Number of Vehicles - multiple variants (CSV: "Number of Vehicles", Airtable: "Vehicle Count")
            {
                "target_field": "number_of_vehicles",
                "source_field": "Number of Vehicles",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0
                }
            },
            {
                "target_field": "number_of_vehicles",
                "source_field": "Vehicle Count",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0
                }
            },
            # Number of Drivers - multiple variants (CSV: "Number of Drivers", Airtable: "Driver Count")
            {
                "target_field": "number_of_drivers",
                "source_field": "Number of Drivers",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0
                }
            },
            {
                "target_field": "number_of_drivers",
                "source_field": "Driver Count",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0
                }
            },
            # Agency Name - multiple variants (CSV: "Agency Name", Airtable: "Agency")
            {
                "target_field": "agency_name",
                "source_field": "Agency Name",
                "type": "string",
                "required": False
            },
            {
                "target_field": "agency_name",
                "source_field": "Agency",
                "type": "string",
                "required": False
            },
            {
                "target_field": "producer_code",
                "source_field": "Producer Code",
                "type": "string",
                "required": False
            },
            # Underwriting Company - multiple variants (CSV: "Underwriting Company", Airtable: "UW Company")
            {
                "target_field": "underwriting_company",
                "source_field": "Underwriting Company",
                "type": "string",
                "required": False
            },
            {
                "target_field": "underwriting_company",
                "source_field": "UW Company",
                "type": "string",
                "required": False
            },
            # Cancellation Date - multiple variants (CSV: "Cancellation Date", Airtable: "Cancel Date")
            {
                "target_field": "cancellation_date",
                "source_field": "Cancellation Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "cancellation_date",
                "source_field": "Cancel Date",
                "type": "date",
                "required": False
            },
            # Cancellation Reason - multiple variants (CSV: "Cancellation Reason", Airtable: "Cancel Reason")
            {
                "target_field": "cancellation_reason",
                "source_field": "Cancellation Reason",
                "type": "string",
                "required": False
            },
            {
                "target_field": "cancellation_reason",
                "source_field": "Cancel Reason",
                "type": "string",
                "required": False
            }
        ]
    },
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
                "target_field": "carrier",
                "source_field": "Carrier",
                "type": "string",
                "required": False
            },
            {
                "target_field": "product",
                "source_field": "Product",
                "type": "string",
                "required": False
            },
            {
                "target_field": "line_of_business",
                "source_field": "LOB",
                "type": "string",
                "required": False
            },
            {
                "target_field": "named_insured",
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
                "target_field": "status",
                "source_field": "Status",
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
                "target_field": "term_months",
                "source_field": "Term Months",
                "type": "integer",
                "required": False
            },
            {
                "target_field": "billing_plan",
                "source_field": "Billing Plan",
                "type": "string",
                "required": False
            },
            {
                "target_field": "payment_method",
                "source_field": "Payment Method",
                "type": "string",
                "required": False
            },
            {
                "target_field": "policy_premium",
                "source_field": "Premium",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "fees",
                "source_field": "Fees",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "total_written_premium",
                "source_field": "Total Written",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "number_of_vehicles",
                "source_field": "Vehicle Count",
                "type": "integer",
                "required": False
            },
            {
                "target_field": "number_of_drivers",
                "source_field": "Driver Count",
                "type": "integer",
                "required": False
            },
            {
                "target_field": "agency_name",
                "source_field": "Agency",
                "type": "string",
                "required": False
            },
            {
                "target_field": "producer_code",
                "source_field": "Producer Code",
                "type": "string",
                "required": False
            },
            {
                "target_field": "underwriting_company",
                "source_field": "UW Company",
                "type": "string",
                "required": False
            },
            {
                "target_field": "cancellation_date",
                "source_field": "Cancel Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "cancellation_reason",
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
                "source_field": "POLICY_NUMBER",
                "type": "string",
                "required": True
            },
            {
                "target_field": "carrier",
                "source_field": "CARRIER",
                "type": "string",
                "required": False
            },
            {
                "target_field": "product",
                "source_field": "PRODUCT",
                "type": "string",
                "required": False
            },
            {
                "target_field": "line_of_business",
                "source_field": "LINE_OF_BUSINESS",
                "type": "string",
                "required": False
            },
            {
                "target_field": "named_insured",
                "source_field": "NAMED_INSURED",
                "type": "string",
                "required": False
            },
            {
                "target_field": "effective_date",
                "source_field": "EFFECTIVE_DATE",
                "type": "date",
                "required": False
            },
            {
                "target_field": "expiration_date",
                "source_field": "EXPIRATION_DATE",
                "type": "date",
                "required": False
            },
            {
                "target_field": "status",
                "source_field": "STATUS",
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
                "target_field": "term_months",
                "source_field": "TERM_MONTHS",
                "type": "integer",
                "required": False
            },
            {
                "target_field": "billing_plan",
                "source_field": "BILLING_PLAN",
                "type": "string",
                "required": False
            },
            {
                "target_field": "payment_method",
                "source_field": "PAYMENT_METHOD",
                "type": "string",
                "required": False
            },
            {
                "target_field": "policy_premium",
                "source_field": "POLICY_PREMIUM",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "fees",
                "source_field": "FEES",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "total_written_premium",
                "source_field": "TOTAL_WRITTEN_PREMIUM",
                "type": "decimal",
                "required": False
            },
            {
                "target_field": "number_of_vehicles",
                "source_field": "NUMBER_OF_VEHICLES",
                "type": "integer",
                "required": False
            },
            {
                "target_field": "number_of_drivers",
                "source_field": "NUMBER_OF_DRIVERS",
                "type": "integer",
                "required": False
            },
            {
                "target_field": "agency_name",
                "source_field": "AGENCY_NAME",
                "type": "string",
                "required": False
            },
            {
                "target_field": "producer_code",
                "source_field": "PRODUCER_CODE",
                "type": "string",
                "required": False
            },
            {
                "target_field": "underwriting_company",
                "source_field": "UNDERWRITING_COMPANY",
                "type": "string",
                "required": False
            },
            {
                "target_field": "cancellation_date",
                "source_field": "CANCELLATION_DATE",
                "type": "date",
                "required": False
            },
            {
                "target_field": "cancellation_reason",
                "source_field": "CANCELLATION_REASON",
                "type": "string",
                "required": False
            }
        ]
    },
    {
        "id": "source_pdf_policies",
        "metadata": {
            "source_name": "PDF Policy Documents",
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
                "ai_instruction": "Extract the policy number from the document. Look for labels like 'Policy Number', 'Policy #', 'Policy ID', 'Policy', or policy identifiers typically displayed prominently on policy documents, declarations pages, or policy headers."
            },
            {
                "target_field": "carrier",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the insurance carrier or company name from the document. Look for labels like 'Carrier', 'Insurance Company', 'Company', 'Insurer', or carrier names in document headers, letterheads, or company information sections."
            },
            {
                "target_field": "product",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the insurance product type from the document. Look for labels like 'Product', 'Product Type', 'Coverage Type', or product descriptions (e.g., 'Personal Auto', 'Commercial Auto', 'High-Risk Auto') in policy information sections."
            },
            {
                "target_field": "named_insured",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the named insured from the document. Look for labels like 'Named Insured', 'Insured', 'Policyholder', 'Account Holder', or names of individuals or entities listed as the primary insured party."
            },
            {
                "target_field": "effective_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the policy effective date from the document. Look for labels like 'Effective Date', 'Effective', 'Policy Start Date', 'Inception Date', or dates labeled as the start of coverage period."
            },
            {
                "target_field": "expiration_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the policy expiration date from the document. Look for labels like 'Expiration Date', 'Expiration', 'Policy End Date', 'Expiry Date', or dates labeled as the end of coverage period."
            },
            {
                "target_field": "status",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Active", "Cancelled", "Expired", "Quoted", "Pending", "active", "cancelled", "expired", "quoted", "pending", ""]
                },
                "ai_instruction": "Extract the policy status from the document. Look for labels like 'Status', 'Policy Status', 'Current Status', or status indicators. Common values include Active, Cancelled, Expired, Quoted, or Pending."
            },
            {
                "target_field": "state",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the state where the policy is issued from the document. Look for labels like 'State', 'Policy State', 'Issuing State', or state abbreviations (2 letters) in policy information sections."
            },
            {
                "target_field": "policy_premium",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the policy premium amount from the document. Look for labels like 'Premium', 'Policy Premium', 'Annual Premium', 'Total Premium', or premium amounts (numeric values, may include currency symbols). Remove currency symbols and return only the numeric value."
            },
            {
                "target_field": "total_written_premium",
                "source_field": "",
                "type": "decimal",
                "required": False,
                "ai_instruction": "Extract the total written premium from the document. Look for labels like 'Total Written Premium', 'Total Premium', 'Written Premium', or total premium amounts. Remove currency symbols and return only the numeric value."
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
                "ai_instruction": "Search the text for a policy number. It may be mentioned explicitly with keywords like 'Policy Number:', 'Policy #:', 'Policy:', 'Policy ID:', or may appear as a unique identifier associated with policy information."
            },
            {
                "target_field": "carrier",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Identify the insurance carrier or company name from the text. Look for carrier names mentioned with keywords like 'Carrier:', 'Insurance Company:', 'Company:', or insurance company names in policy descriptions."
            },
            {
                "target_field": "product",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Identify the insurance product type from the text. Look for product descriptions mentioned with keywords like 'Product:', 'Product Type:', or product names (e.g., 'Personal Auto', 'Commercial Auto') in policy descriptions."
            },
            {
                "target_field": "named_insured",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the named insured from the text. Look for names mentioned with keywords like 'Named Insured:', 'Insured:', 'Policyholder:', or names of individuals or entities listed as the primary insured party."
            },
            {
                "target_field": "effective_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the policy effective date from the text. Look for dates mentioned with keywords like 'Effective Date:', 'Effective:', 'Policy Start:', or dates representing the start of coverage period."
            },
            {
                "target_field": "expiration_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the policy expiration date from the text. Look for dates mentioned with keywords like 'Expiration Date:', 'Expiration:', 'Policy End:', or dates representing the end of coverage period."
            },
            {
                "target_field": "status",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Active", "Cancelled", "Expired", "Quoted", "Pending", "active", "cancelled", "expired", "quoted", "pending", ""]
                },
                "ai_instruction": "Identify the policy status from the text. Look for status indicators mentioned with keywords like 'Status:', 'Policy Status:', or status descriptions. Common values include Active, Cancelled, Expired, Quoted, or Pending."
            },
            {
                "target_field": "state",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the state where the policy is issued from the text. Look for state abbreviations (2 letters like 'CA', 'TX') or state names mentioned with keywords like 'State:', 'Policy State:', or in policy location information."
            }
        ]
    },
    {
        "id": "source_image_policies",
        "metadata": {
            "source_name": "Policy Image Analysis",
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
                "ai_instruction": "Analyze the image for a policy number. The policy number may appear: on policy declarations pages, on policy documents, on ID cards, or in any policy information visible in the image. Use OCR to extract the policy identifier, which is typically displayed prominently."
            },
            {
                "target_field": "carrier",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Identify the insurance carrier from the image. Look for: 1) Company names on policy documents or letterheads. 2) Carrier logos or branding visible in the image. 3) Insurance company names in document headers or company information sections."
            },
            {
                "target_field": "named_insured",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the named insured from the image. Look for: 1) Names on policy declarations pages. 2) Named insured fields on policy documents. 3) Policyholder names in document headers or policy information sections."
            },
            {
                "target_field": "effective_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the policy effective date from the image. Look for: 1) Effective date fields on policy declarations pages. 2) Dates labeled as 'Effective Date', 'Effective', or 'Policy Start Date' on policy documents. 3) Start dates in policy period information."
            },
            {
                "target_field": "expiration_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the policy expiration date from the image. Look for: 1) Expiration date fields on policy declarations pages. 2) Dates labeled as 'Expiration Date', 'Expiration', or 'Policy End Date' on policy documents. 3) End dates in policy period information."
            },
            {
                "target_field": "status",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["Active", "Cancelled", "Expired", "Quoted", "Pending", "active", "cancelled", "expired", "quoted", "pending", ""]
                },
                "ai_instruction": "Determine the policy status from the image. Look for: 1) Status fields on policy documents. 2) Status indicators like 'Active', 'Cancelled', 'Expired' on declarations pages. 3) Status information in policy headers or summary sections."
            }
        ]
    }
]
