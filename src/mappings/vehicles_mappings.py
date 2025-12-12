"""
Vehicle/Car data mappings from multiple data sources to standardized schema

Each source represents a different vendor, system, or data provider
"""

VEHICLES_MAPPINGS = [
    {
        "id": "source_google_sheet_vehicles",
        "metadata": {
            "source_name": "Google Sheets Vehicle Inventory",
            "source_type": "google_sheet",
            "connection_type": "google_api",
            "notes": "Collaborative spreadsheet, manually maintained by sales team",
            "connection_config": {
                "spreadsheet_id": "1rzGjivKD0-FoaXdDzQGeFtPrGRH3ALA7zECfYYgwlc8",
                "sheet_name": "Sheet1",
                "data_range": "A1:Z1000"
            }
        },
        "mappings": [
            # VIN - multiple variants (CSV: "VIN Number", Airtable: "VIN")
            {
                "target_field": "vin",
                "source_field": "VIN Number",
                "type": "string",
                "required": True,
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                },
                "transform": "uppercase"
            },
            {
                "target_field": "vin",
                "source_field": "VIN",
                "type": "string",
                "required": False,  # Not required for this variant since first one is required
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                },
            },
            {
                "target_field": "vehicle_id",
                "source_field": "Vehicle ID",
                "type": "string",
                "required": False
            },
            # Year - multiple variants (CSV: "Year", Airtable: "Model Year")
            {
                "target_field": "year",
                "source_field": "Year",
                "type": "integer",
                "required": True,
                "validation": {
                    "min": 1900,
                    "max": 2030
                }
            },
            {
                "target_field": "year",
                "source_field": "Model Year",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 1900,
                    "max": 2030
                }
            },
            # Make - multiple variants (CSV: "Make", Airtable: "Vehicle Make")
            {
                "target_field": "make",
                "source_field": "Make",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 2,
                    "max_length": 50
                }
            },
            {
                "target_field": "make",
                "source_field": "Vehicle Make",
                "type": "string",
                "required": False,
                "validation": {
                    "min_length": 2,
                    "max_length": 50
                }
            },
            # Model - multiple variants (CSV: "Model", Airtable: "Vehicle Model")
            {
                "target_field": "model",
                "source_field": "Model",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                }
            },
            {
                "target_field": "model",
                "source_field": "Vehicle Model",
                "type": "string",
                "required": False,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                }
            },
            {
                "target_field": "effective_date",
                "source_field": "Effective Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "notes",
                "source_field": "Notes",
                "type": "string",
                "required": False
            },
            # Color - multiple variants (CSV: "Exterior Color", Airtable: "Color")
            {
                "target_field": "color",
                "source_field": "Exterior Color",
                "type": "string",
                "required": False,
                "validation": {
                    "max_length": 30
                }
            },
            {
                "target_field": "color",
                "source_field": "Color",
                "type": "string",
                "required": False,
                "validation": {
                    "max_length": 30
                }
            },
            # Mileage - multiple variants (CSV: "Mileage", Airtable: "Current Mileage")
            {
                "target_field": "mileage",
                "source_field": "Mileage",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 999999
                }
            },
            {
                "target_field": "mileage",
                "source_field": "Current Mileage",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 999999
                }
            },
            # Trim - multiple variants (CSV: "Trim Level", Airtable: "Trim")
            {
                "target_field": "trim",
                "source_field": "Trim Level",
                "type": "string",
                "required": False
            },
            {
                "target_field": "trim",
                "source_field": "Trim",
                "type": "string",
                "required": False
            },
            # Body Style - multiple variants (CSV: "Body Style", Airtable: "Body Type")
            {
                "target_field": "body_style",
                "source_field": "Body Style",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["sedan", "coupe", "suv", "truck", "van", "convertible", "wagon", "hatchback", "crossover", "boat"]
                }
            },
            {
                "target_field": "body_style",
                "source_field": "Body Type",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["sedan", "coupe", "suv", "truck", "van", "convertible", "wagon", "hatchback", "crossover", "boat"]
                }
            },
            # Fuel Type - multiple variants (CSV: "Fuel Type", Airtable: "Fuel")
            {
                "target_field": "fuel_type",
                "source_field": "Fuel Type",
                "type": "string",
                "required": False,
                "transform": "standardize_fuel_type",
                "validation": {
                    "enum": ["gasoline", "diesel", "electric", "hybrid", "plug-in hybrid", "Gas"]
                }
            },
            {
                "target_field": "fuel_type",
                "source_field": "Fuel",
                "type": "string",
                "required": False,
                "transform": "standardize_fuel_type",
                "validation": {
                    "enum": ["gasoline", "diesel", "electric", "hybrid", "plug-in hybrid", "Gas"]
                }
            },
            # Transmission - multiple variants (CSV: "Transmission", Airtable: "Transmission Type")
            {
                "target_field": "transmission",
                "source_field": "Transmission",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["automatic", "manual", "cvt", "AUTO"]
                }
            },
            {
                "target_field": "transmission",
                "source_field": "Transmission Type",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["automatic", "manual", "cvt", "AUTO"]
                }
            },
            {
                "target_field": "owner_email",
                "source_field": "Owner Email",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                "transform": "lowercase"
            },
            {
                "target_field": "weight",
                "source_field": "Weight",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 99999
                }
            }
        ]
    },
    {
        "id": "source_airtable_vehicles",
        "metadata": {
            "source_name": "Airtable Fleet Vehicles",
            "source_type": "airtable",
            "connection_type": "rest_api",
            "notes": "Well-structured database with rich field types and validations",
            "connection_config": {
                "base_id": "UNKNOWN_BASE_ID",
                "table_name": "Fleet Vehicles",
                "view_name": "All Vehicles"
            }
        },
        "mappings": [
            {
                "target_field": "vin",
                "source_field": "VIN",
                "type": "string",
                "required": True,
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                },
                "transform": "uppercase"
            },
            {
                "target_field": "vehicle_id",
                "source_field": "Vehicle ID",
                "type": "string",
                "required": False
            },
            {
                "target_field": "year",
                "source_field": "Model Year",
                "type": "integer",
                "required": True,
                "validation": {
                    "min": 1900,
                    "max": 2030
                }
            },
            {
                "target_field": "make",
                "source_field": "Vehicle Make",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 2,
                    "max_length": 50
                }
            },
            {
                "target_field": "model",
                "source_field": "Vehicle Model",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                }
            },
            {
                "target_field": "effective_date",
                "source_field": "Effective Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "notes",
                "source_field": "Notes",
                "type": "string",
                "required": False
            },
            {
                "target_field": "trim",
                "source_field": "Trim",
                "type": "string",
                "required": False
            },
            # Color
            {
                "target_field": "color",
                "source_field": "Color",
                "type": "string",
                "required": False,
                "validation": {
                    "max_length": 30
                }
            },
            # Mileage
            {
                "target_field": "mileage",
                "source_field": "Current Mileage",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 999999
                }
            },
            # Body Style
            {
                "target_field": "body_style",
                "source_field": "Body Type",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["sedan", "coupe", "suv", "truck", "van", "convertible", "wagon", "hatchback", "crossover", "boat"]
                }
            },
            # Fuel Type
            {
                "target_field": "fuel_type",
                "source_field": "Fuel",
                "type": "string",
                "required": False,
                "transform": "standardize_fuel_type",
                "validation": {
                    "enum": ["gasoline", "diesel", "electric", "hybrid", "plug-in hybrid", "Gas"]
                }
            },
            # Transmission
            {
                "target_field": "transmission",
                "source_field": "Transmission Type",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["automatic", "manual", "cvt", "AUTO"]
                }
            },
            # Owner Email
            {
                "target_field": "owner_email",
                "source_field": "Owner Email",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                "transform": "lowercase"
            },
            {
                "target_field": "weight",
                "source_field": "Weight",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 99999
                }
            }
        ]
    },
    {
        "id": "source_xlsx_vehicles",
        "metadata": {
            "source_name": "Excel Vehicle Export",
            "source_type": "xlsx_file",
            "connection_type": "file_upload",
            "notes": "Batch export file, column headers may vary slightly between exports",
            "connection_config": {
                "download_link": "",
                "sheet_name": "Vehicle Data",
                "data_range": "A1:Z1000"
            }
        },
        "mappings": [
            # VIN - multiple variants (Excel: "VIN Number")
            {
                "target_field": "vin",
                "source_field": "VIN Number",
                "type": "string",
                "required": True,
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                },
                "transform": "uppercase"
            },
            {
                "target_field": "vehicle_id",
                "source_field": "Vehicle ID",
                "type": "string",
                "required": False
            },
            # Year
            {
                "target_field": "year",
                "source_field": "Year",
                "type": "integer",
                "required": True,
                "validation": {
                    "min": 1900,
                    "max": 2030
                }
            },
            # Make
            {
                "target_field": "make",
                "source_field": "Make",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 2,
                    "max_length": 50
                }
            },
            # Model
            {
                "target_field": "model",
                "source_field": "Model",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                }
            },
            {
                "target_field": "effective_date",
                "source_field": "Effective Date",
                "type": "date",
                "required": False
            },
            {
                "target_field": "notes",
                "source_field": "Notes",
                "type": "string",
                "required": False
            },
            # Color
            {
                "target_field": "color",
                "source_field": "Exterior Color",
                "type": "string",
                "required": False,
                "validation": {
                    "max_length": 30
                }
            },
            # Mileage
            {
                "target_field": "mileage",
                "source_field": "Mileage",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 999999
                }
            },
            # Trim
            {
                "target_field": "trim",
                "source_field": "Trim Level",
                "type": "string",
                "required": False
            },
            # Body Style
            {
                "target_field": "body_style",
                "source_field": "Body Style",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["sedan", "coupe", "suv", "truck", "van", "convertible", "wagon", "hatchback", "crossover", "boat"]
                }
            },
            # Fuel Type
            {
                "target_field": "fuel_type",
                "source_field": "Fuel Type",
                "type": "string",
                "required": False,
                "transform": "standardize_fuel_type",
                "validation": {
                    "enum": ["gasoline", "diesel", "electric", "hybrid", "plug-in hybrid", "Gas"]
                }
            },
            # Transmission
            {
                "target_field": "transmission",
                "source_field": "Transmission",
                "type": "string",
                "required": False,
                "validation": {
                    "enum": ["automatic", "manual", "cvt", "AUTO"]
                }
            },
            {
                "target_field": "owner_email",
                "source_field": "Owner Email",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                "transform": "lowercase"
            },
            {
                "target_field": "weight",
                "source_field": "Weight",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 99999
                }
            }
        ]
    },
    {
        "id": "source_pdf_vehicles",
        "metadata": {
            "source_name": "PDF Vehicle Documents",
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
                "target_field": "vin",
                "source_field": "",
                "type": "string",
                "required": True,
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                },
                "transform": "uppercase",
                "ai_instruction": "Extract the Vehicle Identification Number (VIN) from the document. Look for a 17-character alphanumeric code, often labeled as 'VIN', 'VIN Number', 'Vehicle Identification Number', or found near vehicle details. The VIN excludes the letters I, O, and Q to avoid confusion with numbers."
            },
            {
                "target_field": "year",
                "source_field": "",
                "type": "integer",
                "required": True,
                "validation": {
                    "min": 1900,
                    "max": 2030
                },
                "ai_instruction": "Extract the vehicle model year as a 4-digit number. Look for labels like 'Year', 'Model Year', 'Year of Manufacture', or near the make and model information."
            },
            {
                "target_field": "make",
                "source_field": "",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 2,
                    "max_length": 50
                },
                "ai_instruction": "Extract the vehicle manufacturer or make (e.g., Toyota, Ford, Honda, BMW). Look for labels like 'Make', 'Manufacturer', 'Brand', or in the vehicle description section."
            },
            {
                "target_field": "model",
                "source_field": "",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                },
                "ai_instruction": "Extract the vehicle model name (e.g., Camry, F-150, Civic, 3 Series). Look for labels like 'Model', 'Model Name', or in the vehicle description following the make."
            },
            {
                "target_field": "effective_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the effective date from the document. Look for labels like 'Effective Date', 'Start Date', 'Date', or dates near policy or registration information."
            },
            {
                "target_field": "notes",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract ONLY the vehicle-specific notes.\n\nRULES:\n- Keep each sentence EXACTLY once\n- Remove bullets ('-', '*', '•') and list format\n- Keep original ordering\n- No duplication\n- Output a clean paragraph, not a list"
            },
            {
                "target_field": "vehicle_id",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the vehicle ID if present. Look for labels like 'Vehicle ID', 'ID', 'Vehicle Number', or similar identifiers."
            },
            {
                "target_field": "color",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "max_length": 30
                },
                "ai_instruction": "Extract the vehicle color. Look for labels like 'Color', 'Exterior Color', 'Paint Color', or color descriptions in the vehicle details."
            },
            {
                "target_field": "mileage",
                "source_field": "",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 999999
                },
                "ai_instruction": "Extract the vehicle mileage/odometer reading. Look for labels like 'Mileage', 'Odometer', 'Current Mileage', or numeric values with units like 'miles' or 'mi'."
            },
            {
                "target_field": "trim",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the vehicle trim level (e.g., SE, XLT, EX-L, Sport Line). Look for labels like 'Trim', 'Trim Level', 'Package', or in the model description."
            },
            {
                "target_field": "body_style",
                "source_field": "",
                "type": "string",
                "required": False,
                "transform": "lowercase",
                "validation": {
                    "enum": ["sedan", "coupe", "suv", "truck", "van", "convertible", "wagon", "hatchback", "crossover", "boat"]
                },
                "ai_instruction": "Extract the vehicle body style (e.g., sedan, truck, SUV, coupe). Look for labels like 'Body Style', 'Body Type', 'Type', or infer from vehicle description."
            },
            {
                "target_field": "fuel_type",
                "source_field": "",
                "type": "string",
                "required": False,
                "transform": "standardize_fuel_type",
                "validation": {
                    "enum": ["gasoline", "diesel", "electric", "hybrid", "plug-in hybrid", "Gas", "gas"]
                },
                "ai_instruction": "Extract the fuel type (e.g., gas, gasoline, diesel, electric, hybrid). Look for labels like 'Fuel Type', 'Fuel', 'Engine Type', or in vehicle specifications. Return the raw value as written - normalization will be applied."
            },
            {
                "target_field": "transmission",
                "source_field": "",
                "type": "string",
                "required": False,
                "transform": "lowercase",
                "validation": {
                    "enum": ["automatic", "manual", "cvt", "AUTO"]
                },
                "ai_instruction": "Extract the transmission type (e.g., automatic, manual, CVT). Look for labels like 'Transmission', 'Transmission Type', 'Trans', or in vehicle specifications."
            },
            {
                "target_field": "owner_email",
                "source_field": "",
                "type": "string",
                "required": False,
                "flag": ["pii"],
                "transform": "lowercase",
                "ai_instruction": "Extract the owner's email EXACTLY as written.\n- ALWAYS return the literal string, even if invalid\n- NEVER return null or empty\n- Do NOT guess or correct the format"
            },
            {
                "target_field": "weight",
                "source_field": "",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 99999
                },
                "ai_instruction": "Extract the vehicle weight if present. Look for labels like 'Weight', 'Curb Weight', 'Gross Weight', or numeric values with units like 'lbs' or 'kg'."
            }
        ]
    },
    {
        "id": "source_raw_text_vehicles",
        "metadata": {
            "source_name": "Raw Text Vehicle Data",
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
                "target_field": "vin",
                "source_field": "",
                "type": "string",
                "required": True,
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                },
                "transform": "uppercase",
                "ai_instruction": "Extract the VIN for THIS vehicle only.\nIgnore all example/template VINs in the document."
            },
            {
                "target_field": "year",
                "source_field": "",
                "type": "integer",
                "required": True,
                "validation": {
                    "min": 1900,
                    "max": 2030
                },
                "ai_instruction": "Extract the model year of the vehicle from the actual vehicle description. Look for a 4-digit year in the vehicle description itself (e.g., '2024 Toyota Camry', '2020 Ford F-150'). Do NOT extract years from prefixes like 'Vehicle V001:' - only extract the year from the actual vehicle description."
            },
            {
                "target_field": "make",
                "source_field": "",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 2,
                    "max_length": 50
                },
                "ai_instruction": "Extract the make for THIS vehicle.\nIgnore example text above the actual entry."
            },
            {
                "target_field": "model",
                "source_field": "",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                },
                "ai_instruction": "Extract the model for THIS vehicle.\nDo NOT reuse or infer from example blocks."
            },
            {
                "target_field": "effective_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the effective date from the text. Look for dates mentioned with keywords like 'effective', 'start date', 'date', or dates near policy or registration information."
            },
            {
                "target_field": "notes",
                "source_field": "",
                "type": "string",
                "required": False,
                "ai_instruction": "Extract the descriptive text ONLY for THIS vehicle.\n- Ignore all example blocks\n- Ignore template text\n- Do NOT copy text from previous vehicles\n- Produce one clean narrative sentence"
            },
            {
                "target_field": "color",
                "source_field": "",
                "type": "string",
                "required": False,
                "validation": {
                    "max_length": 30
                },
                "ai_instruction": "Extract the vehicle color from the text. Look for color descriptions like 'painted blue', 'red', 'white', 'black', 'silver', etc."
            },
            {
                "target_field": "mileage",
                "source_field": "",
                "type": "integer",
                "required": False,
                "validation": {
                    "min": 0,
                    "max": 999999
                },
                "ai_instruction": "Extract the vehicle mileage/odometer reading from the text. Look for patterns like '12,345 miles', 'about 45,210 miles', or 'X miles on the odometer'."
            },
            {
                "target_field": "body_style",
                "source_field": "",
                "type": "string",
                "required": False,
                "transform": "lowercase",
                "validation": {
                    "enum": ["sedan", "coupe", "suv", "truck", "van", "convertible", "wagon", "hatchback", "crossover", "boat"]
                },
                "ai_instruction": "Extract the vehicle body style from the text. Look for terms like 'sedan', 'truck', 'SUV', 'crossover', 'coupe', etc."
            },
            {
                "target_field": "fuel_type",
                "source_field": "",
                "type": "string",
                "required": False,
                "transform": "standardize_fuel_type",
                "validation": {
                    "enum": ["gasoline", "diesel", "electric", "hybrid", "plug-in hybrid", "Gas"]
                },
                "ai_instruction": "Extract the fuel type from the text. Look for patterns like 'Fuel: gas', 'Fuel: electric', etc. Return the raw value (normalization will be applied)."
            },
            {
                "target_field": "transmission",
                "source_field": "",
                "type": "string",
                "required": False,
                "transform": "lowercase",
                "validation": {
                    "enum": ["automatic", "manual", "cvt", "AUTO"]
                },
                "ai_instruction": "Extract the transmission type from the text. Look for patterns like '8-speed auto', 'automatic', 'manual', 'CVT', etc."
            },
            {
                "target_field": "owner_email",
                "source_field": "",
                "type": "email",
                "required": False,
                "flag": ["pii"],
                "validation": {
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                },
                "ai_instruction": "Extract the owner email address from the text. Look for patterns like 'Owner contact email: X' or standalone email addresses."
            }
        ]
    },
    {
        "id": "source_image_vehicles",
        "metadata": {
            "source_name": "Vehicle Image Analysis",
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
                "target_field": "vehicle_id",
                "source_field": "vehicle_id",
                "type": "string",
                "required": False
            },
            {
                "target_field": "vin",
                "source_field": "vin",
                "type": "string",
                "required": True,
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                },
                "transform": "uppercase"
            },
            {
                "target_field": "year",
                "source_field": "",
                "type": "integer",
                "required": True,
                "validation": {
                    "min": 1900,
                    "max": 2030
                },
                "ai_instruction": "Extract the model year from the image. Look for: 1) Year printed on window stickers (Monroney labels). 2) Model year on registration or title documents. 3) Year information in document headers or vehicle specifications. 4) The year is typically a 4-digit number and may be labeled as 'Model Year', 'Year', or 'MY'."
            },
            {
                "target_field": "make",
                "source_field": "",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 2,
                    "max_length": 50
                },
                "ai_instruction": "Identify the vehicle make from the image. This can be determined by: 1) Reading text from window stickers, registration documents, or titles visible in the image. 2) Recognizing brand logos or emblems on the vehicle. 3) Reading manufacturer names from document headers. Look for brand names like Toyota, Ford, Honda, BMW, etc."
            },
            {
                "target_field": "model",
                "source_field": "",
                "type": "string",
                "required": True,
                "validation": {
                    "min_length": 1,
                    "max_length": 50
                },
                "ai_instruction": "Identify the vehicle model from the image by: 1) Reading model names from window stickers, documents, or vehicle badges. 2) Extracting model information from visible registration or title documents. 3) Reading model badging on the vehicle itself (often on trunk, fender, or grille). The model typically appears with or after the make."
            },
            {
                "target_field": "effective_date",
                "source_field": "",
                "type": "date",
                "required": False,
                "ai_instruction": "Extract the effective date from any documents visible in the image. Look for dates on registration documents, titles, or window stickers."
            },
            {
                "target_field": "notes",
                "source_field": "",
                "type": "string",
                "required": False,
                "transform": "combine_image_metadata_notes"
            },
        ]
    },
    {
        "id": "source_image_metadata_json_vehicles",
        "metadata": {
            "source_name": "Image Metadata JSON",
            "source_type": "airtable",
            "connection_type": "file_upload",
            "notes": "Simple JSON file with image metadata (image_url, description) for vehicle records",
            "connection_config": {
                "file_path": ""
            }
        },
        "mappings": [
            {
                "target_field": "vehicle_id",
                "source_field": "vehicle_id",
                "type": "string",
                "required": False
            },
            {
                "target_field": "vin",
                "source_field": "vin",
                "type": "string",
                "required": True,
                "validation": {
                    "pattern": r"^[A-HJ-NPR-Z0-9]{17}$",
                    "min_length": 17,
                    "max_length": 17
                },
                "transform": "uppercase"
            },
            {
                "target_field": "image_url",
                "source_field": "image_url",
                "type": "string",
                "required": False
            },
            {
                "target_field": "description",
                "source_field": "description",
                "type": "string",
                "required": False
            },
            {
                "target_field": "notes",
                "source_field": "",
                "type": "string",
                "required": False,
                "transform": "combine_image_metadata_notes"
            }
        ]
    }
]

