# mappings.py

MAPPINGS = {
    "vehicles_basic": {
        # VIN variants
        "vin": "vin",
        "vin_number": "vin",
        "vin number": "vin",
        "vin_#": "vin",          
           
        # Vehicle ID variants
        "vehicle_id": "vehicle_id",
        "vehicle id": "vehicle_id",
        "vehicle_id_#": "vehicle_id",
        "vehicle id #": "vehicle_id",
        "unit_id": "vehicle_id",
        "unit": "vehicle_id",
        "unit_number": "vehicle_id",  

        # Year variants
        "year": "year",
        "model_year": "year",
        "model year": "year",
        "model YEAR": "year",

        # Make
        "make": "make",
        "MAKE": "make",
        "manufacturer": "make",      

        # Model
        "model": "model",
        "car_model": "model",
        "Car Model": "model",
        "vehicle_model": "model",     

        # Effective date
        "effective_date": "effective_date",
        "effective date": "effective_date",
        "EFFECTIVE date": "effective_date",
        "start_date": "effective_date",
        "start_dt": "effective_date",  

        # Notes
        "notes": "notes",
        "extra_note": "notes",
        "extra note": "notes",
        "extra_notes": "notes",       

        # Trim
        "trim": "trim",
        "trim_level": "trim",          

        # Weight / GVW
        "weight": "weight",
        "g_v_w": "weight",
        "gvw": "weight",               
    }
}


def get_mapping(name: str) -> dict:
    if name not in MAPPINGS:
        raise KeyError(f"Mapping '{name}' not found.")
    return MAPPINGS[name]