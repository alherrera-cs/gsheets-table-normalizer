"""
Mappings module - Contains all dataset mapping configurations
"""

from .vehicles_mappings import VEHICLES_MAPPINGS
from .drivers_mappings import DRIVERS_MAPPINGS
from .policies_mappings import POLICIES_MAPPINGS
from .locations_mappings import LOCATIONS_MAPPINGS
from .claims_mappings import CLAIMS_MAPPINGS
from .relationships_mappings import RELATIONSHIPS_MAPPINGS

# Backward compatibility: Old mapping dictionary structure
# This is kept for external_tables.py and other legacy code
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


def get_mapping(name: str):
    """
    Get a mapping dictionary by name (backward compatibility).
    
    This function is kept for backward compatibility with external_tables.py
    and other legacy code that uses the old mapping structure.
    
    Args:
        name: Mapping name (e.g., "vehicles_basic")
        
    Returns:
        Mapping dictionary mapping source field names to target field names
    """
    if name not in MAPPINGS:
        raise KeyError(f"Mapping '{name}' not found.")
    return MAPPINGS[name]


def get_mapping_by_id(mapping_id: str):
    """
    Get a mapping configuration by its ID.
    
    Args:
        mapping_id: The unique ID of the mapping configuration
        
    Returns:
        Mapping configuration dictionary or None if not found
    """
    all_mappings = (
        VEHICLES_MAPPINGS +
        DRIVERS_MAPPINGS +
        POLICIES_MAPPINGS +
        LOCATIONS_MAPPINGS +
        CLAIMS_MAPPINGS +
        RELATIONSHIPS_MAPPINGS
    )
    
    for mapping in all_mappings:
        if mapping.get("id") == mapping_id:
            return mapping
    
    return None


def get_mappings_by_source_type(source_type: str):
    """
    Get all mappings for a specific source type.
    
    Args:
        source_type: The source type (e.g., "google_sheet", "airtable", "pdf")
        
    Returns:
        List of mapping configurations matching the source type
    """
    all_mappings = (
        VEHICLES_MAPPINGS +
        DRIVERS_MAPPINGS +
        POLICIES_MAPPINGS +
        LOCATIONS_MAPPINGS +
        CLAIMS_MAPPINGS +
        RELATIONSHIPS_MAPPINGS
    )
    
    return [
        mapping for mapping in all_mappings
        if mapping.get("metadata", {}).get("source_type") == source_type
    ]

