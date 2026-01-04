"""
External tables processing module.

Converts raw 2D tables → normalized list of dicts → mapped → cleaned.
"""
import os
from pathlib import Path

# Google Sheets imports - only needed for Google Sheets functionality
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False
    # Dummy classes for type hints
    class service_account:
        @staticmethod
        def Credentials(*args, **kwargs):
            raise ImportError("google.oauth2 not available")
    class build:
        @staticmethod
        def __call__(*args, **kwargs):
            raise ImportError("googleapiclient not available")

from mappings import get_mapping

# Service account file path relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
SERVICE_ACCOUNT_FILE = os.getenv(
    "GOOGLE_SERVICE_ACCOUNT_FILE",
    str(PROJECT_ROOT / "cosmic-bonus-434805-m6-0d79573ad277.json"),
)
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Only define these if Google Sheets is available
if not GOOGLE_SHEETS_AVAILABLE:
    SERVICE_ACCOUNT_FILE = None
    SCOPES = []

def clean_header(h: str) -> str:
    """
    Clean and normalize a header string.
    
    Args:
        h: Raw header string
        
    Returns:
        Cleaned header string (lowercased, stripped, spaces replaced with underscores)
    """
    if not h:
        return ""
    return h.strip().lower().replace(" ", "_")

def normalize_header(header: str) -> str:
    """
    Normalize CSV header names to a standardized format.
    
    Applies the following transformations:
    - strip whitespace
    - lowercase everything
    - replace spaces with underscores
    - replace hyphens with underscores
    - remove trailing colons
    - remove non-alphanumeric characters except underscores
    - apply header aliases to map common variants to canonical names
    
    Args:
        header: Raw header string from CSV
        
    Returns:
        Normalized header string
        
    Examples:
        "VIN Number" → "vin" (via alias mapping)
        "VIN" → "vin"
        "Exterior Color" → "exterior_color"
        "Owner Email" → "owner_email"
        "Fuel Type" → "fuel_type"
        "Trim Level" → "trim_level"
        "Field:" → "field"
    """
    if not header:
        return ""
    
    # Convert to string and strip whitespace
    normalized = str(header).strip()
    
    # Lowercase everything
    normalized = normalized.lower()
    
    # Replace spaces with underscores
    normalized = normalized.replace(" ", "_")
    
    # Replace hyphens with underscores
    normalized = normalized.replace("-", "_")
    
    # Remove trailing colons
    normalized = normalized.rstrip(":")
    
    # Remove non-alphanumeric characters except underscores
    normalized = "".join(c if c.isalnum() or c == "_" else "" for c in normalized)
    
    # Apply header aliases to map common variants to canonical names
    # This ensures variants like "VIN Number", "VINNumber", etc. all map to "vin"
    header_aliases = {
        "vin_number": "vin",
        "vinnumber": "vin",
        "vehicle_identification_number": "vin",
        "vinno": "vin"
    }
    
    # Return the alias if it exists, otherwise return the normalized key
    return header_aliases.get(normalized, normalized)

def rows2d_to_objects(values, header_row_index=0):
    """
    Convert a 2D list/array to a list of dictionaries.
    
    Args:
        values: 2D list where first row (or header_row_index) contains headers
        header_row_index: Index of the row containing headers (default: 0)
        
    Returns:
        List of dictionaries, one per data row
    """
    if not values or len(values) <= header_row_index:
        return []
    
    # Get headers and normalize them using normalize_header()
    # This ensures all headers are normalized before creating row dictionaries
    headers = [normalize_header(str(h)) for h in values[header_row_index]]
    
    # Get data rows (everything after header row)
    data_rows = values[header_row_index + 1:]
    
    # Convert each row to a dict
    objects = []
    for row in data_rows:
        obj = {}
        for i, header in enumerate(headers):
            value = row[i] if i < len(row) else None
            obj[header] = value
        objects.append(obj)
    
    return objects
