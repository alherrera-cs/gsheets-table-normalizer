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
    
    # Get headers and clean them
    headers = [clean_header(str(h)) for h in values[header_row_index]]
    
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


def apply_mapping(rows, mapping_dict):
    """
    Apply a mapping dictionary to rename/standardize column names.
    
    Args:
        rows: List of dictionaries (from rows2d_to_objects)
        mapping_dict: Dictionary mapping source column names to target column names
        
    Returns:
        List of dictionaries with mapped column names
    """
    mapped_rows = []
    
    for row in rows:
        mapped_row = {}
        for source_key, value in row.items():
            target_key = mapping_dict.get(source_key, source_key)
            
            if target_key in mapped_row:
                # Prefer non-empty overwrite
                if not mapped_row[target_key] and value:
                    mapped_row[target_key] = value
            else:
                mapped_row[target_key] = value
        
        mapped_rows.append(mapped_row)
    
    return mapped_rows


def clean_year(value):
    """
    Normalize year values to integers.
    
    Args:
        value: Year value (string, int, float, None, or empty)
        
    Returns:
        Integer year or None if invalid/empty
    """
    if value is None:
        return None
    
    # If already an int, return as-is
    if isinstance(value, int):
        return value
    
    # If it's a float, try to convert directly
    if isinstance(value, float):
        try:
            return int(value)
        except (ValueError, TypeError, OverflowError):
            return None
    
    # Convert to string and strip whitespace
    str_value = str(value).strip()
    
    # Return None for empty strings
    if not str_value:
        return None
    
    # Try to convert to integer (handles "2020" and "2020.0")
    try:
        # First try direct conversion
        return int(float(str_value))
    except (ValueError, TypeError, OverflowError):
        return None


def clean_notes(value):
    """
    Normalize notes values.
    
    Args:
        value: Notes value (string, None, or whitespace)
        
    Returns:
        Empty string for None/whitespace, otherwise trimmed string
    """
    if value is None:
        return ""
    
    str_value = str(value).strip()
    return "" if not str_value else str_value


def clean_text(value):
    """
    Normalize text field values.
    
    Args:
        value: Text value (string, None, or whitespace)
        
    Returns:
        Empty string for None/whitespace, otherwise trimmed string
    """
    if value is None:
        return ""
    
    str_value = str(value).strip()
    return "" if not str_value else str_value

def clean_int(value):
    """Normalize numeric fields: convert numeric strings to int, allow None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    value = str(value).strip()
    if value == "":
        return None
    if value.isdigit():
        return int(value)
    return value


def normalize_values(rows):
    """
    Apply final normalization pass to mapped rows.
    Normalizes year, notes, and other text fields.
    
    Args:
        rows: List of dictionaries after mapping
        
    Returns:
        List of dictionaries with normalized values
    """
    normalized = []
    for row in rows:
        cleaned = {}
        for key, value in row.items():

            # Year normalization
            if key == "year":
                cleaned[key] = clean_int(value)
                continue

            # Weight normalization
            if key == "weight":
                cleaned[key] = clean_int(value)
                continue

            # Notes normalization
            if key == "notes":
                cleaned[key] = "" if value is None or str(value).strip() == "" else str(value).strip()
                continue

            # Generic string cleanup
            if isinstance(value, str):
                cleaned[key] = value.strip()
            else:
                cleaned[key] = value

        normalized.append(cleaned)
    return normalized


def drop_empty_rows(rows):
    """
    Remove rows that are empty or contain only empty/None values.
    
    Args:
        rows: List of dictionaries
        
    Returns:
        List of dictionaries with empty rows removed
    """
    filtered = []
    
    for row in rows:
        if any(
            value is not None and str(value).strip() != ""
            for value in row.values()
        ):
            filtered.append(row)
    
    return filtered


def list_sheets(sheet_id: str):
    """
    List all available sheet names in a Google Sheets document.
    
    Args:
        sheet_id: Google Sheets document ID
        
    Returns:
        List of sheet names
    """
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=SCOPES,
    )
    
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    
    metadata = sheet.get(spreadsheetId=sheet_id).execute()
    return [s['properties']['title'] for s in metadata.get('sheets', [])]


def fetch_from_google_sheets(
    mapping_name: str,
    sheet_id: str,
    range_="Sheet1!A:Z",
    header_row_index: int = 0,
):
    """
    Fetch a table from Google Sheets, normalize headers, apply a mapping,
    and drop empty rows.
    """
    mapping_dict = get_mapping(mapping_name)

    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=SCOPES,
    )

    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()

    result = sheet.values().get(
        spreadsheetId=sheet_id,
        range=range_,
        majorDimension="ROWS",
    ).execute()

    values = result.get("values", [])

    rows = rows2d_to_objects(values, header_row_index=header_row_index)
    mapped = apply_mapping(rows, mapping_dict)
    normalized = normalize_values(mapped)
    cleaned = drop_empty_rows(normalized)

    return cleaned


# ---------------------------
# Updated bottom message
# ---------------------------
if __name__ == "__main__":
    dummy_values = [
        ["VIN", "Model Year", "Make", "Model"],
        ["12345", "2020", "Ford", "F-150"],
        ["67890", "2021", "Honda", "Civic"],
    ]
    
    rows = rows2d_to_objects(dummy_values)
    mapping_dict = get_mapping("vehicles_basic")
    mapped = apply_mapping(rows, mapping_dict)
    mapped = drop_empty_rows(mapped)
    
    print("Original 2D data:")
    for row in dummy_values:
        print(row)
    
    print("\nConverted to objects:")
    for row in rows:
        print(row)
    
    print("\nAfter mapping:")
    for row in mapped:
        print(row)
    
    print("\n" + "=" * 50)
    print("Run `python tests/run_gsheets_tests.py` to test real Google Sheets.")
    print("=" * 50)