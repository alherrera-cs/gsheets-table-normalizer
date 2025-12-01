"""
Source detection and data extraction utilities.

Supports multiple data sources: Google Sheets, Airtable, XLSX, raw text, PDF, image.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum


class SourceType(Enum):
    """Supported data source types."""
    GOOGLE_SHEETS = "google_sheet"  # Match mapping structure
    AIRTABLE = "airtable"
    SHARED_TABLE = "shared_table"
    XLSX = "xlsx_file"  # Match mapping structure
    CSV = "csv"
    RAW_TEXT = "raw_text"
    PDF = "pdf"
    IMAGE = "image"
    UNKNOWN = "unknown"


def detect_source_type(source: Union[str, Dict, Path]) -> SourceType:
    """
    Detect the type of data source.
    
    Args:
        source: Source identifier (sheet_id, file path, URL, etc.)
        
    Returns:
        SourceType enum value
    """
    if isinstance(source, dict):
        # Check for explicit source type
        if "source_type" in source:
            try:
                return SourceType(source["source_type"])
            except ValueError:
                pass
        
        # Check for Google Sheets indicators
        if "sheet_id" in source or "spreadsheet_id" in source:
            return SourceType.GOOGLE_SHEETS
        
        # Check for Airtable indicators
        if "base_id" in source or "table_id" in source:
            return SourceType.AIRTABLE
    
    if isinstance(source, (str, Path)):
        source_str = str(source)
        
        # File extensions
        if source_str.endswith(('.xlsx', '.XLSX')):
            return SourceType.XLSX
        elif source_str.endswith(('.csv', '.CSV')):
            return SourceType.CSV
        elif source_str.endswith(('.json', '.JSON')):
            # Check if it's an Airtable export (has "records" key)
            if isinstance(source, Path) or (isinstance(source, str) and not source.startswith('http')):
                try:
                    import json
                    file_path = Path(source) if isinstance(source, (str, Path)) else source
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, dict) and "records" in data:
                                return SourceType.AIRTABLE
                except (json.JSONDecodeError, IOError):
                    pass
            return SourceType.AIRTABLE  # Default for .json files
        elif source_str.endswith(('.pdf', '.PDF')):
            return SourceType.PDF
        elif source_str.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
            return SourceType.IMAGE
        
        # URL patterns
        if 'docs.google.com/spreadsheets' in source_str:
            return SourceType.GOOGLE_SHEETS
        elif 'airtable.com' in source_str:
            return SourceType.AIRTABLE
    
    return SourceType.UNKNOWN


def fetch_from_google_sheets_raw(
    source: Union[str, Dict],
    range_: str = "Sheet1!A:Z",
    header_row_index: int = 0,
) -> List[List[Any]]:
    """
    Fetch raw 2D data from Google Sheets (helper for extract_from_source).
    
    This is a wrapper around the existing Google Sheets functionality.
    Google Sheets support is optional - will raise ImportError if dependencies are missing.
    """
    try:
        from external_tables import (
            SERVICE_ACCOUNT_FILE,
            SCOPES,
            GOOGLE_SHEETS_AVAILABLE,
        )
        
        if not GOOGLE_SHEETS_AVAILABLE:
            raise ImportError(
                "Google Sheets support requires google-auth and google-api-python-client. "
                "Install with: pip install google-auth google-api-python-client"
            )
        
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        
        # Extract sheet_id from source
        if isinstance(source, dict):
            sheet_id = source.get("sheet_id") or source.get("spreadsheet_id")
            range_ = source.get("range", range_)
        else:
            sheet_id = str(source)
        
        if not sheet_id:
            raise ValueError("Google Sheets source must provide sheet_id")
        
        # Authenticate and fetch
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
        
        return result.get("values", [])
    except ImportError as e:
        raise ImportError(
            f"Google Sheets support not available: {e}. "
            "For testing, use CSV/Excel/Airtable files instead."
        )


def extract_from_source(
    source: Union[str, Dict, Path],
    source_type: Optional[SourceType] = None,
    **kwargs
) -> List[List[Any]]:
    """
    Extract raw 2D data from a source.
    
    Args:
        source: Source identifier
        source_type: Optional explicit source type (auto-detected if not provided)
        **kwargs: Additional source-specific parameters
        
    Returns:
        2D list of values (rows x columns)
        
    Raises:
        NotImplementedError: If source type is not yet supported
        ValueError: If source cannot be processed
    """
    if source_type is None:
        source_type = detect_source_type(source)
    
    if source_type == SourceType.GOOGLE_SHEETS:
        return fetch_from_google_sheets_raw(source, **kwargs)
    
    elif source_type == SourceType.XLSX:
        # Load Excel file - try openpyxl first, fallback to zipfile/xml
        # Handle both file paths and dict with file_path
        if isinstance(source, dict):
            file_path = Path(source.get("file_path", source.get("path", "")))
        else:
            file_path = Path(source)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        
        # Try openpyxl first (preferred method)
        try:
            from openpyxl import load_workbook
            workbook = load_workbook(file_path, data_only=True)
            sheet = workbook.active  # Use first sheet
            
            # Extract all rows
            rows = []
            for row in sheet.iter_rows(values_only=True):
                # Convert None to empty string for consistency
                row_values = [str(cell) if cell is not None else "" for cell in row]
                rows.append(row_values)
            
            return rows
        except ImportError:
            # Fallback to zipfile/xml method
            import zipfile
            import xml.etree.ElementTree as ET
            
            rows = []
            with zipfile.ZipFile(file_path, 'r') as z:
                # Read shared strings
                strings = []
                try:
                    with z.open('xl/sharedStrings.xml') as f:
                        tree = ET.parse(f)
                        root = tree.getroot()
                        ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                        strings = [t.text if t.text else '' for t in root.findall('.//main:t', ns)]
                except:
                    pass
                
                # Read all rows from first sheet
                with z.open('xl/worksheets/sheet1.xml') as f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                    
                    for row_elem in root.findall('.//main:row', ns):
                        row = []
                        for cell in row_elem.findall('main:c', ns):
                            val_elem = cell.find('main:v', ns)
                            if val_elem is not None and val_elem.text:
                                val = val_elem.text
                                if val.startswith('str') and strings:
                                    try:
                                        idx = int(val.replace('str', ''))
                                        row.append(strings[idx] if idx < len(strings) else '')
                                    except:
                                        row.append(val)
                                else:
                                    row.append(val)
                            else:
                                row.append('')
                        if row:  # Only add non-empty rows
                            rows.append(row)
            
            return rows
    
    elif source_type == SourceType.CSV:
        import csv
        
        # Handle both file paths and dict with file_path
        if isinstance(source, dict):
            file_path = Path(source.get("file_path", source.get("path", "")))
        else:
            file_path = Path(source)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Read CSV file
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        return rows
    
    elif source_type == SourceType.AIRTABLE:
        # Load Airtable JSON export
        import json
        
        # Handle both file paths and dict with file_path
        if isinstance(source, dict):
            file_path = Path(source.get("file_path", source.get("path", "")))
        else:
            file_path = Path(source)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Airtable JSON file not found: {file_path}")
        
        # Load JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert Airtable "records" format to 2D array
        if not isinstance(data, dict) or "records" not in data:
            raise ValueError("Airtable JSON must have a 'records' key")
        
        records = data["records"]
        if not records:
            return []
        
        # Extract headers from first record's fields
        first_record = records[0]
        if "fields" not in first_record:
            raise ValueError("Airtable records must have 'fields' key")
        
        # Get all unique field names from all records (in case some records have different fields)
        all_fields = set()
        for record in records:
            if "fields" in record:
                all_fields.update(record["fields"].keys())
        
        # Sort for consistency and create a mapping from original to cleaned
        from external_tables import clean_header
        original_headers = sorted(all_fields)
        cleaned_headers = [clean_header(h) for h in original_headers]
        header_map = dict(zip(original_headers, cleaned_headers))
        
        # Build 2D array: header row (cleaned) + data rows
        rows = [cleaned_headers]
        for record in records:
            row = []
            fields = record.get("fields", {})
            for original_header in original_headers:
                value = fields.get(original_header, None)
                # Convert to string for consistency with CSV/Excel
                if value is not None:
                    row.append(str(value))
                else:
                    row.append("")
            rows.append(row)
        
        return rows
    
    elif source_type == SourceType.PDF:
        # TODO: Implement PDF/OCR extraction
        raise NotImplementedError("PDF extraction not yet implemented")
    
    elif source_type == SourceType.IMAGE:
        # TODO: Implement Image OCR extraction
        raise NotImplementedError("Image OCR extraction not yet implemented")
    
    elif source_type == SourceType.RAW_TEXT:
        # TODO: Implement raw text parsing
        raise NotImplementedError("Raw text extraction not yet implemented")
    
    else:
        raise ValueError(f"Unknown or unsupported source type: {source_type}")



