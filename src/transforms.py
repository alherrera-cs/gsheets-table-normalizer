import re
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from decimal import Decimal


class TransformResult:
    """Result of a transform operation"""
    def __init__(self, value: Any, error: Optional[str] = None):
        self.value = value
        self.error = error


def apply_transform(data: Dict[str, Any], transform_string: str) -> TransformResult:
    """
    Applies a transform string to a data object and returns the transformed value
    
    Args:
        data: The data object containing variables
        transform_string: The transform string (e.g., "capitalize(fullName)", "join([firstName, lastName], ' ')")
    
    Returns:
        TransformResult with the transformed value or error message
    """
    try:
        # Handle direct variable reference (no transform)
        if '(' not in transform_string and '[' not in transform_string:
            value = get_nested_value(data, transform_string.strip())
            return TransformResult(value if value is not None else '')
        
        return parse_and_apply_transform(data, transform_string.strip())
    except Exception as e:
        return TransformResult('', f'Transform error: {str(e)}')


def parse_and_apply_transform(data: Dict[str, Any], transform_string: str) -> TransformResult:
    """
    Parses and applies a transform function to the data
    Enhanced with nested expression debugging
    """
    # Log complex nested expressions for debugging
    if os.environ.get('ENVIRONMENT') != 'production':
        depth = transform_string.count('(')
        if depth > 1:
            print(f'Parsing nested expression (depth {depth}): {transform_string}')
    
    # Match function pattern: functionName(args)
    function_match = re.match(r'^(\w+)\((.*)\)$', transform_string)
    if not function_match:
        return TransformResult('', f'Invalid transform format: {transform_string}')
    
    function_name, args_string = function_match.groups()
    args = parse_arguments(args_string)
    
    # Transform function dispatch
    transforms = {
        'capitalize': apply_capitalize,
        'uppercase': apply_uppercase,
        'lowercase': apply_lowercase,
        'date': apply_date,
        'number': apply_number,
        'phone': apply_phone,
        'currency': apply_currency,
        'prepend': apply_prepend,
        'append': apply_append,
        'join': apply_join,
        'regex': apply_regex,
        'index': apply_index,
        'split': apply_split,
        'checkif': apply_checkif,
        'slice': apply_slice,
        'arrayfrom': apply_arrayfrom,
        'length': apply_length,
        'sum': apply_sum,
        'filter': apply_filter,
        'count': apply_count,
        'first': apply_first,
        'last': apply_last,
        'boolean': apply_boolean,
        'if': apply_if,
        'flatten': apply_flatten,
        'standardize_fuel_type': apply_standardize_fuel_type,
        'combine_image_metadata_notes': apply_combine_image_metadata_notes,
    }
    
    transform_func = transforms.get(function_name.lower())
    if not transform_func:
        print(f'Unknown transform function encountered: {function_name} in transform string "{transform_string}"')
        return TransformResult('', f'Unknown transform function: {function_name}')
    
    return transform_func(data, args)


def parse_arguments(args_string: str) -> List[str]:
    """Parses function arguments, handling nested functions and arrays"""
    if not args_string.strip():
        return []
    
    args = []
    current = ''
    depth = 0
    in_quotes = False
    quote_char = ''
    
    for i, char in enumerate(args_string):
        if not in_quotes and char in ('"', "'"):
            in_quotes = True
            quote_char = char
            current += char
        elif in_quotes and char == quote_char and (i == 0 or args_string[i - 1] != '\\'):
            in_quotes = False
            quote_char = ''
            current += char
        elif not in_quotes and char in ('(', '['):
            depth += 1
            current += char
        elif not in_quotes and char in (')', ']'):
            depth -= 1
            current += char
        elif not in_quotes and char == ',' and depth == 0:
            args.append(current.strip())
            current = ''
        else:
            current += char
    
    if current.strip():
        args.append(current.strip())
    
    return args


def get_nested_value(obj: Dict[str, Any], path: str) -> Any:
    """Gets nested value from object using dot notation"""
    current = obj
    for key in path.split('.'):
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
        if current is None:
            return None
    return current


def unquote(s: str) -> str:
    """Removes quotes from string if present"""
    trimmed = s.strip()
    if (trimmed.startswith('"') and trimmed.endswith('"')) or \
       (trimmed.startswith("'") and trimmed.endswith("'")):
        return trimmed[1:-1]
    return trimmed


def validate_expression_structure(expr: str) -> Dict[str, Union[bool, str]]:
    """Validates that parentheses are properly balanced in expressions"""
    open_parens = 0
    open_brackets = 0
    in_quotes = False
    quote_char = ''
    
    for i, char in enumerate(expr):
        prev_char = expr[i - 1] if i > 0 else ''
        
        # Handle quotes
        if not in_quotes and char in ('"', "'"):
            in_quotes = True
            quote_char = char
        elif in_quotes and char == quote_char and prev_char != '\\':
            in_quotes = False
            quote_char = ''
        
        # Skip characters inside quotes
        if in_quotes:
            continue
        
        # Track parentheses and brackets
        if char == '(':
            open_parens += 1
        elif char == ')':
            open_parens -= 1
        elif char == '[':
            open_brackets += 1
        elif char == ']':
            open_brackets -= 1
        
        # Check for negative counts (closing before opening)
        if open_parens < 0:
            return {'is_valid': False, 'error': 'Unmatched closing parenthesis'}
        if open_brackets < 0:
            return {'is_valid': False, 'error': 'Unmatched closing bracket'}
    
    if open_parens > 0:
        return {'is_valid': False, 'error': 'Unclosed parentheses'}
    if open_brackets > 0:
        return {'is_valid': False, 'error': 'Unclosed brackets'}
    if in_quotes:
        return {'is_valid': False, 'error': 'Unclosed quote'}
    
    return {'is_valid': True}


def get_value(data: Dict[str, Any], arg: str) -> Any:
    """
    Gets value from data object, handling both quoted strings and variable references
    Enhanced with better nested expression error handling
    """
    trimmed = arg.strip()
    
    # Check if it's a quoted string
    if (trimmed.startswith('"') and trimmed.endswith('"')) or \
       (trimmed.startswith("'") and trimmed.endswith("'")):
        return unquote(trimmed)
    
    # Check if it's a nested transform
    if '(' in trimmed:
        # Validate expression structure before attempting to parse
        validation = validate_expression_structure(trimmed)
        if not validation['is_valid']:
            print(f'Invalid expression structure in "{trimmed}": {validation.get("error")}')
            return ''
        
        result = parse_and_apply_transform(data, trimmed)
        if result.error:
            # Log nested expression errors for debugging
            print(f'Nested expression error in "{trimmed}": {result.error}')
            return ''
        return result.value
    
    # Treat as variable reference
    return get_nested_value(data, trimmed) or ''


# Transform function implementations

def apply_capitalize(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 1:
        return TransformResult('', 'capitalize() requires exactly 1 argument')
    
    value = get_value(data, args[0])
    s = str(value)
    result = s[0].upper() + s[1:].lower() if s else ''
    return TransformResult(result)


def apply_uppercase(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 1:
        return TransformResult('', 'uppercase() requires exactly 1 argument')
    
    value = get_value(data, args[0])
    return TransformResult(str(value).upper())


def apply_lowercase(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 1:
        return TransformResult('', 'lowercase() requires exactly 1 argument')
    
    value = get_value(data, args[0])
    return TransformResult(str(value).lower())


def apply_date(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 2:
        return TransformResult('', 'date() requires exactly 2 arguments: date(variable, "format")')
    
    value = get_value(data, args[0])
    date_format = unquote(args[1])
    
    if not value or value == '':
        return TransformResult('')
    
    try:
        parsed_date = parse_date(str(value))
        if not parsed_date:
            return TransformResult('', f'Invalid date: {value}')
        
        formatted = format_date(parsed_date, date_format)
        return TransformResult(formatted)
    except Exception as e:
        return TransformResult('', f'Date formatting error: {str(e)}')


def apply_number(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 2:
        return TransformResult('', 'number() requires exactly 2 arguments: number(variable, "format")')
    
    value = get_value(data, args[0])
    number_format = unquote(args[1])
    
    # Remove currency symbols and commas
    cleaned = re.sub(r'[,$]', '', str(value))
    try:
        num = float(cleaned)
    except (ValueError, TypeError):
        return TransformResult('', f'Invalid number: {value}')
    
    formatted = format_number(num, number_format)
    return TransformResult(formatted)


def apply_phone(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 2:
        return TransformResult('', 'phone() requires exactly 2 arguments: phone(variable, "format")')
    
    value = get_value(data, args[0])
    phone_format = unquote(args[1])
    
    digits = re.sub(r'\D', '', str(value))
    formatted = format_phone(digits, phone_format)
    return TransformResult(formatted)


def apply_currency(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 2:
        return TransformResult('', 'currency() requires exactly 2 arguments: currency(variable, "format")')
    
    value = get_value(data, args[0])
    currency_format = unquote(args[1])
    
    # Remove currency symbols and commas
    cleaned = re.sub(r'[,$]', '', str(value))
    try:
        num = float(cleaned)
    except (ValueError, TypeError):
        return TransformResult('', f'Invalid currency amount: {value}')
    
    formatted = format_currency(num, currency_format)
    return TransformResult(formatted)


def apply_prepend(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 2:
        return TransformResult('', 'prepend() requires exactly 2 arguments: prepend(variable, "prefix")')
    
    value = get_value(data, args[0])
    prefix = unquote(args[1])
    
    return TransformResult(prefix + str(value))


def apply_append(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 2:
        return TransformResult('', 'append() requires exactly 2 arguments: append(variable, "suffix")')
    
    value = get_value(data, args[0])
    suffix = unquote(args[1])
    
    return TransformResult(str(value) + suffix)


def apply_join(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 2:
        return TransformResult('', 'join() requires exactly 2 arguments: join([var1, var2], "separator")')
    
    array_arg = args[0].strip()
    separator = unquote(args[1])
    
    # Parse array argument [var1, var2, ...]
    if not array_arg.startswith('[') or not array_arg.endswith(']'):
        return TransformResult('', 'join() first argument must be an array: [var1, var2]')
    
    array_content = array_arg[1:-1]
    variables = parse_arguments(array_content)
    
    values = [str(get_value(data, var_name)) for var_name in variables if get_value(data, var_name)]
    return TransformResult(separator.join(values))


def apply_regex(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 2:
        return TransformResult('', 'regex() requires exactly 2 arguments: regex(variable, "pattern")')
    
    value = get_value(data, args[0])
    pattern = unquote(args[1])
    
    try:
        match = re.search(pattern, str(value))
        return TransformResult(match.group(0) if match else '')
    except re.error:
        return TransformResult('', f'Invalid regex pattern: {pattern}')


def apply_index(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 2:
        return TransformResult('', 'index() requires exactly 2 arguments: index(variable, 0)')
    
    value = get_value(data, args[0])
    index_str = args[1].strip()
    
    try:
        index = int(index_str)
    except ValueError:
        return TransformResult('', f'Invalid index: {index_str}')
    
    # Handle negative indices (Python-style)
    if isinstance(value, list):
        # Python naturally handles negative indices
        if index < -len(value) or index >= len(value):
            print(f'Index {index} out of bounds for array of length {len(value)}')
            return TransformResult('')
        return TransformResult(value[index])
    else:
        # If it's a string result, treat as single element or try comma-separated parsing
        s = str(value)
        if ',' in s:
            parts = [p.strip() for p in s.split(',')]
            if index < -len(parts) or index >= len(parts):
                print(f'Index {index} out of bounds for comma-separated string with {len(parts)} parts')
                return TransformResult('')
            return TransformResult(parts[index])
        # For non-array values, only index 0 makes sense
        return TransformResult(s if index == 0 else '')


def apply_split(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 2:
        return TransformResult('', 'split() requires exactly 2 arguments: split(variable, "separator")')
    
    value = get_value(data, args[0])
    separator = unquote(args[1])
    
    # Ensure we always return a list for consistent downstream processing
    string_value = str(value or '')
    parts = string_value.split(separator)
    
    return TransformResult(parts)


def normalize_value_for_comparison(value: Any) -> str:
    """
    Normalizes a value to a standard format for comparison
    Handles: yes/Yes/YES/true/True/TRUE/1/"1"/boolean true -> "yes"
             no/No/NO/false/False/FALSE/0/"0"/boolean false -> "no"
    """
    # Handle None
    if value is None:
        return ''
    
    # Handle boolean
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    
    # Convert to string and trim
    str_value = str(value).strip().lower()
    
    # Empty string stays empty
    if not str_value:
        return ''
    
    # Normalize common affirmative values
    if str_value in ['yes', 'y', 'true', '1', 'on', 'checked', 'active', 'enabled']:
        return 'yes'
    
    # Normalize common negative values
    if str_value in ['no', 'n', 'false', '0', 'off', 'unchecked', 'inactive', 'disabled']:
        return 'no'
    
    # Return lowercase for other values
    return str_value


def apply_checkif(data: Dict[str, Any], args: List[str]) -> TransformResult:
    """
    checkIf() transform with robust value normalization
    
    Works with ANY variation of yes/no/true/false:
    - checkIf(x, "=", "yes") where x="Yes" → ✓ Match
    - checkIf(x, "=", "Yes") where x="yes" → ✓ Match
    - checkIf(x, "=", "true") where x=true → ✓ Match
    - checkIf(x, "=", "1") where x="yes" → ✓ Match
    
    All variations are normalized before comparison!
    """
    if len(args) != 3:
        return TransformResult('', 'checkIf() requires exactly 3 arguments: checkIf(variable, operator, value)')
    
    raw_value = get_value(data, args[0])
    operator = unquote(args[1])
    raw_expected_value = unquote(args[2])
    
    # CRITICAL: Normalize BOTH the variable value AND the comparison value
    normalized_value = normalize_value_for_comparison(raw_value)
    normalized_expected_value = normalize_value_for_comparison(raw_expected_value)
    
    if operator in ('=', '==', '==='):
        return TransformResult('Yes' if normalized_value == normalized_expected_value else '')
    elif operator in ('!=', '!=='):
        return TransformResult('Yes' if normalized_value != normalized_expected_value else '')
    else:
        return TransformResult('', f'Invalid operator: {operator}')


def apply_slice(data: Dict[str, Any], args: List[str]) -> TransformResult:
    value = get_value(data, args[0])
    start = int(args[1])
    end = int(args[2])
    
    return TransformResult(value[start:end])


def apply_arrayfrom(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 1:
        return TransformResult([], 'arrayFrom() requires exactly 1 argument')
    
    field_path = args[0]
    
    # Parse the field path to extract array field and property
    path_parts = field_path.split('.')
    if len(path_parts) < 2:
        return TransformResult([], 'arrayFrom() requires a property path like "arrayField.property"')
    
    array_field_name = path_parts[0]
    property_path = '.'.join(path_parts[1:])
    
    # Get the array from data
    array_value = get_value(data, array_field_name)
    
    if not isinstance(array_value, list):
        return TransformResult([], f'Field \'{array_field_name}\' is not an array')
    
    # Extract the specified property from each object in the array
    result = []
    for item in array_value:
        if isinstance(item, dict):
            val = get_value(item, property_path)
            if val is not None:
                result.append(val)
    
    return TransformResult(result)


def apply_length(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 1:
        return TransformResult(0, 'length() requires exactly 1 argument')
    
    value = get_value(data, args[0])
    
    if isinstance(value, (list, str)):
        return TransformResult(len(value))
    elif isinstance(value, dict):
        return TransformResult(len(value))
    
    return TransformResult(0)


def apply_sum(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 1:
        return TransformResult(0, 'sum() requires exactly 1 argument')
    
    field_path = args[0]
    
    # Parse the field path to extract array field and property
    path_parts = field_path.split('.')
    if len(path_parts) < 2:
        return TransformResult(0, 'sum() requires a property path like "arrayField.property"')
    
    array_field_name = path_parts[0]
    property_path = '.'.join(path_parts[1:])
    
    array_value = get_value(data, array_field_name)
    
    if not isinstance(array_value, list):
        return TransformResult(0, f'Field \'{array_field_name}\' is not an array')
    
    total = 0
    for item in array_value:
        if isinstance(item, dict):
            value = get_value(item, property_path)
            cleaned = re.sub(r'[,$]', '', str(value))
            try:
                num = float(cleaned)
                total += num
            except (ValueError, TypeError):
                pass
    
    return TransformResult(total)


def apply_filter(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 3:
        return TransformResult([], 'filter() requires exactly 3 arguments: filter(arrayField.property, operator, value)')
    
    field_path = args[0]
    operator = unquote(args[1])
    expected_value = unquote(args[2])
    
    path_parts = field_path.split('.')
    if len(path_parts) < 2:
        return TransformResult([], 'filter() requires a property path like "arrayField.property"')
    
    array_field_name = path_parts[0]
    property_path = '.'.join(path_parts[1:])
    
    array_value = get_value(data, array_field_name)
    
    if not isinstance(array_value, list):
        return TransformResult([], f'Field \'{array_field_name}\' is not an array')
    
    filtered = []
    for item in array_value:
        if isinstance(item, dict):
            value = get_value(item, property_path)
            
            match = False
            if operator in ('==', '==='):
                match = str(value) == str(expected_value)
            elif operator in ('!=', '!=='):
                match = str(value) != str(expected_value)
            elif operator == '>':
                try:
                    match = float(str(value)) > float(str(expected_value))
                except (ValueError, TypeError):
                    pass
            elif operator == '<':
                try:
                    match = float(str(value)) < float(str(expected_value))
                except (ValueError, TypeError):
                    pass
            elif operator == '>=':
                try:
                    match = float(str(value)) >= float(str(expected_value))
                except (ValueError, TypeError):
                    pass
            elif operator == '<=':
                try:
                    match = float(str(value)) <= float(str(expected_value))
                except (ValueError, TypeError):
                    pass
            
            if match:
                filtered.append(item)
    
    return TransformResult(filtered)


def apply_count(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 2:
        return TransformResult(0, 'count() requires exactly 2 arguments: count(arrayField.property, value)')
    
    field_path = args[0]
    expected_value = unquote(args[1])
    
    path_parts = field_path.split('.')
    if len(path_parts) < 2:
        return TransformResult(0, 'count() requires a property path like "arrayField.property"')
    
    array_field_name = path_parts[0]
    property_path = '.'.join(path_parts[1:])
    
    array_value = get_value(data, array_field_name)
    
    if not isinstance(array_value, list):
        return TransformResult(0, f'Field \'{array_field_name}\' is not an array')
    
    count = 0
    for item in array_value:
        if isinstance(item, dict):
            value = get_value(item, property_path)
            if str(value) == str(expected_value):
                count += 1
    
    return TransformResult(count)


def apply_first(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 1:
        return TransformResult(None, 'first() requires exactly 1 argument')
    
    value = get_value(data, args[0])
    
    if isinstance(value, list) and len(value) > 0:
        return TransformResult(value[0])
    
    return TransformResult(None)


def apply_last(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 1:
        return TransformResult(None, 'last() requires exactly 1 argument')
    
    value = get_value(data, args[0])
    
    if isinstance(value, list) and len(value) > 0:
        return TransformResult(value[-1])
    
    return TransformResult(None)


def apply_boolean(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 3:
        return TransformResult(False, 'boolean() requires exactly 3 arguments: boolean(variable, operator, value)')
    
    value = get_value(data, args[0])
    operator = unquote(args[1])
    expected_value = unquote(args[2])
    
    result = False
    if operator in ('=', '==', '==='):
        result = str(value) == str(expected_value)
    elif operator in ('!=', '!=='):
        result = str(value) != str(expected_value)
    elif operator == '>':
        try:
            result = float(str(value)) > float(str(expected_value))
        except (ValueError, TypeError):
            pass
    elif operator == '<':
        try:
            result = float(str(value)) < float(str(expected_value))
        except (ValueError, TypeError):
            pass
    elif operator == '>=':
        try:
            result = float(str(value)) >= float(str(expected_value))
        except (ValueError, TypeError):
            pass
    elif operator == '<=':
        try:
            result = float(str(value)) <= float(str(expected_value))
        except (ValueError, TypeError):
            pass
    elif operator == 'exists':
        result = value not in (None, '', [])
    elif operator == 'not exists':
        result = value in (None, '', [])
    else:
        return TransformResult(False, f'Invalid operator: {operator}')
    
    return TransformResult(result)


def apply_if(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) != 5:
        return TransformResult('', 'if() requires exactly 5 arguments: if(val1, operator, val2, thenvalue, elsevalue)')
    
    val1 = get_value(data, args[0])
    operator = unquote(args[1])
    val2 = get_value(data, args[2])
    then_value = get_value(data, args[3])
    else_value = get_value(data, args[4])
    
    condition = False
    
    if operator in ('=', '==', '==='):
        condition = str(val1) == str(val2)
    elif operator in ('!=', '!=='):
        condition = str(val1) != str(val2)
    elif operator == '>':
        try:
            condition = float(str(val1)) > float(str(val2))
        except (ValueError, TypeError):
            pass
    elif operator == '<':
        try:
            condition = float(str(val1)) < float(str(val2))
        except (ValueError, TypeError):
            pass
    elif operator == '>=':
        try:
            condition = float(str(val1)) >= float(str(val2))
        except (ValueError, TypeError):
            pass
    elif operator == '<=':
        try:
            condition = float(str(val1)) <= float(str(val2))
        except (ValueError, TypeError):
            pass
    elif operator == 'exists':
        condition = val1 not in (None, '', [])
    elif operator == 'not exists':
        condition = val1 in (None, '', [])
    else:
        return TransformResult('', f'Invalid operator: {operator}')
    
    return TransformResult(then_value if condition else else_value)


def apply_flatten(data: Dict[str, Any], args: List[str]) -> TransformResult:
    if len(args) < 1 or len(args) > 3:
        return TransformResult('', 'flatten() requires 1-3 arguments: flatten(objectField[, separator, "deep"])')
    
    value = get_value(data, args[0])
    separator = unquote(args[1]) if len(args) > 1 else ', '
    is_deep = unquote(args[2]).lower() == 'deep' if len(args) > 2 else False
    
    if not isinstance(value, dict) or value is None:
        return TransformResult(str(value))
    
    def flatten_deep(obj: Dict[str, Any], prefix: str = '') -> List[str]:
        """Helper function for deep flattening"""
        results = []
        
        for key, val in obj.items():
            if val in (None, '', []):
                continue
            
            if isinstance(val, dict):
                # Recursively flatten nested objects
                new_prefix = f'{prefix}.{key}' if prefix else key
                results.extend(flatten_deep(val, new_prefix))
            elif isinstance(val, list):
                # For arrays, just stringify them
                import json
                results.append(json.dumps(val))
            else:
                # Add primitive values
                results.append(str(val))
        
        return results
    
    if is_deep:
        # Deep flatten - recursively process nested objects
        values = flatten_deep(value)
        return TransformResult(separator.join(values))
    else:
        # Shallow flatten - only top-level values, ignore nested objects
        values = []
        for v in value.values():
            if v in (None, '', []):
                continue
            # Skip nested objects in shallow mode
            if isinstance(v, dict):
                continue
            values.append(str(v))
        
        return TransformResult(separator.join(values))


def apply_combine_image_metadata_notes(data: Dict[str, Any], args: List[str]) -> TransformResult:
    """
    Combine image_url + description into a clean notes string.
    Should output: "Image: <url> | Desc: <desc>"
    Must NOT include 'None' anywhere.
    """
    if len(args) != 0:
        return TransformResult('', 'combine_image_metadata_notes() takes no arguments')
    
    image_url = data.get("image_url")
    description = data.get("description")
    
    # Clean inputs - filter out None, empty strings, and the literal string "None"
    cleaned_url = None
    cleaned_desc = None
    
    if image_url is not None:
        url_str = str(image_url).strip()
        if url_str != "" and url_str.lower() != "none":
            cleaned_url = url_str
    
    if description is not None:
        desc_str = str(description).strip()
        if desc_str != "" and desc_str.lower() != "none":
            cleaned_desc = desc_str
    
    # Build result - only include parts that are valid
    parts = []
    if cleaned_url:
        parts.append(f"Image: {cleaned_url}")
    if cleaned_desc:
        parts.append(f"Desc: {cleaned_desc}")
    
    if not parts:
        return TransformResult("")
    
    result = " | ".join(parts)
    return TransformResult(result)


def apply_standardize_fuel_type(data: Dict[str, Any], args: List[str]) -> TransformResult:
    """
    Standardizes fuel type values to canonical forms.
    Converts: "Gas", "gas", "GAS" → "gasoline"
    Keeps: "diesel", "electric", "hybrid", "plug-in hybrid" as-is (lowercase)
    """
    if len(args) != 1:
        return TransformResult('', 'standardize_fuel_type() requires exactly 1 argument')
    
    value = get_value(data, args[0])
    
    if value is None or value == '':
        return TransformResult('')
    
    value_str = str(value).strip()
    value_lower = value_str.lower()
    
    # Normalize "gas" or "gasoline" to "gasoline"
    if value_lower in ["gas", "gasoline"]:
        return TransformResult("gasoline")
    
    # Keep other valid fuel types as lowercase
    if value_lower in ["diesel", "electric", "hybrid", "plug-in hybrid"]:
        return TransformResult(value_lower)
    
    # Return original value if not recognized (will trigger validation warning)
    return TransformResult(value_str)


# Date/Time helper functions

def parse_date(date_str: str) -> Optional[datetime]:
    """Comprehensive date parser that handles multiple common date string formats"""
    if not date_str or not isinstance(date_str, str):
        return None
    
    s = date_str.strip()
    
    if not s:
        return None
    
    # Try parsing as timestamp (numeric string)
    if re.match(r'^\d+$', s):
        timestamp = int(s)
        # Handle both seconds and milliseconds timestamps
        try:
            if timestamp > 10000000000:
                return datetime.fromtimestamp(timestamp / 1000)
            else:
                return datetime.fromtimestamp(timestamp)
        except (ValueError, OSError):
            pass
    
    # Common date patterns to try in order of specificity
    patterns = [
        # ISO formats
        (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z?$', '%Y-%m-%dT%H:%M:%S'),
        (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),
        
        # MM/DD/YYYY format
        (r'^(\d{1,2})/(\d{1,2})/(\d{4})$', None),  # Custom handler
        
        # MM-DD-YYYY format
        (r'^(\d{1,2})-(\d{1,2})-(\d{4})$', None),  # Custom handler
        
        # YYYY/MM/DD format
        (r'^(\d{4})/(\d{1,2})/(\d{1,2})$', '%Y/%m/%d'),
        
        # Month name formats
        (r'^(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})$', '%B %d, %Y'),
        (r'^(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})$', '%d %B %Y'),
        
        # Shorter year formats MM/DD/YY
        (r'^(\d{1,2})/(\d{1,2})/(\d{2})$', None),  # Custom handler
    ]
    
    for pattern, fmt in patterns:
        match = re.match(pattern, s, re.IGNORECASE)
        if match:
            try:
                if fmt is None:
                    # Custom date parsing for ambiguous formats
                    groups = match.groups()
                    if len(groups) == 3:
                        first, second, year = groups
                        first_num = int(first)
                        second_num = int(second)
                        year_num = int(year)
                        
                        # Handle 2-digit years
                        if year_num < 100:
                            year_num = 2000 + year_num if year_num <= 30 else 1900 + year_num
                        
                        # Smart format detection
                        if first_num > 12:
                            # Must be DD/MM/YYYY
                            return datetime(year_num, second_num, first_num)
                        elif second_num > 12:
                            # Must be MM/DD/YYYY
                            return datetime(year_num, first_num, second_num)
                        else:
                            # Assume MM/DD/YYYY (US format)
                            return datetime(year_num, first_num, second_num)
                else:
                    # Try multiple format variants
                    for date_fmt in [fmt, fmt.replace('%B', '%b')]:
                        try:
                            return datetime.strptime(s, date_fmt)
                        except ValueError:
                            continue
            except (ValueError, IndexError):
                continue
    
    # Try native datetime parsing as fallback
    try:
        from dateutil import parser
        return parser.parse(s)
    except:
        pass
    
    return None


def format_date(date: datetime, date_format: str) -> str:
    """Formats a datetime object according to the specified format"""
    # Use placeholders to avoid token collision during replacement
    # Placeholders must not contain M, D, or Y to avoid being replaced themselves
    result = date_format
    
    # Replace in specific order with temporary placeholders
    result = result.replace('YYYY', '___1___')
    result = result.replace('YY', '___2___')
    result = result.replace('MMMM', '___3___')
    result = result.replace('MMM', '___4___')
    result = result.replace('MM', '___5___')
    result = result.replace('M', '___6___')
    result = result.replace('DD', '___7___')
    result = result.replace('D', '___8___')
    
    # Now replace placeholders with actual values
    result = result.replace('___1___', date.strftime('%Y'))
    result = result.replace('___2___', date.strftime('%y'))
    result = result.replace('___3___', date.strftime('%B'))
    result = result.replace('___4___', date.strftime('%b'))
    result = result.replace('___5___', date.strftime('%m'))
    result = result.replace('___6___', str(date.month))
    result = result.replace('___7___', date.strftime('%d'))
    result = result.replace('___8___', str(date.day))
    
    return result


def format_number(num: float, number_format: str) -> str:
    """Formats a number according to the specified format"""
    if number_format == 'integer':
        return str(round(num))
    elif 'decimal' in number_format:
        match = re.search(r'\d+', number_format)
        decimals = int(match.group(0)) if match else 2
        return f'{num:.{decimals}f}'
    elif number_format == 'percentage':
        return f'{num * 100:.1f}%'
    elif number_format == 'ordinal':
        n = round(num)
        last_digit = n % 10
        last_two_digits = n % 100
        
        if 11 <= last_two_digits <= 13:
            return f'{n}th'
        
        if last_digit == 1:
            return f'{n}st'
        elif last_digit == 2:
            return f'{n}nd'
        elif last_digit == 3:
            return f'{n}rd'
        else:
            return f'{n}th'
    
    return str(num)


def format_phone(digits: str, phone_format: str) -> str:
    """Formats a phone number according to the specified format"""
    if phone_format in ('US', '(XXX) XXX-XXXX'):
        if len(digits) == 10:
            return f'({digits[:3]}) {digits[3:6]}-{digits[6:]}'
        elif len(digits) == 11 and digits.startswith('1'):
            return f'+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}'
    elif phone_format == 'XXX-XXX-XXXX':
        if len(digits) == 10:
            return f'{digits[:3]}-{digits[3:6]}-{digits[6:]}'
    
    return digits


def format_currency(num: float, currency_format: str) -> str:
    """Formats a currency value according to the specified format"""
    # Handle simple currency codes
    if currency_format in ('USD', '$'):
        return f'${num:,.2f}'
    elif currency_format in ('EUR', '€'):
        return f'€{num:,.2f}'
    
    # Handle format patterns like '$,0', '$,0.00', etc.
    if currency_format.startswith('$,'):
        decimal_part = currency_format[2:]
        
        if decimal_part == '0':
            return f'${num:,.0f}'
        elif re.match(r'^0\.0+$', decimal_part):
            decimal_places = len(decimal_part) - 2
            return f'${num:,.{decimal_places}f}'
    
    # Handle format patterns like '€,0', '€,0.00', etc.
    if currency_format.startswith('€,'):
        decimal_part = currency_format[2:]
        
        if decimal_part == '0':
            return f'€{num:,.0f}'
        elif re.match(r'^0\.0+$', decimal_part):
            decimal_places = len(decimal_part) - 2
            return f'€{num:,.{decimal_places}f}'
    
    # Default fallback
    return f'${num:.2f}'


def apply_transforms(data: Dict[str, Any], transforms: Dict[str, str]) -> Dict[str, Any]:
    """
    Batch apply multiple transforms to a data object
    
    Args:
        data: The data object
        transforms: Object mapping field names to transform strings
    
    Returns:
        Object with transformed values
    """
    result = {}
    
    for field, transform_string in transforms.items():
        transform_result = apply_transform(data, transform_string)
        result[field] = transform_result.value
        
        if transform_result.error:
            print(f'Transform error for field {field}: {transform_result.error}')
    
    return result

