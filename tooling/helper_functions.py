import re

def extract_messages(conversation_list):
    formatted_history = []
    
    for msg in conversation_list:
        if msg.role in ['user', 'assistant']:
            formatted_history.append({
                "role": msg.role,
                "content": msg.content
            })
    
    return formatted_history


def extract_ticket_info(data):
    try:
        # Check if the request was successful
        if data.get('code') != 200:
            return f"Error: {data.get('message', 'Unknown error')}"
        
        # Extract ticket information
        tickets = []
        for ticket in data.get('data', []):
            # Map state values to readable format
            state_mapping = {
                '1': 'New',
                '2': 'In Progress',
                '3': 'On Hold',
                '4': 'Resolved',
                '5': 'Closed',
                '6': 'Resolved',
                '7': 'Cancelled'
            }
            
            state_value = ticket.get('state', '')
            state_readable = state_mapping.get(state_value, f"Unknown ({state_value})")
            
            tickets.append({
                'incident_number': ticket.get('number', ''),
                'state': state_readable,
                'short_description': ticket.get('short_description', ''),
                'description': ticket.get('description', ''),
                'opened_at': ticket.get('opened_at', '')
            })
        
        return tickets
    
    except Exception as e:
        return f"Error: {str(e)}"


def validate_dtmf_employee_id(dtmf_input):
    """
    Validates DTMF input for employee ID.
    Employee ID must be exactly 6 digits long and contain only numeric digits (0-9).
    
    Args:
        dtmf_input (str): The DTMF input string to validate
    
    Returns:
        str or None: Returns the employee ID if valid, None otherwise
    """
    # Check if input is None or empty
    if not dtmf_input:
        return None
    
    # Check if exactly 6 characters long
    if len(dtmf_input) != 6:
        return None
    
    # Check if all characters are digits (0-9)
    if not dtmf_input.isdigit():
        return None
    
    # If all validations pass, return the employee ID
    return dtmf_input


async def verify_employee_dtmf(userdata, employeeId):
    """
    Verify employee by ID and return their details.
    
    Args:
        userdata: User data object containing snow instance
        employeeId: Employee ID to verify
        
    Returns:
        dict: Employee details with full_name and sys_id, or None values if not found
        
    Raises:
        ValueError: If required parameters are missing or invalid
        Exception: For other unexpected errors
    """

    if hasattr(userdata, 'inactivity_monitor'):
        userdata.inactivity_monitor.start_function_call("verify_employee")

    try:      
        employeeId = str(employeeId).strip()          
        snow = userdata.snow
        
        # Step 1: Get user sys_id by employee number
        try:
            user_sys_id = await snow.get_user_sys_id_by_employee_number_dict(employeeId)
        except Exception as e:
            if hasattr(userdata, 'inactivity_monitor'):
                userdata.inactivity_monitor.end_function_call("verify_employee")

            return {
                "full_name": None,
                "sys_id": None,
                "error": f"Employee ID {employeeId} not found"
            }
        
        # Check if sys_id was found
        if not user_sys_id:
            logging.warning(f"No sys_id found for employee ID: {employeeId}")
            if hasattr(userdata, 'inactivity_monitor'):
                userdata.inactivity_monitor.end_function_call("verify_employee")

            return {
                "full_name": None,
                "sys_id": None,
                "error": f"Employee ID {employeeId} not found"
            }
        
        # Step 2: Get user details by sys_id
        try:
            user_details_response = await snow.get_user_by_user_sys_id(user_sys_id)
        except Exception as e:
            if hasattr(userdata, 'inactivity_monitor'):
                userdata.inactivity_monitor.end_function_call("verify_employee")

            return {
                "full_name": None,
                "sys_id": user_sys_id,
                "error": f"Failed to retrieve user details"
            }
        
        # Step 3: Parse user details
        if not user_details_response:
            if hasattr(userdata, 'inactivity_monitor'):
                userdata.inactivity_monitor.end_function_call("verify_employee")

            return {
                "full_name": None,
                "sys_id": user_sys_id,
                "error": "Empty user details response"
            }
        
        # Extract user data
        user_data = user_details_response.get('data', {})
        
        if not isinstance(user_data, dict):
            if hasattr(userdata, 'inactivity_monitor'):
                userdata.inactivity_monitor.end_function_call("verify_employee")

            return {
                "full_name": None,
                "sys_id": user_sys_id,
                "error": "Invalid user data format"
            }
        
        # Check for records
        if 'records' not in user_data:
            if hasattr(userdata, 'inactivity_monitor'):
                userdata.inactivity_monitor.end_function_call("verify_employee")

            return {
                "full_name": None,
                "sys_id": user_sys_id,
                "error": "No records found in user data"
            }
        
        records = user_data['records']
        if not records or not isinstance(records, list):
            if hasattr(userdata, 'inactivity_monitor'):
                userdata.inactivity_monitor.end_function_call("verify_employee")

            return {
                "full_name": None,
                "sys_id": user_sys_id,
                "error": "No user records found"
            }
        
        # Extract user details
        user_record = records[0]
        if not isinstance(user_record, dict):
            if hasattr(userdata, 'inactivity_monitor'):
                userdata.inactivity_monitor.end_function_call("verify_employee")

            return {
                "full_name": None,
                "sys_id": user_sys_id,
                "error": "Invalid user record format"
            }
        
        user_full_name = user_record.get('name', '').strip()
        
        if not user_full_name:
            if hasattr(userdata, 'inactivity_monitor'):
                userdata.inactivity_monitor.end_function_call("verify_employee")

            return {
                "full_name": None,
                "sys_id": user_sys_id,
                "error": "User name not found"
            }
        
        user_details = {
            "full_name": user_full_name,
            "sys_id": user_sys_id,
            "error": None
        }
        if hasattr(userdata, 'inactivity_monitor'):
            userdata.inactivity_monitor.end_function_call("verify_employee")

        return user_details
        
    except ValueError as ve:
        if hasattr(userdata, 'inactivity_monitor'):
            userdata.inactivity_monitor.end_function_call("verify_employee")

        return {
            "full_name": None,
            "sys_id": None,
            "error": str(ve)
        }
    except Exception as e:
        if hasattr(userdata, 'inactivity_monitor'):
            userdata.inactivity_monitor.end_function_call("verify_employee")

        return {
            "full_name": None,
            "sys_id": None,
            "error": f"System error during verification: {str(e)}"
        }


def process_llm_output_for_tts(text):
    """
    Process LLM output to replace special symbols, short forms, URLs, and service tickets that may cause TTS issues.
    
    Args:
        text (str): The original LLM output text
        
    Returns:
        str: Processed text with special symbols, short forms, URLs, and service tickets replaced
    """
    # Handle service ticket patterns (INC followed by digits)
    def spell_out_ticket(match):
        """Convert ticket number to spelled out format"""
        ticket = match.group(0)
        # Split into characters and join with spaces
        spelled_out = ','.join(list(ticket))
        return spelled_out
    
    # Apply service ticket transformation
    processed_text = re.sub(r'\bINC\d+\b', spell_out_ticket, text, flags=re.IGNORECASE)
    
    # You can add more ticket patterns here if needed
    # For example, if you have other ticket types like REQ, CHG, etc.
    processed_text = re.sub(r'\bREQ\d+\b', spell_out_ticket, processed_text, flags=re.IGNORECASE)
    processed_text = re.sub(r'\bCHG\d+\b', spell_out_ticket, processed_text, flags=re.IGNORECASE)
    processed_text = re.sub(r'\bSR\d+\b', spell_out_ticket, processed_text, flags=re.IGNORECASE)
    
    # Handle URLs and domain patterns
    # Handle full URLs (http/https)
    processed_text = re.sub(
        r'https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)',
        lambda m: f"h t t p {'s' if m.group(0).startswith('https') else ''} {m.group(1).replace('.', ' dot ').replace('/', ' slash ')}",
        processed_text
    )
    
    # Handle domain patterns (word.word format)
    processed_text = re.sub(
        r'\b([a-zA-Z0-9-]+)\.([a-zA-Z]{2,})\b',
        r'\1 dot \2',
        processed_text
    )
    
    # Handle email addresses
    processed_text = re.sub(
        r'\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
        lambda m: f"{m.group(1).replace('.', ' dot ')} at {m.group(2).replace('.', ' dot ')}",
        processed_text
    )
    
    # Handle multi-character symbol combinations
    multi_char_replacements = {
        ':/': ' colon slash ',
        ':;': ' colon semicolon ',
        ';:': ' semicolon colon ',
        ';&': ' semicolon and ',
        ':&': ' colon and ',
        '://': ' colon slash slash ',
    }
    
    # Apply multi-character replacements
    for combo, replacement in multi_char_replacements.items():
        processed_text = processed_text.replace(combo, replacement)
    
    # Dictionary for single character symbol replacements
    single_char_replacements = {
        ';': ' semicolon ',
        '&': ' and ',
        '@': ' at ',
        '%': ' percent ',
        '#': ' hash ',
        '$': ' dollar ',
        '+': ' plus ',
        '=': ' equals ',
    }
    
    # Apply single character replacements
    for symbol, replacement in single_char_replacements.items():
        processed_text = processed_text.replace(symbol, replacement)
    
    # Dictionary for short forms and contractions (case-insensitive)
    short_forms = {
        # Common contractions
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bshan't\b": "shall not",
        r"\bain't\b": "am not",
        r"\bn't\b": " not",
        r"\b'll\b": " will",
        r"\b're\b": " are",
        r"\b've\b": " have",
        r"\b'd\b": " would",
        r"\bI'm\b": "I am",
        r"\bhe's\b": "he is",
        r"\bshe's\b": "she is",
        r"\bit's\b": "it is",
        r"\bthat's\b": "that is",
        r"\bwhat's\b": "what is",
        r"\bwhere's\b": "where is",
        r"\bwhen's\b": "when is",
        r"\bwho's\b": "who is",
        r"\bhow's\b": "how is",
        r"\blet's\b": "let us",
        
        # Common abbreviations
        r"\betc\b\.?": "et cetera",
        r"\be\.g\.\b": "for example",
        r"\bi\.e\.\b": "that is",
        r"\bvs\b\.?": "versus",
        r"\bw/\b": "with",
        r"\bw/o\b": "without",
        r"\bb/c\b": "because",
        r"\bb4\b": "before",
        r"\bu\b": "you",
        r"\bur\b": "your",
        r"\br\b": "are",
        r"\bu're\b": "you are",
        r"\bthru\b": "through",
        r"\btho\b": "though",
    }
    
    # Apply short form replacements (case-insensitive)
    for pattern, replacement in short_forms.items():
        processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    processed_text = re.sub(r'\s+', ' ', processed_text)
    
    # Strip leading/trailing whitespace
    processed_text = processed_text.strip()
    
    return processed_text

