from typing import Dict, List, Optional, TypedDict
import json
import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")


def analyze_tickets_from_transcript(
    transcript: List[Dict],
    tickets: List[Dict],
) -> List[Dict]:
    """
    Analyzes call transcript and returns updated ticket information.

    Args:
        transcript (List[Dict]): List of conversation messages
        tickets (List[Dict]): List of ServiceNow tickets

    Returns:
        List[Dict]: List of tickets with updated information for update_ticket function
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    updated_tickets = []

    # Format the transcript to be more readable and relevant
    formatted_transcript = format_transcript(transcript)

    for ticket in tickets:
        # Create a properly formatted ticket dictionary
        ticket_info = {
            "number": ticket.get("incident_number", ""),
            "short_description": ticket.get("subject", ""),
            "description": ticket.get("description", ""),
            "state": ticket.get("state", ""),
            "category": ticket.get("category", ""),
            "subcategory": ticket.get("subcategory", ""),
            "urgency": ticket.get("urgency", ""),
            "impact": ticket.get("impact", ""),
        }

        prompt = f"""
        Analyze this IT service desk conversation and ticket to determine necessary updates.
                    
        Current Ticket Information:
        Ticket Number: {ticket_info['number']}
        Subject: {ticket_info['short_description']}
        Description: {ticket_info['description']}
        State: {ticket_info['state']}
        Category: {ticket_info['category']}
        Subcategory: {ticket_info['subcategory']}
        Urgency: {ticket_info['urgency']}
        Impact: {ticket_info['impact']}

        Conversation Transcript:
        {formatted_transcript}

        Based on the conversation, determine:
        1. If the subject or description needs updating
        2. If the ticket state should change or not present (use EXACT ServiceNow states: 'New', 'Resolved', 'Open', 'Closed', 'Assigned', 'Work in Progress', 'On Hold', 'Canceled')
        - If changing to "Resolved" or "Closed", you MUST include:
            * resolution_code (choose from: SOLVED, WORKAROUND, NOT_RESOLVED, DUPLICATE)
            * close_notes (summary of resolution)
        3. If category/subcategory needs updating
        4. If urgency/impact needs adjusting (1=High, 2=Medium, 3=Low)
        5. What comments should be added based on the conversation
        6. What work_notes should be added (technical notes not visible to end user)

        Return updated in JSON format:
        {{{{
            "incident_number": "{ticket.get('incident_number', '')}",
            "subject": "updated subject if needed",
            "description": "updated description if needed",
            "state": "updated state (MUST be one of: 'New', 'Resolved', 'Open', 'Closed', 'Assigned', 'Work in Progress', 'On Hold', 'Canceled')",
            "category": "updated category if needed",
            "subcategory": "updated subcategory if needed",
            "urgency": "updated urgency if needed (1, 2, or 3)",
            "impact": "updated impact if needed (1, 2, or 3)",
            "comments": "additional context from the conversation for the end user",
            "work_notes": "technical notes from the conversation for internal use",
            "resolution_code": "REQUIRED if state=Resolved or state=Closed",
            "close_notes": "REQUIRED if state=Resolved or state=Closed - provide resolution summary"
        }}}}

        If changing state to Resolved or Closed, you MUST include both resolution_code and close_notes.
        Resolution code must be one of: SOLVED, WORKAROUND, NOT_RESOLVED, DUPLICATE.
        Close notes should summarize how the issue was resolved based on the conversation.
        State values must be EXACTLY as listed above (case-sensitive).
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an IT service desk analyst. Analyze tickets and conversations to improve ticket accuracy and completeness. Always use exact ServiceNow state names and required fields.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            # Parse the JSON response
            analysis = json.loads(response.choices[0].message.content)

            # Add the incident_number if not present
            if "incident_number" not in analysis or not analysis["incident_number"]:
                analysis["incident_number"] = ticket.get("incident_number", "")

            # # Ensure proper state names (validate and normalize if needed)
            # if 'state' in analysis:
            #     # Map common variations to exact ServiceNow states
            #     state_mappings = {
            #         'new': 'New',
            #         'open': 'Open',
            #         'assigned': 'Assigned',
            #         'work in progress': 'Work in Progress',
            #         'in progress': 'Work in Progress',
            #         'workinprogress': 'Work in Progress',
            #         'on hold': 'On Hold',
            #         'onhold': 'On Hold',
            #         'resolved': 'Resolved',
            #         'closed': 'Closed',
            #         'canceled': 'Canceled',
            #         'cancelled': 'Canceled'
            #     }

            #     # Normalize state if needed
            #     given_state = analysis['state'].lower().strip()
            #     if given_state in state_mappings:
            #         analysis['state'] = state_mappings[given_state]

            # Validate required fields for resolved/closed states
            if analysis.get("state") in ["Resolved", "Closed"]:
                if "resolution_code" not in analysis or not analysis["resolution_code"]:
                    print(
                        f"WARNING: Missing resolution_code for {analysis.get('state')} state - setting default"
                    )
                    analysis["resolution_code"] = "SOLVED"

                if "close_notes" not in analysis or not analysis["close_notes"]:
                    print(
                        f"WARNING: Missing close_notes for {analysis.get('state')} state - setting default"
                    )
                    analysis["close_notes"] = "Issue resolved based on conversation"

            # Only include fields that were returned by the analysis
            update_data = {
                k: v for k, v in analysis.items() if v is not None and v != ""
            }

            updated_tickets.append(update_data)

        except Exception as e:
            print(
                f"Error analyzing ticket {ticket.get('incident_number', '')}: {str(e)}"
            )
            continue  # Continue with next ticket instead of returning None

    return updated_tickets


def format_transcript(transcript: List[Dict]) -> str:
    """
    Formats the transcript to make it more readable and removes unnecessary information like timestamps.

    Args:
        transcript (List[Dict]): List of conversation messages

    Returns:
        str: Formatted transcript as a string
    """
    formatted_lines = []

    for message in transcript:
        role = message.get("role", "")

        # Get content and handle different possible formats
        content = message.get("content", "")
        if isinstance(content, list) and len(content) > 0:
            content = content[0]

        # Skip empty messages
        if not content:
            continue

        # Format each message as "Role: Content"
        formatted_lines.append(f"{role.capitalize()}: {content}")

    # Join all formatted lines with newlines
    return "\n".join(formatted_lines)


def generate_call_summary(transcript: List[Dict]) -> Dict:
    """
    Generates a comprehensive summary of a customer service call.

    Args:
        transcript (List[Dict]): List of conversation messages with 'role' and 'content'

    Returns:
        Dict: Summary of the call containing key information
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Format the transcript to be more readable
    formatted_transcript = format_transcript(transcript)

    # Extract the ticket number if mentioned in the conversation
    ticket_number = extract_ticket_number(transcript)

    prompt = f"""
    Analyze this IT service desk conversation transcript and provide a comprehensive summary.
    
    Conversation Transcript:
    {formatted_transcript}
    
    Please provide a summary that includes:
    1. Customer's main issue or request
    2. Key details provided by the customer
    3. Solutions offered or steps taken by the agent
    4. Resolution status (resolved, escalated, pending, etc.)
    5. Any follow-up actions or commitments made
    6. Overall sentiment/tone of the interaction
    
    Format your response as a structured JSON with these keys:
    {{
        "issue_summary": "Brief description of the main issue",
        "customer_details": "Key information provided by the customer",
        "actions_taken": "Solutions or steps taken by the agent",
        "resolution_status": "Current status (Resolved, Pending, etc.)",
        "resolution_method": "How the issue was resolved (if applicable)",
        "follow_up_needed": true/false,
        "follow_up_details": "Any scheduled actions or commitments",
        "ticket_number": "Ticket number if mentioned",
        "call_duration_estimate": "Approximate call duration in minutes based on timestamps (if available)",
        "sentiment": "Customer satisfaction level (Satisfied, Neutral, Dissatisfied)"
    }}
    
    Keep each field concise but comprehensive - aim for 1-3 sentences per field.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert call analyzer for IT service desks. Provide accurate, concise summaries of support calls.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        # Parse the JSON response
        summary = json.loads(response.choices[0].message.content)

        # If ticket number was extracted but not included in the summary, add it
        if ticket_number and not summary.get("ticket_number"):
            summary["ticket_number"] = ticket_number

        # Add metadata
        summary["generated_at"] = datetime.datetime.now().isoformat()
        summary["transcript_messages_count"] = len(transcript)

        return summary

    except Exception as e:
        print(f"Error generating call summary: {str(e)}")
        return {
            "error": "Failed to generate call summary",
            "error_details": str(e),
            "generated_at": datetime.datetime.now().isoformat(),
        }


def extract_ticket_number(transcript: List[Dict]) -> str:
    """
    Extracts the ticket number from the conversation if mentioned.

    Args:
        transcript (List[Dict]): List of conversation messages

    Returns:
        str: Ticket number if found, empty string otherwise
    """
    ticket_pattern = (
        r"(?:ticket|case|incident)(?:\s+number)?(?:\s+is)?\s+(INC\d+|(?:SR|REQ)\d+)"
    )

    for message in transcript:
        content = message.get("content", "")
        if isinstance(content, list) and len(content) > 0:
            content = content[0]

        if not content:
            continue

        # Case insensitive search for ticket numbers
        match = re.search(ticket_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)

    return ""


def estimate_call_duration(transcript: List[Dict]) -> int:
    """
    Estimates the call duration in minutes based on timestamps if available.

    Args:
        transcript (List[Dict]): List of conversation messages with timestamps

    Returns:
        int: Estimated call duration in minutes, or 0 if cannot be determined
    """
    timestamps = []

    for message in transcript:
        timestamp = message.get("timestamp", "")
        if timestamp:
            try:
                # Convert timestamp string to datetime object
                dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                timestamps.append(dt)
            except ValueError:
                continue

    if len(timestamps) >= 2:
        # Sort timestamps and calculate duration between first and last
        timestamps.sort()
        duration = timestamps[-1] - timestamps[0]
        return round(duration.total_seconds() / 60)

    return 0
