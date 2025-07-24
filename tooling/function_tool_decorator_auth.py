"""
These function tool only works with apigee


"""

import json
import traceback
from typing import Dict, Any, Optional, List
from livekit.agents import function_tool, RunContext
from livekit.agents import ChatMessage, ChatContext
import logging
from tooling.helper_functions import extract_ticket_info, validate_dtmf_employee_id
from servicenow.servicenow_apigee import ServiceNowError, ServiceNowNotFoundError

# transfer calls
from livekit import api
from livekit.protocol.sip import TransferSIPParticipantRequest

logger = logging.getLogger(__name__)


@function_tool
async def verify_employee(context: RunContext, employeeId: str):
    """Verifies if an employee exists in the system using their employee ID

    This function queries the ServiceNow API to validate if the provided employee ID
    exists in the database. It notifies the user that their ID is being verified and
    returns the employee's full name if verified successfully.

    Args:
        employeeId: The employee's unique identification number in the system

    Returns:
        A JSON object containing verification result and employee name if successful
    """
    try:
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.start_function_call(
                "verify_employee"
            )

        if hasattr(context.session.userdata, "function_tool_tracker"):
            tracker = context.session.userdata.function_tool_tracker

        dtmf_input = context.session.userdata.dtmf_input
        logger.info(f"DTMF Signal recived = {dtmf_input}")
        dtmf_employeeId = validate_dtmf_employee_id(dtmf_input)

        if dtmf_employeeId:
            employeeId = dtmf_employeeId

        # Start tracking
        call_id = tracker.start_function_call(
            "verify_employee", inputs={"User Employee ID": employeeId}
        )

        context.session.say("Please wait while I verify your ID", add_to_chat_ctx=False)
        snow = context.session.userdata.snow
        if not snow:
            logger.error("ServiceNow client not initialized in session")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "verify_employee"
                )

            return {
                "result": json.dumps(
                    {"result": "Error", "message": "Internal system error occurred"}
                )
            }

        # Use the get_user_sys_id_by_employee_number method from the updated client
        user_sys_id = await snow.get_user_by_id(employeeId)

        if not user_sys_id:
            logger.warning(f"No employee found with ID: {employeeId}")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "verify_employee"
                )

            return {
                "result": json.dumps(
                    {
                        "result": "Employee Not Verified",
                        "message": "No employee records found",
                    }
                )
            }

        # Now get the full user details using the sys_id
        user_details_response = await snow.get_user_by_user_sys_id(user_sys_id)

        if user_details_response.get("code") == 200:
            user_data = user_details_response.get("data", {})

            if (
                isinstance(user_data, dict)
                and "records" in user_data
                and user_data["records"]
            ):
                user_record = user_data["records"][0]
                user_full_name = user_record.get("name")

            # Store user_sys_id in session for later use
            context.session.userdata.user_sys_id = user_sys_id
            context.session.userdata.full_name = user_full_name
            await context.session.userdata.update_call_user_info(
                full_name=user_full_name, user_sys_id=user_sys_id
            )

            user_details = {
                "full_name": user_full_name,
                "sys_id": user_sys_id,
            }
            logger.info(f"Employee Verified===============>: {user_details}")
            logger.info(
                f"User sys_id set to ===============>(verify_employee): {context.session.userdata.user_sys_id}"
            )

            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "verify_employee"
                )

            # update tracker
            tracker.complete_function_call(call_id, user_details)

            return {
                "result": json.dumps(
                    {
                        "result": "Employee Verified, greet the user with their name",
                        "employee_name": user_full_name,
                    }
                )
            }
        else:
            error_message = user_details_response.get("message", "Verification failed")
            logger.error(
                f"Employee verification failed with code {user_details_response.get('code')}: {error_message}"
            )

            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "verify_employee"
                )

            return {
                "result": json.dumps(
                    {"result": "Employee Not Verified", "message": error_message}
                )
            }

    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Exception in verify_employee: {str(e)}\n{error_detail}")
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "verify_employee"
            )

        tracker.fail_function_call(call_id, str(e))

        return {
            "result": json.dumps(
                {
                    "result": "Error",
                    "message": "An unexpected error occurred while verifying employee",
                }
            )
        }


@function_tool
async def retrieve_previous_ticket(context: RunContext):
    """Retrieves most recent incident tickets filed by an employee

    This function queries the ServiceNow database for the tickets associated with
    the provided employee ID. It notifies the user that information is being gathered and
    returns details about each ticket including ticket numbers, descriptions, status, and dates.

    Returns:
        A dictionary containing formatted information about the employee's previous tickets
    """
    try:
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.start_function_call(
                "retrieve_previous_ticket"
            )

        snow = context.session.userdata.snow
        if not snow:
            logger.error("ServiceNow client not initialized in session")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "retrieve_previous_ticket"
                )

            return {
                "Previous Tickets": [],
                "Status": "Failed",
                "Message": "Internal system error occurred",
            }
        logger.info(
            f"User sys_id set to ===============>(retrieve_previous_ticket): {context.session.userdata.user_sys_id}"
        )
        user_sys_id = context.session.userdata.user_sys_id
        if not user_sys_id:
            logger.error("User sys_id not initialized in session")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "retrieve_previous_ticket"
                )

            return {
                "Incident Number": None,
                "Status": "Failed",
                "Message": "Internal system error occurred",
            }
        context.session.say("Let me gather some details for you", add_to_chat_ctx=False)

        # Use the new get_recent_tickets_by_creator method with a limit of 3
        previous_tickets_response = await snow.get_recent_tickets_by_creator(
            user_sys_id=user_sys_id, limit=3
        )

        # Check the response for success
        if (
            not previous_tickets_response
            or previous_tickets_response.get("code") != 200
        ):
            error_message = (
                previous_tickets_response.get("message", "Failed to retrieve tickets")
                if isinstance(previous_tickets_response, dict)
                else "No ticket data returned"
            )
            logger.error(f"Failed to retrieve tickets: {error_message}")

            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "retrieve_previous_ticket"
                )

            return {
                "Previous Tickets": [],
                "Status": "Failed",
                "Message": "Unable to retrieve previous tickets",
            }

        # Extract ticket data from the response
        tickets_data = previous_tickets_response.get("data", [])

        # Format the ticket information
        formatted_previous_tickets = []
        for ticket in tickets_data:
            # Extract values using _safe_extract_field helper for safer field extraction
            incident_number = snow._safe_extract_field(
                ticket.get("number"),
                "display_value",
                snow._safe_extract_field(ticket.get("number"), "value", "N/A"),
            )

            short_description = snow._safe_extract_field(
                ticket.get("short_description"),
                "display_value",
                snow._safe_extract_field(
                    ticket.get("short_description"), "value", "N/A"
                ),
            )

            state = snow._safe_extract_field(
                ticket.get("state"),
                "display_value",
                snow._safe_extract_field(ticket.get("state"), "value", "N/A"),
            )

            created_date = snow._safe_extract_field(
                ticket.get("sys_created_on"),
                "display_value",
                snow._safe_extract_field(ticket.get("sys_created_on"), "value", "N/A"),
            )

            # Get link directly if available
            link = ticket.get("link", "#")

            formatted_ticket = {
                "Incident Number": incident_number,
                "Short Description": short_description,
                "Status": state,
                "Created Date": created_date,
                "Link": link,
            }
            formatted_previous_tickets.append(formatted_ticket)

        logger.info(
            f"FORMATTED PREVIOUS TICKET ==================> {formatted_previous_tickets}"
        )

        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "retrieve_previous_ticket"
            )

        return {
            "Previous Tickets": formatted_previous_tickets,
            "Status": "Fetched tickets successfully, see if these tickets are related to current issue user is facing",
        }
    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Exception in retrieve_previous_ticket: {str(e)}\n{error_detail}")

        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "retrieve_previous_ticket"
            )

        return {
            "Previous Tickets": [],
            "Status": "Error",
            "Message": "An unexpected error occurred while retrieving previous tickets",
        }


@function_tool
async def create_ticket(
    context: RunContext, subject: str, description: str, urgency: str, impact: str
):
    """Creates a new incident ticket in the ServiceNow system based on the user's issue

    This function submits a new incident ticket to ServiceNow with the provided details.
    The function requires specific information about the issue including subject, detailed
    description, urgency level (e.g., "high", "medium", "low"), impact level, and the
    employee's ID. It stores the ticket in the system and returns the ticket number for reference.

    Args:
        subject: A brief title describing the issue
        description: Detailed explanation of the problem being experienced
        urgency: Priority level indicating how quickly the issue needs resolution (1-3, where 1 is highest)
        impact: Assessment of how severely the issue affects the employee (1-3, where 1 is highest)

    Returns:
        A dictionary containing the incident number and creation status
    """
    try:
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.start_function_call(
                "create_ticket"
            )

        context.session.say(
            "Please be with me, while I work on this", add_to_chat_ctx=False
        )
        snow = context.session.userdata.snow
        if not snow:
            logger.error("ServiceNow client not initialized in session")

            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "create_ticket"
                )

            return {
                "Incident Number": None,
                "Status": "Failed",
                "Message": "Internal system error occurred",
            }

        user_sys_id = context.session.userdata.user_sys_id
        if not user_sys_id:
            logger.error("User sys_id not initialized in session")

            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "create_ticket"
                )

            return {
                "Incident Number": None,
                "Status": "Failed",
                "Message": "Internal system error occurred",
            }

        logger.info(
            f"User sys_id set to ===============>(create_ticket): {context.session.userdata.user_sys_id}"
        )

        # Validate inputs
        if not all([subject, description, urgency, impact]):
            missing_fields = []
            if not subject:
                missing_fields.append("subject")
            if not description:
                missing_fields.append("description")
            if not urgency:
                missing_fields.append("urgency")
            if not impact:
                missing_fields.append("impact")

            logger.error(
                f"Missing required fields for ticket creation: {', '.join(missing_fields)}"
            )
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "create_ticket"
                )

            return {
                "Incident Number": None,
                "Status": "Failed",
                "Message": f"Missing required information: {', '.join(missing_fields)}",
            }

        # Convert string urgency/impact to numeric if needed
        urgency_map = {"high": "1", "medium": "2", "low": "3"}
        impact_map = {"high": "1", "medium": "2", "low": "3"}

        numeric_urgency = (
            urgency_map.get(urgency.lower(), urgency)
            if isinstance(urgency, str)
            else urgency
        )
        numeric_impact = (
            impact_map.get(impact.lower(), impact)
            if isinstance(impact, str)
            else impact
        )

        # Use the new create_incident method from the updated client
        # Map fields to the expected names for create_incident
        incident_data = {
            "short_description": subject,
            "description": description,
            "urgency": numeric_urgency,
            "impact": numeric_impact,
            "caller_id": user_sys_id,
        }

        logger.info(f"Creating incident with data: {incident_data}")
        result = await snow.create_incident(**incident_data)

        logger.debug(f"Incident Creation Result: {result}")

        # Check if result has the expected structure
        if not result:
            logger.error("No result returned from incident creation")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "create_ticket"
                )

            return {
                "Incident Number": None,
                "Status": "Failed",
                "Message": "Unable to create ticket",
            }

        # Extract incident number and sys_id using _safe_extract_field helper
        incident_number = snow._safe_extract_field(
            result.get("number"),
            "display_value",
            snow._safe_extract_field(result.get("number"), "value", None),
        )

        incident_sys_id = snow._safe_extract_field(result.get("sys_id"), "value")

        if not incident_number or not incident_sys_id:
            logger.error("Missing incident number or sys_id in response")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "create_ticket"
                )

            return {
                "Incident Number": None,
                "Status": "Failed",
                "Message": "Ticket created but missing incident number or ID",
            }

        logger.info(f"Adding ticket details =========> {result}")

        # Store ticket data with proper field extraction
        stored_ticket_data = {
            "incident_number": incident_number,
            "sys_id": incident_sys_id,
            "subject": snow._safe_extract_field(
                result.get("short_description"),
                "display_value",
                snow._safe_extract_field(
                    result.get("short_description"), "value", subject
                ),
            ),
            "description": snow._safe_extract_field(
                result.get("description"),
                "display_value",
                snow._safe_extract_field(
                    result.get("description"), "value", description
                ),
            ),
            "state": snow._safe_extract_field(
                result.get("state"),
                "display_value",
                snow._safe_extract_field(result.get("state"), "value", "1"),
            ),
            "category": snow._safe_extract_field(
                result.get("category"),
                "display_value",
                snow._safe_extract_field(result.get("category"), "value", None),
            ),
            "subcategory": snow._safe_extract_field(
                result.get("subcategory"),
                "display_value",
                snow._safe_extract_field(result.get("subcategory"), "value", None),
            ),
            "urgency": snow._safe_extract_field(
                result.get("urgency"),
                "display_value",
                snow._safe_extract_field(
                    result.get("urgency"), "value", numeric_urgency
                ),
            ),
            "impact": snow._safe_extract_field(
                result.get("impact"),
                "display_value",
                snow._safe_extract_field(result.get("impact"), "value", numeric_impact),
            ),
            "created_on": snow._safe_extract_field(
                result.get("sys_created_on"),
                "display_value",
                snow._safe_extract_field(result.get("sys_created_on"), "value", None),
            ),
            "link": result.get("link", None),  # Include the link if available
        }

        # Safely append to incident_tickets
        # Initialize incident_tickets if it doesn't exist or is None
        if context.session.userdata.incident_tickets is None:
            context.session.userdata.incident_tickets = []

        context.session.userdata.incident_tickets.append(stored_ticket_data)

        # Update call record with ticket information
        if context.session.userdata.db_operations and context.session.userdata.call_id:
            try:
                # Get all incident ticket IDs
                incident_ticket_ids = [
                    ticket.get("incident_number")
                    for ticket in context.session.userdata.incident_tickets
                    if ticket.get("incident_number")
                ]

                # Use the first ticket as the primary ticket_id
                primary_ticket_id = (
                    incident_ticket_ids[0] if incident_ticket_ids else None
                )

                await context.session.userdata.db_operations.update_call_ticket_info(
                    call_id=context.session.userdata.call_id,
                    ticket_id=primary_ticket_id,
                    incident_ticket_ids=incident_ticket_ids,
                )
                logger.info(f"Call record updated with ticket info: {incident_number}")
            except Exception as e:
                logger.warning(f"Could not update call record with ticket info: {e}")

        logger.info(f"New ticket created =========> {incident_number}")

        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "create_ticket"
            )

        return {
            "Incident Number": incident_number,
            "Status": "Ticket created successfully, inform the ticket number to the user",
        }

    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Exception in create_ticket: {str(e)}\n{error_detail}")
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "create_ticket"
            )

        return {
            "Incident Number": None,
            "Status": "Error",
            "Message": "An unexpected error occurred while creating the ticket",
        }


@function_tool
async def update_ticket(
    context: RunContext,
    incident_number: str,
    comments: str = None,
    work_notes: str = None,
):
    """Updates an existing incident ticket in the ServiceNow system

    This function updates an existing ticket in ServiceNow with additional information
    such as comments or work notes. It requires the incident number and at least one
    field to update.

    Args:
        incident_number: The reference number of the incident to update
        comments: Optional comments to add to the ticket (visible to end user)
        work_notes: Optional technical notes to add to the ticket (internal use)

    Returns:
        A dictionary containing the update status and message
    """
    try:
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.start_function_call(
                "update_ticket"
            )

        snow = context.session.userdata.snow
        if not snow:
            logger.error("ServiceNow client not initialized in session")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_ticket"
                )

            return {"Status": "Failed", "Message": "Internal system error occurred"}

        # Validate inputs
        if not incident_number:
            logger.error("Missing incident number for ticket update")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_ticket"
                )

            return {"Status": "Failed", "Message": "Missing incident number"}

        if not comments and not work_notes:
            logger.error("No update data provided for ticket update")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_ticket"
                )

            return {"Status": "Failed", "Message": "No update data provided"}

        # First, get the incident sys_id from its number
        try:
            incident_data = await snow.get_incident_by_number_internal(incident_number)

            # Extract the sys_id using the _safe_extract_field helper
            incident_sys_id = snow._safe_extract_field(
                incident_data.get("sys_id"), "value"
            )

            if not incident_sys_id:
                logger.error(
                    f"Could not find sys_id for incident number {incident_number}"
                )
                if hasattr(context.session.userdata, "inactivity_monitor"):
                    context.session.userdata.inactivity_monitor.end_function_call(
                        "update_ticket"
                    )

                return {
                    "Status": "Failed",
                    "Message": f"Incident {incident_number} not found",
                }

        except ServiceNowNotFoundError:
            logger.error(f"Incident {incident_number} not found")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_ticket"
                )

            return {
                "Status": "Failed",
                "Message": f"Incident {incident_number} not found",
            }
        except Exception as e:
            logger.error(f"Error retrieving incident details: {str(e)}")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_ticket"
                )

            return {"Status": "Failed", "Message": "Error retrieving incident details"}

        # Prepare update data
        update_data = {}
        if comments:
            update_data["comments"] = comments
        if work_notes:
            update_data["work_notes"] = work_notes

        logger.info(f"Updating incident {incident_number} with data: {update_data}")

        # Use the update_incident method to apply the changes
        updated_incident = await snow.update_incident(incident_sys_id, **update_data)

        if not updated_incident:
            logger.error(f"Failed to update incident {incident_number}")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_ticket"
                )

            return {"Status": "Failed", "Message": "Failed to update incident"}

        # Update was successful
        logger.info(f"Successfully updated incident {incident_number}")

        # Find and update the incident in context.session.userdata.incident_tickets if it exists
        if context.session.userdata.incident_tickets:
            for idx, ticket in enumerate(context.session.userdata.incident_tickets):
                if (
                    ticket.get("incident_number") == incident_number
                    or ticket.get("sys_id") == incident_sys_id
                ):
                    # Update the stored ticket data
                    if comments:
                        context.session.userdata.incident_tickets[idx]["comments"] = (
                            comments
                        )
                    if work_notes:
                        context.session.userdata.incident_tickets[idx]["work_notes"] = (
                            work_notes
                        )
                    logger.info(f"Updated incident in session data: {incident_number}")
                    break

        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "update_ticket"
            )

        return {
            "Status": "Success",
            "Message": f"Incident {incident_number} updated successfully",
            "Updated Fields": list(update_data.keys()),
        }

    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Exception in update_ticket: {str(e)}\n{error_detail}")
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "update_ticket"
            )

        return {
            "Status": "Error",
            "Message": "An unexpected error occurred while updating the ticket",
        }


@function_tool
async def update_ticket_state(
    context: RunContext,
    incident_number: str,
    state: str,
    comment: str = None,
    work_note: str = None,
):
    """Updates the state of an existing incident ticket in the ServiceNow system

    This function updates the state of an existing ticket in ServiceNow and optionally
    adds comments or work notes. Valid states include: "New", "Open", "In Progress",
    "On Hold", "Resolved", "Closed", "Canceled".

    Args:
        incident_number: The reference number of the incident to update
        state: The new state to set for the incident
        comment: Optional comment to add (visible to end user)
        work_note: Optional work note to add (internal use)

    Returns:
        A dictionary containing the update status and message
    """
    try:
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.start_function_call(
                "update_ticket_state"
            )

        snow = context.session.userdata.snow
        if not snow:
            logger.error("ServiceNow client not initialized in session")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_ticket_state"
                )

            return {"Status": "Failed", "Message": "Internal system error occurred"}

        # Validate inputs
        if not incident_number:
            logger.error("Missing incident number for state update")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_ticket_state"
                )

            return {"Status": "Failed", "Message": "Missing incident number"}

        if not state:
            logger.error("Missing state for ticket state update")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_ticket_state"
                )

            return {"Status": "Failed", "Message": "Missing state information"}

        # Normalize state names to match expected values
        # Map common alternative names to the exact state names expected by ServiceNow
        state_map = {
            "in progress": "Work in Progress",
            "inprogress": "Work in Progress",
            "working": "Work in Progress",
            "wip": "Work in Progress",
            "hold": "On Hold",
            "onhold": "On Hold",
            "complete": "Resolved",
            "fixed": "Resolved",
            "done": "Resolved",
            "finish": "Closed",
            "finished": "Closed",
            "cancel": "Canceled",
            "cancelled": "Canceled",
        }

        # Normalize state (preserve original casing if not in map)
        normalized_state = state_map.get(state.lower(), state)

        # First, get the incident sys_id from its number
        try:
            incident_data = await snow.get_incident_by_number_internal(incident_number)

            # Extract the sys_id using the _safe_extract_field helper
            incident_sys_id = snow._safe_extract_field(
                incident_data.get("sys_id"), "value"
            )

            if not incident_sys_id:
                logger.error(
                    f"Could not find sys_id for incident number {incident_number}"
                )
                if hasattr(context.session.userdata, "inactivity_monitor"):
                    context.session.userdata.inactivity_monitor.end_function_call(
                        "update_ticket_state"
                    )

                return {
                    "Status": "Failed",
                    "Message": f"Incident {incident_number} not found",
                }

        except ServiceNowNotFoundError:
            logger.error(f"Incident {incident_number} not found")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_ticket_state"
                )

            return {
                "Status": "Failed",
                "Message": f"Incident {incident_number} not found",
            }
        except Exception as e:
            logger.error(f"Error retrieving incident details: {str(e)}")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_ticket_state"
                )

            return {"Status": "Failed", "Message": "Error retrieving incident details"}

        logger.info(
            f"Updating incident {incident_number} state to '{normalized_state}'"
        )

        # Use the update_incident_state method to apply the state change
        state_update_result = await snow.update_incident_state(
            incident_sys_id=incident_sys_id,
            action=normalized_state,
            comment=comment if comment else "",
            work_note=work_note if work_note else "",
        )

        # Check response
        if not state_update_result or state_update_result.get("code") != 200:
            error_message = (
                state_update_result.get("message", "Failed to update state")
                if isinstance(state_update_result, dict)
                else "No data returned"
            )
            logger.error(f"Failed to update incident state: {error_message}")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_ticket_state"
                )

            return {"Status": "Failed", "Message": "Unable to update incident state"}

        # State update was successful
        logger.info(
            f"Successfully updated incident {incident_number} state to {normalized_state}"
        )

        # Find and update the incident in context.session.userdata.incident_tickets if it exists
        if context.session.userdata.incident_tickets:
            for idx, ticket in enumerate(context.session.userdata.incident_tickets):
                if (
                    ticket.get("incident_number") == incident_number
                    or ticket.get("sys_id") == incident_sys_id
                ):
                    # Update the stored ticket data
                    context.session.userdata.incident_tickets[idx]["state"] = (
                        normalized_state
                    )
                    if comment:
                        context.session.userdata.incident_tickets[idx]["comments"] = (
                            comment
                        )
                    if work_note:
                        context.session.userdata.incident_tickets[idx]["work_notes"] = (
                            work_note
                        )
                    logger.info(
                        f"Updated incident state in session data: {incident_number}"
                    )
                    break

        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "update_ticket_state"
            )

        return {
            "Status": "Success",
            "Message": f"Incident {incident_number} state updated to '{normalized_state}' successfully",
            "New State": normalized_state,
        }

    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Exception in update_ticket_state: {str(e)}\n{error_detail}")
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "update_ticket_state"
            )

        return {
            "Status": "Error",
            "Message": "An unexpected error occurred while updating the ticket state",
        }


@function_tool
async def update_snow_ticket(
    context: RunContext,
    incident_number: str,
    comments: str = None,
    work_notes: str = None,
    action: str = None,
):
    """Updates an existing incident ticket in the ServiceNow system

    This function updates an existing ticket in ServiceNow with additional information
    such as comments, work notes, or state changes. It requires the incident number 
    and at least one field to update.

    Args:
        incident_number: The reference number of the incident to update
        comments: Optional comments to add to the ticket (visible to end user)
        work_notes: Optional technical notes to add to the ticket (internal use)
        action: Optional state action (e.g., 'close', 'reopen', 'resolve')

    Returns:
        A dictionary containing the update status and message
    """
    try:
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.start_function_call(
                "update_snow_ticket"
            )

        snow = context.session.userdata.snow
        if not snow:
            logger.error("ServiceNow client not initialized in session")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_snow_ticket"
                )

            return {"Status": "Failed", "Message": "Internal system error occurred"}

        # Validate inputs
        if not incident_number:
            logger.error("Missing incident number for ticket update")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_snow_ticket"
                )

            return {"Status": "Failed", "Message": "Missing incident number"}

        if not comments and not work_notes and not action:
            logger.error("No update data provided for ticket update")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_snow_ticket"
                )

            return {"Status": "Failed", "Message": "No update data provided"}

        # First, get the incident sys_id from its number
        try:
            incident_data = await snow.get_incident_by_number_internal(incident_number)

            # Extract the sys_id using the _safe_extract_field helper
            incident_sys_id = snow._safe_extract_field(
                incident_data.get("sys_id"), "value"
            )

            if not incident_sys_id:
                logger.error(
                    f"Could not find sys_id for incident number {incident_number}"
                )
                if hasattr(context.session.userdata, "inactivity_monitor"):
                    context.session.userdata.inactivity_monitor.end_function_call(
                        "update_snow_ticket"
                    )

                return {
                    "Status": "Failed",
                    "Message": f"Incident {incident_number} not found",
                }

        except ServiceNowNotFoundError:
            logger.error(f"Incident {incident_number} not found")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_snow_ticket"
                )

            return {
                "Status": "Failed",
                "Message": f"Incident {incident_number} not found",
            }
        except Exception as e:
            logger.error(f"Error retrieving incident details: {str(e)}")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "update_snow_ticket"
                )

            return {"Status": "Failed", "Message": "Error retrieving incident details"}

        # Track what fields are being updated for response
        updated_fields = []
        if comments:
            updated_fields.append("comments")
        if work_notes:
            updated_fields.append("work_notes")
        if action:
            updated_fields.append("state")

        logger.info(f"Updating incident {incident_number} with action: {action}, comments: {bool(comments)}, work_notes: {bool(work_notes)}")

        # Use the merged update_incident method
        updated_incident = await snow.update_incident(
            incident_sys_id, 
            action=action, 
            comment=comments, 
            work_note=work_notes
        )

        # Handle response based on whether it's a state update or general update
        if action:
            # State updates return ResponseHandler wrapped responses
            if isinstance(updated_incident, dict):
                if updated_incident.get("status") == "success":
                    logger.info(f"Successfully updated incident {incident_number}")
                elif updated_incident.get("status") == "error":
                    logger.error(f"Failed to update incident {incident_number}: {updated_incident.get('message')}")
                    if hasattr(context.session.userdata, "inactivity_monitor"):
                        context.session.userdata.inactivity_monitor.end_function_call(
                            "update_snow_ticket"
                        )
                    return {
                        "Status": "Failed", 
                        "Message": updated_incident.get("message", "Failed to update incident")
                    }
                else:
                    logger.error(f"Unexpected response format for incident {incident_number}")
                    if hasattr(context.session.userdata, "inactivity_monitor"):
                        context.session.userdata.inactivity_monitor.end_function_call(
                            "update_snow_ticket"
                        )
                    return {"Status": "Failed", "Message": "Unexpected response format"}
            else:
                logger.error(f"Invalid response from update_incident for {incident_number}")
                if hasattr(context.session.userdata, "inactivity_monitor"):
                    context.session.userdata.inactivity_monitor.end_function_call(
                        "update_snow_ticket"
                    )
                return {"Status": "Failed", "Message": "Invalid response from update"}
        else:
            # General updates return raw data
            if not updated_incident:
                logger.error(f"Failed to update incident {incident_number}")
                if hasattr(context.session.userdata, "inactivity_monitor"):
                    context.session.userdata.inactivity_monitor.end_function_call(
                        "update_snow_ticket"
                    )
                return {"Status": "Failed", "Message": "Failed to update incident"}
            
            logger.info(f"Successfully updated incident {incident_number}")

        # Find and update the incident in context.session.userdata.incident_tickets if it exists
        if hasattr(context.session.userdata, "incident_tickets") and context.session.userdata.incident_tickets:
            for idx, ticket in enumerate(context.session.userdata.incident_tickets):
                if (
                    ticket.get("incident_number") == incident_number
                    or ticket.get("sys_id") == incident_sys_id
                ):
                    # Update the stored ticket data
                    if comments:
                        context.session.userdata.incident_tickets[idx]["comments"] = comments
                    if work_notes:
                        context.session.userdata.incident_tickets[idx]["work_notes"] = work_notes
                    
                    logger.info(f"Updated incident in session data: {incident_number}")
                    break

        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "update_snow_ticket"
            )

        return {
            "Status": "Success",
            "Message": f"Incident {incident_number} updated successfully",
            "Updated Fields": updated_fields,
        }

    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Exception in update_snow_ticket: {str(e)}\n{error_detail}")
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "update_snow_ticket"
            )

        return {
            "Status": "Error",
            "Message": "An unexpected error occurred while updating the ticket",
        }


@function_tool
async def rag_agent_function(context: RunContext, user_query: str):
    """Provides troubleshooting solutions using a Retrieval Augmented Generation (RAG) system

    This function processes the user's technical issue through a sophisticated RAG pipeline
    to provide accurate troubleshooting steps. It performs query rewriting based on conversation
    context, generates embeddings, retrieves relevant knowledge base articles, reranks results,
    and generates a tailored response that addresses the user's specific technical problem.

    Args:
        user_query: The user's description of the technical issue they're experiencing

    Returns:
        A JSON string containing the recommended troubleshooting steps and solutions
    """
    try:
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.start_function_call(
                "rag_agent_function"
            )

        if (
            not user_query
            or not isinstance(user_query, str)
            or len(user_query.strip()) == 0
        ):
            logger.error("Empty or invalid user query provided to RAG function")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "rag_agent_function"
                )

            return json.dumps({"result": json.dumps({"error": "No query provided"})})

        # Get query engine from context
        query_engine = context.session.userdata.query_engine
        if not query_engine:
            logger.error("Query engine not initialized in session")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "rag_agent_function"
                )

            return json.dumps(
                {"result": json.dumps({"error": "Internal system error occurred"})}
            )

        context.session.say("Just a sec", add_to_chat_ctx=False)

        try:
            rag_response = query_engine.query(user_query)

            if rag_response:
                print(f"RAG RESPONSE =====>{rag_response}")
                if hasattr(context.session.userdata, "inactivity_monitor"):
                    context.session.userdata.inactivity_monitor.end_function_call(
                        "rag_agent_function"
                    )

                return {"result": json.dumps(rag_response["answer"])}
            else:
                print("No answer found in rag")
                if hasattr(context.session.userdata, "inactivity_monitor"):
                    context.session.userdata.inactivity_monitor.end_function_call(
                        "rag_agent_function"
                    )

                return {"result": json.dumps("No relavant knowledge found")}

                # 'rag_instruction': "ALWAYS confirm if user is ready. ALWAYS tell one step(or sub-step) at a time."

        except Exception as analyzer_error:
            logger.error(f"Error in analyzing the query: {analyzer_error}")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "rag_agent_function"
                )

            return {"result": "Error", "Message": f"Failed to analyze the query"}

    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Exception in rag_agent_function: {str(e)}\n{error_detail}")
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "rag_agent_function"
            )

        return json.dumps(
            {
                "result": json.dumps(
                    {
                        "error": "An unexpected error occurred while processing your query"
                    }
                )
            }
        )


@function_tool
async def transfer_call(context: RunContext):
    """
    Escalates the issue to a service desk specialist for further handling when user issue is not getting resolved.

    Returns:
        None
    """
    try:
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.start_function_call(
                "transfer_call"
            )

        # Get transfer details from context
        transfer_number = context.session.userdata.transfer_number
        room_name = context.session.userdata.room_name
        participant_identity = context.session.userdata.participant_details

        # Validate required parameters
        if not all([transfer_number, room_name, participant_identity]):
            missing_params = []
            if not transfer_number:
                missing_params.append("transfer_number")
            if not room_name:
                missing_params.append("room_name")
            if not participant_identity:
                missing_params.append("participant_identity")

            logger.error(
                f"Missing required parameters for call transfer: {', '.join(missing_params)}"
            )
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "transfer_call"
                )

            return {
                "Status": "Failed",
                "Message": f"Missing required information for transfer",
            }

        # Prepare transfer number
        # transfer_to = 'tel:' + transfer_number

        try:
            # async with api.LiveKitAPI() as livekit_api:
            #     # Create transfer request
            #     transfer_request = TransferSIPParticipantRequest(
            #         participant_identity=participant_identity,
            #         room_name=room_name,
            #         transfer_to=transfer_to,
            #         play_dialtone=False
            #     )
            #     logger.debug(f"Transfer request: {transfer_request}")

            #     # Transfer caller
            #     await livekit_api.sip.transfer_sip_participant(transfer_request)
            #     logger.info(f"Successfully transferred participant {participant_identity}")

            #     if hasattr(context.session.userdata, 'inactivity_monitor'):
            #         context.session.userdata.inactivity_monitor.end_function_call("transfer_call")

            #     return {"Status": "Success", "Message": "Call successfully transferred"}

            # async with api.LiveKitAPI() as livekit_api:
            #     await livekit_api.room.delete_room(
            #         api.DeleteRoomRequest(room=room_name))
            #     logger.info("Room deleted successfully")

            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "transfer_call"
                )

            return {"Status": "Success", "Message": "Call successfully transferred"}
        except Exception as api_error:
            logger.error(f"LiveKit API error during transfer: {str(api_error)}")
            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "transfer_call"
                )

            return {"Status": "Failed", "Message": "Failed to transfer call"}

    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Exception in transfer_call: {str(e)}\n{error_detail}")
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call(
                "transfer_call"
            )

        return {
            "Status": "Error",
            "Message": "An unexpected error occurred during call transfer",
        }


@function_tool()
async def end_call(context: RunContext):
    """Gracefully terminates the current call session

    This function should be used to properly end the call when the interaction is complete
    or when the user has not provided consent to continue. It delivers a closing message to
    the user, then uses the LiveKit API to delete the room and terminate all connections.

    Returns:
        None
    """
    try:
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.start_function_call("end_call")

        session = context.session
        room_name = context.session.userdata.room_name

        if not room_name:
            logger.error("Room name not found in session")

            if hasattr(context.session.userdata, "inactivity_monitor"):
                context.session.userdata.inactivity_monitor.end_function_call(
                    "end_call"
                )

            return {"Status": "Failed", "Message": "Missing room information"}

        # try:
        #     await session.say("Thank you for your time, have a wonderful day.")
        # except Exception as say_error:
        #     logger.error(f"Error delivering closing message: {str(say_error)}")
        # Continue with room deletion even if message fails

        # try:
        #     async with api.LiveKitAPI() as livekit_api:
        #         await livekit_api.room.delete_room(
        #             api.DeleteRoomRequest(room=room_name))
        #         logger.info("Room deleted successfully")

        #         if hasattr(context.session.userdata, 'inactivity_monitor'):
        #             context.session.userdata.inactivity_monitor.end_function_call("end_call")

        #         return {"Status": "Success", "Message": "Call ended successfully"}
        # except Exception as api_error:
        #     logger.error(f"LiveKit API error during room deletion: {str(api_error)}")
        #     if hasattr(context.session.userdata, 'inactivity_monitor'):
        #         context.session.userdata.inactivity_monitor.end_function_call("end_call")

        #     return {"Status": "Failed", "Message": "Failed to properly end the call"}

        return {"Status": "Success"}

    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Exception in end_call: {str(e)}\n{error_detail}")
        if hasattr(context.session.userdata, "inactivity_monitor"):
            context.session.userdata.inactivity_monitor.end_function_call("end_call")

        return {
            "Status": "Error",
            "Message": "An unexpected error occurred while ending the call",
        }
