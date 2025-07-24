from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Import your session classes
from session.vva_session_info import VVASessionInfo
from session.user_data import UserData


@dataclass
class CallStats:
    """Simple container for call statistics"""

    call_id: str
    phone_number: str
    org_id: str
    duration_minutes: Optional[int] = None
    total_messages: int = 0
    user_messages: int = 0
    agent_messages: int = 0
    is_multi_agent: bool = False
    agents_used: List[str] = None
    tickets_processed: int = 0
    issue_resolved: bool = False

    def __post_init__(self):
        if self.agents_used is None:
            self.agents_used = []


@dataclass
class ProcessingResults:
    """Container for processing results with actual outputs"""

    summary_created: bool = False
    summary_data: Optional[Dict[str, Any]] = None
    summary_error: Optional[str] = None
    tickets_updated: int = 0
    ticket_update_results: List[Dict[str, Any]] = None
    ticket_analysis_data: Optional[List[Dict[str, Any]]] = None
    ticket_analysis_error: Optional[str] = None
    conversation_saved: bool = False
    conversation_save_result: Optional[Any] = None
    conversation_save_error: Optional[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.ticket_update_results is None:
            self.ticket_update_results = []


class PostCallProcessor:
    """
    Simplified post-call processor that focuses on clarity over complexity.
    Takes session data and provides clean statistics and processing results.
    """

    def __init__(self, session_data, logger=None):
        """
        Initialize with session data (VVASessionInfo or UserData)

        Args:
            session_data: Your VVASessionInfo or UserData object
            logger: Optional logger (creates one if not provided)
        """
        self.session = session_data
        self.logger = logger or logging.getLogger(__name__)

        # Simple flags
        self.is_multi_agent = isinstance(session_data, UserData)

        # Initialize results containers
        self.stats = None
        self.results = ProcessingResults()

        # Validate we have basic required data
        if not self._has_required_data():
            raise ValueError(
                "Session data missing required fields (call_id, phone_number, org_id)"
            )

    def _has_required_data(self) -> bool:
        """Check if we have the minimum required data"""
        required = ["call_id", "phone_number", "org_id"]
        return all(getattr(self.session, field, None) for field in required)

    def get_call_stats(self) -> CallStats:
        """Extract and return simple call statistics"""
        if self.stats:
            return self.stats

        # Basic session info
        conversation = getattr(self.session, "conversation_history", []) or []
        user_transcripts = getattr(self.session, "user_transcripts", []) or []
        agent_transcripts = getattr(self.session, "agent_transcripts", []) or []
        tickets = getattr(self.session, "incident_tickets", []) or []

        # Multi-agent specific data
        agents_used = []
        if self.is_multi_agent:
            used_agents = getattr(self.session, "used_agents", {})
            agents_used = list(used_agents.keys()) if used_agents else []

        # Create stats object
        self.stats = CallStats(
            call_id=self.session.call_id,
            phone_number=self.session.phone_number,
            org_id=self.session.org_id,
            total_messages=len(conversation),
            user_messages=len(user_transcripts),
            agent_messages=len(agent_transcripts),
            is_multi_agent=self.is_multi_agent,
            agents_used=agents_used,
            tickets_processed=len(tickets),
        )

        return self.stats

    async def process_call(self) -> ProcessingResults:
        """
        Main processing method - does the actual work
        Returns simple results about what was accomplished
        """
        self.logger.info(f"Processing call {self.session.call_id}")

        # 1. Create call summary
        await self._create_summary()

        # 2. Update any tickets
        await self._update_tickets()

        # 3. Save conversation
        await self._save_conversation()

        # 4. Log what we did
        self._log_summary()

        return self.results

    async def _create_summary(self):
        """Create a call summary and store the detailed results"""
        try:
            conversation = getattr(self.session, "conversation_history", [])
            if not conversation:
                self.logger.info("No conversation to summarize")
                return

            # Use your existing summary function
            from session.post_call_process_helper_function import generate_call_summary

            summary = generate_call_summary(conversation)
            print(f"summary: {summary}")

            if summary and "error" not in summary:
                self.results.summary_created = True
                # Store the complete summary data
                self.results.summary_data = summary
                self.logger.info("Call summary created successfully")

                # Log key details from the summary
                self.logger.info(f"Issue: {summary.get('issue_summary', 'N/A')}")
                self.logger.info(
                    f"Resolution Status: {summary.get('resolution_status', 'N/A')}"
                )
                self.logger.info(f"Sentiment: {summary.get('sentiment', 'N/A')}")
                self.logger.info(
                    f"Follow-up Needed: {summary.get('follow_up_needed', False)}"
                )

            else:
                error_msg = "Failed to create summary or summary contains errors"
                self.results.summary_error = error_msg
                self.results.errors.append(error_msg)
                # Still store the data even if it has errors for debugging
                if summary:
                    self.results.summary_data = summary

        except Exception as e:
            error_msg = f"Summary creation failed: {str(e)}"
            self.results.summary_error = error_msg
            self.results.errors.append(error_msg)
            self.logger.error(error_msg)

    async def _update_tickets(self):
        """Update any incident tickets"""
        try:
            tickets = getattr(self.session, "incident_tickets", [])
            snow_client = getattr(self.session, "snow", None)

            if not tickets:
                self.logger.info("No tickets to update")
                return

            if not snow_client:
                self.results.errors.append("No ServiceNow client available")
                return

            # Use your existing ticket analysis function
            from session.post_call_process_helper_function import (
                analyze_tickets_from_transcript,
            )

            conversation = getattr(self.session, "conversation_history", [])
            ticket_updates = analyze_tickets_from_transcript(conversation, tickets)

            # Store the ticket analysis data
            if ticket_updates:
                self.results.ticket_analysis_data = ticket_updates

                updated_count = 0
                for update in ticket_updates:
                    try:
                        # Simplified ticket update - adjust based on your ServiceNow client
                        incident_id = update.get("incident_number", "")
                        update_result = {
                            "ticket_id": incident_id,
                            "success": False,
                            "update_data": update,
                            "error": None,
                        }

                        if "state" in update:
                            await snow_client.update_incident_state(
                                incident_sys_id=incident_id,
                                action=update["state"],
                                comment=update.get(
                                    "comments", "Updated from call analysis"
                                ),
                            )
                            updated_count += 1
                            update_result["success"] = True

                        self.results.ticket_update_results.append(update_result)

                    except Exception as e:
                        error_msg = f"Failed to update ticket {incident_id}: {str(e)}"
                        update_result["error"] = str(e)
                        self.results.ticket_update_results.append(update_result)
                        self.results.errors.append(error_msg)

                self.results.tickets_updated = updated_count
                self.logger.info(f"Updated {updated_count} tickets")
            else:
                self.results.ticket_analysis_error = (
                    "No ticket updates generated from analysis"
                )

        except Exception as e:
            error_msg = f"Ticket update failed: {str(e)}"
            self.results.ticket_analysis_error = error_msg
            self.results.errors.append(error_msg)
            self.logger.error(error_msg)

    async def _save_conversation(self):
        """Save conversation to database and store the result"""
        try:
            # Try to save using DatabaseOperations if available
            if hasattr(self.session, "db_operations") and self.session.db_operations:
                self.logger.info("Saving conversation using DatabaseOperations...")
                try:
                    # Prepare conversation data
                    conversation_data = {
                        "call_id": self.session.call_id,
                        "phone_number": self.session.phone_number,
                        "org_id": self.session.org_id,
                        "user_sys_id": self.session.user_sys_id,
                        "timestamp": datetime.now().isoformat(),
                        "conversation_history": getattr(
                            self.session, "conversation_history", []
                        ),
                        "user_transcripts": getattr(
                            self.session, "user_transcripts", []
                        ),
                        "agent_transcripts": getattr(
                            self.session, "agent_transcripts", []
                        ),
                    }

                    # Save using DatabaseOperations
                    result = await self.session.db_operations.save_conversation(
                        conversation_data
                    )

                    if result:
                        self.results.conversation_saved = True
                        self.results.conversation_save_result = {
                            "status": "success",
                            "message": "Conversation saved successfully",
                        }
                        self.logger.info(
                            "Conversation saved to database using DatabaseOperations"
                        )
                    else:
                        error_msg = (
                            "DatabaseOperations.save_conversation returned False"
                        )
                        self.results.conversation_save_error = error_msg
                        self.results.errors.append(error_msg)
                        self.logger.error(error_msg)

                except Exception as e:
                    error_msg = (
                        f"Exception in DatabaseOperations.save_conversation: {str(e)}"
                    )
                    self.results.conversation_save_error = error_msg
                    self.results.errors.append(error_msg)
                    self.logger.error(error_msg, exc_info=True)

            # Fallback to session's save_conversation_to_db method
            elif hasattr(self.session, "save_conversation_to_db"):
                self.logger.info("Calling save_conversation_to_db function...")
                try:
                    result = await self.session.save_conversation_to_db()

                    if result:
                        self.results.conversation_saved = True
                        self.results.conversation_save_result = {
                            "status": "success",
                            "message": "Conversation saved successfully",
                        }
                        self.logger.info("Conversation saved to database")
                    else:
                        error_msg = "save_conversation_to_db returned None or False"
                        self.results.conversation_save_error = error_msg
                        self.results.errors.append(error_msg)

                except Exception as e:
                    error_msg = f"Exception in save_conversation_to_db: {str(e)}"
                    self.results.conversation_save_error = error_msg
                    self.results.errors.append(error_msg)
                    self.logger.error(error_msg, exc_info=True)
            else:
                message = "No database operations available for conversation saving"
                self.logger.info(message)
                self.results.conversation_save_result = {
                    "status": "skipped",
                    "message": message,
                }

        except Exception as e:
            error_msg = f"Conversation save process failed: {str(e)}"
            self.results.conversation_save_error = error_msg
            self.results.errors.append(error_msg)
            self.logger.error(error_msg, exc_info=True)

    def _log_summary(self):
        """Log a detailed summary of what was done with actual stored data"""
        stats = self.get_call_stats()

        self.logger.info("=== Call Processing Complete ===")
        self.logger.info(f"Call ID: {stats.call_id}")
        self.logger.info(
            f"Messages: {stats.total_messages} total ({stats.user_messages} user, {stats.agent_messages} agent)"
        )

        if stats.is_multi_agent:
            self.logger.info(f"Agents used: {', '.join(stats.agents_used)}")

        # Log summary details with actual stored data
        if self.results.summary_created and self.results.summary_data:
            summary = self.results.summary_data
            self.logger.info(f"Summary created: YES")
            self.logger.info(f"  - Issue: {summary.get('issue_summary', 'N/A')}")
            self.logger.info(f"  - Customer: {summary.get('customer_details', 'N/A')}")
            self.logger.info(f"  - Actions: {summary.get('actions_taken', 'N/A')}")
            self.logger.info(
                f"  - Resolution Status: {summary.get('resolution_status', 'N/A')}"
            )
            self.logger.info(
                f"  - Resolution Method: {summary.get('resolution_method', 'N/A')}"
            )
            self.logger.info(
                f"  - Follow-up Needed: {summary.get('follow_up_needed', False)}"
            )
            if summary.get("follow_up_needed"):
                self.logger.info(
                    f"  - Follow-up Details: {summary.get('follow_up_details', 'N/A')}"
                )
            self.logger.info(
                f"  - Ticket Number: {summary.get('ticket_number', 'None')}"
            )
            self.logger.info(
                f"  - Call Duration: {summary.get('call_duration_estimate', 'N/A')}"
            )
            self.logger.info(f"  - Sentiment: {summary.get('sentiment', 'N/A')}")
            self.logger.info(
                f"  - Messages Processed: {summary.get('transcript_messages_count', 'N/A')}"
            )
            self.logger.info(f"  - Generated At: {summary.get('generated_at', 'N/A')}")
        else:
            self.logger.info(f"Summary created: NO")
            if self.results.summary_error:
                self.logger.error(f"  - Error: {self.results.summary_error}")
            if self.results.summary_data:
                self.logger.debug(f"  - Raw summary data: {self.results.summary_data}")

        # Log ticket analysis and update details
        if self.results.ticket_analysis_data:
            self.logger.info(
                f"Ticket analysis: Generated {len(self.results.ticket_analysis_data)} updates"
            )
            self.logger.debug(f"  - Analysis data: {self.results.ticket_analysis_data}")
        elif self.results.ticket_analysis_error:
            self.logger.error(
                f"Ticket analysis failed: {self.results.ticket_analysis_error}"
            )

        if self.results.tickets_updated > 0:
            self.logger.info(f"Tickets updated: {self.results.tickets_updated}")
            for ticket_result in self.results.ticket_update_results:
                if ticket_result.get("success"):
                    self.logger.info(
                        f"  - ✓ {ticket_result['ticket_id']}: {ticket_result['update_data'].get('state', 'Updated')}"
                    )
                else:
                    self.logger.error(
                        f"  - ✗ {ticket_result['ticket_id']}: {ticket_result['error']}"
                    )
        else:
            self.logger.info(f"Tickets updated: 0")

        # Log conversation save details
        if self.results.conversation_saved:
            self.logger.info(f"Conversation saved: YES")
            if self.results.conversation_save_result:
                self.logger.debug(
                    f"  - Save details: {self.results.conversation_save_result}"
                )
        else:
            self.logger.info(f"Conversation saved: NO")
            if self.results.conversation_save_error:
                self.logger.error(
                    f"  - Save error: {self.results.conversation_save_error}"
                )
            elif self.results.conversation_save_result:
                self.logger.debug(
                    f"  - Save details: {self.results.conversation_save_result}"
                )

        # Log errors
        if self.results.errors:
            self.logger.warning(f"Errors encountered: {len(self.results.errors)}")
            for error in self.results.errors:
                self.logger.error(f"  - {error}")
        else:
            self.logger.info("No errors encountered")

    def get_simple_report(self) -> Dict[str, Any]:
        """Get a comprehensive report with all actual data outputs"""
        stats = self.get_call_stats()

        report = {
            # Basic call info
            "call_info": {
                "id": stats.call_id,
                "phone": stats.phone_number,
                "org": stats.org_id,
                "type": "Multi-agent" if stats.is_multi_agent else "Single-agent",
            },
            # Conversation stats
            "conversation": {
                "total_messages": stats.total_messages,
                "user_messages": stats.user_messages,
                "agent_messages": stats.agent_messages,
            },
            # Multi-agent info (if applicable)
            "agents": {"count": len(stats.agents_used), "names": stats.agents_used}
            if stats.is_multi_agent
            else None,
            # Processing results with actual data
            "processing": {
                "summary": {
                    "created": self.results.summary_created,
                    "data": self.results.summary_data,
                    "error": self.results.summary_error,
                },
                "tickets": {
                    "analysis_data": self.results.ticket_analysis_data,
                    "analysis_error": self.results.ticket_analysis_error,
                    "updated_count": self.results.tickets_updated,
                    "update_details": self.results.ticket_update_results,
                },
                "conversation": {
                    "saved": self.results.conversation_saved,
                    "save_result": self.results.conversation_save_result,
                    "save_error": self.results.conversation_save_error,
                },
                "errors": self.results.errors if self.results.errors else None,
            },
            # Timestamp
            "processed_at": datetime.now().isoformat(),
        }

        # Remove None values for cleaner output
        if report["agents"] is None:
            del report["agents"]
        if report["processing"]["errors"] is None:
            del report["processing"]["errors"]

        return report

    def get_detailed_outputs(self) -> Dict[str, Any]:
        """Get just the detailed outputs from all processing functions"""
        return {
            "call_summary": self.results.summary_data,
            "ticket_analysis": self.results.ticket_analysis_data,
            "ticket_updates": self.results.ticket_update_results,
            "conversation_save": self.results.conversation_save_result,
            "errors": self.results.errors,
            "processing_timestamp": datetime.now().isoformat(),
        }

    def get_summary_details(self) -> Dict[str, Any]:
        """Get just the summary data with easy access to key fields"""
        if not self.results.summary_data:
            return {}

        summary = self.results.summary_data
        return {
            "issue_summary": summary.get("issue_summary"),
            "customer_details": summary.get("customer_details"),
            "actions_taken": summary.get("actions_taken"),
            "resolution_status": summary.get("resolution_status"),
            "resolution_method": summary.get("resolution_method"),
            "follow_up_needed": summary.get("follow_up_needed", False),
            "follow_up_details": summary.get("follow_up_details"),
            "ticket_number": summary.get("ticket_number"),
            "call_duration_estimate": summary.get("call_duration_estimate"),
            "sentiment": summary.get("sentiment"),
            "transcript_messages_count": summary.get("transcript_messages_count"),
            "generated_at": summary.get("generated_at"),
            "raw_summary": summary,  # Include full summary for completeness
        }
