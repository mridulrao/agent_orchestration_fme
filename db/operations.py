"""
Database operations module for VVA (Voice Virtual Agent) system.
This module contains all database interaction methods for storing and retrieving
call data, conversations, summaries, and session information using the actual
database schema.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import asdict

from prisma.models import organizations
from db.handler import DatabaseHandler
from db.config import OrganizationConfig


class DatabaseOperations:
    """
    Database operations class that handles all VVA-related database interactions.
    Uses the DatabaseHandler for connection management and works with actual schema.
    """

    def __init__(self, db_handler: DatabaseHandler, logger=None):
        """
        Initialize with a DatabaseHandler instance.

        Args:
            db_handler: DatabaseHandler instance for connection management
            logger: Optional logger instance
        """
        self.db_handler = db_handler
        self.logger = logger or logging.getLogger(__name__)

    # Organization Operations

    async def get_organization(self, org_id: str) -> organizations:
        """
        Retrieves an organization from the database by its ID.

        Args:
            org_id (str): The ID of the organization to retrieve.

        Returns:
            Organization: The retrieved organization object, or None if not found.
        """
        try:
            async with self.db_handler.get_connection() as client:
                org = await client.organizations.find_unique(where={"id": org_id})

                if org:
                    # Get associated phone number
                    phone_number = await client.phone_number.find_many(
                        where={"org_id": org_id, "app_type": "voice_copilot"}
                    )
                    org.phone = phone_number[0].number if phone_number else None

                    # Parse the configuration
                    if org.config:
                        org.config = OrganizationConfig(**json.loads(org.config))
                    self.logger.debug(
                        f"Organization config loaded for org_id: {org_id}"
                    )

                return org
        except Exception as e:
            self.logger.error(
                f"Error retrieving organization with ID {org_id}: {str(e)}"
            )
            raise

    # VVA Call Operations (using voice_virtual_agent_calls table)

    async def create_vva_call(self, call_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new VVA call record in voice_virtual_agent_calls table.

        Args:
            call_data: Dictionary containing call information
                - caller: Phone number
                - user_name: User name from VVASessionInfo
                - user_sys_id: ServiceNow user system ID from VVASessionInfo
                - incident_ticket_ids: List of incident ticket IDs from VVASessionInfo
                - status: Call status (default: "ongoing")

        Returns:
            str: Call ID if created successfully, None otherwise
        """
        self.logger.info(f"create_vva_call called with data: {call_data}")
        try:
            self.logger.info("Getting database connection...")
            async with self.db_handler.get_connection() as client:
                self.logger.info(
                    "Database connection obtained, creating call record..."
                )
                # Create call record with proper field mapping
                create_data = {
                    "caller": call_data.get("caller"),  # Phone number
                    "user_name": call_data.get("user_name") or "Anonymous User",
                    "user_sys_id": call_data.get("user_sys_id"),
                    "called_at": call_data.get("called_at"),  # Blank for now
                    "voice_virtual_agent_id": None,  # Blank for now
                    "incident_ticket_ids": call_data.get("incident_ticket_ids", []),
                    "interaction_id": None,  # Blank for now
                    "status": call_data.get("status", "ongoing"),
                    "details_url": None,  # Empty for now
                    "ticket_id": call_data.get("ticket_id"),  # From incident_tickets
                    "recordingUrl": call_data.get("recording_url"),  # File name
                }
                self.logger.info(f"Creating call record with data: {create_data}")

                call_record = await client.voice_virtual_agent_calls.create(
                    data=create_data
                )

                self.logger.info(f"VVA call created with ID: {call_record.id}")
                return call_record.id

        except Exception as e:
            self.logger.error(f"Error creating VVA call: {str(e)}")
            self.logger.error(f"Exception type: {type(e)}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def update_call_status(self, call_id: str, status: str) -> bool:
        """
        Update VVA call status in voice_virtual_agent_calls table.

        Args:
            call_id: Call ID to update
            status: New status ("ongoing", "ended", etc.)

        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            async with self.db_handler.get_connection() as client:
                await client.voice_virtual_agent_calls.update(
                    where={"id": call_id}, data={"status": status}
                )

                self.logger.info(
                    f"VVA call status updated to '{status}' for ID: {call_id}"
                )
                return True

        except Exception as e:
            self.logger.error(
                f"Error updating VVA call status for ID {call_id}: {str(e)}"
            )
            return False

    async def update_call_user_info(
        self, call_id: str, user_name: str, user_sys_id: str
    ) -> bool:
        """
        Update VVA call with user information after employee verification.

        Args:
            call_id: Call ID to update
            user_name: User's full name
            user_sys_id: ServiceNow user system ID

        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            async with self.db_handler.get_connection() as client:
                await client.voice_virtual_agent_calls.update(
                    where={"id": call_id},
                    data={"user_name": user_name, "user_sys_id": user_sys_id},
                )

                self.logger.info(f"VVA call user info updated for ID: {call_id}")
                return True

        except Exception as e:
            self.logger.error(
                f"Error updating VVA call user info for ID {call_id}: {str(e)}"
            )
            return False

    async def update_call_ticket_info(
        self, call_id: str, ticket_id: str, incident_ticket_ids: List[str]
    ) -> bool:
        """
        Update VVA call with ticket information when tickets are created.

        Args:
            call_id: Call ID to update
            ticket_id: Primary ticket ID
            incident_ticket_ids: List of all incident ticket IDs

        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            async with self.db_handler.get_connection() as client:
                await client.voice_virtual_agent_calls.update(
                    where={"id": call_id},
                    data={
                        "ticket_id": ticket_id,
                        "incident_ticket_ids": incident_ticket_ids,
                    },
                )

                self.logger.info(f"VVA call ticket info updated for ID: {call_id}")
                return True

        except Exception as e:
            self.logger.error(
                f"Error updating VVA call ticket info for ID {call_id}: {str(e)}"
            )
            return False

    async def update_vva_call_summary_and_sentiment(
        self, call_id: str, summary: str, sentiment: str
    ) -> bool:
        """
        Update VVA call with summary and sentiment in voice_virtual_agent_calls table.

        Args:
            call_id: Call ID to update
            summary: Call summary
            sentiment: Call sentiment

        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            async with self.db_handler.get_connection() as client:
                await client.voice_virtual_agent_calls.update(
                    where={"id": call_id},
                    data={"summary": summary, "sentiment": sentiment},
                )

                self.logger.info(
                    f"VVA call summary and sentiment updated for ID: {call_id}"
                )
                return True

        except Exception as e:
            self.logger.error(
                f"Error updating VVA call summary/sentiment for ID {call_id}: {str(e)}"
            )
            return False

    async def end_vva_call(
        self,
        call_id: str,
        ended_reason: str,
        ended_at: datetime = None,
        recording_url: str = None,
    ) -> bool:
        """
        End a VVA call by updating status, end time, and recording URL.

        Args:
            call_id: Call ID to end
            ended_reason: Reason for ending the call ("agent_session_ended" or "customer-ended-call")
            ended_at: End timestamp (defaults to current time)
            recording_url: Recording file name

        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            async with self.db_handler.get_connection() as client:
                update_data = {"status": "ended", "ended_reason": ended_reason}

                if ended_at:
                    update_data["ended_at"] = ended_at
                else:
                    update_data["ended_at"] = datetime.now()

                if recording_url:
                    update_data["recordingUrl"] = recording_url

                await client.voice_virtual_agent_calls.update(
                    where={"id": call_id}, data=update_data
                )

                self.logger.info(
                    f"VVA call ended for ID: {call_id} with reason: {ended_reason}"
                )
                return True

        except Exception as e:
            self.logger.error(f"Error ending VVA call for ID {call_id}: {str(e)}")
            return False

    async def get_vva_call_by_id(self, call_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve VVA call data by ID.

        Args:
            call_id: Call ID

        Returns:
            Dict containing call data or None if not found
        """
        try:
            async with self.db_handler.get_connection() as client:
                call = await client.voice_virtual_agent_calls.find_unique(
                    where={"id": call_id}
                )

                if call:
                    call_data = {
                        "id": call.id,
                        "caller": call.caller,
                        "user_name": call.user_name,
                        "user_sys_id": call.user_sys_id,
                        "called_at": call.called_at,
                        "voice_virtual_agent_id": call.voice_virtual_agent_id,
                        "incident_ticket_ids": call.incident_ticket_ids,
                        "interaction_id": call.interaction_id,
                        "summary": call.summary,
                        "sentiment": call.sentiment,
                        "status": call.status,
                        "ended_at": call.ended_at,
                        "ended_reason": call.ended_reason,
                        "recordingUrl": call.recordingUrl,
                        "details_url": call.details_url,
                        "ticket_id": call.ticket_id,
                        "created_at": call.created_at,
                    }
                    return call_data

                return None

        except Exception as e:
            self.logger.error(f"Error retrieving VVA call for ID {call_id}: {str(e)}")
            return None

    # VVA Call Status Operations (using voice_virtual_agent_call_status table for transcripts)

    async def save_transcript_entry(
        self,
        call_id: str,
        role: str,
        content: str,
        content_type: str = "FinalTranscript",
    ) -> bool:
        """
        Save a transcript entry to voice_virtual_agent_call_status table.

        Args:
            call_id: Call ID
            role: Role (user/assistant)
            content: Transcript content
            content_type: Content type (default: "FinalTranscript")

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            async with self.db_handler.get_connection() as client:
                # Format content as required
                formatted_content = json.dumps(
                    {
                        "message_type": "FinalTranscript",
                        "text": content,
                    }
                )

                transcript_record = await client.voice_virtual_agent_call_status.create(
                    data={
                        "call_id": call_id,  # Link to parent call
                        "role": role,
                        "content": formatted_content,
                        "content_type": content_type,
                    }
                )

                self.logger.info(
                    f"Transcript entry saved with ID: {transcript_record.id}"
                )
                return True

        except Exception as e:
            self.logger.error(f"Error saving transcript entry: {str(e)}")
            return False

    async def save_conversation_transcripts(
        self, call_id: str, conversation_data: Dict[str, Any]
    ) -> bool:
        """
        Save conversation transcripts to voice_virtual_agent_call_status table.

        Args:
            call_id: Call ID
            conversation_data: Dictionary containing conversation information
                - conversation_history: List of conversation entries
                - user_transcripts: List of user transcripts
                - agent_transcripts: List of agent transcripts

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            async with self.db_handler.get_connection() as client:
                conversation_history = conversation_data.get("conversation_history", [])
                user_transcripts = conversation_data.get("user_transcripts", [])
                agent_transcripts = conversation_data.get("agent_transcripts", [])

                # Save conversation history entries
                for entry in conversation_history:
                    await self.save_transcript_entry(
                        call_id=call_id,
                        role=entry.get("role", "unknown"),
                        content=entry.get("content", ""),
                        content_type="FinalTranscript",
                    )

                # Save user transcripts
                for transcript in user_transcripts:
                    await self.save_transcript_entry(
                        call_id=call_id,
                        role="user",
                        content=transcript.get("content", ""),
                        content_type="FinalTranscript",
                    )

                # Save agent transcripts
                for transcript in agent_transcripts:
                    await self.save_transcript_entry(
                        call_id=call_id,
                        role="assistant",
                        content=transcript.get("content", ""),
                        content_type="FinalTranscript",
                    )

                self.logger.info(f"All transcripts saved for call_id: {call_id}")
                return True

        except Exception as e:
            self.logger.error(f"Error saving conversation transcripts: {str(e)}")
            return False

    async def get_call_transcripts(self, call_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all transcript entries for a call.

        Args:
            call_id: Call ID

        Returns:
            List of transcript entries
        """
        try:
            async with self.db_handler.get_connection() as client:
                transcripts = await client.voice_virtual_agent_call_status.find_many(
                    where={"call_id": call_id}, order={"created_at": "asc"}
                )

                result = []
                for transcript in transcripts:
                    transcript_data = {
                        "id": transcript.id,
                        "call_id": transcript.call_id,
                        "role": transcript.role,
                        "content": transcript.content,
                        "content_type": transcript.content_type,
                        "created_at": transcript.created_at,
                    }
                    result.append(transcript_data)

                return result

        except Exception as e:
            self.logger.error(
                f"Error retrieving transcripts for call_id {call_id}: {str(e)}"
            )
            return []

    # Legacy method for backward compatibility
    async def save_conversation(self, conversation_data: Dict[str, Any]) -> bool:
        """
        Legacy method for saving conversation data.
        This method adapts the old format to work with the actual database schema.

        Args:
            conversation_data: Dictionary containing conversation information
                - call_id: Unique call identifier
                - phone_number: Phone number used for the call
                - org_id: Organization ID
                - user_sys_id: ServiceNow user system ID
                - timestamp: Call timestamp
                - conversation_history: Full conversation history
                - user_transcripts: User-specific transcripts
                - agent_transcripts: Agent-specific transcripts

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            call_id = conversation_data.get("call_id")
            if not call_id:
                self.logger.error("call_id is required for saving conversation")
                return False

            # Save transcripts to voice_virtual_agent_call_status
            transcript_result = await self.save_conversation_transcripts(
                call_id, conversation_data
            )

            if transcript_result:
                self.logger.info(
                    f"Conversation saved successfully for call_id: {call_id}"
                )
                return True
            else:
                self.logger.error(f"Failed to save conversation for call_id: {call_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error in legacy save_conversation: {str(e)}")
            return False

    # Utility methods

    async def get_organization_calls(
        self, org_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent calls for an organization.

        Args:
            org_id: Organization ID
            limit: Maximum number of calls to retrieve

        Returns:
            List of call data dictionaries
        """
        try:
            async with self.db_handler.get_connection() as client:
                # First get voice virtual agents for the org
                agents = await client.voice_virtual_agents.find_many(
                    where={"org_id": org_id}
                )

                agent_ids = [agent.id for agent in agents]

                if not agent_ids:
                    return []

                # Get calls for these agents
                calls = await client.voice_virtual_agent_calls.find_many(
                    where={"voice_virtual_agent_id": {"in": agent_ids}},
                    order={"created_at": "desc"},
                    take=limit,
                )

                result = []
                for call in calls:
                    call_data = {
                        "id": call.id,
                        "caller": call.caller,
                        "user_name": call.user_name,
                        "user_sys_id": call.user_sys_id,
                        "status": call.status,
                        "summary": call.summary,
                        "sentiment": call.sentiment,
                        "created_at": call.created_at,
                        "ended_at": call.ended_at,
                    }
                    result.append(call_data)

                return result

        except Exception as e:
            self.logger.error(f"Error retrieving calls for org_id {org_id}: {str(e)}")
            return []

    async def update_call_recording_url(self, call_id: str, recording_url: str) -> bool:
        """
        Update call with recording URL.

        Args:
            call_id: Call ID
            recording_url: Recording URL

        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            async with self.db_handler.get_connection() as client:
                await client.voice_virtual_agent_calls.update(
                    where={"id": call_id}, data={"recordingUrl": recording_url}
                )

                self.logger.info(f"Recording URL updated for call_id: {call_id}")
                return True

        except Exception as e:
            self.logger.error(
                f"Error updating recording URL for call_id {call_id}: {str(e)}"
            )
            return False

    async def update_call_ticket_ids(self, call_id: str, ticket_ids: List[str]) -> bool:
        """
        Update call with incident ticket IDs.

        Args:
            call_id: Call ID
            ticket_ids: List of ticket IDs

        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            async with self.db_handler.get_connection() as client:
                await client.voice_virtual_agent_calls.update(
                    where={"id": call_id}, data={"incident_ticket_ids": ticket_ids}
                )

                self.logger.info(f"Ticket IDs updated for call_id: {call_id}")
                return True

        except Exception as e:
            self.logger.error(
                f"Error updating ticket IDs for call_id {call_id}: {str(e)}"
            )
            return False
