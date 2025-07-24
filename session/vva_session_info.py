import re
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from livekit.agents import AgentSession
from livekit.agents.voice.events import ConversationItemAddedEvent
from db.handler import DatabaseHandler
from db.operations import DatabaseOperations
from servicenow.servicenow_apigee import ServiceNow
from rag.unified_rag_agent_local_follow_up import KnowledgeQueryEngine
from session.monitor_inactivity import InactivityMonitor
from tooling.function_tool_analytics import FunctionToolTracker


@dataclass
class VVASessionInfo:
    # sip details
    org_id: str | None = None
    phone_number: str | None = None
    user_sys_id: str | None = None
    full_name: str | None = None
    call_id: str | None = None
    # db handler
    db_handler: DatabaseHandler | None = None
    db_operations: DatabaseOperations | None = None
    # snow
    snow: ServiceNow | None = None
    incident_tickets: List | None = None
    # query engine
    query_engine: KnowledgeQueryEngine | None = None
    # conversation tracking
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_transcripts: List[Dict[str, Any]] = field(default_factory=list)
    agent_transcripts: List[Dict[str, Any]] = field(default_factory=list)
    # inactivity monitor
    _listeners_registered: bool = False
    monitor_inactivity: InactivityMonitor | None = None
    # call transfer details
    room_name: str | None = None
    transfer_number: str | None = None
    participant_details: str | None = None
    # dtmf signal
    dtmf_input: str | None = None
    # tracker
    funtion_tool_tracker: FunctionToolTracker | None = None
    # current issue - For TruU and GADI agent
    current_user_issue: str | None = None

    def register_conversation_listeners(self, session: AgentSession):
        """Register event listeners for conversation tracking."""
        if not self._listeners_registered:
            session.on("conversation_item_added", self._on_conversation_item_added)
            logging.info(
                f"Registered conversation listeners for call ID: {self.call_id}"
            )
            self._listeners_registered = True

    def _on_conversation_item_added(self, event: ConversationItemAddedEvent):
        """Process and store new conversation items."""
        item = event.item

        # Create a formatted transcript entry
        entry = {
            "role": item.role,
            "content": item.content,
            "timestamp": datetime.now().isoformat(),
        }

        # Add to the full conversation history
        self.conversation_history.append(entry)

        # Add to the role-specific transcript list
        if item.role == "user":
            self.user_transcripts.append(entry)
            logging.info(f"Added user transcript: {item.content[:50]}...")
        elif item.role == "assistant":
            self.agent_transcripts.append(entry)
            logging.info(f"Added agent transcript: {item.content[:50]}...")

        # Save to database in real-time
        self._save_conversation_turn_to_db(item.role, item.content)

    def _save_conversation_turn_to_db(self, role: str, content: str):
        """Save a conversation turn to the database in real-time."""
        if not self.db_operations or not self.call_id:
            logging.warning(
                f"Cannot save conversation turn: db_operations={self.db_operations is not None}, call_id={self.call_id is not None}"
            )
            return

        try:
            # Use asyncio to run the async database operation
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a task
                asyncio.create_task(
                    self.db_operations.save_transcript_entry(
                        call_id=self.call_id,
                        role=role,
                        content=content,
                        content_type="FinalTranscript",
                    )
                )
            else:
                # If we're not in an async context, run the coroutine
                loop.run_until_complete(
                    self.db_operations.save_transcript_entry(
                        call_id=self.call_id,
                        role=role,
                        content=content,
                        content_type="FinalTranscript",
                    )
                )

            logging.debug(
                f"Saved conversation turn to database: {role} - {content[:50]}..."
            )

        except Exception as e:
            logging.error(f"Error saving conversation turn to database: {e}")
            # Don't raise the exception to avoid disrupting the conversation flow

    def get_conversation_history(self):
        """Returns the full conversation history."""
        return self.conversation_history

    def get_user_transcripts(self):
        """Returns only the user transcripts."""
        return self.user_transcripts

    def get_agent_transcripts(self):
        """Returns only the agent transcripts."""
        return self.agent_transcripts

    def get_formatted_history_for_tools(self):
        """
        Returns the conversation history in a format suitable for function tools.
        Most tools expect a list of {"role": role, "content": content} dictionaries.
        """
        return [
            {"role": item["role"], "content": item["content"]}
            for item in self.conversation_history
        ]

    async def save_conversation_to_db(self):
        """Save the conversation history to the database."""
        if not self.db_handler:
            logging.warning("Cannot save conversation: db_handler is not initialized")
            return False

        logging.info(
            f"Attempting to save conversation with db_operations={self.db_operations is not None}, call_id={self.call_id}"
        )

        try:
            # Prepare conversation data for storage
            conversation_data = {
                "call_id": self.call_id,
                "phone_number": self.phone_number,
                "org_id": self.org_id,
                "user_sys_id": self.user_sys_id,
                "timestamp": datetime.now().isoformat(),
                "conversation_history": self.conversation_history,
                "user_transcripts": self.user_transcripts,
                "agent_transcripts": self.agent_transcripts,
            }

            # Use DatabaseOperations to save the conversation
            if hasattr(self, "db_operations") and self.db_operations:
                result = await self.db_operations.save_conversation(conversation_data)
            else:
                # Fallback to direct database operations if db_operations not available
                print(f"Conversation data => {conversation_data}")
                result = False  # Placeholder - implement actual save logic

            logging.info(
                f"Saved conversation with {len(self.conversation_history)} entries to database"
            )
            return result

        except Exception as e:
            logging.error(f"Error saving conversation to database: {e}", exc_info=True)
            return False

    def get_org_id(self, metadata: str) -> str:
        # Parse the metadata string to a dictionary
        try:
            metadata_dict = json.loads(metadata)
            # Note: The key is 'orgId' not 'org_id'
            return metadata_dict.get("orgId", "")
        except json.JSONDecodeError:
            # Handle the case where metadata is not valid JSON
            return ""

    def get_transfer_number(self, metadata: str) -> str:
        # Parse the metadata string to a dictionary
        try:
            metadata_dict = json.loads(metadata)
            # Note: The key is 'orgId' not 'org_id'
            return metadata_dict.get("transferNumber", "")
        except json.JSONDecodeError:
            # Handle the case where metadata is not valid JSON
            return ""

    def get_call_id_and_phone_number(self, room_name: str) -> tuple[str, str]:
        parts = room_name.split("_")

        # For a room name like "number-_+12132716774_V2hcfLfgU3ds"
        # The phone number is the second part
        phone_number = parts[1] if len(parts) > 1 else ""

        # The call ID is the third part
        call_id = parts[2] if len(parts) > 2 else ""

        return call_id, phone_number

    async def update_call_user_info(self, full_name: str, user_sys_id: str):
        """Update the call user info in the database."""
        if not self.db_operations or not self.call_id:
            logging.warning(
                f"Cannot update call user info: db_operations={self.db_operations is not None}, call_id={self.call_id is not None}"
            )
            return False

        try:
            await self.db_operations.update_call_user_info(
                call_id=self.call_id,
                user_name=full_name,
                user_sys_id=user_sys_id,
            )
            logging.info(f"Call record updated with user info: {full_name}")
        except Exception as e:
            logging.error(f"Error updating call user info: {e}")
            return False

        return True
