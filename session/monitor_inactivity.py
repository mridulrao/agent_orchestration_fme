import asyncio
import logging
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.events import (
    AgentEvent, AgentState, AgentStateChangedEvent, CloseEvent,
    ConversationItemAddedEvent, EventTypes, UserState, UserStateChangedEvent,
)

# Set up a logger
logger = logging.getLogger("inactivity_monitor")
logger.setLevel(logging.DEBUG)

class InactivityMonitor:
    """Monitors user inactivity and triggers actions after specified timeouts."""
    
    def __init__(self, session: AgentSession, 
                 prompt_timeout: float = 30.0,  # Changed to 10 seconds as requested
                 hangup_timeout: float = 20.0,  # Time after prompt before hanging up
                 room_name = None,
                 livekit_api = None):
        self.prompt_timeout = prompt_timeout  # Time before prompting user
        self.hangup_timeout = hangup_timeout  # Time after prompt before hanging up
        self.inactivity_timer = None
        self.hangup_timer = None  # Separate timer for hangup
        self.prompt_sent = False
        self.hangup_phase = False  # New flag to track if we're in the hangup phase
        self.livekit_api = livekit_api
        self.context = None  # Will be set when started
        self.session = session
        # NEW: Function call tracking
        self.function_call_in_progress = False
        self.function_call_start_time = None
        self.room_name = room_name
        
    async def start(self, context):
        """Start monitoring for inactivity."""
        self.context = context
        
        # Listen for state changes
        self.session.on("user_state_changed", self._on_user_state_changed)
        self.session.on("agent_state_changed", self._on_agent_state_changed)
        self.session.on("conversation_item_added", self._on_conversation_item_added)

        self._reset_inactivity_timer()
    
    # NEW METHODS: Function call tracking
    def start_function_call(self, function_name: str = None):
        """Call this when a function/tool call begins."""
        logger.info(f"Function call started: {function_name or 'unknown'}")
        self.function_call_in_progress = True
        self.function_call_start_time = asyncio.get_event_loop().time()
        
        # Cancel inactivity timer while function is executing
        # but keep hangup timer if we're in hangup phase
        if not self.hangup_phase:
            logger.info("Cancelling inactivity timer during function call")
            self._cancel_inactivity_timer()
    
    def end_function_call(self, function_name: str = None):
        """Call this when a function/tool call completes."""
        if self.function_call_in_progress:
            duration = asyncio.get_event_loop().time() - self.function_call_start_time if self.function_call_start_time else 0
            logger.info(f"Function call completed: {function_name or 'unknown'} (took {duration:.2f}s)")
            
            self.function_call_in_progress = False
            self.function_call_start_time = None
            
            # Only restart inactivity timer if we're not in hangup phase
            if not self.hangup_phase:
                logger.info("Restarting inactivity timer after function call completion")
                self._reset_inactivity_timer()
        else:
            logger.warning(f"end_function_call called but no function call was in progress")
    
    def is_function_call_active(self) -> bool:
        """Check if a function call is currently in progress."""
        return self.function_call_in_progress
        
    def _reset_inactivity_timer(self):
        """Reset the inactivity timer that leads to prompting."""
        logger.debug("Resetting inactivity timer")
        self._cancel_inactivity_timer()
        
        # Only cancel hangup timer if we're not in the hangup phase
        if not self.hangup_phase:
            self._cancel_hangup_timer()
        
        # MODIFIED: Don't start timer if function call is in progress
        if not self.function_call_in_progress:
            self.inactivity_timer = asyncio.create_task(self._inactivity_timer())
            logger.debug("New inactivity timer started")
        else:
            logger.debug("Function call in progress, skipping inactivity timer start")
        
    def _cancel_inactivity_timer(self):
        if self.inactivity_timer and not self.inactivity_timer.done():
            self.inactivity_timer.cancel()
        else:
            logger.debug("No active inactivity timer to cancel")
            
    def _start_hangup_timer(self):
        """Start the timer for hanging up after prompt."""
        self._cancel_hangup_timer()  # Cancel any existing hangup timer
        self.hangup_phase = True  # Set flag to indicate we're in hangup phase
        self.hangup_timer = asyncio.create_task(self._hangup_timer())
        logger.debug("New hangup timer started")
        
    def _cancel_hangup_timer(self):
        """Cancel the current hangup timer if it exists."""
        if self.hangup_timer and not self.hangup_timer.done():
            self.hangup_timer.cancel()
            logger.debug("Hangup timer cancelled")
        else:
            logger.debug("No active hangup timer to cancel")
        
    async def _inactivity_timer(self):
        """Timer that triggers prompt after timeout."""
        try:
            await asyncio.sleep(self.prompt_timeout)
            
            # MODIFIED: Check if function call started during our sleep
            if self.function_call_in_progress:
                logger.info("Function call detected during inactivity timeout, skipping prompt")
                return
            
            # Check current states
            logger.info(f"Prompt timeout reached. Current user state: {self.session._user_state}")
            logger.info(f"Current agent state: {self.session._agent_state}")
            logger.info(f"Current prompt_sent: {self.prompt_sent}, hangup_phase: {self.hangup_phase}")
            logger.info(f"Function call in progress: {self.function_call_in_progress}")
            
            # Check if we should prompt the user
            if not self.prompt_sent and not self.hangup_phase:
                logger.info(f"No user activity detected for {self.prompt_timeout}s, prompting user")
                await self._prompt_user()
                
                # Start the hangup timer after prompt is sent
                logger.info(f"Starting hangup timer for {self.hangup_timeout}s")
                self._start_hangup_timer()
            elif self.prompt_sent and not self.hangup_phase:
                # If prompt was sent but we're not in hangup phase, this is abnormal
                logger.warning("Abnormal state: prompt was sent but not in hangup phase. Entering hangup phase now.")
                self.hangup_phase = True
                self._start_hangup_timer()
            
        except asyncio.CancelledError:
            # Timer was reset
            logger.debug("Inactivity timer was cancelled")
        except Exception as e:
            logger.error(f"Error in inactivity timer: {e}", exc_info=True)
    
    async def _hangup_timer(self):
        """Timer that triggers hangup after timeout."""
        try:
            logger.debug(f"Hangup timer started, waiting {self.hangup_timeout}s before hangup")
            # Wait for hangup timeout
            await asyncio.sleep(self.hangup_timeout)
            
            # Check current states
            logger.info(f"Hangup timeout reached. Current user state: {self.session._user_state}")
            logger.info(f"Current agent state: {self.session._agent_state}")
            logger.info(f"Function call in progress: {self.function_call_in_progress}")
            
            # Hang up regardless of state - if user had responded, this timer would have been cancelled
            logger.info(f"No user response after prompt, initiating hangup")
            await self._hangup()
            
        except asyncio.CancelledError:
            # Timer was reset
            logger.debug("Hangup timer was cancelled")
            self.hangup_phase = False  # Reset hangup phase if cancelled
        except Exception as e:
            logger.error(f"Error in hangup timer: {e}", exc_info=True)
            
    async def _prompt_user(self):
        """Prompt the user to check if they're still there."""
        try:
            logger.info("Sending inactivity prompt to user")
            # Set prompt_sent and hangup_phase before sending the prompt
            self.prompt_sent = True
            self.hangup_phase = True
            logger.info("Setting prompt_sent=True and hangup_phase=True")
            
            # Send the prompt but don't wait for it to complete
            await self.session.say("Are you still there?", allow_interruptions=True)
            logger.info("Inactivity prompt sent")
            
        except Exception as e:
            logger.error(f"Error prompting user: {e}", exc_info=True)
            
    async def _hangup(self):
        """End the call due to inactivity."""
        try:    
            #await self.session.say("Ending the call due to inactivity. Goodbye!")

            logger.info("DELETING ROOM")
            
            # Then delete the room
            from livekit import api
            async with api.LiveKitAPI() as livekit_api:  
                await livekit_api.room.delete_room(api.DeleteRoomRequest(room=self.room_name))

            logger.info("ROOM DELETED")
                
            # Reset all state
            self.prompt_sent = False
            self.hangup_phase = False
            self.function_call_in_progress = False  # NEW: Reset function call state
            
        except Exception as e:
            logger.error(f"Error while ending call: {e}", exc_info=True)
            
    def _on_user_state_changed(self, event: UserStateChangedEvent):
        """Handle user state changes."""
        logger.info(f"User state changed: {event.old_state} -> {event.new_state}")
        
        if event.new_state == "speaking":
            logger.info("User started speaking, cancelling all timers")
            self._cancel_inactivity_timer()
            self._cancel_hangup_timer()  # Cancel hangup if user responds after prompt
            self.prompt_sent = False  # Reset prompt status when user speaks
            self.hangup_phase = False  # Exit hangup phase when user speaks
        elif event.new_state == "idle" and event.old_state == "speaking":
            logger.info("User stopped speaking, starting new inactivity timer")
            self._reset_inactivity_timer()
            
    def _on_agent_state_changed(self, event: AgentStateChangedEvent):
        """Handle agent state changes."""
        logger.info(f"Agent state changed: {event.old_state} -> {event.new_state}")
        
        if event.old_state == "speaking" and event.new_state == "listening":
            # Check if we're in the hangup phase
            if self.hangup_phase:
                logger.info("In hangup phase after agent finished 'Are you still there?' prompt")
                # Don't start any new timers, the hangup timer should already be running
                if not self.hangup_timer or self.hangup_timer.done():
                    logger.info("No active hangup timer found, starting one now")
                    self._start_hangup_timer()
                else:
                    logger.info("Hangup timer is already active, waiting for timeout or user response")
            else:
                logger.info("Agent finished speaking and is now listening (normal conversation)")
                
                # Double-check if prompt was sent, which means we should be in hangup phase
                if self.prompt_sent:
                    logger.info("Prompt was already sent, entering hangup phase")
                    self.hangup_phase = True
                    self._start_hangup_timer()
                else:
                    logger.info("Starting normal inactivity timer")
                    self._reset_inactivity_timer()
                
        elif event.new_state == "thinking" or event.new_state == "speaking":
            # Don't cancel timers if we're in the hangup phase OR if function call is in progress
            if not self.hangup_phase and not self.function_call_in_progress:
                logger.info(f"Agent is now {event.new_state}, cancelling inactivity timer")
                self._cancel_inactivity_timer()
            else:
                reason = "hangup phase" if self.hangup_phase else "function call in progress"
                logger.info(f"Agent is now {event.new_state} during {reason}, keeping timers as-is")
            
    def _on_conversation_item_added(self, event: ConversationItemAddedEvent):
        """Handle new conversation items."""
        logger.info(f"Conversation item added: {event.item.role} - {event.item.content[:50]}...")
        
        # Reset timers when user adds something to the conversation
        if event.item.role == "user":
            logger.info("User added conversation item, cancelling all timers")
            self._cancel_inactivity_timer()
            self._cancel_hangup_timer()
            self.prompt_sent = False  # Reset prompt status when user responds
            self.hangup_phase = False  # Exit hangup phase when user responds