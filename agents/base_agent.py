import logging
import asyncio
import random
import json
import os
from typing import Optional
from typing import AsyncIterable
from livekit.agents import ChatContext, ChatMessage
from livekit.agents.voice import Agent, RunContext
from livekit import rtc
from livekit.agents.voice import ModelSettings
from livekit.agents import stt, JobProcess
from livekit.plugins import openai
from openai import OpenAI
import re
from livekit import rtc
from livekit.agents.voice import ModelSettings
from livekit.agents import tts
from typing import AsyncIterable
from livekit.agents import FunctionTool, Agent
from livekit.agents import llm

# session
from session.user_data import UserData
from session.noise_filteration import NoiseFilter

# agents
from agents.stt_agent import STTValidator

from instructions.primary_agent_instructions import (
    prepare_rag_agent_function_instructions,
)

# tooling
from tooling.helper_functions import process_llm_output_for_tts

logger = logging.getLogger("BaseAgent")

RunContext_T = RunContext[UserData]


"""

Chatcontext operations - ['add_message', 'copy', 'empty', 'find_insertion_index', 'from_dict', 'get_by_id', 'index_by_id', 'insert', 'items', 'readonly', 'to_dict', 'to_provider_format', 'truncate']
"""


class BaseAgent(Agent):
    def __init__(self, agent_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_name = agent_name
        self.transfer_status = None
        self.user_issue = ""
        self.noise_filter = NoiseFilter()
        self.stt_validator = STTValidator()

        self.filler_words = [
            ", Okay",
            ", I see",
            ", Understood",
            ", Got it",
            ", Let me check",
            ", One moment",
            ", Alright",
        ]

    async def play_filler_words(self, duration: float = 1.0, interval: float = 0.5):
        """
        Play filler words during processing delays to maintain engagement
        """
        # start_time = asyncio.get_event_loop().time()

        filler = random.choice(self.filler_words)
        self.session.say(filler, add_to_chat_ctx=False)

        # while (asyncio.get_event_loop().time() - start_time) < duration:
        #     filler = random.choice(self.filler_words)
        #     self.session.say(filler, add_to_chat_ctx=False)
        #     await asyncio.sleep(interval)

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        # Check if rag_agent_function was recently called in chat context
        rag_function_called = False
        transfer_to_truu_agent_called = False
        transfer_to_gadi_agent_called = False
        transfer_to_service_agent_called = False

        for item in reversed(chat_ctx.items):  # Check recent items first
            if item.type == "function_call" and item.name == "rag_agent_function":
                rag_function_called = True
                break
            # # Stop looking if we hit a user message (new conversation turn)
            # if item.type == "message" and item.role == "user":
            #     break

            if item.type == "function_call" and item.name == "truu_query_function":
                transfer_to_truu_agent_called = True
                break

            if item.type == "function_call" and item.name == "gadi_query_function":
                transfer_to_gadi_agent_called = True
                break

            if (
                item.type == "function_call"
                and item.name == "update_user_issue_status"
            ):
                transfer_to_service_agent_called = True
                break

        if rag_function_called:
            current_function_rag_agent = True
            # Create modified context with your system message
            modified_chat_ctx = chat_ctx.copy()
            # message = "INTERNAL_THOUGHT: Now that the ticket is created(I have to inform the user) and the issue is classified as general IT, I must strictly follow the one-step-at-a-time resolution guidance, using the RAG output to guide the user. I will ask for confirmation after each step or a sub-step"
            message = prepare_rag_agent_function_instructions()
            modified_chat_ctx.add_message(role="system", content=message)

            # Use the modified context
            async for chunk in Agent.default.llm_node(
                self, modified_chat_ctx, tools, model_settings
            ):
                yield chunk

        elif transfer_to_truu_agent_called:
            modified_chat_ctx = chat_ctx.copy()
            message = "INTERNAL_THOUGHT: I will inform the user their ticket details and then I will use truu_query_function"
            modified_chat_ctx.add_message(role="assistant", content=message)

            # Use the modified context
            async for chunk in Agent.default.llm_node(
                self, modified_chat_ctx, tools, model_settings
            ):
                yield chunk

        elif transfer_to_gadi_agent_called:
            modified_chat_ctx = chat_ctx.copy()
            message = "INTERNAL_THOUGHT: I will inform the user their ticket details and then I will use gadi_query_function"
            modified_chat_ctx.add_message(role="assistant", content=message)

            # Use the modified context
            async for chunk in Agent.default.llm_node(
                self, modified_chat_ctx, tools, model_settings
            ):
                yield chunk

        elif transfer_to_service_agent_called:
            modified_chat_ctx = chat_ctx.copy()
            message = "INTERNAL_THOUGHT: I helped the user, and need to confirm is the issue is resolved or not"
            modified_chat_ctx.add_message(role="assistant", content=message)

            # Use the modified context
            async for chunk in Agent.default.llm_node(
                self, modified_chat_ctx, tools, model_settings
            ):
                yield chunk

        else:
            # Use original context
            async for chunk in Agent.default.llm_node(
                self, chat_ctx, tools, model_settings
            ):
                yield chunk

    async def process_speech_events(self, audio, model_settings):
        async for event in Agent.default.stt_node(self, audio, model_settings):
            # Access the transcript from the alternatives array
            if event.alternatives and event.alternatives[0].text:
                print(f"Unprocessed transcript: {event.alternatives[0].text}")
                processed_transcript = self.stt_validator.validate_message(
                    event.alternatives[0].text
                )
                print(f"Processed transcript: {processed_transcript}")
                # Check if validation failed
                if processed_transcript == "I didn't get what you said":
                    # Make the agent say the error message
                    self.session.say(
                        "Sorry I was not able to understand, can you repeat?"
                    )
                    # Optionally skip yielding this event to prevent further processing
                    continue

                event.alternatives[0].text = processed_transcript

            yield event

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        # print(f"TTS Node -> {text}")

        # Accumulate all text chunks
        full_text = ""
        async for chunk in text:
            full_text += chunk

        # Process the complete text
        processed_text = process_llm_output_for_tts(full_text)
        print(f"Processed text returned -> {processed_text}")

        # Convert back to async generator
        async def text_generator():
            yield processed_text

        async for frame in Agent.default.tts_node(
            self, text_generator(), model_settings
        ):
            yield frame

    async def transcription_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[str]:
        final_transcript = ""
        async for delta in text:
            final_transcript += delta

        # print(f"Final Transcript collected {final_transcript}")
        processed_text = process_llm_output_for_tts(final_transcript)
        yield processed_text

    async def on_enter(self) -> None:
        logger.info(f"Entering {self.agent_name}")
        userdata: UserData = self.session.userdata
        if userdata.ctx and userdata.ctx.room:
            await userdata.ctx.room.local_participant.set_attributes(
                {"agent": self.agent_name}
            )

        # Register conversation listeners if not already done
        # Note: Listeners are now registered in main.py before session starts
        # This is just a safety check in case they weren't registered
        if not userdata._listeners_registered:
            userdata.register_conversation_listeners(self.session)
            logger.info(
                f"Conversation listeners registered from {self.agent_name} (fallback)"
            )

        # for TruU and GADI agent
        user_issue = userdata.current_user_issue

        # Get the agent's specific chat context
        agent_chat_ctx = userdata.get_agent_chat_context(self.agent_name)

        # Check if this is ServiceAgent and there's an update message
        update_message = None
        if self.agent_name == "ServiceAgent":
            update_message = userdata.get_and_clear_service_agent_update()

        # If this agent has previous conversation history, restore it
        if agent_chat_ctx.items:
            logger.info(
                f"Restoring previous chat context for {self.agent_name} with {len(agent_chat_ctx.items)} items"
            )
            await self.update_chat_ctx(agent_chat_ctx)

            # For non-ServiceAgent with previous history, add the issue message
            if self.agent_name != "ServiceAgent":
                chat_ctx = self.chat_ctx.copy()
                print(f"User issue found for {self.agent_name} = {user_issue}")
                user_message = f"I have an issue with {user_issue}"
                chat_ctx.add_message(role="user", content=user_message)
                await self.update_chat_ctx(chat_ctx)
                logger.info(
                    f"Added 'I have an issue' message to {self.agent_name} with previous history"
                )

            # If there's an update message for ServiceAgent, add it
            if update_message:
                chat_ctx = self.chat_ctx.copy() 
                # assistant message 
                chat_ctx.add_message(role="assistant", content="User performed the troubleshooting steps")               
                # user message
                chat_ctx.add_message(role="user", content=update_message)
                await self.update_chat_ctx(chat_ctx)
                logger.info(
                    f"Added update message to {self.agent_name}: {update_message}"
                )

            # Generate reply to continue conversation
            self.session.generate_reply()
        else:
            # First time entering this agent
            chat_ctx = self.chat_ctx.copy()

            if self.agent_name == "ServiceAgent":
                # No need to add any message to chat_ctx for ServiceAgent
                logger.info(
                    f"Initializing Service agent without any user message (already greeted the user)"
                )
            else:
                # Default initialization message for other agents
                print(f"User issue found for {self.agent_name} = {user_issue}")
                user_message = f"I have an issue with {user_issue}"
                chat_ctx.add_message(role="user", content=user_message)
                logger.info(f"Initialized {self.agent_name} with default message")

            await self.update_chat_ctx(chat_ctx)

            # Generate reply to continue conversation (only for non-ServiceAgent)
            if self.agent_name != "ServiceAgent":
                self.session.generate_reply()

        # Track the agent and save context
        userdata.track_agent_usage(self.agent_name, self.agent_name)
        # Use the current chat_ctx instead of the local variable
        userdata.save_agent_chat_context(self.agent_name, self.chat_ctx)

    async def on_exit(self) -> None:
        """Save the current chat context and generate update for ServiceAgent if needed"""
        userdata: UserData = self.session.userdata
        current_chat_ctx = self.chat_ctx

        # Save this agent's chat context
        userdata.save_agent_chat_context(self.agent_name, current_chat_ctx)

        # If this is NOT ServiceAgent, generate update message for ServiceAgent
        if self.agent_name != "ServiceAgent":
            await self._generate_service_agent_update(userdata, current_chat_ctx)

        logger.info(
            f"Exiting {self.agent_name}, saved chat context with {len(current_chat_ctx.items)} items"
        )

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        await self.play_filler_words()

    async def _generate_service_agent_update(
        self, userdata: UserData, chat_ctx: ChatContext
    ):
        """Generate update message for ServiceAgent using LLM analysis"""
        try:
            # Store the update message for ServiceAgent
            if self.transfer_status == "issue resolved":
                update_message = "I followed the steps and my issue has been resolved, thanks"
            elif self.transfer_status == "issue new":
                update_message = "I have another issue as well"
            elif self.transfer_status == "issue diverged":
                update_message = (
                    "I think the issue I have is not related to current topic"
                )
            elif self.transfer_status == "issue unresolved":
                update_message = "My issue is not resolved"
            else:
                update_message = "My issue is not resolved"

            userdata.set_service_agent_update(update_message)
            logger.info(
                f"Generated ServiceAgent update from {self.agent_name}: {update_message}"
            )

        except Exception as e:
            logger.warning(
                f"Could not generate ServiceAgent update from {self.agent_name}: {e}"
            )
            # Fallback message
            fallback_message = (
                f"I followed the troubleshooting steps"
            )
            userdata.set_service_agent_update(fallback_message)

    async def _transfer_to_agent(
        self, name: str, context: RunContext_T, status: str, user_issue: str
    ) -> Agent:
        """Transfer to another agent while preserving individual contexts"""
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.personas[name]

        # Save current agent's state
        userdata.prev_agent = current_agent

        # save the status
        self.transfer_status = status

        userdata.current_user_issue = user_issue
        print(f"Reported Issue {userdata.current_user_issue}")

        # Trigger on_exit for current agent to save its context
        if hasattr(current_agent, "on_exit"):
            await current_agent.on_exit()

        return next_agent

    async def add_temporary_system_message(self, message: str):
        """Add a temporary system message to current context"""
        chat_ctx = self.chat_ctx.copy()
        message = "Ask user if they are ready to perform troubleshooting step or not. Always provide the rag-responce one step at a time. And keep it as short as possilbe"
        chat_ctx.add_message(role="system", content=message)
        await self.update_chat_ctx(chat_ctx)
        logger.info(f"Added temporary system message: {message}")

        print(f"Chat Context ==> {self.chat_ctx.to_dict()}")
