#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import datetime
import io
import os
import sys
import wave

import aiofiles
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext  # Changed!
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

# Import the correct AWS Nova Sonic service
from pipecat.services.aws_nova_sonic import AWSNovaSonicLLMService 

# function tooling 
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from function_tooling import verify_employee, create_ticket, rag_agent_function

# instruction
from instructions import primary_instructions



load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = (
            f"{server_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save")


async def run_bot(websocket_client: WebSocket, stream_sid: str, call_sid: str, testing: bool):
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    # Corrected AWS Nova Sonic LLM Service initialization
    llm = AWSNovaSonicLLMService(
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        region="us-east-1",  # Use supported region
        voice_id="tiffany",  # Voices: matthew, tiffany, amy
        # Note: No model_id parameter needed according to official code
    )

    # System instruction with special trigger instruction
    system_instruction = (
        primary_instructions
        f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
    )

    # Register the function
    llm.register_direct_function(verify_employee)
    llm.register_direct_function(create_ticket)
    llm.register_direct_function(rag_agent_function)


    messages = [
        {
            "role": "system",
            "content": system_instruction,
        },
    ]

    tools = ToolsSchema(standard_tools=[verify_employee, create_ticket, rag_agent_function])

    # Use OpenAILLMContext instead of AWSNovaSonicLLMContext
    context = OpenAILLMContext(messages=messages, tools=tools)
    context_aggregator = llm.create_context_aggregator(context)

    # Audio buffer for recording (optional)
    audiobuffer = AudioBufferProcessor()

    # Pipeline for Nova Sonic (speech-to-speech)
    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            context_aggregator.user(),  # User context processing
            llm,  # AWS Nova Sonic LLM (handles STT, processing, and TTS)
            transport.output(),  # Websocket output to client
            audiobuffer,  # Audio recording buffer
            context_aggregator.assistant(),  # Assistant context processing
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Start recording
        await audiobuffer.start_recording()
        
        # Kick off the conversation - this is the key part!
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        
        # CRITICAL: This special trigger is required for AWS Nova Sonic
        # This is what was missing in your original implementation!
        await llm.trigger_assistant_response()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        server_name = f"server_{websocket_client.client.port}"
        #await save_audio(server_name, audio, sample_rate, num_channels)

    # Pipeline runner configuration
    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)