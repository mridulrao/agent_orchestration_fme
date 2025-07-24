from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, RunContext
from livekit.plugins import openai, silero
from livekit.plugins import azure
from livekit.plugins import elevenlabs
from livekit.plugins import deepgram

#from livekit.plugins.turn_detector.english import EnglishModel

# agents
from agents.base_agent import BaseAgent

# session
from session.user_data import UserData

# Import instructions
from instructions.primary_agent_instructions import primary_instructions
from instructions.whisper_instructions import instructions as whisper_instructions
from livekit.plugins.azure.tts import ProsodyConfig

from tooling.function_tool_decorator_apigee import (
    verify_employee,
    retrieve_previous_ticket,
    create_ticket,
    update_snow_ticket,
    rag_agent_function,
    transfer_call,
    end_call,
)
import os

from dotenv import load_dotenv

load_dotenv()
stt_azure_endpoint = os.environ.get("STT_AZURE_OPENAI_ENDPOINT", "")
llm_azure_endpoint = os.environ.get("LLM_AZURE_OPENAI_ENDPOINT", "")
tts_azure_endpoint = os.environ.get("TTS_AZURE_OPENAI_ENDPOINT", "")
azure_speech = os.environ.get("AZURE_SPEECH")
azure_speech_region = os.environ.get("AZURE_SPEECH_REGION")

RunContext_T = RunContext[UserData]


class ServiceAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="ServiceAgent",
            instructions=primary_instructions,
            #turn_detection=EnglishModel(),
            vad=silero.VAD.load(
                min_speech_duration=0.09, # 0.07
                min_silence_duration=1.0, # 0.7
                activation_threshold=0.7, # 0.5 
            ),
            #stt=openai.STT.with_azure(model="gpt-4o-mini-transcribe", azure_endpoint = stt_azure_endpoint, prompt = whisper_instructions),
            #stt=openai.STT(model="gpt-4o-mini-transcribe"),
            # stt=azure.STT(speech_key=azure_speech, speech_region=azure_speech_region,),
            stt = deepgram.STT(model="nova-3", language = "en-US", keyterms = ["TruU", "GADI"]),
            llm=openai.LLM.with_azure(model="gpt-4o", azure_endpoint=llm_azure_endpoint),
            # llm=openai.LLM(model="gpt-4o"),
            # tts=openai.TTS.with_azure(model="gpt-4o-mini-tts", voice="fable", azure_endpoint = tts_azure_endpoint),
            # tts=openai.TTS(model="gpt-4o-mini-tts", voice="fable", instructions="Speak in a friendly and conversational tone.",),
            # tts=azure.TTS(
            #     speech_key=azure_speech,
            #     speech_region=azure_speech_region,
            #     voice="en-US-NovaTurboMultilingualNeural",
            #     prosody=ProsodyConfig(rate=0.85),
            # ),
            tts=elevenlabs.TTS(voice_id="9BWtsMINqrJLrRacOk9x",model="eleven_flash_v2"),
            tools=[
                verify_employee,
                retrieve_previous_ticket,
                create_ticket,
                update_snow_ticket,
                rag_agent_function,
                transfer_call,
                end_call,
            ],
        )

    @function_tool
    async def truu_query_function(
        self, context: RunContext_T, user_issue: str
    ) -> Agent:
        """
        Handles TruU-specific support queries.

        This function is used to address TruU-related issues such as account 
        access, device registration, authentication problems, or other functionality 
        specific to TruU. 

        Args:
            user_issue: A concise summary of the user's specific issue related to TruU.
        """
        status = "truu_issue_pending_resolution"
        return await self._transfer_to_agent("truu", context, status, user_issue)


    @function_tool
    async def gadi_query_function(
        self, context: RunContext_T, user_issue: str
    ) -> Agent:
        """
        Handles GADI-specific support queries or clitrix related queries.

        This function is used to address issues related to GADI data access, 
        integration, reporting, or data discrepancies. 

        Args:
            user_issue: A concise summary of the user's specific issue related to GADI.
        """
        status = "gadi_issue_pending_resolution"
        return await self._transfer_to_agent("gadi", context, status, user_issue)

