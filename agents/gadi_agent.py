from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, RunContext
from livekit.plugins import openai, silero
from livekit.plugins import azure
from livekit.plugins import elevenlabs
from livekit.plugins import deepgram
from livekit.plugins.azure.tts import ProsodyConfig
#from livekit.plugins.turn_detector.english import EnglishModel


from agents.base_agent import BaseAgent
from session.user_data import UserData

# Import instructions
from instructions.gadi_agent_instructions import instructions as gadi_instructions
from instructions.whisper_instructions import instructions as whisper_instructions

# function tool
from tooling.function_tool_decorator_apigee import retrieve_previous_ticket

RunContext_T = RunContext[UserData]

import os

from dotenv import load_dotenv

load_dotenv()
stt_azure_endpoint = os.environ.get("STT_AZURE_OPENAI_ENDPOINT", "")
llm_azure_endpoint = os.environ.get("LLM_AZURE_OPENAI_ENDPOINT", "")
tts_azure_endpoint = os.environ.get("TTS_AZURE_OPENAI_ENDPOINT", "")
azure_speech = os.environ.get("AZURE_SPEECH")
azure_speech_region = os.environ.get("AZURE_SPEECH_REGION")


class GADIAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="GADIAgent",
            instructions=gadi_instructions,
            #turn_detection=EnglishModel(),
            vad=silero.VAD.load(
                min_speech_duration=0.09, # 0.07
                min_silence_duration=1, # 0.7
                activation_threshold=0.5, # 0.5 
            ),
            # stt=openai.STT.with_azure(
            #     model="gpt-4o-mini-transcribe",
            #     azure_endpoint=stt_azure_endpoint,
            #     prompt=whisper_instructions,
            # ),
            stt = deepgram.STT(model="nova-3", language = "en-US", keyterms = ["TruU", "GADI"]),
            # stt=azure.STT(speech_key='4wZxlIOVeO85g36vOPjitwij2kAJaJW4v5uW8SLFGQpvTIemhPEJJQQJ99BFACYeBjFXJ3w3AAAYACOGBfd1', speech_region='eastus'),
            llm=openai.LLM.with_azure(
                model="gpt-4o", azure_endpoint=llm_azure_endpoint
            ),
            # llm=openai.LLM(model="gpt-4o"),
            # tts=openai.TTS.with_azure(model="gpt-4o-mini-tts", voice="fable", azure_endpoint = tts_azure_endpoint),
            # tts=azure.TTS(
            #     speech_key=azure_speech,
            #     speech_region=azure_speech_region,
            #     voice="en-US-NovaTurboMultilingualNeural",
            #     prosody=ProsodyConfig(rate=0.85),
            # ),
            tts=elevenlabs.TTS(voice_id="9BWtsMINqrJLrRacOk9x",model="eleven_flash_v2"),
            tools=[
                retrieve_previous_ticket,
            ],
        )

    @function_tool
    async def update_user_issue_status(
        self, context: RunContext_T, status: str
    ) -> Agent:
        """
            Updates the status of the user's issue
            This function updates the current status of the user's issue

            Args:
                status (str): A short description of the current state of the issue 
                              (e.g., user is stuck, issue unresolved, requires help).
        """
        user_issue = ""
        print(f"Status of transfer {status}")
        return await self._transfer_to_agent("service", context, status, user_issue)
