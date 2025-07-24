from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, RunContext
from livekit.plugins import openai, silero


from agents.base_agent import BaseAgent
from session.user_data import UserData

# Import instructions
from instructions.laptop_agent_instructions import (
    instructions as laptop_agent_instructions,
)
from instructions.whisper_instructions import instructions as whisper_instructions

import os

from dotenv import load_dotenv

load_dotenv()
stt_azure_endpoint = os.environ.get("STT_AZURE_OPENAI_ENDPOINT", "")
llm_azure_endpoint = os.environ.get("LLM_AZURE_OPENAI_ENDPOINT", "")
tts_azure_endpoint = os.environ.get("TTS_AZURE_OPENAI_ENDPOINT", "")

RunContext_T = RunContext[UserData]


class LaptopTroubleshootinAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="LaptopTroubleshootinAgent",
            instructions=laptop_agent_instructions,
            vad=silero.VAD.load(
                min_speech_duration=0.07,
                min_silence_duration=0.7,
                activation_threshold=0.5,
            ),
            stt=openai.STT.with_azure(
                model="gpt-4o-mini-transcribe",
                azure_endpoint=stt_azure_endpoint,
                prompt=whisper_instructions,
            ),
            llm=openai.LLM.with_azure(
                model="gpt-4o", azure_endpoint=llm_azure_endpoint
            ),
            tts=openai.TTS.with_azure(
                model="gpt-4o-mini-tts",
                voice="fable",
                azure_endpoint=tts_azure_endpoint,
            ),
        )

    @function_tool
    async def transfer_to_service_agent(
        self, context: RunContext_T, status: str
    ) -> Agent:
        """
        Transfers the user to a service agent.

        This function facilitates the call transfer to service agent.
        The **status** parameter is crucial as it conveys the current state of the
        interaction , enabling the service agent to be appropriately informed and
        prepared to continue the conversation seamlessly.

        Args:
            status (str): A description of the current state of the conversation
                          to inform the service agent.

        Returns:
            Agent: The service agent object to which the user is transferred.
        """
        print(f"Status of transfer {status}")
        self.session.say("Connecting you to service agent, please hold on")
        return await self._transfer_to_agent("service", context, status)
