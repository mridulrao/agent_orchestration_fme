from dataclasses import dataclass, field
from livekit.agents import JobContext
from livekit.agents.llm import ChatContext
from livekit.agents.voice import Agent
from typing import Dict, Optional, List, Tuple
from datetime import datetime


from session.vva_session_info import VVASessionInfo
from db.operations import DatabaseOperations


@dataclass
class UserData(VVASessionInfo):
    """Stores data and agents to be shared across the session - extends VVASessionInfo"""

    # IT Support specific fields
    personas: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None
    ctx: Optional[JobContext] = None

    # Store separate chat contexts for each agent
    agent_chat_contexts: Dict[str, ChatContext] = field(default_factory=dict)

    # Track all agents used during the session
    used_agents: Dict[str, Agent] = field(default_factory=dict)

    # Track agent usage history with timestamps
    agent_usage_history: List[Tuple[str, datetime]] = field(default_factory=list)

    # Message for ServiceAgent updates from other agents
    service_agent_update_message: Optional[str] = None

    def summarize(self) -> str:
        agent_count = len(self.used_agents)
        agent_names = list(self.used_agents.keys())
        return f"User data: IT support triage system - {agent_count} agents used: {', '.join(agent_names)}"

    def get_agent_chat_context(self, agent_name: str) -> ChatContext:
        """Get or create chat context for specific agent"""
        if agent_name not in self.agent_chat_contexts:
            self.agent_chat_contexts[agent_name] = ChatContext()
        return self.agent_chat_contexts[agent_name]

    def save_agent_chat_context(self, agent_name: str, chat_ctx: ChatContext) -> None:
        """Save chat context for specific agent"""
        self.agent_chat_contexts[agent_name] = chat_ctx.copy()

    def track_agent_usage(self, agent_name: str, agent: Agent) -> None:
        """Track that an agent was used during this session"""
        # Store the agent if not already tracked
        if agent_name not in self.used_agents:
            self.used_agents[agent_name] = agent

        # Record usage timestamp
        self.agent_usage_history.append((agent_name, datetime.now()))

    def get_used_agents(self) -> Dict[str, Agent]:
        """Get all agents that were used during this session"""
        return self.used_agents.copy()

    def get_used_agents_with_contexts(self) -> Dict[str, Tuple[Agent, ChatContext]]:
        """Get all used agents along with their chat contexts"""
        result = {}
        for agent_name, agent in self.used_agents.items():
            chat_ctx = self.agent_chat_contexts.get(agent_name, ChatContext())
            result[agent_name] = (agent, chat_ctx.to_dict())
        return result

    def get_agent_usage_summary(self) -> Dict[str, dict]:
        """Get summary of agent usage including timestamps and context info"""
        summary = {}
        for agent_name, agent in self.used_agents.items():
            # Get usage timestamps for this agent
            usage_times = [
                timestamp
                for name, timestamp in self.agent_usage_history
                if name == agent_name
            ]

            # Get chat context info
            # chat_ctx = self.agent_chat_contexts.get(agent_name)
            # context_message_count = len(chat_ctx.messages) if chat_ctx else 0

            summary[agent_name] = {
                "agent": agent,
                "first_used": min(usage_times) if usage_times else None,
                "last_used": max(usage_times) if usage_times else None,
                "usage_count": len(usage_times),
                # 'context_message_count': context_message_count,
                # 'has_chat_context': chat_ctx is not None
            }
        return summary

    def set_service_agent_update(self, message: str):
        """Set update message for ServiceAgent from other agents"""
        self.service_agent_update_message = message

    def get_and_clear_service_agent_update(self) -> Optional[str]:
        """Get and clear the ServiceAgent update message"""
        message = self.service_agent_update_message
        self.service_agent_update_message = None
        return message
