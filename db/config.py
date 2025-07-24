from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from enum import Enum
from prisma.bases import Baseorganizations


class LandingPageConfig(BaseModel):
    """
    Landing page configuration for the organization.
    """

    header_message: str
    example_questions: Optional[List[str]] = None


class SpecialistAgents(Enum):
    """
    Specialist agents for the organization.
    """

    Ticketing = "TicketingAgent"
    RAG = "RAGAgent"
    # TODO: Mark as deprecated
    Independent = "IndependentAgent"
    # TODO: Mark as deprecated
    Employee = "EmployeeAgent"

    Citations = "CitationsAgent"
    # TODO: Mark as deprecated
    Followup = "FollowUpAgent"

    LanguageDetection = "LanguageDetectionAgent"

    Handover = "HandoverAgent"

    SessionTitle = "SessionTitleAgent"
    # TODO: Mark as deprecated , as this is an internal agent
    QueryOptimizer = "QueryOptimizerAgent"
    # TODO: Mark as deprecated , as this is an internal agent
    DuplicateQuestion = "DuplicateQuestionsAgent"
    # TODO: Mark as deprecated , as this is an internal agent
    CallTranscript = "CallTranscriptAgent"
    SoftwareServicerequest = "SoftwareServicerequestAgent"
    WebSearch = "WebSearchAgent"
    # TODO: Mark as deprecated , as this is an internal agent
    TicketQueryRewritter = "TicketQueryRewritterAgent"

    # New Agents
    IncidentAgent = "incident_agent"
    NudgeIncidentAgent = "nudge_incident_agent"
    ServiceRequestAgent = "service_request_agent"
    ChangeRequestAgent = "change_request_agent"
    DistributionListAgent = "distribution_list_agent"
    # This is to mark whether the previous incidents must be taken into context or not
    PreviousIncidentsContextAgent = "previous_incidents_context_agent"
    # This is to mark whether the query must be rewritten or not
    RewriteQueryAgent = "rewrite_query_agent"
    QuickRepliesAgent = "quick_replies_agent"


class TicketingService(Enum):
    """
    Ticketing service for the organization.
    """

    ServiceNow = "service_now"
    FreshService = "fresh_service"


class MeetingBotType(Enum):
    Recall = "recall"
    RecallPoll = "recall_poll"


class SharepointRAGConfig(BaseModel):
    top_k: Optional[int] = 3
    collection_name: Optional[str] = None


class Meta(BaseModel):
    sharepoint: Optional[SharepointRAGConfig] = None


class RagConfigs(BaseModel):
    """
    RAG configuration for the organization.
    """

    top_k: Optional[int] = 3
    collection_name: Optional[str] = None
    meta: Optional[Meta] = None


class TicketingAuth(BaseModel):
    """
    Ticketing authentication for the organization.
    """

    username: Optional[str] = None
    password: Optional[str] = None
    instance_name: str
    use_user_id: Optional[bool] = False
    extra_fields: Optional[bool] = False
    client_id: Optional[str] = None
    client_secret: Optional[str] = None


class SharepointAuth(BaseModel):
    tenant_id: str
    client_id: str
    client_secret: str
    site_url: str
    drive_name: str
    site_urls: Optional[List[str]] = None
    drive_names: Optional[Dict[str, List[str]]] = None


class kbSyncAuth(BaseModel):
    """
    Knowledgebase sync authentication for the organization.
    """

    username: Optional[str] = None
    password: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    instance_name: Optional[str] = None
    sharepoint: Optional[SharepointAuth] = None
    sync_interval: Optional[int] = 60 * 60 * 12


class kbSync(BaseModel):
    """
    Knowledgebase sync for the organization.
    """

    platform: Optional[str] = (
        None  # Platform name, for example, "ServiceNow", "FreshService"
    )
    kb_name: Optional[str] = None  # Knowledgebase name, might be used in the future
    enabled: Optional[bool] = False
    last_synced: Optional[str] = None


class SubmoduleConfig(BaseModel):
    """
    Submodule configuration for the organization.
    """

    name: Optional[str] = None
    enabled: Optional[bool] = False
    agents: Optional[List[str]] = None
    rag_config: Optional[RagConfigs] = None
    ticketing_service: Optional[str] = None
    ticketing_auth: Optional[TicketingAuth] = None
    features: Optional[dict] = None
    knowledgebase_id: Optional[str] = None
    landing_page_config: Optional[LandingPageConfig] = None
    endpoint_auth: Optional[str] = None
    kb_sync: Optional[kbSync] = None
    meta: Optional[Any] = None  # meta data for the module for any adhoc changes
    version: Optional[str] = "v1"


class SubModules(BaseModel):
    chat_assist: Optional[SubmoduleConfig] = None


class EntraIDAuth(BaseModel):
    """
    Ticketing authentication for the organization.
    """

    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None


class ModuleConfig(BaseModel):
    """
    Module configuration for the organization.
    """

    name: Optional[str] = None
    enabled: Optional[bool] = False
    agents: Optional[List[str]] = None
    rag_config: Optional[RagConfigs] = None
    ticketing_service: Optional[str] = None
    ticketing_auth: Optional[TicketingAuth] = None
    features: Optional[dict] = None
    knowledgebase_id: Optional[str] = None
    landing_page_config: Optional[LandingPageConfig] = None
    endpoint_auth: Optional[str] = None
    kb_sync: Optional[kbSync] = None
    meta: Optional[Any] = None  # meta data for the module for any adhoc changes
    version: Optional[str] = "v1"
    # submodules is a new field for the module
    submodules: Optional[SubModules] = None
    type: Optional[str] = None
    identity: Optional[str] = None
    restrict_response_to_knowledge_base: Optional[bool] = False
    meeting_bot_type: Optional[MeetingBotType] = None
    entra_id_auth: Optional[EntraIDAuth] = None  # Or create a proper model for this
    verify_by_entra_id: Optional[bool] = False



class Modules(BaseModel):
    """
    Modules for the organization.
    """

    agent_assist: Optional[ModuleConfig] = None
    voice_copilot: Optional[ModuleConfig] = None
    noc_copilot: Optional[ModuleConfig] = None
    voice_virtual_agent: Optional[ModuleConfig] = None
    chat_virtual_agent: Optional[ModuleConfig] = None
    expert_copilot: Optional[ModuleConfig] = None
    voice_p1_bridge: Optional[ModuleConfig] = None
    knowledge_assist: Optional[ModuleConfig] = None
    train_assist: Optional[ModuleConfig] = None
    anomaly_detection: Optional[ModuleConfig] = None


class Integrations(str, Enum):
    """
    Available integrations that can be shown in the organization.
    uses the format of <provider>_<integration_type>
    """

    SERVICENOW_ITSM = "SERVICENOW_ITSM"
    AZURE_RUNBOOKS = "AZURE_RUNBOOKS"
    PROMETHEUS_MONITORING = "PROMETHEUS_MONITORING"


class OrganizationConfig(BaseModel):
    """
    Organization configuration for the organization.
    """

    Name: str
    ShowHeader: bool = True
    Description: Optional[str] = None
    Modules: Optional[Modules]
    KbSyncAuth: Optional[kbSyncAuth] = None
    KbTemplate: Optional[str] = None
    KbLanguages: Optional[List[str]] = ["en", "es", "fr", "de"]
    ShowIntegrations: Optional[List[Integrations]] = None


class ClientConfig(BaseModel):
    """
    Client configuration for the organization.
    """

    Name: str
    Modules: Optional[Modules]


class AIOrgIdConfig(BaseModel):
    """
    AI organization configuration for the organization.
    """

    agents: list[SpecialistAgents]
    rag_config: RagConfigs
    blob_container_id: Optional[str] = None
    knowledgebase_id: Optional[str] = None
    ticketing_service: Optional[TicketingService] = None
    ticketing_auth: Optional[TicketingAuth] = None
    meta: Optional[Any] = None  # meta data for the module for any adhoc changes
    version: Optional[str] = "v1"
    # identity is a new field for the module
    identity: Optional[str] = None
    # restrict_response_to_knowledge_base is a new field for the module
    restrict_response_to_knowledge_base: Optional[bool] = False


# Example usage
example_config = OrganizationConfig(
    Name="cariard",
    OrgId=123,
    Modules=Modules(
        agent_assist=ModuleConfig(
            agents=[
                "TicketingAgent",
                "RAGAgent",
                "IndependentAgent",
                "EmployeeAgent",
                "CitationsAgent",
                "FollowUpAgent",
            ],
            rag_config=RagConfigs(
                top_k=7,
                collection_name="507169ef-97c4-407c-bb0a-b57ef1e96f5d_agent_assist",
            ),
            ticketing_service="ServiceNow",
            blob_container_id="0262799b-de56-4926-871b-35ba47c654ff",
        ),
        voice_copilot=None,
    ),
)


def agent_mapper(module_config: ModuleConfig) -> ModuleConfig:
    # if the fetched version is v2 , then map the agent names to the new agents
    if module_config.version == "v2":
        # Map Ticketing Agent to Incident Agent if present
        if SpecialistAgents.Ticketing in module_config.agents:
            module_config.agents.remove(SpecialistAgents.Ticketing)
            if SpecialistAgents.IncidentAgent not in module_config.agents:
                module_config.agents.append(SpecialistAgents.IncidentAgent)
        if SpecialistAgents.SoftwareServicerequest in module_config.agents:
            module_config.agents.remove(SpecialistAgents.SoftwareServicerequest)
            if SpecialistAgents.ServiceRequestAgent not in module_config.agents:
                module_config.agents.append(SpecialistAgents.ServiceRequestAgent)
        if SpecialistAgents.Employee in module_config.agents:
            module_config.agents.remove(SpecialistAgents.Employee)
            if SpecialistAgents.ServiceRequestAgent not in module_config.agents:
                module_config.agents.append(SpecialistAgents.ServiceRequestAgent)
    return module_config


class KnowledgebaseCollectionError(Exception):
    error_message: str

    def __init__(self, error_message: str):
        self.error_message = error_message


class OrganisationWithConfig(Baseorganizations):
    id: str
    config: Optional[str] = None
