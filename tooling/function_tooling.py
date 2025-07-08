from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

async def verify_employee(params: FunctionCallParams, employee_id: str, department: str = None):
    """Verify employee information in ServiceNow.
    Args:
        employee_id: The employee ID or email to verify.
        department: Optional department filter for verification.
    """
    print("Employee verification function called")
    
    # Sample employee data
    employee_data = {
        "employee_id": employee_id,
        "name": "John Doe",
        "department": department or "IT",
        "status": "active",
        "email": f"{employee_id}@company.com",
        "manager": "Jane Smith",
        "hire_date": "2022-03-15",
        "location": "San Francisco, CA",
        "verified": True
    }
    
    await params.result_callback(employee_data)

async def create_ticket(params: FunctionCallParams, title: str, description: str, priority: str = "medium", category: str = "general"):
    """Create a new ticket in ServiceNow.
    Args:
        title: The ticket title/summary.
        description: Detailed description of the issue.
        priority: Ticket priority (low, medium, high, critical).
        category: Ticket category (general, hardware, software, network).
    """
    print("Create ticket function called")
    
    # Generate sample ticket data
    ticket_data = {
        "ticket_number": "INC0012345",
        "title": title,
        "description": description,
        "priority": priority,
        "category": category,
        "status": "new",
        "assigned_to": "IT Support Team",
        "created_by": "system_agent",
        "created_date": "2025-07-07T10:30:00Z",
        "expected_resolution": "2025-07-09T17:00:00Z",
        "url": "https://company.service-now.com/nav_to.do?uri=incident.do?sys_id=abc123"
    }
    
    await params.result_callback(ticket_data)

async def rag_agent_function(params: FunctionCallParams, query: str, knowledge_base: str = "general", max_results: int = 5):
    """Query the knowledge base using RAG (Retrieval-Augmented Generation).
    Args:
        query: The search query for the knowledge base.
        knowledge_base: The knowledge base to search (general, technical, hr, policies).
        max_results: Maximum number of results to return.
    """
    print("RAG agent function called")
    
    # Sample knowledge base responses
    rag_response = {
        "query": query,
        "knowledge_base": knowledge_base,
        "results": [
            {
                "id": "kb001",
                "title": "Password Reset Procedure",
                "content": "To reset your password, navigate to the self-service portal and click 'Forgot Password'.",
                "relevance_score": 0.95,
                "source": "IT Knowledge Base"
            },
            {
                "id": "kb002", 
                "title": "VPN Setup Guide",
                "content": "Download the VPN client from the IT portal and use your employee credentials to connect.",
                "relevance_score": 0.87,
                "source": "Network Documentation"
            },
            {
                "id": "kb003",
                "title": "Software Installation Policy",
                "content": "All software installations must be approved by IT security before deployment.",
                "relevance_score": 0.73,
                "source": "Security Policies"
            }
        ][:max_results],
        "total_results": 3,
        "search_time_ms": 45
    }
    
    await params.result_callback(rag_response)


    