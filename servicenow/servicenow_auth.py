"""
ServiceNow API Integration Module

This module provides a comprehensive interface for interacting with ServiceNow instances.
It handles various ServiceNow operations such as incidents, requests, approvals, and file management.
"""

import os
import base64
import io
import json
import mimetypes
from abc import ABCMeta, abstractmethod
from enum import Enum
from logging import getLogger
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from opentelemetry import trace
from pydantic import BaseModel, Field, validator
from requests.auth import HTTPBasicAuth

# Initialize logging and tracing
logger = getLogger(__name__)
tracer = trace.get_tracer(__name__)


# Status code definitions as Enum for better type safety
class StatusCode(Enum):
    SUCCESS = 200
    CREATED = 201
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    NOT_FOUND = 404
    VALIDATION_ERROR = 422
    INTERNAL_SERVER_ERROR = 500


# Standardized error and success responses
class ServiceNowResponse(BaseModel):
    """Standardized response model for ServiceNow API operations"""
    code: int
    message: str
    data: Optional[Any] = None
    details: Optional[str] = None
    errors: Optional[List[str]] = None

    @classmethod
    def success(cls, data=None, message="The request was successful"):
        """Create a success response"""
        return cls(code=StatusCode.SUCCESS.value, message=message, data=data)

    @classmethod
    def created(cls, data=None, message="The resource was created successfully"):
        """Create a created response"""
        return cls(code=StatusCode.CREATED.value, message=message, data=data)

    @classmethod
    def error(cls, code=StatusCode.INTERNAL_SERVER_ERROR.value, 
              message="An error occurred", details=None, errors=None):
        """Create an error response"""
        return cls(code=code, message=message, details=details, errors=errors)

    def to_dict(self):
        """Convert the model to a dictionary"""
        return self.dict(exclude_none=True)


# Authentication configuration with environment variable fallbacks
class ServiceNowAuth(BaseModel):
    """ServiceNow authentication configuration"""
    username: str = Field(..., description="ServiceNow username")
    password: str = Field(..., description="ServiceNow password")
    instance_name: str = Field(..., description="ServiceNow instance name")

    @validator('username', 'password', 'instance_name', pre=True)
    def check_empty(cls, v):
        """Validate that fields are not empty"""
        if not v:
            raise ValueError("This field cannot be empty")
        return v

    @classmethod
    def from_env(cls):
        """Load authentication from environment variables"""
        return cls(
            username=os.getenv('SERVICENOW_USERNAME', ''),
            password=os.getenv('SERVICENOW_PASSWORD', ''),
            instance_name=os.getenv('SERVICENOW_INSTANCE', '')
        )


# State mappings for various ServiceNow entities
class ServiceNowStateMapping:
    """Mappings for ServiceNow states and priorities"""
    # Priority mapping according to ServiceNow default values
    PRIORITY = {
        "1": "1 - Critical",
        "2": "2 - High",
        "3": "3 - Moderate",
        "4": "4 - Low",
        "5": "5 - Planning",
    }

    # Incident state mapping
    INCIDENT_STATE = {
        "1": "New",
        "2": "In Progress",
        "3": "On Hold",
        "6": "Resolved",
        "7": "Closed",
        "8": "Canceled",
    }

    # Request Item state mapping
    RITM_STATE = {
        "-5": "Pending",
        "-4": "Draft",
        "-3": "Review",
        "-2": "Requested",
        "-1": "Waiting for Approval",
        "0": "Not Requested",
        "1": "Open",
        "2": "Work in Progress",
        "3": "Closed Complete",
        "4": "Closed Incomplete",
        "5": "Closed Cancelled",
        "6": "Closed Rejected",
        "7": "Closed Skipped",
        "8": "On Hold",
    }

    # Impact mapping
    IMPACT = {
        "1": "1 - High",
        "2": "2 - Medium",
        "3": "3 - Low"
    }

    # Urgency mapping
    URGENCY = {
        "1": "1 - High",
        "2": "2 - Medium",
        "3": "3 - Low"
    }


class ServiceNowBase(metaclass=ABCMeta):
    """
    Abstract base class for interacting with ServiceNow instances.
    This class defines the interface for common ServiceNow operations.
    """

    @abstractmethod
    async def get_user_by_id(self, user_id: str) -> ServiceNowResponse:
        """
        Retrieves a user by their ID.

        Args:
            user_id: The unique identifier of the user in ServiceNow.

        Returns:
            ServiceNowResponse containing user details if found
        """
        pass

    @abstractmethod
    async def get_tickets_by_user_id(self, user_sys_id: str, num_tickets: int) -> ServiceNowResponse:
        """
        Retrieves incidents associated with a given user's system ID.

        Args:
            user_sys_id: The system ID of the user.
            num_tickets: The number of tickets to retrieve.

        Returns:
            ServiceNowResponse containing a list of incidents associated with the user
        """
        pass

    @abstractmethod
    async def create_ticket(
        self, 
        subject: str, 
        description: str, 
        urgency: int, 
        impact: int, 
        user_id: str, 
        **kwargs
    ) -> ServiceNowResponse:
        """
        Creates a new incident with the specified details.

        Args:
            subject: A brief description of the incident.
            description: A detailed description of the incident.
            urgency: The urgency of the incident.
            impact: The impact of the incident.
            user_id: The ID of the user creating the incident.
            **kwargs: Additional parameters for incident creation.

        Returns:
            ServiceNowResponse containing the result of the incident creation operation
        """
        pass

    @abstractmethod
    async def update_ticket(
        self,
        incident_sys_id: str,
        **kwargs
    ) -> ServiceNowResponse:
        """
        Updates an existing incident with new information.

        Args:
            incident_sys_id: The sys_id of the ticket to update.
            **kwargs: Fields to update on the incident.

        Returns:
            ServiceNowResponse containing the result of the incident update operation
        """
        pass

    @abstractmethod
    async def get_resolver_groups(self) -> ServiceNowResponse:
        """
        Retrieves a list of resolver groups.

        Returns:
            ServiceNowResponse containing a list of resolver groups
        """
        pass

    @abstractmethod
    async def create_request(
        self,
        short_description: str,
        description: str,
        caller_id: str,
        requested_for: str,
        **kwargs
    ) -> ServiceNowResponse:
        """
        Creates a service request with the specified details.

        Args:
            short_description: A brief description of the service request.
            description: Detailed description of the service request.
            caller_id: The identifier of the person making the request.
            requested_for: The identifier of the person for whom the request is being made.
            **kwargs: Additional parameters for request creation.

        Returns:
            ServiceNowResponse containing the result of the service request creation operation
        """
        pass

    @abstractmethod
    async def update_request(
        self,
        sys_id: str,
        **kwargs
    ) -> ServiceNowResponse:
        """
        Updates an existing service request with new information.

        Args:
            sys_id: The sys_id of the service request to be updated.
            **kwargs: Fields to update on the request.

        Returns:
            ServiceNowResponse containing the result of the service request update operation
        """
        pass


class ServiceNowClient:
    """
    HTTP client for ServiceNow API with support for both sync and async operations.
    """
    def __init__(self, auth: ServiceNowAuth):
        """
        Initialize the ServiceNow client with authentication details.
        
        Args:
            auth: ServiceNow authentication configuration
        """
        self.auth = auth
        self.username = auth.username
        self.password = auth.password
        self.instance_name = auth.instance_name
        
        self.basic_auth = HTTPBasicAuth(self.username, self.password)
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.base_url = f"https://{self.instance_name}.service-now.com/api/now/table/"
        
        # Initialize session for async requests
        self._session = None
        logger.info(f"Initialized ServiceNowClient for instance: {self.instance_name}")

    async def get_session(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp session for async requests.
        
        Returns:
            aiohttp.ClientSession: The session object
        """
        if self._session is None or self._session.closed:
            auth = aiohttp.BasicAuth(self.username, self.password)
            self._session = aiohttp.ClientSession(auth=auth, headers=self.headers)
        return self._session

    async def close_session(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    @tracer.start_as_current_span("servicenow_client.make_request_sync")
    def make_request_sync(self, method: str, url: str, data: Dict = None) -> ServiceNowResponse:
        """
        Make a synchronous HTTP request to ServiceNow.
        
        Args:
            method: HTTP method (GET, POST, PUT, PATCH)
            url: The URL to request
            data: JSON data to send (for POST, PUT, PATCH)
            
        Returns:
            ServiceNowResponse with the API response
        """
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, auth=self.basic_auth)
            elif method == "POST":
                response = requests.post(url, json=data, headers=self.headers, auth=self.basic_auth)
            elif method == "PUT":
                response = requests.put(url, json=data, headers=self.headers, auth=self.basic_auth)
            elif method == "PATCH":
                response = requests.patch(url, json=data, headers=self.headers, auth=self.basic_auth)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return ServiceNowResponse.error(
                    message=f"Unsupported HTTP method: {method}"
                )

            if response.status_code in [200, 201]:
                result = response.json().get("result")
                if not result:
                    logger.warning("Empty result from ServiceNow")
                    return ServiceNowResponse.error(
                        code=StatusCode.VALIDATION_ERROR.value,
                        message="No data returned by ServiceNow",
                        details="Empty result set"
                    )
                return ServiceNowResponse.success(data=result)
            else:
                logger.error(f"Error response from ServiceNow: {response.status_code} - {response.text}")
                if response.status_code == 404:
                    return ServiceNowResponse.error(
                        code=StatusCode.NOT_FOUND.value,
                        message="The requested resource was not found",
                        details=response.text
                    )
                elif response.status_code == 401:
                    return ServiceNowResponse.error(
                        code=StatusCode.UNAUTHORIZED.value,
                        message="You are not authorized to access this resource",
                        details=response.text
                    )
                else:
                    return ServiceNowResponse.error(
                        code=response.status_code,
                        message="Error from ServiceNow API",
                        details=response.text
                    )

        except requests.exceptions.RequestException as error:
            logger.exception("Request exception")
            return ServiceNowResponse.error(
                message="Failed to connect to ServiceNow",
                details=str(error)
            )
        except json.JSONDecodeError:
            logger.exception("JSON parsing error")
            return ServiceNowResponse.error(
                message="Failed to parse ServiceNow response as JSON",
                details=f"Status code: {response.status_code}, Response: {response.text}"
            )
        except Exception as error:
            logger.exception("Unexpected error")
            return ServiceNowResponse.error(
                message="An unexpected error occurred",
                details=str(error)
            )

    @tracer.start_as_current_span("servicenow_client.make_request_async")
    async def make_request_async(self, method: str, url: str, data: Dict = None) -> ServiceNowResponse:
        """
        Make an asynchronous HTTP request to ServiceNow.
        
        Args:
            method: HTTP method (GET, POST, PUT, PATCH)
            url: The URL to request
            data: JSON data to send (for POST, PUT, PATCH)
            
        Returns:
            ServiceNowResponse with the API response
        """
        session = await self.get_session()
        
        try:
            if method == "GET":
                async with session.get(url) as response:
                    return await self._process_response(response)
            elif method == "POST":
                async with session.post(url, json=data) as response:
                    return await self._process_response(response)
            elif method == "PUT":
                async with session.put(url, json=data) as response:
                    return await self._process_response(response)
            elif method == "PATCH":
                async with session.patch(url, json=data) as response:
                    return await self._process_response(response)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return ServiceNowResponse.error(
                    message=f"Unsupported HTTP method: {method}"
                )
                
        except aiohttp.ClientError as error:
            logger.exception("aiohttp client error")
            return ServiceNowResponse.error(
                message="Failed to connect to ServiceNow",
                details=str(error)
            )
        except json.JSONDecodeError:
            logger.exception("JSON parsing error")
            return ServiceNowResponse.error(
                message="Failed to parse ServiceNow response as JSON"
            )
        except Exception as error:
            logger.exception("Unexpected error")
            return ServiceNowResponse.error(
                message="An unexpected error occurred",
                details=str(error)
            )

    async def _process_response(self, response: aiohttp.ClientResponse) -> ServiceNowResponse:
        """
        Process an aiohttp response into a ServiceNowResponse.
        
        Args:
            response: The aiohttp response object
            
        Returns:
            ServiceNowResponse containing the processed result
        """
        if response.status in [200, 201]:
            try:
                json_data = await response.json()
                result = json_data.get("result")
                if not result:
                    logger.warning("Empty result from ServiceNow")
                    return ServiceNowResponse.error(
                        code=StatusCode.VALIDATION_ERROR.value,
                        message="No data returned by ServiceNow",
                        details="Empty result set"
                    )
                return ServiceNowResponse.success(data=result)
            except json.JSONDecodeError:
                logger.exception("JSON parsing error")
                text = await response.text()
                return ServiceNowResponse.error(
                    message="Failed to parse ServiceNow response as JSON",
                    details=f"Response: {text}"
                )
        else:
            text = await response.text()
            logger.error(f"Error response from ServiceNow: {response.status} - {text}")
            
            if response.status == 404:
                return ServiceNowResponse.error(
                    code=StatusCode.NOT_FOUND.value,
                    message="The requested resource was not found",
                    details=text
                )
            elif response.status == 401:
                return ServiceNowResponse.error(
                    code=StatusCode.UNAUTHORIZED.value,
                    message="You are not authorized to access this resource",
                    details=text
                )
            else:
                return ServiceNowResponse.error(
                    code=response.status,
                    message="Error from ServiceNow API",
                    details=text
                )


class ServiceNow(ServiceNowBase):
    """
    Implementation of the ServiceNow base interface for interacting with ServiceNow instances.
    Provides methods for common ServiceNow operations like managing incidents, requests, and approvals.
    """
    
    def __init__(self, auth: Optional[ServiceNowAuth] = None):
        """
        Initialize the ServiceNow client.
        
        Args:
            auth: ServiceNow authentication configuration. If None, loads from environment variables.
        """
        if auth is None:
            auth = ServiceNowAuth.from_env()
            
        self.client = ServiceNowClient(auth)
        logger.info(f"Initialized ServiceNow client for instance: {auth.instance_name}")

    async def close(self):
        """Close all connections and resources."""
        await self.client.close_session()

    # User Management Methods
    
    @tracer.start_as_current_span("service_now.get_user_sys_id_by_employee_number")
    async def get_user_sys_id_by_employee_number(self, employee_number: str) -> Optional[str]:
        """
        Get the sys_id of a user by their employee number.
        
        Args:
            employee_number: The employee number to look up
            
        Returns:
            The sys_id if found, None otherwise
        """
        url = f"{self.client.base_url}sys_user?sysparm_query=employee_number={employee_number}&sysparm_fields=sys_id"
        response = await self.client.make_request_async("GET", url)
        
        if response.code == StatusCode.SUCCESS.value and response.data:
            return response.data[0]["sys_id"]
        return None

    @tracer.start_as_current_span("service_now.get_user_sys_id_by_user_id")
    async def get_user_sys_id_by_user_id(self, user_name: str) -> Optional[str]:
        """
        Get the sys_id of a user by their username.
        
        Args:
            user_name: The username to look up
            
        Returns:
            The sys_id if found, None otherwise
        """
        url = f"{self.client.base_url}sys_user?sysparm_query=user_name={user_name}&sysparm_fields=sys_id"
        response = await self.client.make_request_async("GET", url)
        
        if response.code == StatusCode.SUCCESS.value and response.data:
            return response.data[0]["sys_id"]
        return None

    @tracer.start_as_current_span("service_now.get_me_user_id")
    async def get_me_user_id(self) -> Optional[str]:
        """
        Get the sys_id of the currently authenticated user.
        
        Returns:
            The sys_id if found, None otherwise
        """
        url = f"{self.client.base_url}sys_user?sysparm_query=user_name={self.client.username}&sysparm_fields=sys_id"
        response = await self.client.make_request_async("GET", url)
        
        if response.code == StatusCode.SUCCESS.value and response.data:
            return response.data[0]["sys_id"]
        return None

    @tracer.start_as_current_span("service_now.get_user_by_id")
    async def get_user_by_id(self, employee_number: str) -> ServiceNowResponse:
        """
        Get a user's details by their employee number.
        
        Args:
            employee_number: The employee number to look up
            
        Returns:
            ServiceNowResponse containing the user details
        """
        user_sys_id = await self.get_user_sys_id_by_employee_number(employee_number)
        if user_sys_id:
            url = f"{self.client.base_url}sys_user?sysparm_query=sys_id={user_sys_id}"
            return await self.client.make_request_async("GET", url)
        return ServiceNowResponse.error(
            code=StatusCode.NOT_FOUND.value,
            message=f"No user found with employee number: {employee_number}"
        )

    @tracer.start_as_current_span("service_now.get_user_by_user_sys_id")
    async def get_user_by_user_sys_id(self, user_sys_id: str) -> ServiceNowResponse:
        """
        Get a user's details by their sys_id.
        
        Args:
            user_sys_id: The sys_id of the user to look up
            
        Returns:
            ServiceNowResponse containing the user details
        """
        url = f"{self.client.base_url}sys_user?sysparm_query=sys_id={user_sys_id}"
        return await self.client.make_request_async("GET", url)

    # Ticket Management Methods
    
    @tracer.start_as_current_span("service_now.get_tickets_by_agent_id")
    async def get_tickets_by_agent_id(self, agent_sys_id: str, in_last: int) -> ServiceNowResponse:
        """
        Get tickets assigned to a specific agent.
        
        Args:
            agent_sys_id: The sys_id of the agent
            in_last: Get tickets updated in the last X minutes
            
        Returns:
            ServiceNowResponse containing the tickets
        """
        if agent_sys_id:
            query = (f"incident?sysparm_query=assigned_to={agent_sys_id}"
                     f"^sys_updated_on>=javascript:gs.minutesAgoStart({in_last})"
                     f"^ORDERBYsys_updated_on")
        else:
            query = "incident?sysparm_query=ORDERBYDESCsys_created_on"
            
        url = f"{self.client.base_url}{query}"
        return await self.client.make_request_async("GET", url)

    @tracer.start_as_current_span("service_now.get_tickets_by_user_id")
    async def get_tickets_by_user_id(self, employee_number: str, limit: int = 1) -> ServiceNowResponse:
        """
        Get tickets created by a specific user.
        
        Args:
            employee_number: The employee number of the user
            limit: Maximum number of tickets to return
            
        Returns:
            ServiceNowResponse containing the tickets
        """
        user_sys_id = await self.get_user_sys_id_by_employee_number(employee_number)
        logger.debug(f"Found user sys_id: {user_sys_id} for employee: {employee_number}")
        
        if user_sys_id:
            query = (
                f"incident?sysparm_query=caller_id={user_sys_id}^ORDERBYDESCsys_created_on"
                f"&sysparm_limit={limit}"
                f"&sysparm_fields=number,short_description,description,sys_created_on,state,priority"
            )
            url = f"{self.client.base_url}{query}"
            return await self.client.make_request_async("GET", url)
            
        return ServiceNowResponse.error(
            code=StatusCode.NOT_FOUND.value,
            message=f"No user found with employee number: {employee_number}"
        )

    @tracer.start_as_current_span("service_now.create_ticket")
    async def create_ticket(
        self,
        subject: str,
        description: str,
        urgency: int,
        impact: int,
        employee_number: Optional[str] = None,
        emp_sys_id: Optional[str] = None,
        **kwargs
    ) -> ServiceNowResponse:
        """
        Create a new incident ticket.
        
        Args:
            subject: Short description of the incident
            description: Detailed description
            urgency: Urgency level (1-3)
            impact: Impact level (1-3)
            employee_number: Employee number of the caller
            emp_sys_id: System ID of the caller (alternative to employee_number)
            **kwargs: Additional ticket fields
            
        Returns:
            ServiceNowResponse with the created ticket details
        """
        if employee_number:
            user_sys_id = await self.get_user_sys_id_by_employee_number(employee_number)
            if user_sys_id is None:
                return ServiceNowResponse.error(
                    code=StatusCode.NOT_FOUND.value,
                    message=f"No user found for employee {employee_number}"
                )
        else:
            user_sys_id = emp_sys_id
            
        if not user_sys_id:
            return ServiceNowResponse.error(
                code=StatusCode.BAD_REQUEST.value,
                message="Either employee_number or emp_sys_id must be provided"
            )
            
        url = f"{self.client.base_url}incident"
        data = {
            "short_description": subject,
            "description": description,
            "urgency": urgency,
            "impact": impact,
            "caller_id": user_sys_id,
            **{k: v for k, v in kwargs.items() if v is not None}
        }
        
        response = await self.client.make_request_async("POST", url, data)
        
        # Add link to the incident in the response
        if response.code == StatusCode.SUCCESS.value and response.data:
            incident_sys_id = response.data.get("sys_id")
            incident_number = response.data.get("number")
            if incident_sys_id:
                incident_link = f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=incident.do?sys_id={incident_sys_id}"
                response.data["link"] = incident_link
                logger.info(f"Created incident {incident_number} with sys_id: {incident_sys_id}")
                
        return response

    @tracer.start_as_current_span("service_now.get_ticket_by_id")
    async def get_ticket_by_id(self, ticket_id: str, user_sys_id: Optional[str] = None) -> ServiceNowResponse:
        """
        Retrieve specific incident details by incident ID.
        
        Args:
            ticket_id: The ticket number (e.g., INC0010005)
            user_sys_id: Optional user sys_id to check permissions
            
        Returns:
            ServiceNowResponse containing the ticket details
        """
        url = f"{self.client.base_url}incident?sysparm_query=number={ticket_id}"
        response = await self.client.make_request_async("GET", url)
        
        if response.code == StatusCode.SUCCESS.value and response.data:
            # Check if user has permission to view this ticket
            if user_sys_id and response.data[0].get("caller_id", {}).get("value") != user_sys_id:
                return ServiceNowResponse.error(
                    code=StatusCode.VALIDATION_ERROR.value,
                    message=f"Access denied: The employee does not have permission to view or modify ticket {ticket_id}"
                )
                
            # Add link to the incident
            for incident in response.data:
                incident_sys_id = incident.get("sys_id")
                if incident_sys_id:
                    incident["link"] = f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=incident.do?sys_id={incident_sys_id}"
                    
            return response
            
        return ServiceNowResponse.error(
            code=StatusCode.NOT_FOUND.value,
            message=f"No ticket found with incident id: {ticket_id}"
        )

    @tracer.start_as_current_span("service_now.update_ticket")
    async def update_ticket(
        self,
        incident_sys_id: str,
        **kwargs
    ) -> ServiceNowResponse:
        """
        Update an existing incident ticket.
        
        Args:
            incident_sys_id: System ID of the incident to update
            **kwargs: Fields to update
            
        Returns:
            ServiceNowResponse with the updated ticket details
        """
        url = f"{self.client.base_url}incident/{incident_sys_id}"
        data = {k: v for k, v in kwargs.items() if v is not None}
        
        if not data:
            return ServiceNowResponse.error(
                code=StatusCode.BAD_REQUEST.value,
                message="No update data provided"
            )
            
        return await self.client.make_request_async("PATCH", url, data)

    @tracer.start_as_current_span("service_now.update_incident")
    async def update_incident(
        self, 
        incident_number: str, 
        update_data: Dict[str, Any], 
        user_sys_id: Optional[str] = None
    ) -> ServiceNowResponse:
        """
        Update an incident by incident number.
        
        Args:
            incident_number: The incident number (e.g., INC0010005)
            update_data: Dictionary of fields to update
            user_sys_id: Optional user sys_id to check permissions
            
        Returns:
            ServiceNowResponse with the update result
        """
        # First, get the sys_id of the incident
        if user_sys_id:
            get_result = await self.get_ticket_by_id(incident_number, user_sys_id)
        else:
            get_result = await self.get_ticket_by_id(incident_number)
            
        if get_result.code != StatusCode.SUCCESS.value:
            return get_result  # Return the error if incident not found

        sys_id = get_result.data[0]["sys_id"]
        url = f"{self.client.base_url}incident/{sys_id}"
        
        return await self.client.make_request_async("PUT", url, data=update_data)

    # Helper Methods
    
    @tracer.start_as_current_span("service_now.add_comment_to_incident")
    async def add_comment_to_incident(self, incident_number: str, comment: str, agent: str = "TVA") -> ServiceNowResponse:
        """
        Add a comment to an incident.
        
        Args:
            incident_number: The incident number (e.g., INC0010005)
            comment: The comment text to add
            agent: The agent to use for the comment (default: TVA)
            
        Returns:
            ServiceNowResponse with the result
        """
        # First, get the sys_id of the incident
        incident_url = f"{self.client.base_url}incident?sysparm_query=number={incident_number}&sysparm_fields=sys_id"
        incident_response = await self.client.make_request_async("GET", incident_url)

        if incident_response.code != StatusCode.SUCCESS.value or not incident_response.data:
            return incident_response

        incident_sys_id = incident_response.data[0]["sys_id"]

        # Now, add the comment to the incident
        comment_url = f"{self.client.base_url}incident/{incident_sys_id}"
        comment_data = {"comments": comment}

        # Use agent credentials if specified
        if agent == "TVA":
            agent_auth = ServiceNowAuth(
                username=os.getenv("TVA_USERNAME", "ticketingvirtualagent@futurepath.ai"),
                password=os.getenv("TVA_PASSWORD", "Futurepath@789"),
                instance_name=self.client.instance_name
            )
        elif agent == "NVA":
            agent_auth = ServiceNowAuth(
                username=os.getenv("NVA_USERNAME", "nocvirtualanlyst@futurepath.ai"),
                password=os.getenv("NVA_PASSWORD", ""),
                instance_name=self.client.instance_name
            )
        else:
            return ServiceNowResponse.error(
                code=StatusCode.BAD_REQUEST.value,
                message=f"Invalid agent type: {agent}"
            )

        agent_client = ServiceNowClient(agent_auth)
        try:
            return await agent_client.make_request_async("PATCH", comment_url, comment_data)
        finally:
            await agent_client.close_session()

    @tracer.start_as_current_span("service_now.read_incident_comments")
    async def read_incident_comments(self, incident_number: str) -> ServiceNowResponse:
        """
        Read all comments from an incident.
        
        Args:
            incident_number: The incident number (e.g., INC0010005)
            
        Returns:
            ServiceNowResponse containing the comments
        """
        # First, get the sys_id of the incident
        incident_url = f"{self.client.base_url}incident?sysparm_query=number={incident_number}&sysparm_fields=sys_id"
        incident_response = await self.client.make_request_async("GET", incident_url)

        if incident_response.code != StatusCode.SUCCESS.value or not incident_response.data:
            return incident_response

        incident_sys_id = incident_response.data[0]["sys_id"]

        # Now, fetch the comments for this incident
        comments_url = (f"{self.client.base_url}sys_journal_field"
                       f"?sysparm_query=element_id={incident_sys_id}^element=comments"
                       f"&sysparm_fields=value,sys_created_on,sys_created_by"
                       f"&sysparm_order_by_desc=sys_created_on")

        return await self.client.make_request_async("GET", comments_url)

    @tracer.start_as_current_span("service_now.get_resolver_groups")
    async def get_resolver_groups(self) -> ServiceNowResponse:
        """
        Get all resolver groups from ServiceNow.
        
        Returns:
            ServiceNowResponse containing the resolver groups
        """
        url = f"{self.client.base_url}group?sysparm_query=type=resolver"
        return await self.client.make_request_async("GET", url)

    @tracer.start_as_current_span("service_now.get_assignment_groups")
    async def get_assignment_groups(self) -> ServiceNowResponse:
        """
        Get all assignment groups from ServiceNow.
        
        Returns:
            ServiceNowResponse containing the assignment groups
        """
        url = f"{self.client.base_url}sys_user_group"
        response = await self.client.make_request_async("GET", url)

        if response.code == StatusCode.SUCCESS.value and response.data:
            formatted_groups = []
            for group in response.data:
                formatted_groups.append({
                    "name": group.get("name"),
                    "description": group.get("description"),
                    "sys_id": group.get("sys_id"),
                })
            return ServiceNowResponse.success(data=formatted_groups)
        return response

    @tracer.start_as_current_span("service_now.get_assignment_group_members")
    async def get_assignment_group_members(self, assignment_group_sys_id: str) -> ServiceNowResponse:
        """
        Get all members of an assignment group.
        
        Args:
            assignment_group_sys_id: The sys_id of the assignment group
            
        Returns:
            ServiceNowResponse containing the group members
        """
        members_endpoint = (
            f"{self.client.base_url}sys_user_grmember"
            f"?sysparm_query=group={assignment_group_sys_id}"
            f"&sysparm_fields=user.user_name,user.email,user.first_name,user.last_name,user.active,user.sys_id"
        )
        return await self.client.make_request_async("GET", members_endpoint)

    @tracer.start_as_current_span("service_now.get_all_user_groups_with_members")
    async def get_all_user_groups_with_members(self) -> ServiceNowResponse:
        """
        Get all user groups with their members.
        
        Returns:
            ServiceNowResponse containing the groups and members
        """
        endpoint = (
            f"{self.client.base_url}sys_user_grmember"
            f"?sysparm_fields=group.name,group.sys_id,user.first_name,user.last_name,"
            f"user.sys_id,user.user_name,user.email,user.active"
        )
        return await self.client.make_request_async("GET", endpoint)

    # Service Request Methods
    
    @tracer.start_as_current_span("service_now.create_request")
    async def create_request(
        self,
        short_description: str,
        description: str,
        employee_number: str,
        requested_for: str,
        approval: str = "not requested",
        priority: int = 3,
        special_instructions: Optional[str] = None,
    ) -> ServiceNowResponse:
        """
        Create a new service request.
        
        Args:
            short_description: Brief description of the request
            description: Detailed description
            employee_number: Employee number of the caller
            requested_for: User ID for whom the request is being made
            approval: Approval status (default: "not requested")
            priority: Priority level (1-5, default: 3)
            special_instructions: Additional instructions
            
        Returns:
            ServiceNowResponse with the created request details
        """
        caller_sys_id = await self.get_user_sys_id_by_employee_number(employee_number)

        if not caller_sys_id:
            return ServiceNowResponse.error(
                code=StatusCode.NOT_FOUND.value,
                message=f"No ServiceNow user found with employee number: {employee_number}"
            )

        url = f"{self.client.base_url}sc_request"
        data = {
            "short_description": short_description,
            "description": description,
            "caller_id": caller_sys_id,
            "requested_for": requested_for,
            "approval": approval,
            "priority": priority,
        }
        
        if special_instructions:
            data["special_instructions"] = special_instructions
            
        response = await self.client.make_request_async("POST", url, data)
        
        # Add link to the request in the response
        if response.code == StatusCode.SUCCESS.value and response.data:
            request_sys_id = response.data.get("sys_id")
            if request_sys_id:
                request_link = f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=sc_request.do?sys_id={request_sys_id}"
                response.data["link"] = request_link
                
        return response

    @tracer.start_as_current_span("service_now.update_request")
    async def update_request(
        self,
        sys_id: str,
        **kwargs
    ) -> ServiceNowResponse:
        """
        Update an existing service request.
        
        Args:
            sys_id: System ID of the request to update
            **kwargs: Fields to update
            
        Returns:
            ServiceNowResponse with the updated request details
        """
        url = f"{self.client.base_url}sc_request/{sys_id}"
        data = {k: v for k, v in kwargs.items() if v is not None}
        
        if not data:
            return ServiceNowResponse.error(
                code=StatusCode.BAD_REQUEST.value,
                message="No update data provided"
            )
            
        return await self.client.make_request_async("PATCH", url, data)

    @tracer.start_as_current_span("service_now.get_request_by_number")
    async def get_request_by_number(
        self, 
        request_number: str, 
        user_sys_id: Optional[str] = None
    ) -> ServiceNowResponse:
        """
        Get a service request by its number.
        
        Args:
            request_number: The request number (e.g., REQ0010005)
            user_sys_id: Optional user sys_id to check permissions
            
        Returns:
            ServiceNowResponse containing the request details
        """
        url = f"{self.client.base_url}sc_request?number={request_number}"
        logger.debug(f"Fetching ServiceNow request from URL: {url}")

        response = await self.client.make_request_async("GET", url)

        if response.code == StatusCode.VALIDATION_ERROR.value:
            return ServiceNowResponse.error(
                code=StatusCode.NOT_FOUND.value,
                message=f"No request found with number: {request_number}"
            )

        if response.code == StatusCode.SUCCESS.value:
            logger.debug("Request fetched successfully")
            
            # Check if user has permission to view this request
            if user_sys_id and response.data[0].get("requested_for", {}).get("value") != user_sys_id:
                return ServiceNowResponse.error(
                    code=StatusCode.VALIDATION_ERROR.value,
                    message=f"Access denied: The employee does not have permission to view or modify request {request_number}"
                )
                
            # Add link to the request
            for request in response.data:
                request_sys_id = request.get("sys_id")
                if request_sys_id:
                    request["link"] = f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=sc_request.do?sys_id={request_sys_id}"
                    
            return ServiceNowResponse.success(data=response.data)
        else:
            logger.debug(f"Failed to fetch request. Status Code: {response.code}")
            return response

    @tracer.start_as_current_span("service_now.update_request_by_sys_id")
    async def update_request_by_sys_id(
        self, 
        request_number: str, 
        data: Dict[str, Any], 
        user_sys_id: Optional[str] = None
    ) -> ServiceNowResponse:
        """
        Update a service request by its number.
        
        Args:
            request_number: The request number (e.g., REQ0010005)
            data: Dictionary of fields to update
            user_sys_id: Optional user sys_id to check permissions
            
        Returns:
            ServiceNowResponse with the update result
        """
        # Get the request details first
        if user_sys_id:
            request_response = await self.get_request_by_number(request_number, user_sys_id)
        else:
            request_response = await self.get_request_by_number(request_number)
            
        logger.debug(f"Request details: {request_response}")

        # Check for error responses
        if request_response.code != StatusCode.SUCCESS.value:
            return request_response

        # Verify we have valid request data
        if not request_response.data or "sys_id" not in request_response.data[0]:
            logger.error(f"Failed to fetch request or sys_id not found for request number: {request_number}")
            return ServiceNowResponse.error(
                code=StatusCode.NOT_FOUND.value,
                message="Request not found or invalid"
            )

        # Extract sys_id and update the request
        sys_id = request_response.data[0]["sys_id"]
        logger.debug(f"Request sys_id: {sys_id}")

        url = f"{self.client.base_url}sc_request/{sys_id}"
        logger.debug(f"Updating ServiceNow request with URL: {url} and data: {data}")

        return await self.client.make_request_async("PUT", url, data)

    # Approval Methods
    
    @tracer.start_as_current_span("service_now.approve_request")
    async def approve_request(self, approval_sys_id: str) -> ServiceNowResponse:
        """
        Approve a request.
        
        Args:
            approval_sys_id: The sys_id of the approval
            
        Returns:
            ServiceNowResponse with the result
        """
        url = f"{self.client.base_url}sysapproval_approver/{approval_sys_id}"
        data = {"state": "approved"}
        return await self.client.make_request_async("PATCH", url, data)

    @tracer.start_as_current_span("service_now.reject_request")
    async def reject_request(
        self, 
        approval_sys_id: str, 
        comments: Optional[str] = None
    ) -> ServiceNowResponse:
        """
        Reject a request.
        
        Args:
            approval_sys_id: The sys_id of the approval
            comments: Optional comments explaining the rejection
            
        Returns:
            ServiceNowResponse with the result
        """
        url = f"{self.client.base_url}sysapproval_approver/{approval_sys_id}"
        data = {"state": "rejected"}
        if comments:
            data["comments"] = comments
        return await self.client.make_request_async("PATCH", url, data)

    @tracer.start_as_current_span("service_now.add_comment_to_approval")
    async def add_comment_to_approval(self, approval_sys_id: str, comment: str) -> ServiceNowResponse:
        """
        Add a comment to an approval request.
        
        Args:
            approval_sys_id: The sys_id of the approval
            comment: The comment text
            
        Returns:
            ServiceNowResponse with the result
        """
        url = f"{self.client.base_url}sysapproval_approver/{approval_sys_id}"
        data = {"comments": comment}
        return await self.client.make_request_async("PATCH", url, data)

    @tracer.start_as_current_span("service_now.get_all_requested_approvals")
    async def get_all_requested_approvals(self) -> ServiceNowResponse:
        """
        Get all requested approvals that are pending.
        
        Returns:
            ServiceNowResponse containing the approvals
        """
        fields = [
            "sys_id", "approver.name", "approver.email", "approver.sys_id",
            "due_date", "state", "sys_created_on", "sysapproval.number",
            "sysapproval.impact", "sysapproval.priority", "sysapproval.requested_by",
            "sysapproval.requested_by.name", "sysapproval.requested_by.email",
            "sysapproval.requested_by.sys_id", "sysapproval.sys_created_on",
            "sysapproval.approval", "sysapproval.description", "sysapproval.state",
            "sysapproval.change_type", "sysapproval.short_description",
            "sysapproval.number", "sysapproval.caller_id.name",
            "sysapproval.caller_id.email", "sysapproval.caller_id.sys_id",
            "sysapproval.requested_for.name", "sysapproval.requested_for.email",
            "sysapproval.requested_for.sys_id", "sysapproval.opened_by.name",
            "sysapproval.opened_by.email", "sysapproval.opened_by.sys_id",
            "sysapproval.type", "approval_for", "approval_id", "source_table",
        ]
        
        url = f"{self.client.base_url}sysapproval_approver?sysparm_query=state=requested"
        url += f"&sysparm_fields={','.join(fields)}"
        url += "&sysparm_display_value=all"
        
        return await self.client.make_request_async("GET", url)

    @tracer.start_as_current_span("service_now.get_latest_pending_approvals")
    async def get_latest_pending_approvals(
        self, 
        user_sys_id: str, 
        limit: int = 3
    ) -> ServiceNowResponse:
        """
        Get the latest pending approvals for a specific user.
        
        Args:
            user_sys_id: The sys_id of the user
            limit: Maximum number of approvals to return
            
        Returns:
            ServiceNowResponse containing the approvals
        """
        fields = [
            "sys_id", "sysapproval.number", "sysapproval.short_description",
            "sysapproval.description", "sysapproval.sys_created_on",
            "sysapproval.priority", "sysapproval.urgency", 
            "sysapproval.requested_for", "state", "approver", "document_id",
            "due_date", "sysapproval.sys_class_name", "sysapproval.sys_id",
        ]

        query = (
            f"sysapproval_approver"
            f"?sysparm_query=state=requested^approver={user_sys_id}^ORDERBYDESCsys_created_on"
            f"&sysparm_fields={','.join(fields)}"
            f"&sysparm_limit={limit}"
            f"&sysparm_display_value=true"
        )

        url = f"{self.client.base_url}{query}"
        response = await self.client.make_request_async("GET", url)

        if response.code == StatusCode.SUCCESS.value and response.data:
            approvals = []

            for item in response.data:
                # Get the approval type and sys_id
                approval_type = item.get("sysapproval.sys_class_name", "").lower()
                approval_sys_id = item.get("sysapproval.sys_id")

                # Construct the link based on approval type
                if approval_type == "requested item":
                    link = f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=sc_req_item.do?sys_id={approval_sys_id}"
                elif approval_type == "incident":
                    link = f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=incident.do?sys_id={approval_sys_id}"
                elif approval_type == "sc_request":
                    link = f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=sc_request.do?sys_id={approval_sys_id}"
                elif approval_type == "change_request":
                    link = f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=change_request.do?sys_id={approval_sys_id}"
                else:
                    link = f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=task.do?sys_id={approval_sys_id}"

                approval = {
                    "number": item.get("sysapproval.number", "N/A"),
                    "short_description": item.get("sysapproval.short_description", "N/A"),
                    "description": item.get("sysapproval.description", "N/A"),
                    "created": item.get("sysapproval.sys_created_on", "N/A"),
                    "priority": item.get("sysapproval.priority", "N/A"),
                    "urgency": item.get("sysapproval.urgency", "N/A"),
                    "requested_for": item.get("sysapproval.requested_for", {}).get("display_value", "N/A"),
                    "state": item.get("state", "N/A"),
                    "due_date": item.get("due_date", "N/A"),
                    "document_id": item.get("document_id", {}).get("display_value", "N/A"),
                    "link": link,
                }

                approvals.append(approval)

            return ServiceNowResponse.success(data=approvals)
        else:
            return ServiceNowResponse.success(
                data=[],
                message="No pending approvals found"
            )

    # File Attachment Methods

    @tracer.start_as_current_span("service_now.create_incident_and_attach_files_with_blob")
    async def create_incident_and_attach_files_with_blob(
        self,
        files: List[Dict[str, Any]],
        short_description: str = "New Incident created via API",
        description: str = "Initial incident creation",
        comments: str = "Initial incident creation",
        service_now_user_sys_id: Optional[str] = None
    ) -> ServiceNowResponse:
        """
        Creates a new incident and attaches files to it.
        
        Args:
            files: List of dictionaries with file details (file_blob and file_name)
            short_description: Brief description of the incident
            comments: Initial comments
            service_now_user_sys_id: User sys_id to associate with the incident
            
        Returns:
            ServiceNowResponse with the result
        """
        logger.info(
            f"Creating new incident with short_description: '{short_description}' and comments: '{comments}'"
        )
        
        # URL for incident creation
        create_incident_url = f"https://{self.client.instance_name}.service-now.com/api/now/table/incident"
        payload = {
            "short_description": short_description,
            "comments": comments,
            "description": description,
        }
        
        if service_now_user_sys_id:
            payload["caller_id"] = service_now_user_sys_id
        
        try:
            # Create the incident
            create_response = requests.post(
                create_incident_url,
                headers=self.client.headers,
                json=payload,
                auth=self.client.basic_auth,
            )
            
            if create_response.status_code not in [200, 201]:
                logger.error(
                    f"Failed to create incident. Status code: {create_response.status_code}, Response: {create_response.text}"
                )
                return ServiceNowResponse.error(
                    code=create_response.status_code,
                    message="Failed to create incident",
                    details=create_response.text
                )
                
            incident_data = create_response.json()
            incident_sys_id = incident_data["result"]["sys_id"]
            incident_number = incident_data["result"]["number"]
            incident_link = f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=incident.do?sys_id={incident_sys_id}"
            
            logger.info(f"Incident created with sys_id: {incident_sys_id}")
            
            # Attach files to the incident
            attachment_url = f"https://{self.client.instance_name}.service-now.com/api/now/attachment/file"
            headers = self.client.headers.copy()
            headers.pop("Content-Type", None)  # Let requests set Content-Type for file uploads

            successful_uploads = 0
            failed_uploads = 0
            
            for file_item in files:
                file_blob = file_item.get("file_blob")
                file_name = file_item.get("file_name")
                
                if not file_blob:
                    logger.error(f"No file blob provided for {file_name}")
                    failed_uploads += 1
                    continue

                try:
                    # If file_blob is a bytes object, wrap it in a BytesIO
                    if isinstance(file_blob, bytes):
                        file_obj = io.BytesIO(file_blob)
                    else:
                        file_obj = file_blob  # Assume it's already a file-like object

                    # Prepare the file payload
                    files_payload = {"file": (file_name, file_obj, "application/octet-stream")}
                    params = {
                        "table_name": "incident",
                        "table_sys_id": incident_sys_id,
                        "file_name": file_name,
                    }

                    attachment_response = requests.post(
                        attachment_url,
                        files=files_payload,
                        params=params,
                        headers=headers,
                        auth=self.client.basic_auth,
                    )

                    if attachment_response.status_code == 201:
                        logger.info(f"Successfully attached file '{file_name}' to incident {incident_sys_id}")
                        successful_uploads += 1
                    else:
                        failed_uploads += 1
                        logger.error(
                            f"Failed to attach file '{file_name}' to incident {incident_sys_id}. "
                            f"Status: {attachment_response.status_code}"
                        )
                except Exception as e:
                    failed_uploads += 1
                    logger.error(f"Error attaching file '{file_name}' to incident {incident_sys_id}: {str(e)}")

            message = f"Incident created with sys_id {incident_sys_id}. Successfully attached {successful_uploads} file(s)"
            if failed_uploads:
                message += f", failed to attach {failed_uploads} file(s)."

            return ServiceNowResponse.created(
                data={
                    "incident_sys_id": incident_sys_id,
                    "incident_link": incident_link,
                    "incident_number": incident_number,
                    "successful_uploads": successful_uploads,
                    "failed_uploads": failed_uploads
                },
                message=message
            )
            
        except Exception as e:
            logger.error(f"Error creating incident: {str(e)}")
            return ServiceNowResponse.error(
                message=f"Error creating incident: {str(e)}",
                details=str(e)
            )

    @tracer.start_as_current_span("service_now.get_incident_attachments")
    async def get_incident_attachments(self, incident_number: str) -> ServiceNowResponse:
        """
        Get all attachments for a specific incident.
        
        Args:
            incident_number: The incident number to get attachments for
            
        Returns:
            ServiceNowResponse containing the attachments
        """
        # First get the sys_id for the incident
        incident_url = f"{self.client.base_url}incident?sysparm_query=number={incident_number}"
        incident_response = await self.client.make_request_async("GET", incident_url)

        if not incident_response.data:
            return ServiceNowResponse.error(
                code=StatusCode.NOT_FOUND.value,
                message="Incident not found",
                details=f"No incident found with number {incident_number}"
            )

        incident_sys_id = incident_response.data[0]["sys_id"]

        attachment_url = (
            f"{self.client.base_url}sys_attachment?"
            f"sysparm_query=table_sys_id={incident_sys_id}^table_name=incident"
            f"&sysparm_fields=sys_id,file_name,size_bytes,content_type"
        )
        response = await self.client.make_request_async("GET", attachment_url)

        if response.code == StatusCode.SUCCESS.value and response.data:
            formatted_attachments = []
            for attachment in response.data:
                formatted_attachments.append({
                    "incident_number": incident_number,
                    "file_name": attachment.get("file_name"),
                    "size_bytes": attachment.get("size_bytes"),
                    "content_type": attachment.get("content_type"),
                    "sys_id": attachment.get("sys_id"),
                    "download_link": (
                        f"https://{self.client.instance_name}.service-now.com/sys_attachment.do?"
                        f"sys_id={attachment.get('sys_id')}&sysparm_download=true"
                    ),
                })
            return ServiceNowResponse.success(
                message="Successfully retrieved attachments",
                data=formatted_attachments
            )
        else:
            return ServiceNowResponse.success(
                message="No attachments found", 
                data=[]
            )

    @tracer.start_as_current_span("service_now.attach_file_to_incident")
    async def attach_file_to_incident(
        self,
        incident_number: str,
        file_names: List[str],
        file_ids: str,
        user_sys_id: str,
        chat_virtual_agent: Any,  # ChatVirtualAgent
    ) -> ServiceNowResponse:
        """
        Downloads files from SharePoint and attaches them to a ServiceNow incident.
        
        Args:
            incident_number: The incident number (e.g. INC0012536)
            file_names: List of file names to attach
            file_ids: File IDs to attach to the incident
            user_sys_id: User sys_id for permission checking
            chat_virtual_agent: Virtual agent for SharePoint integration
            
        Returns:
            ServiceNowResponse with the result
        """
        logger.info(
            f"Attaching file to incident {incident_number} with file ID {file_ids} and file names {file_names}"
        )

        # Initialize file_paths list at the start
        file_paths = []

        # Convert single incident number to list for consistent processing
        incident_numbers = [incident_number] if isinstance(incident_number, str) else incident_number

        try:
            incident_sys_ids = []
            incident_links = []

            # Get incident sys_ids for all incident numbers
            for inc_number in incident_numbers:
                incident_lookup_url = f"https://{self.client.instance_name}.service-now.com/api/now/table/incident"
                params = {
                    "sysparm_query": f"number={inc_number}^caller_id={user_sys_id}",
                    "sysparm_fields": "sys_id",
                    "sysparm_limit": 1,
                }

                response = requests.get(
                    incident_lookup_url,
                    headers=self.client.headers,
                    params=params,
                    auth=self.client.basic_auth,
                )

                if response.status_code != 200:
                    logger.error(
                        f"Failed to lookup incident {inc_number}. Status code: {response.status_code}. Response: {response.text}"
                    )
                    return ServiceNowResponse.error(
                        code=response.status_code,
                        message=f"Failed to lookup incident {inc_number}"
                    )

                incident_data = response.json()
                if not incident_data.get("result"):
                    logger.error(f"No incident found for {inc_number}")
                    return ServiceNowResponse.error(
                        code=StatusCode.NOT_FOUND.value,
                        message=f"No incident found for {inc_number}"
                    )

                incident_sys_id = incident_data["result"][0]["sys_id"]
                incident_sys_ids.append(incident_sys_id)
                incident_links.append(
                    f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=incident.do?sys_id={incident_sys_id}"
                )

            if not incident_sys_ids:
                return ServiceNowResponse.error(
                    code=StatusCode.NOT_FOUND.value,
                    message="No valid incidents found"
                )

            successful_uploads = 0
            failed_uploads = 0

            # Process each file: download from SharePoint and upload to ServiceNow
            for file_name in file_names:
                try:
                    # Get file attachment info
                    attachments = await chat_virtual_agent.get_file_attachments_by_session(file_ids, file_name)
                    if not attachments:
                        failed_uploads += 1
                        logger.error(f"No attachment found for file {file_name}")
                        continue

                    for attachment in attachments:
                        file_url = attachment["content_url"]
                        file_path = await chat_virtual_agent.download_file_from_sharepoint(file_url, attachment["name"])

                        if not file_path:
                            failed_uploads += 1
                            logger.error(f"Failed to download {file_name} from SharePoint")
                            continue

                        file_paths.append((file_path, attachment["name"]))

                        # Upload file to each incident using the multipart endpoint
                        for incident_sys_id in incident_sys_ids:
                            servicenow_attachment_url = f"https://{self.client.instance_name}.service-now.com/api/now/attachment/upload"
                            headers = self.client.headers.copy()
                            # Remove Content-Type header so that requests sets multipart boundaries correctly
                            headers.pop("Content-Type", None)
                            # Set Accept header to request JSON response
                            headers["Accept"] = "application/json"

                            # Prepare form data for multipart POST
                            data = {
                                "table_name": "incident",
                                "table_sys_id": incident_sys_id,
                            }

                            with open(file_path, "rb") as file_obj:
                                original_filename = attachment["name"]
                                mime_type, _ = mimetypes.guess_type(original_filename)
                                if not mime_type:
                                    mime_type = "application/octet-stream"

                                # "uploadFile" is the key for the file content in the multipart form.
                                files = {
                                    "uploadFile": (
                                        original_filename,
                                        file_obj,
                                        mime_type,
                                    )
                                }

                                upload_response = requests.post(
                                    servicenow_attachment_url,
                                    headers=headers,
                                    data=data,
                                    files=files,
                                    auth=self.client.basic_auth,
                                )

                            if upload_response.status_code == 201:
                                successful_uploads += 1
                            else:
                                failed_uploads += 1
                                logger.error(
                                    f"Failed to upload {original_filename} to incident {incident_sys_id}. "
                                    f"Status: {upload_response.status_code}. Response: {upload_response.text}"
                                )

                except Exception as e:
                    failed_uploads += 1
                    logger.error(f"Error processing {file_name}: {str(e)}")

            if successful_uploads > 0:
                message = f"Successfully attached {successful_uploads} file(s)"
                if failed_uploads > 0:
                    message += f", failed to attach {failed_uploads} file(s)"
                    
                return ServiceNowResponse.created(
                    data={
                        "incident_links": incident_links,
                        "successful_uploads": successful_uploads,
                        "failed_uploads": failed_uploads
                    },
                    message=message
                )
            else:
                return ServiceNowResponse.error(
                    code=StatusCode.INTERNAL_SERVER_ERROR.value,
                    message="Failed to attach any files"
                )

        except Exception as e:
            logger.error(f"An error occurred while attaching files: {str(e)}")
            return ServiceNowResponse.error(
                code=StatusCode.INTERNAL_SERVER_ERROR.value,
                message=f"Failed to attach files: {str(e)}"
            )

        finally:
            # Clean up any temporary files that were created
            for file_path, _ in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Temporary file {file_path} deleted")

    # Service Catalog Methods
    
    @tracer.start_as_current_span("service_now.get_all_service_catalogs")
    async def get_all_service_catalogs(self) -> ServiceNowResponse:
        """
        Fetch all service catalog forms with their names and sys_id.
        
        Returns:
            ServiceNowResponse containing the service catalogs
        """
        url = (
            f"{self.client.base_url}sc_cat_item"
            f"?sysparm_fields=sys_id,name,description,short_description"
            f"&sysparm_display_value=true"
        )

        try:
            response = await self.client.make_request_async("GET", url)
            if response.code == StatusCode.SUCCESS.value:
                catalogs = response.data
                result = []
                for cat in catalogs:
                    # Ensure all values are strings and handle potential None values
                    catalog_entry = {
                        "sys_id": str(cat.get("sys_id", "")),
                        "name": str(cat.get("name", "")),
                        "description": str(cat.get("description", "")),
                        "short_description": str(cat.get("short_description", "")),
                    }
                    result.append(catalog_entry)
                return ServiceNowResponse.success(data=result)
            else:
                error_message = f"Failed to fetch service catalogs: {response.message}"
                logger.error(error_message)
                return response
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            logger.error(f"Detailed error: {error_message}")
            return ServiceNowResponse.error(
                message=error_message,
                details=str(e)
            )

    @tracer.start_as_current_span("service_now.generate_service_catalog_link")
    async def generate_service_catalog_link(self, catalog_sys_id: str) -> str:
        """
        Generates a service catalog link based on the catalog sys_id.
        
        Args:
            catalog_sys_id: The sys_id of the catalog item
            
        Returns:
            The URL to the service catalog item
        """
        return f"https://{self.client.instance_name}.service-now.com/sp?id=sc_cat_item&sys_id={catalog_sys_id}"

    @tracer.start_as_current_span("service_now.get_catalog_item_variables")
    async def get_catalog_item_variables(self, catalog_sys_id: str) -> ServiceNowResponse:
        """
        Fetches catalog item variables for a catalog item.
        
        Args:
            catalog_sys_id: The sys_id of the catalog item
            
        Returns:
            ServiceNowResponse containing the variables
        """
        logger.debug(f"Fetching catalog item variables for catalog sys_id: {catalog_sys_id}")
        url = (
            f"{self.client.base_url}item_option_new"
            f"?sysparm_query=cat_item={catalog_sys_id}"
            f"&sysparm_fields=name,mandatory,question_text,type,choices"
        )
        response = await self.client.make_request_async("GET", url)

        logger.debug(f"#Full response: {response}")  # Log the entire response

        if response.code == StatusCode.SUCCESS.value:
            variables = response.data
            if not variables:
                logger.warning(f"No variables found for catalog item: {catalog_sys_id}")
            logger.debug(f"Catalog item variables fetched: {variables}")
            return ServiceNowResponse.success(data=variables)
        else:
            logger.error(
                f"Failed to fetch catalog item variables. Status code: {response.code}, Message: {response.message}"
            )
            return response

    @tracer.start_as_current_span("service_now.get_catalog_item_variables_with_metadata")
    async def get_catalog_item_variables_with_metadata(self, catalog_sys_id: str) -> ServiceNowResponse:
        """
        Fetches catalog item variables with their metadata including choice values.
        
        Args:
            catalog_sys_id: The sys_id of the catalog item
            
        Returns:
            ServiceNowResponse containing the variables with metadata
        """
        if not catalog_sys_id or not isinstance(catalog_sys_id, str):
            return ServiceNowResponse.error(
                code=StatusCode.BAD_REQUEST.value,
                message="Invalid catalog_sys_id provided"
            )

        # Base URL for variable definitions
        variables_url = f"{self.client.base_url}item_option_new"

        # Get both catalog item details and variables in a single query
        query_params = {
            "sysparm_query": f"cat_item={catalog_sys_id}^active=true^ORDERBYorder",
            "sysparm_fields": (
                "sys_id,name,type,mandatory,help_text,label,order,question_text,"
                "default_value,max_length,min_length,regex_pattern,reference_table,"
                "reference_key,lookup_table,lookup_key,max_value,min_value,"
                "cat_item.name,cat_item.description,cat_item.short_description,"
                "cat_item.picture,cat_item.price,cat_item.recurring_price,"
                "cat_item.recurring_frequency,cat_item.delivery_plan,"
                "cat_item.workflow,cat_item.availability,cat_item.category,"
                "cat_item.show_price,cat_item.show_quantity,cat_item.max_quantity,"
                "cat_item.min_quantity,cat_item.order_guide,cat_item.active"
            ),
            "sysparm_display_value": "true",
        }

        # Construct the URL with query parameters
        query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
        variables_url = f"{variables_url}?{query_string}"
        
        response = await self.client.make_request_async("GET", variables_url)

        if response.code != StatusCode.SUCCESS.value:
            return response

        variables = []

        # Define field types that can have choices
        choice_field_types = {
            "Select Box": {"multi_select": False},
            "Multiple Choice": {"multi_select": False},
            "Checkbox": {"multi_select": False},
            "Radio Button": {"multi_select": False},
            "List Collector": {"multi_select": True},
        }

        # Supported field types with clear descriptive names
        supported_field_types = [
            "Date", "Datetime", "Single Line Text", "Multi Line Text",
            "Integer", "Float", "Select Box", "Multiple Choice",
            "Checkbox", "Radio Button", "List Collector", "Reference",
            "URL", "Email", "Wide Single Line Text", "Lookup Select Box",
        ]

        # Process each variable
        for variable in response.data:
            variable_metadata = {
                "catalog_item": {
                    "title": variable.get("cat_item.name"),
                    "heading": variable.get("cat_item.short_description"),
                    "description": variable.get("cat_item.description"),
                    "picture": variable.get("cat_item.picture"),
                    "price": variable.get("cat_item.price"),
                    "recurring_price": variable.get("cat_item.recurring_price"),
                    "recurring_frequency": variable.get("cat_item.recurring_frequency"),
                    "availability": variable.get("cat_item.availability"),
                    "category": variable.get("cat_item.category"),
                    "workflow": variable.get("cat_item.workflow"),
                },
                "field": {
                    "name": variable.get("name"),
                    "sys_id": variable.get("sys_id"),
                    "type": variable.get("type"),
                    "mandatory": variable.get("mandatory") == "true",
                    "help_text": variable.get("help_text", ""),
                    "label": variable.get("label") or variable.get("question_text"),
                    "order": variable.get("order"),
                    "default_value": variable.get("default_value", ""),
                },
                "validation": {
                    "max_length": variable.get("max_length"),
                    "min_length": variable.get("min_length"),
                    "regex_pattern": variable.get("regex_pattern"),
                    "max_value": variable.get("max_value"),
                    "min_value": variable.get("min_value"),
                },
            }

            # Handle choice fields
            if variable.get("type") in choice_field_types:
                choices_url = f"{self.client.base_url}question_choice"
                choices_query = {
                    "sysparm_query": f"question={variable.get('sys_id')}^active=true^ORDERBYorder",
                    "sysparm_fields": "value,text,order,image,selected_by_default,inactive",
                    "sysparm_display_value": "true",
                }

                choices_query_string = "&".join([f"{k}={v}" for k, v in choices_query.items()])
                choices_url = f"{choices_url}?{choices_query_string}"
                
                choices_response = await self.client.make_request_async("GET", choices_url)

                if choices_response.code == StatusCode.SUCCESS.value:
                    choices = []
                    for choice in choices_response.data:
                        if choice.get("inactive") != "true":
                            choices.append({
                                "value": choice.get("value"),
                                "text": choice.get("text"),
                                "order": choice.get("order"),
                                "image": choice.get("image", ""),
                                "selected_by_default": choice.get("selected_by_default") == "true",
                            })
                    
                    choices.sort(key=lambda x: int(x.get("order", 0) or 0))
                    variable_metadata["choices"] = choices
                    variable_metadata["is_multi_select"] = choice_field_types[variable.get("type")]["multi_select"]

            # Handle reference fields
            if variable.get("type") in ["Reference", "Lookup Select Box"]:
                reference_table = variable.get("reference_table") or variable.get("lookup_table")
                reference_key = variable.get("reference_key") or variable.get("lookup_key")

                reference_url = f"{self.client.base_url}{reference_table}"
                reference_query = {
                    "sysparm_fields": f"sys_id,{reference_key or 'name'}",
                    "sysparm_limit": 100,
                    "sysparm_display_value": "true",
                }

                reference_query_string = "&".join([f"{k}={v}" for k, v in reference_query.items()])
                reference_url = f"{reference_url}?{reference_query_string}"
                
                reference_response = await self.client.make_request_async("GET", reference_url)

                reference_values = []
                if reference_response.code == StatusCode.SUCCESS.value:
                    for ref in reference_response.data:
                        reference_values.append({
                            "value": ref.get("sys_id"),
                            "text": ref.get(reference_key or "name"),
                        })

                variable_metadata["reference"] = {
                    "table": reference_table,
                    "key": reference_key,
                    "values": reference_values,
                }

            # Handle unsupported fields
            if variable.get("type") not in supported_field_types:
                variable_metadata["unsupported"] = True
                variable_metadata["reason"] = "Field type is not supported"

            variables.append(variable_metadata)

        # Sort variables by order
        variables.sort(key=lambda x: int(x.get("field", {}).get("order", 0) or 0))

        return ServiceNowResponse.success(
            message="Successfully retrieved catalog item variables",
            data={
                "variables": variables,
                "catalog_sys_id": catalog_sys_id
            }
        )

    @tracer.start_as_current_span("service_now.submit_catalog_form")
    async def submit_catalog_form(
        self, 
        catalog_sys_id: str, 
        variables: Dict[str, Any], 
        user_sys_id: str
    ) -> ServiceNowResponse:
        """
        Submit a service catalog form with the provided variables.
        
        Args:
            catalog_sys_id: The sys_id of the catalog item
            variables: Form field values to submit
            user_sys_id: The sys_id of the user submitting the form
            
        Returns:
            ServiceNowResponse containing the result
        """
        try:
            base_url = f"https://{self.client.instance_name}.service-now.com/api/sn_sc/servicecatalog"

            # Add the item to the cart
            add_item_url = f"{base_url}/items/{catalog_sys_id}/add_to_cart"
            cart_item_data = {
                "sysparm_quantity": "1",
                "variables": variables,
                "requested_for": user_sys_id if user_sys_id else None,
            }

            add_item_response = await self.client.make_request_async("POST", add_item_url, cart_item_data)
            if add_item_response.code not in [StatusCode.SUCCESS.value, StatusCode.CREATED.value]:
                logger.error(f"Failed to add item to cart: {add_item_response}")
                return ServiceNowResponse.error(
                    code=add_item_response.code,
                    message="Failed to add item to cart",
                    errors=[add_item_response.message],
                    details=add_item_response.details
                )

            # Submit the cart
            submit_url = f"{base_url}/cart/submit_order"
            submit_data = {}
            if user_sys_id:
                submit_data["sysparm_requested_for"] = user_sys_id

            submit_response = await self.client.make_request_async("POST", submit_url, submit_data)
            if submit_response.code != StatusCode.SUCCESS.value:
                logger.error(f"Failed to submit cart: {submit_response}")
                return ServiceNowResponse.error(
                    code=submit_response.code,
                    message="Failed to submit cart",
                    errors=[submit_response.message],
                    details=submit_response.details
                )

            logger.debug(f"Submit response: {submit_response}")

            request_data = submit_response.data or {}
            request_number = request_data.get("request_number")
            request_id = request_data.get("request_id")
            
            if not request_number or not request_id:
                logger.error("Missing request details in successful submission")
                return ServiceNowResponse.error(
                    code=StatusCode.INTERNAL_SERVER_ERROR.value,
                    message="Submission succeeded but missing request details",
                    errors=["Missing request_number or request_id in the response"]
                )

            req_link = (
                f"https://{self.client.instance_name}.service-now.com/now/nav/ui/classic/"
                f"params/target/sc_request.do%3Fsys_id%3D{request_id}"
            )

            return ServiceNowResponse.success(
                message="Catalog form submitted successfully",
                data={
                    "request_number": request_number,
                    "sys_id": request_id,
                    "req_link": req_link,
                }
            )

        except Exception as e:
            logger.error(f"Error submitting catalog form: {str(e)}", exc_info=True)
            return ServiceNowResponse.error(
                code=StatusCode.INTERNAL_SERVER_ERROR.value,
                message=f"Error submitting catalog form: {str(e)}",
                errors=[str(e)]
            )

    # Change Request Methods
    
    @tracer.start_as_current_span("service_now.create_change_request")
    async def create_change_request(
        self,
        short_description: str,
        description: str,
        category: str,
        type: str,
        risk: str,
        impact: str,
        priority: str,
        assignment_group: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        requested_by: Optional[str] = None,
    ) -> ServiceNowResponse:
        """
        Creates a new change request in ServiceNow.
        
        Args:
            short_description: Brief description of the change
            description: Detailed description
            category: Category of the change
            type: Type of change (normal, standard, emergency)
            risk: Risk level (low, medium, high)
            impact: Impact level
            priority: Priority level
            assignment_group: Group assigned to the change
            start_date: Planned start date (YYYY-MM-DD HH:MM:SS)
            end_date: Planned end date (YYYY-MM-DD HH:MM:SS)
            requested_by: User who requested the change
            
        Returns:
            ServiceNowResponse with the created change request
        """
        url = f"{self.client.base_url}change_request"

        data = {
            "short_description": short_description,
            "description": description,
            "type": type,
            "category": category,
            "risk": risk,
            "impact": impact,
            "priority": priority,
        }

        if requested_by:
            data["requested_by"] = requested_by
        if assignment_group:
            data["assignment_group"] = assignment_group
        if start_date:
            data["start_date"] = start_date
        if end_date:
            data["end_date"] = end_date

        response = await self.client.make_request_async("POST", url, data)
        
        # Add link to the change request in the response
        if response.code == StatusCode.SUCCESS.value and response.data:
            change_sys_id = response.data.get("sys_id")
            change_number = response.data.get("number")
            if change_sys_id:
                change_link = f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=change_request.do?sys_id={change_sys_id}"
                response.data["link"] = change_link
                logger.info(f"Created change request {change_number} with sys_id: {change_sys_id}")
                
        return response

    @tracer.start_as_current_span("service_now.get_change_request_by_number")
    async def get_change_request_by_number(self, change_request_number: str) -> ServiceNowResponse:
        """
        Retrieves a change request by its number.
        
        Args:
            change_request_number: The number of the change request
            
        Returns:
            ServiceNowResponse containing the change request
        """
        url = f"{self.client.base_url}change_request?number={change_request_number}"
        response = await self.client.make_request_async("GET", url)
        
        # Add link to the change request in the response
        if response.code == StatusCode.SUCCESS.value and response.data:
            for change in response.data:
                change_sys_id = change.get("sys_id")
                if change_sys_id:
                    change["link"] = f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=change_request.do?sys_id={change_sys_id}"
                    
        return response

    @tracer.start_as_current_span("service_now.update_change_request")
    async def update_change_request(
        self,
        change_request_number: str,
        **kwargs
    ) -> ServiceNowResponse:
        """
        Updates an existing change request.
        
        Args:
            change_request_number: The number of the change request
            **kwargs: Fields to update
            
        Returns:
            ServiceNowResponse with the update result
        """
        # Get the change request details using the number
        get_response = await self.get_change_request_by_number(change_request_number)

        logger.debug(f"Response from fetching change request: {get_response}")

        if get_response.code != StatusCode.SUCCESS.value or not get_response.data:
            logger.error(f"Failed to fetch change request {change_request_number}")
            return get_response

        sys_id = get_response.data[0]["sys_id"]

        # Now update the change request using the sys_id
        update_url = f"{self.client.base_url}change_request/{sys_id}"
        
        # Filter out None values and reserved parameter names
        data = {
            k: v for k, v in kwargs.items() 
            if v is not None and k not in ["self", "change_request_number", "sys_id"]
        }

        if not data:
            return ServiceNowResponse.error(
                code=StatusCode.BAD_REQUEST.value,
                message="No update data provided"
            )
            
        logger.debug(f"Data to update: {data}")
        return await self.client.make_request_async("PATCH", update_url, data)

    # User Management Methods for System Integration
    
    @tracer.start_as_current_span("service_now.create_user")
    async def create_user(self, phone_number: str) -> ServiceNowResponse:
        """
        Create a new user with the given phone number.
        
        Args:
            phone_number: The phone number for the new user
            
        Returns:
            ServiceNowResponse with the created user
        """
        try:
            # User data to create
            user_data = {
                "first_name": "Anonymous",
                "last_name": "User",
                "mobile_phone": phone_number,
            }

            url = f"{self.client.base_url}sys_user"
            response = await self.client.make_request_async("POST", url, user_data)
            
            if response.code == StatusCode.SUCCESS.value:
                return ServiceNowResponse.success(
                    message="User created successfully",
                    data=response.data
                )
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return ServiceNowResponse.error(
                message="Error creating user",
                details=str(e)
            )

    @tracer.start_as_current_span("service_now.get_user")
    async def get_user(self, phone_number: str) -> ServiceNowResponse:
        """
        Get a user by their phone number.
        
        Args:
            phone_number: The phone number to look up
            
        Returns:
            ServiceNowResponse with the user details
        """
        try:
            # Encode the phone number
            encoded_phone_number = requests.utils.quote(phone_number)

            url = f"{self.client.base_url}sys_user?sysparm_query=mobile_phone={encoded_phone_number}"
            response = await self.client.make_request_async("GET", url)
            
            if response.code == StatusCode.SUCCESS.value:
                return ServiceNowResponse.success(
                    message="User retrieved successfully",
                    data=response.data
                )
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error fetching user: {e}")
            return ServiceNowResponse.error(
                message="Error fetching user",
                details=str(e)
            )

    @tracer.start_as_current_span("service_now.fetch_or_create_user_by_phone_number")
    async def fetch_or_create_user_by_phone_number(self, phone_number: str) -> ServiceNowResponse:
        """
        Get a user by phone number or create if not found.
        
        Args:
            phone_number: The phone number to look up or use for creation
            
        Returns:
            ServiceNowResponse with the user details
        """
        try:
            # Try to fetch the user by phone number
            user_response = await self.get_user(phone_number)

            if user_response.code == StatusCode.SUCCESS.value and user_response.data:
                users = user_response.data
                if users:  # Check if users list is not empty
                    user = users[0]
                    user_name = f"{user['first_name']} {user['last_name']}"
                    user_sys_id = user["sys_id"]
                    logger.debug(f"User found: {user_name} (ID: {user_sys_id})")
                    return ServiceNowResponse.success(
                        data={"full_name": user_name, "sys_id": user_sys_id}
                    )
                else:
                    logger.debug("No user found, creating new user.")
                    # Create a new user if not found
                    create_response = await self.create_user(phone_number)
                    if create_response.code == StatusCode.SUCCESS.value:
                        user = create_response.data
                        user_name = f"{user['first_name']} {user['last_name']}"
                        user_sys_id = user["sys_id"]
                        logger.debug(f"User created: {user_name} (ID: {user_sys_id})")
                        return ServiceNowResponse.success(
                            data={"full_name": user_name, "sys_id": user_sys_id}
                        )
                    else:
                        return create_response
            else:
                return user_response
                
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return ServiceNowResponse.error(
                message="Error processing user",
                details=str(e)
            )

    @tracer.start_as_current_span("service_now.get_locations")
    async def get_locations(self) -> List[Dict[str, str]]:
        """
        Get all locations from ServiceNow.
        
        Returns:
            List of locations with sys_id and name
        """
        url = f"{self.client.base_url}cmn_location?sysparm_fields=sys_id,name"
        response = await self.client.make_request_async("GET", url)

        if response.code == StatusCode.SUCCESS.value:
            locations = []
            for location in response.data:
                locations.append({
                    "sys_id": location.get("sys_id"), 
                    "name": location.get("name")
                })
            return locations
        return []

    @tracer.start_as_current_span("service_now.get_user_email")
    async def get_user_email(self, emp_sys_id: str) -> str:
        """
        Fetch user's email address from their employee record.
        
        Args:
            emp_sys_id: The system ID of the employee
            
        Returns:
            The email address of the employee
        """
        endpoint = f"{self.client.base_url}sys_user/{emp_sys_id}?sysparm_fields=email"
        response = await self.client.make_request_async("GET", endpoint)
        
        if response.code == StatusCode.SUCCESS.value:
            return response.data.get("email", "")
        else:
            return ""

    @tracer.start_as_current_span("service_now.get_all_categories")
    async def get_all_categories(self) -> List[Dict[str, str]]:
        """
        Retrieve all categories from ServiceNow.
        
        Returns:
            List of categories with value and label
        """
        endpoint = f"{self.client.base_url}sys_choice?sysparm_query=name=incident^element=category&sysparm_fields=value,label"
        response = await self.client.make_request_async("GET", endpoint)

        if response.code == StatusCode.SUCCESS.value:
            categories = []
            for category in response.data:
                categories.append({
                    "value": category.get("value"), 
                    "name": category.get("label")
                })
            return categories
        else:
            return []

    @tracer.start_as_current_span("service_now.get_all_open_tickets")
    async def get_all_open_tickets(self, user_sys_id: str, limit: int = 3) -> ServiceNowResponse:
        """
        Fetches all open items from ServiceNow including:
        - Incidents (INC)
        - Request Items (RITM)

        Open items are those with state not equal to Closed or Resolved.

        Args:
            user_sys_id: The sys_id of the user in ServiceNow to filter items by
            limit: Maximum number of items to retrieve per type. Defaults to 3.

        Returns:
            ServiceNowResponse containing lists of all open incidents and requests
        """
        # Incident state mapping according to ServiceNow default values
        incident_state_mapping = {
            "1": "New",
            "2": "In Progress",
            "3": "On Hold",
            "6": "Resolved",
            "7": "Closed",
            "8": "Canceled",
        }

        # RITM (Request Item) state mapping according to ServiceNow default values
        ritm_state_mapping = {
            "-5": "Pending",
            "-4": "Draft",
            "-3": "Review",
            "-2": "Requested",
            "-1": "Waiting for Approval",
            "0": "Not Requested",
            "1": "Open",
            "2": "Work in Progress",
            "3": "Closed Complete",
            "4": "Closed Incomplete",
            "5": "Closed Cancelled",
            "6": "Closed Rejected",
            "7": "Closed Skipped",
            "8": "On Hold",
        }

        # Fields to retrieve
        fields = "number,short_description,description,sys_created_on,state,priority,sys_id,impact,urgency"

        # Build base URLs
        incident_url = (
            f"{self.client.base_url}incident?"
            f"sysparm_query=caller_id={user_sys_id}^stateNOT IN6,7^ORDERBYDESCsys_created_on&"
            f"sysparm_limit={limit}&sysparm_fields={fields}"
        )
        ritm_url = (
            f"{self.client.base_url}sc_req_item?"
            f"sysparm_query=caller_id={user_sys_id}^stateNOT IN3,4,7^ORDERBYDESCsys_created_on&"
            f"sysparm_limit={limit}&sysparm_fields={fields}"
        )

        # Make requests for incidents and RITMs
        incident_response = await self.client.make_request_async("GET", incident_url)
        request_response = await self.client.make_request_async("GET", ritm_url)

        # Process responses and add links
        result = {"incidents": [], "requests": []}

        # Process incidents
        if incident_response.code == StatusCode.SUCCESS.value and incident_response.data:
            incidents = []
            for item in incident_response.data:
                processed_item = {
                    "number": item["number"],
                    "link": f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=incident.do?sys_id={item['sys_id']}",
                    "description": item["description"],
                    "short_description": item["short_description"],
                    "sys_created_on": item["sys_created_on"],
                    "state": incident_state_mapping.get(str(item["state"]), f"State-{item['state']}"),
                }
                incidents.append(processed_item)
            result["incidents"] = incidents

        # Process requests
        if request_response.code == StatusCode.SUCCESS.value and request_response.data:
            requests = []
            for item in request_response.data:
                processed_item = {
                    "number": item["number"],
                    "link": f"https://{self.client.instance_name}.service-now.com/nav_to.do?uri=sc_req_item.do?sys_id={item['sys_id']}",
                    "description": item["description"],
                    "short_description": item["short_description"],
                    "sys_created_on": item["sys_created_on"],
                    "state": ritm_state_mapping.get(str(item["state"]), f"State-{item['state']}"),
                }
                requests.append(processed_item)
            result["requests"] = requests

        return ServiceNowResponse.success(data=result)


# Factory function to create ServiceNow client with the right configuration
def create_service_now_client(
    username: Optional[str] = None,
    password: Optional[str] = None,
    instance_name: Optional[str] = None
) -> ServiceNow:
    """
    Create a ServiceNow client with the provided credentials or from environment variables.
    
    Args:
        username: ServiceNow username (optional)
        password: ServiceNow password (optional)
        instance_name: ServiceNow instance name (optional)
        
    Returns:
        Configured ServiceNow client
    """
    if all([username, password, instance_name]):
        auth = ServiceNowAuth(
            username=username,
            password=password,
            instance_name=instance_name
        )
    else:
        auth = ServiceNowAuth.from_env()
        
    return ServiceNow(auth)


# Example usage:
# async def main():
#     """Example of how to use the ServiceNow client."""
#     try:
#         # Create client with environment variables
#         sn_client = create_service_now_client()
        
#         # Get the current user
#         current_user_id = await sn_client.get_me_user_id()
#         logger.info(f"Current user ID: {current_user_id}")
        
#         # Get open tickets for the current user
#         if current_user_id:
#             tickets = await sn_client.get_all_open_tickets(current_user_id, limit=5)
#             logger.info(f"Found {len(tickets.data.get('incidents', []))} open incidents")
#             logger.info(f"Found {len(tickets.data.get('requests', []))} open requests")
            
#     except Exception as e:
#         logger.error(f"Error in main: {e}", exc_info=True)
#     finally:
#         # Always close the client when done
#         await sn_client.close()


# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())