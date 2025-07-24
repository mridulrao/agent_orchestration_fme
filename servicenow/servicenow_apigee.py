# -*- coding: utf-8 -*-
"""
ServiceNow Client Module
========================

Provides an asynchronous client for interacting with ServiceNow via a SPARC API gateway,
following specified structural patterns including Enums, helper classes, and separation of concerns.

Includes all functionalities from the original provided script, adapted to the new structure,
preserving original public method names and parameters where feasible, and incorporates error fixes.
"""

import base64
import io
import json
import logging
import mimetypes
import os
import collections
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Type, Callable

import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry
from opentelemetry import trace
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# --- Basic Setup ---
# Configure logging
urllib3.disable_warnings(InsecureRequestWarning)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
use_colors = os.getenv("USE_COLORS", "true").lower() == "true"

# Setup logging with the same format as in logging_config.py
log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger(__name__)

if not log.handlers:
    console_handler = logging.StreamHandler()

    class LogColors:
        GREY = "\x1b[38;21m"
        BLUE = "\x1b[38;5;39m"
        YELLOW = "\x1b[38;5;226m"
        RED = "\x1b[38;5;196m"
        BOLD_RED = "\x1b[31;1m"
        RESET = "\x1b[0m"

    class ColoredFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: LogColors.GREY,
            logging.INFO: LogColors.BLUE,
            logging.WARNING: LogColors.YELLOW,
            logging.ERROR: LogColors.RED,
            logging.CRITICAL: LogColors.BOLD_RED
        }

        def __init__(self, fmt, datefmt="%Y-%m-%d %H:%M:%S", use_colors=True):
            super().__init__(fmt, datefmt)
            self.use_colors = use_colors

        def format(self, record):
            import copy
            colored_record = copy.copy(record)
            color = None
            if self.use_colors:
                color = self.COLORS.get(colored_record.levelno)
            if color:
                colored_record.levelname = f"{color}{colored_record.levelname}{LogColors.RESET}"

            message = colored_record.getMessage()
            colored_record.message = message
            original_getMessage = colored_record.getMessage
            colored_record.getMessage = lambda: colored_record.message
            formatted_message = super().format(colored_record)
            colored_record.getMessage = original_getMessage
            return formatted_message
    formatter = ColoredFormatter(fmt=log_format, use_colors=use_colors)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    log.setLevel(getattr(logging, log_level, logging.INFO))
    log.propagate = False
tracer = trace.get_tracer(__name__)
if not log.handlers:
    _default_handler = logging.StreamHandler()
    _default_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    _default_handler.setFormatter(_default_formatter)
    log.addHandler(_default_handler)
    log.setLevel(logging.INFO)
    log.propagate = False


## Constants and Enums #########################################################
class HTTPMethod(Enum): GET = "GET"; POST = "POST"; PUT = "PUT"; DELETE = "DELETE"; PATCH = "PATCH"
class ServiceNowTable(Enum): INCIDENT = "incident"; REQ_ITEM = "sc_req_item"; CHANGE_REQUEST = "change_request"; SYS_USER = "sys_user"; SYS_APPROVAL = "sysapproval_approver"; CMN_LOCATION = "cmn_location"; SYS_ATTACHMENT = "sys_attachment"; SC_REQUEST = "sc_request"; SC_CAT_ITEM = "sc_cat_item"; ITEM_OPTION = "item_option_new"; QUESTION_CHOICE = "question_choice"; SC_ITEM_OPTION = "sc_item_option"; TASK = "task"; SYS_USER_GROUP = "sys_user_group"; SYS_USER_GRMEMBER = "sys_user_grmember"
class ApprovalState(Enum): REQUESTED = "requested"; APPROVED = "approved"; REJECTED = "rejected"; CANCELLED = "cancelled"; NOT_REQUIRED = "not_required"
class IncidentState(Enum): NEW = "New"; RESOLVED = "Resolved"; OPEN = "Open"; CLOSED = "Closed"; ASSIGNED = "Assigned"; WORK_IN_PROGRESS = "Work in Progress"; ON_HOLD = "On Hold"; CANCELED = "Canceled"
class RITMState(Enum): OPEN = "Open"; CLOSED_COMPLETE = "Closed Complete"; WORK_IN_PROGRESS = "Work in Progress"; CLOSED_SKIPPED = "Closed Skipped"; CLOSED_INCOMPLETE = "Closed Incomplete"; PENDING = "Pending"; REQUEST_CANCELLED = "Request Cancelled"

## Data Classes & Exceptions ##################################################
@dataclass
class Config:
    client_id: str
    client_secret: str
    env: str="sparcdev2"
    base_url: str="https://api-test.gaim.gilead.com/v1"
    oauth_token_path: str="/oauth/generatetoken"
    table_api_path: str="/SPARCDetails/table"
    incident_create_path: str="/SPARCDetails/createIncident"
    attachment_api_path: str="/SPARCDetails/attachment" # Keep for reference, maybe POST uses it?
    file_attachment_api_path: str="/SPARCDetails/fileAttachment" # For multipart POST
    attachment_metadata_path: str="/SPARCDetails/GetAttachmentMetaData" # <<< ADD THIS LINE
    ssl_verify: Union[bool, str]=False
    default_assignment_group: str="75f2a04e0f6a0600ae6e22d8b1050e70"
    default_contact_type: str="G.Bot"
    retry_attempts: int=3
    retry_statuses: set=field(default_factory=lambda:{500,502,503,504})
    retry_exceptions: set=field(default_factory=lambda:{aiohttp.ClientConnectionError,asyncio.TimeoutError})
    incident_state_mapping_str_key: Dict[str, str]=field(default_factory=lambda:{"New":"100","Resolved":"900","Open":"200","Closed":"1000","Assigned":"300","Work in Progress":"400","On Hold":"500","Canceled":"1030"})
    ritm_state_mapping_str_key: Dict[str, str]=field(default_factory=lambda:{"Open":"100","Closed Complete":"1000","Work in Progress":"400","Closed Skipped":"1020","Closed Incomplete":"1010","Pending":"200","Request Cancelled":"1040"})
    @classmethod
    def from_env(cls): # Definition unchanged
        client_id="J7utPRd2CueMA0DJTRvsqlxpx2HT195U" # Example value, should use os.getenv in production
        client_secret="e4yPW0idUApK0MkE" # Example value, should use os.getenv in production
        env=os.getenv("SPARC_ENV","sparcdev2")
        base_url=os.getenv("SPARC_BASE_URL","https://api-test.gaim.gilead.com/v1")
        ssl_verify_env=os.getenv("SPARC_SSL_VERIFY","false").lower(); ssl_verify:Union[bool,str]
        if ssl_verify_env=="false": ssl_verify=False
        elif ssl_verify_env=="true": ssl_verify=True
        else: ssl_verify=ssl_verify_env
        return cls(client_id=client_id,client_secret=client_secret,env=env,base_url=base_url,ssl_verify=ssl_verify,default_assignment_group=os.getenv("SPARC_DEFAULT_ASSIGNMENT_GROUP","75f2a04e0f6a0600ae6e22d8b1050e70"),default_contact_type=os.getenv("SPARC_DEFAULT_CONTACT_TYPE","G.Bot"),retry_attempts=int(os.getenv("SPARC_RETRY_ATTEMPTS",3)))

# (Exceptions: ServiceNowError, etc. - definitions unchanged)
class ServiceNowError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, details: Any = None): super().__init__(message); self.status_code = status_code; self.details = details
    def __str__(self): details_str = str(self.details)[:200] + '...' if self.details else 'None'; return f"{self.__class__.__name__}(status={self.status_code}, msg='{self.args[0]}', details={details_str})"
class ServiceNowAuthError(ServiceNowError): pass
class ServiceNowNotFoundError(ServiceNowError): pass
class ServiceNowPermissionError(ServiceNowError): pass
class ServiceNowBadRequestError(ServiceNowError): pass
class ServiceNowInvalidResponseError(ServiceNowError): pass
class ServiceNowRateLimitError(ServiceNowError): pass


## Helper Classes and Interfaces ##############################################
# (Interfaces: IRequestHandler, IAttachmentHandler - definitions unchanged)
class IRequestHandler(ABC):
    @abstractmethod
    async def make_request(self, method: HTTPMethod, endpoint: str, params: Optional[Dict]=None, data: Optional[Any]=None, headers: Optional[Dict]=None) -> Any: pass
class IAttachmentHandler(ABC):
    @abstractmethod
    async def add_attachment(self, table: ServiceNowTable, sys_id: str, file_content: Any, file_name: str, content_type: Optional[str]=None) -> Dict[str, Any]: pass
    @abstractmethod
    async def get_attachments(self, table: ServiceNowTable, sys_id: str, limit: Optional[int]=None, offset: int=0) -> List[Dict[str, Any]]: pass

# (Helper Implementations: StateMapper, ResponseHandler, RequestHandlerImpl, TableRequestHandler - unchanged)
class StateMapper: # Definition unchanged
    def __init__(self, config: Config): self.config = config
    def get_incident_state_code(self, state_name: str) -> str:
        code = self.config.incident_state_mapping_str_key.get(state_name);
        if code is None: raise ValueError(f"Invalid Incident state name: '{state_name}'. Known: {list(self.config.incident_state_mapping_str_key.keys())}")
        return code
    def get_ritm_state_code(self, state_name: str) -> str:
        code = self.config.ritm_state_mapping_str_key.get(state_name);
        if code is None: raise ValueError(f"Invalid RITM state name: '{state_name}'. Known: {list(self.config.ritm_state_mapping_str_key.keys())}")
        return code
class ResponseHandler: # Definition unchanged
    @staticmethod
    def success(data: Optional[Any]=None, message: str="Success", code: int=200) -> Dict: return {"code": code, "message": message, "data": data}
    @staticmethod
    def error(code: int, message: str, details: Optional[Any]=None) -> Dict: return {"code": code, "message": message, "details": details}
    @staticmethod
    def not_found(message: str="Not Found") -> Dict: return ResponseHandler.error(404, message)
    @staticmethod
    def unauthorized(message: str="Unauthorized") -> Dict: return ResponseHandler.error(401, message)
    @staticmethod
    def bad_request(message: str="Bad Request", details: Optional[Any]=None) -> Dict: return ResponseHandler.error(400, message, details)
    @staticmethod
    def internal_error(message: str="Internal Error", details: Optional[Any]=None) -> Dict: return ResponseHandler.error(500, message, details)


class RequestHandlerImpl(IRequestHandler):
    def __init__(self, client: 'ServiceNow'):
        self._client = client

    async def make_request(self, method: HTTPMethod, endpoint: str, params: Optional[Dict]=None, data: Optional[Any]=None, headers: Optional[Dict]=None, is_retry: bool=False) -> Tuple[int, Optional[Dict[str, str]], Any]:
        """
        Makes the HTTP request and returns status, headers (as dict), and parsed body.
        Headers are returned only on success (2xx).
        """
        await self._client._ensure_token()
        request_url = f"{self._client.config.base_url}{endpoint}"
        req_params = params.copy() if params else {}
        req_params.setdefault('client_id', self._client.config.client_id)
        req_params.setdefault('envName', self._client.config.env)

        final_headers = self._client._get_headers(include_auth=True)
        if headers:
            final_headers.update(headers) # Merge custom headers

        json_payload: Optional[Dict] = None
        data_payload: Any = data
        # (Keep existing logic for determining json_payload vs data_payload)
        if isinstance(data,dict) and final_headers.get('Content-Type','').startswith('application/json'): json_payload=data; data_payload=None
        elif not isinstance(data,(str,bytes,io.IOBase,aiohttp.FormData)) and data is not None: final_headers.setdefault('Content-Type','application/json'); json_payload=data; data_payload=None
        if data_payload is not None and not json_payload and 'Content-Type' in final_headers: pass
        elif data_payload is not None and not json_payload: final_headers.setdefault('Content-Type','application/octet-stream')


        log.debug(f"ReqHandler: {method.value} {request_url} Params: {req_params} Headers: {list(final_headers.keys())}")
        session = await self._client._get_session()

        try:
            async with session.request(method.value, request_url, params=req_params, headers=final_headers, json=json_payload, data=data_payload, raise_for_status=False) as response:
                status = response.status
                response_headers = dict(response.headers) # Capture headers as dict
                text = await response.text(errors='ignore')
                log.debug(f"ReqHandler Resp: {status}")

                # --- Process based on status code ---
                if 200 <= status < 300:
                    if status == 204 or not text:
                        return status, response_headers, None # Success, return headers, no body
                    try:
                        result_data = json.loads(text)
                        # Check for SPARC API specific structure first
                        if isinstance(result_data, dict) and 'result' in result_data:
                           return status, response_headers, result_data['result']
                        if isinstance(result_data, dict) and result_data.get('status') == 'failure':
                            err_msg = result_data.get("error", {}).get("message", "Unknown SPARC failure")
                            log.error(f"API Error (Status {status}): {err_msg}")
                            raise ServiceNowInvalidResponseError(f"API Error: {err_msg}", status, result_data)
                        # Return raw data if not the SPARC 'result' structure
                        return status, response_headers, result_data # Success, return headers and body
                    except json.JSONDecodeError:
                        raise ServiceNowInvalidResponseError("Invalid JSON", status, text)
                # --- Handle Errors (Raise exceptions as before) ---
                elif status == 401:
                    if not is_retry:
                        log.warning("401 Unauthorized. Refreshing token...")
                        await self._client._ensure_token(force_refresh=True)
                        # Important: Retry the request by calling make_request again
                        # The result of the retry (status, headers, body) should be returned
                        return await self.make_request(method, endpoint, params, data, headers, is_retry=True)
                    else:
                        raise ServiceNowAuthError("Auth failed after retry.", status, text)
                elif status == 400: raise ServiceNowBadRequestError(f"Bad Request", status, text)
                elif status == 403: raise ServiceNowPermissionError(f"Permission Denied", status, text)
                elif status == 404: raise ServiceNowNotFoundError(f"Not Found: {endpoint}", status, text)
                elif status == 422: raise ServiceNowBadRequestError(f"Validation Error", status, text)
                elif status == 429: raise ServiceNowRateLimitError(f"Rate Limit", status, text)
                elif status >= 500: raise ServiceNowError(f"Server Error", status, text)
                else: raise ServiceNowError(f"Unexpected Status", status, text)

        except aiohttp.ClientError as e: raise ServiceNowError(f"Client error: {e}") from e
        except asyncio.TimeoutError as e: raise ServiceNowError(f"Timeout: {e}") from e
# In servicenow_client.py

class TableRequestHandler:
    def __init__(self, client: 'ServiceNow'):
        self.client = client
        self.request_handler: IRequestHandler = RequestHandlerImpl(client)

    async def table_request(
        self,
        table: ServiceNowTable,
        method: HTTPMethod,
        query: Optional[str] = None,
        data: Optional[Dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        fields: Optional[List[str]] = None,
        display_value: bool = False,
        view: Optional[str] = None
    ) -> Dict[str, Any]: # Changed return type hint
        """
        Makes a request to the Table API endpoint.
        Returns a dictionary containing 'records' (list) and 'total_count' (int or None).
        """
        params = {'table_name': table.value}
        endpoint = self.client.config.table_api_path
        log.debug(f"Table Request Params Base: {params}") # Use debug

        if method == HTTPMethod.GET and query: params['sysparm_query'] = query
        elif method in (HTTPMethod.PUT, HTTPMethod.DELETE) and query: params['table_pathID'] = query

        if method == HTTPMethod.GET:
            # Standard ServiceNow pagination params
            if limit is not None: params['sysparm_limit'] = limit
            if offset is not None: params['sysparm_offset'] = offset
            if fields: params['sysparm_fields'] = ','.join(fields)
            if display_value: params['sysparm_display_value'] = 'true'
            if view: params['sysparm_view'] = view
            # Tell ServiceNow we want the total count header
            params['sysparm_no_count'] = 'false' # Explicitly ask for count header

        # Call make_request which now returns (status, headers, body)
        # Exceptions are raised directly by make_request on error
        status, headers, body = await self.request_handler.make_request(
            method, endpoint, params, data
        )

        # Process successful response (status is guaranteed 2xx here)
        records = body if isinstance(body, list) else ([] if body is None else [body])
        total_count: Optional[int] = None

        if headers:
            # Check for the total count header (case-insensitive)
            count_header = headers.get('X-Total-Count') or headers.get('x-total-count')
            if count_header is not None:
                try:
                    total_count = int(count_header)
                    log.debug(f"Found X-Total-Count header: {total_count}")
                except (ValueError, TypeError):
                    log.warning(f"Could not parse X-Total-Count header value: '{count_header}'")

        return {"records": records, "total_count": total_count}
    
# --- MODIFIED AttachmentHandlerImpl ---
# --- AttachmentHandlerImpl ---
class AttachmentHandlerImpl(IAttachmentHandler):
    """
    Handles adding and retrieving attachments via the SPARC API gateway.
    - add_attachment uses the dedicated file upload endpoint (/SPARCDetails/fileAttachment).
    - get_attachments uses the dedicated /SPARCDetails/GetAttachmentMetaData endpoint,
      based on the provided working cURL command.
    """
    def __init__(self, client: 'ServiceNow'):
        """
        Initializes the AttachmentHandlerImpl.

        Args:
            client: The main ServiceNow client instance.
        """
        self.client = client
        # RequestHandlerImpl is needed for both add and get using dedicated endpoints
        self.request_handler: IRequestHandler = RequestHandlerImpl(client)
        # TableRequestHandler is not directly used by these methods anymore
        # self.table_handler = client.table_handler # Remove or comment out if not used

    @staticmethod
    def _get_content_type_for_file(file_name: str) -> str:
        """Guesses the content type for a file based on its name."""
        ct, _ = mimetypes.guess_type(file_name)
        return ct or "application/octet-stream" # Default if guess fails

    def _safe_extract_field(self, field_data: Optional[Union[Dict, str]], prefer: str = 'value', default: Any = None) -> Any:
        """
        Safely extracts a value from a potentially nested dictionary structure
        (common in ServiceNow API responses with value/display_value pairs).
        """
        # (Implementation is unchanged)
        if isinstance(field_data, dict):
            preferred_key = 'display_value' if prefer == 'display_value' else 'value'
            fallback_key = 'value' if prefer == 'display_value' else 'display_value'
            if preferred_key in field_data and field_data[preferred_key] is not None:
                return field_data[preferred_key]
            if fallback_key in field_data and field_data[fallback_key] is not None:
                return field_data[fallback_key]
            return default
        return field_data if field_data is not None else default

    async def add_attachment(self, table: 'ServiceNowTable', sys_id: str, file_content: Any, file_name: str, content_type: Optional[str]=None) -> Dict[str, Any]:
        """
        Adds an attachment using the direct file upload method (multipart/form-data)
        via the /SPARCDetails/fileAttachment endpoint. This matches the working Postman request.

        (Implementation remains unchanged - it works based on demo output)
        """
        log.info(f"Attaching '{file_name}' to {table.value} {sys_id} using direct file upload endpoint {self.client.config.file_attachment_api_path}.")
        file_bytes: bytes
        try: # (Conversion logic unchanged)
            if hasattr(file_content, 'read'):
                content_read = file_content.read()
                file_bytes = content_read.encode('utf-8') if isinstance(content_read, str) else content_read
            elif isinstance(file_content, bytes): file_bytes = file_content
            elif isinstance(file_content, str): file_bytes = file_content.encode('utf-8')
            else: raise ValueError(f"Invalid file_content type: {type(file_content)}. Must be bytes, str, or have a read() method.")
            if not isinstance(file_bytes, bytes): raise ValueError(f"Could not convert file_content to bytes")
        except Exception as e:
            log.error(f"Error reading or converting file_content for '{file_name}': {e}", exc_info=True)
            raise ValueError(f"Could not read or convert file_content for '{file_name}': {e}") from e

        final_ct = content_type or self._get_content_type_for_file(file_name)
        log.debug(f"Using Content-Type: {final_ct} for file '{file_name}'")
        params = { "table_name": table.value, "table_sys_id": sys_id, "file_name": file_name }
        file_headers = self.client._get_headers(include_auth=True, content_type=final_ct)
        # file_headers.pop('Content-Type', None) # Consider if needed

        log.debug(f"Attempting direct attachment upload to {self.client.config.file_attachment_api_path} for {table.value} {sys_id}. Params: {list(params.keys())}")
        try:
            response_data = await self.request_handler.make_request(
                method=HTTPMethod.POST,
                endpoint=self.client.config.file_attachment_api_path,
                params=params, data=file_bytes, headers=file_headers
            )
            log.info(f"Successfully attached '{file_name}' to {table.value} {sys_id} via {self.client.config.file_attachment_api_path}")
            return response_data if response_data is not None else {} # Return details if available
        except ServiceNowError as e:
            log.error(f"Direct attachment upload failed for '{file_name}' to {table.value} {sys_id}: {e}")
            raise e
        except Exception as e:
            log.exception(f"Unexpected error during direct attachment upload for '{file_name}' to {table.value} {sys_id}")
            raise ServiceNowError(f"Unexpected error attaching file: {e}", details=str(e)) from e

    # --- REVISED get_attachments (using GetAttachmentMetaData based on cURL) ---
    async def get_attachments(self, table: 'ServiceNowTable', sys_id: str, limit: Optional[int]=None, offset: int=0) -> List[Dict[str, Any]]:
        """
        Gets attachment metadata for a specific parent record using the dedicated
        /SPARCDetails/GetAttachmentMetaData endpoint, based on the provided cURL command.

        NOTE: This endpoint uses 'table_sys_id' but NOT 'table_name' in query params.
        It requires 'client_secret' in the header, unlike typical GET requests.
        Pagination parameters (limit, offset) are likely unsupported and are IGNORED.

        Args:
            table: The table the parent record belongs to (e.g., ServiceNowTable.INCIDENT).
                   NOTE: This parameter is logged but NOT sent to this specific endpoint.
            sys_id: The sys_id of the parent record (REQUIRED).
            limit: Maximum number of attachments to return (IGNORED).
            offset: Offset for pagination (IGNORED).

        Returns:
            A list of dictionaries, each representing an attachment, or an empty list if none found.

        Raises:
            ServiceNowError: If the API call fails for reasons other than Not Found.
        """
        # Use the specific path from Config, added in step 1
        endpoint_path = self.client.config.attachment_metadata_path
        log.info(f"Getting attachments for parent record {sys_id} (Table context: {table.value}) using endpoint {endpoint_path}")

        if limit is not None or offset != 0:
            log.warning(f"Pagination parameters (limit={limit}, offset={offset}) are ignored by the {endpoint_path} endpoint.")

        # Parameters required by the GetAttachmentMetaData endpoint URL (from cURL)
        # envName and client_id are added automatically by make_request's default param handling
        params = {
            "table_sys_id": sys_id,
        }

        # Specific headers required by the GetAttachmentMetaData endpoint (from cURL)
        # Includes the unusual client_secret for GET.
        # These will be merged with defaults (Authorization, Accept, default client_id) by make_request.
        custom_headers = {
            'client_secret': self.client.config.client_secret
            # 'client_id' is also in the cURL headers; make_request adds it by default,
            # but adding it here explicitly ensures it matches the cURL precisely if needed.
            # If default handling causes issues (e.g. duplicate headers), manage it here.
            # 'client_id': self.client.config.client_id # Explicitly add if needed
        }

        log.debug(f"Calling GET {endpoint_path} with params: {params} and custom headers: {list(custom_headers.keys())}")

        try:
            # Use the RequestHandlerImpl directly
            raw_attachments_data = await self.request_handler.make_request(
                method=HTTPMethod.GET,
                endpoint=endpoint_path,
                params=params,
                headers=custom_headers # Pass the specific extra headers required by this endpoint
            )

            # Process the results
            status_code, headers, attachment_data = raw_attachments_data
            log.info(f"attachment X-Total-Count: {headers.get('X-Total-Count')}")
            
            # Check if we have valid attachment data
            if status_code == 200 and isinstance(attachment_data, list):
                formatted = []
                for att in attachment_data:
                    # Extract attachment fields directly from the list items
                    try:
                        formatted.append({
                            "sys_id": att.get("sys_id", ""),
                            "file_name": att.get("file_name", ""),
                            "size_bytes": att.get("size_bytes", ""),
                            "content_type": att.get("content_type", ""),
                            "created_on": att.get("sys_created_on", ""),
                            "created_by": att.get("sys_created_by", ""),
                            "download_link": f"https://{self.client.config.env}.service-now.com/sys_attachment.do?view=true&sys_id={att.get('sys_id')}"
                        })
                    except Exception as e:
                        log.exception(f"Error processing attachment: {e} - Data: {att}")
                
                log.info(f"Successfully retrieved {len(formatted)} attachments for parent {sys_id} via {endpoint_path}")
                return formatted
            else:
                log.warning(f"Unexpected response format or status code: {status_code}")
                return []

        except ServiceNowNotFoundError:
             # Should not happen if the parent sys_id exists, but handle defensively.
             # More likely indicates an issue with the API call itself returning 404 inappropriately.
             log.info(f"No attachments found via {endpoint_path} for parent {sys_id} (or endpoint returned 404)")
             return []
        except ServiceNowError as e:
             # Catch specific ServiceNow errors from make_request
             log.error(f"Error retrieving attachments for {sys_id}: {e}", exc_info=False)
             raise e
        except Exception as e_unexpected:
             # Catch any other unexpected Python errors
             log.exception(f"Unexpected Python error while getting attachments via {endpoint_path} for parent {sys_id}")
             raise ServiceNowError(f"Unexpected Python error getting attachments: {e_unexpected}", details=str(e_unexpected)) from e_unexpected

## Main ServiceNow Client Class ###############################################
# (ServiceNow class and all its methods remain unchanged from your provided script)
class ServiceNow:
    """
    ServiceNow client using SPARC API gateway with improved structure.
    Provides methods for interacting with various ServiceNow modules.
    """
    def __init__(self, config: Config):
        self.config = config
        log.info(f"Initializing ServiceNow for env: {self.config.env}, base_url: {self.config.base_url}")
        if not self.config.client_id or not self.config.client_secret: raise ValueError("Client ID/Secret required.")
        self._access_token: Optional[str]=None; self._session: Optional[aiohttp.ClientSession]=None; self._retry_client:Optional[RetryClient]=None
        self.state_mapper=StateMapper(config); self.request_handler:IRequestHandler=RequestHandlerImpl(self); self.table_handler=TableRequestHandler(self); self.attachment_handler:IAttachmentHandler=AttachmentHandlerImpl(self) # Uses the modified AttachmentHandlerImpl

    # --- Context Management & Session ---
    async def _get_session(self) -> RetryClient:
        if self._retry_client is None:
            log.debug("Creating new Session/RetryClient"); connector=aiohttp.TCPConnector(ssl=self.config.ssl_verify); self._session=aiohttp.ClientSession(connector=connector)
            retry_options=ExponentialRetry(attempts=self.config.retry_attempts,statuses=self.config.retry_statuses,exceptions=self.config.retry_exceptions); self._retry_client=RetryClient(client_session=self._session,retry_options=retry_options); log.info(f"RetryClient configured.")
        return self._retry_client
    async def close(self):
        if self._retry_client: await self._retry_client.close(); self._retry_client=None; self._session=None
        elif self._session: await self._session.close(); self._session=None
    async def __aenter__(self): await self._get_session(); return self
    async def __aexit__(self,exc_type,exc_val,exc_tb): await self.close()

    # --- Authentication & Headers ---
    def _get_headers(self, include_auth: bool=True, content_type: str="application/json") -> Dict[str,str]:
        headers={'Accept':'application/json','Content-Type':content_type,'client_id':self.config.client_id}
        if include_auth:
            if not self._access_token: raise ServiceNowAuthError("Token required.")
            headers['Authorization']=f'Bearer {self._access_token}'
        return headers
    @tracer.start_as_current_span("servicenow_client._ensure_token")
    async def _ensure_token(self, force_refresh: bool=False) -> str:
        if force_refresh or not self._access_token:
            log.info("%s OAuth token.","Refreshing" if force_refresh else "Obtaining"); token_url=f"{self.config.base_url}{self.config.oauth_token_path}?grant_type=client_credentials"
            # IMPORTANT: Auth headers ONLY contain client_id and client_secret for the token request itself
            auth_headers={'client_id':self.config.client_id,'client_secret':self.config.client_secret};
            if self._session is None: self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.config.ssl_verify))
            base_session = self._session
            try:
                async with base_session.get(token_url, headers=auth_headers, ssl=self.config.ssl_verify) as response:
                    if response.status == 200: token_response=await response.json(); self._access_token=token_response.get('access_token'); log.info("Token obtained.")
                    else: error_text=await response.text(); raise ServiceNowAuthError(f"Token fetch failed: {response.status}",response.status,error_text)
                    if not self._access_token: raise ServiceNowAuthError("Token missing in response.")
            except(aiohttp.ClientError,json.JSONDecodeError,asyncio.TimeoutError) as e: raise ServiceNowAuthError(f"Token retrieval error: {e}") from e
        if not self._access_token: raise ServiceNowAuthError("Token retrieval failed.")
        return self._access_token

    # --- Helper Methods ---
    def _generate_record_link(self, table: Union[ServiceNowTable, str], sys_id: str, is_service_portal: bool=True) -> str:
        if not sys_id: return "#"
        table_name = table.value if isinstance(table, ServiceNowTable) else table
        instance_url = f"https://{self.config.env}.service-now.com"
        if is_service_portal:
            sp_id = "ticket"
            # Corrected SP link format assuming 'id=ticket' is the standard page
            return f"{instance_url}/sp?sys_id={sys_id}&view=sp&id={sp_id}&table={table_name}"
        else:
            return f"{instance_url}/nav_to.do?uri={table_name}.do?sys_id={sys_id}"
    def _safe_extract_field(self, field_data: Optional[Union[Dict, str]], prefer: str = 'value', default: Any = None) -> Any:
        # Unchanged _safe_extract_field logic
        if isinstance(field_data, dict): preferred_key='display_value' if prefer=='display_value' else 'value'; fallback_key='value' if prefer=='display_value' else 'display_value';
        if isinstance(field_data, dict) and preferred_key in field_data: return field_data[preferred_key]
        if isinstance(field_data, dict) and fallback_key in field_data: return field_data[fallback_key]
        if isinstance(field_data, dict): return default
        return field_data if field_data is not None else default
    async def _fetch_all_pages(self, table: ServiceNowTable, query: str, fields: List[str], page_limit: int = 1000) -> List[Dict]:
        all_records=[]; offset=0;
        while True:
            log.debug(f"Fetching page for {table.value}: offset={offset}, limit={page_limit}")
            page_data=await self.table_handler.table_request(table,HTTPMethod.GET,query,limit=page_limit,offset=offset,fields=fields,display_value=True)
            records=page_data if isinstance(page_data,list) else []
            if not records: log.debug("No more records found."); break
            all_records.extend(records)
            if len(records)<page_limit: log.debug("Last page fetched."); break
            offset+=page_limit
        log.debug(f"Fetched total {len(all_records)} records for query.")
        return all_records

    # --- Instance Info ---
    @tracer.start_as_current_span("sparc_service_now.get_instance_name")
    async def get_instance_name(self) -> str:
        return self.config.env

    # --- User Operations --- #
    @tracer.start_as_current_span("sparc_service_now.get_me_user_id")
    async def get_me_user_id(self) -> Optional[str]:
        log.info("Fetching sys_id for current API user via 'source=system' query")
        try:
            # Assuming 'source=system' is a valid way to identify the API user record
            response = await self.table_handler.table_request(ServiceNowTable.SYS_USER, HTTPMethod.GET, 'source=system', fields=['sys_id'], limit=1)
            if response and isinstance(response, list) and response:
                sys_id = self._safe_extract_field(response[0].get("sys_id"), 'value')
                log.info(f"Found API user sys_id: {sys_id}")
                return sys_id
            log.warning("Could not determine sys_id for current API user using 'source=system'.")
            return None
        except ServiceNowError as e:
            log.error(f"Error fetching current API user sys_id: {e}")
            return None # Return None on error as per original logic indication
    @tracer.start_as_current_span("sparc_service_now.get_user_sys_id_by_field")
    async def get_user_sys_id_by_field(self, field_name: str, field_value: str) -> Optional[str]:
        log.info(f"Fetching user sys_id by {field_name} = '{field_value}'")
        try:
            response = await self.table_handler.table_request(ServiceNowTable.SYS_USER, HTTPMethod.GET, f"{field_name}={field_value}", fields=['sys_id'], limit=1)
            if response and isinstance(response, list) and response:
                 sys_id=self._safe_extract_field(response[0].get("sys_id"), 'value')
                 log.info(f"Found user sys_id: {sys_id}")
                 return sys_id
            log.warning(f"User not found with {field_name} = '{field_value}'"); return None
        except ServiceNowNotFoundError: log.warning(f"User not found (404)"); return None
        except ServiceNowError as e: log.exception(f"Error fetching user: {e}"); raise # Re-raise other errors
    # same function above -- but with dict
    @tracer.start_as_current_span("sparc_service_now.get_user_sys_id_by_field")
    async def get_user_sys_id_by_field_dict(self, field_name: str, field_value: str) -> Optional[str]:
        log.info(f"Fetching user sys_id by {field_name} = '{field_value}'")
        try:
            response = await self.table_handler.table_request(ServiceNowTable.SYS_USER, HTTPMethod.GET, f"{field_name}={field_value}", fields=['sys_id'], limit=1)
            if response and isinstance(response, dict) and 'records' in response and response['records']:
                user_record = response['records'][0]
                sys_id = user_record.get('sys_id')
                if sys_id:
                    log.info(f"Found user sys_id: {sys_id}")
                    return str(sys_id) # Ensure it's returned as a string
                else:
                    # This case means the record existed but didn't have a 'sys_id' key/value
                    log.warning(f"User record found for {field_name} = '{field_value}' but 'sys_id' key is missing or empty.")
                    return None
            else:
                # Handle cases where response is invalid or 'records' is empty
                log.warning(f"User not found or empty response for {field_name} = '{field_value}'")
                return None
        except ServiceNowNotFoundError:
            log.warning(f"User not found with {field_name} = '{field_value}' (ServiceNow 404)")
            return None
        except ServiceNowError as e:
            log.exception(f"ServiceNowError fetching user sys_id for {field_name} = '{field_value}': {e}")
            raise # Re-raise other ServiceNow errors
        except (KeyError, IndexError, TypeError) as e:
            # Catch potential errors during response processing (e.g., 'records' key missing, list index out of bounds)
            log.exception(f"Error processing ServiceNow response structure for {field_name} = '{field_value}': {e}")
            return None
        except Exception as e:
            # Catch any other unexpected errors
            log.exception(f"Unexpected error fetching user sys_id for {field_name} = '{field_value}': {e}")
            raise

    @tracer.start_as_current_span("sparc_service_now.get_user_sys_id_by_employee_number")
    async def get_user_sys_id_by_employee_number(self, employee_number: str) -> Optional[str]:
        return await self.get_user_sys_id_by_field("employee_number", employee_number)

    # same function above -- but with dict 
    @tracer.start_as_current_span("sparc_service_now.get_user_sys_id_by_employee_number")
    async def get_user_sys_id_by_employee_number_dict(self, employee_number: str) -> Optional[str]:
        return await self.get_user_sys_id_by_field_dict("employee_number", employee_number)

    @tracer.start_as_current_span("sparc_service_now.get_user_by_user_sys_id")
    async def get_user_by_user_sys_id(self, user_sys_id: str) -> Dict[str, Any]:
        log.info(f"Fetching user details for sys_id: {user_sys_id}")
        try:
            user_data = await self.table_handler.table_request(ServiceNowTable.SYS_USER, HTTPMethod.GET, f"sys_id={user_sys_id}", display_value=True, limit=1)
            if not user_data or (isinstance(user_data, list) and not user_data):
                raise ServiceNowNotFoundError(f"User {user_sys_id} not found.")
            result = user_data[0] if isinstance(user_data, list) else user_data
            # Flatten value/display_value pairs for easier consumption if needed, or return raw
            # Example flattening (optional):
            # formatted_result = {k: self._safe_extract_field(v, 'display_value', self._safe_extract_field(v, 'value')) for k, v in result.items()}
            # return ResponseHandler.success(data=formatted_result)
            return ResponseHandler.success(data=result) # Return raw structure for now
        except ServiceNowNotFoundError as e: return ResponseHandler.not_found(str(e))
        except ServiceNowError as e: return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)

    # --- Location Operations --- #
    @tracer.start_as_current_span("sparc_service_now.get_locations")
    async def get_locations(self) -> List[Dict[str, str]]:
        log.info("Fetching all locations")
        # Assuming the table API returns a list for GET requests
        locations_data = await self.table_handler.table_request(ServiceNowTable.CMN_LOCATION, HTTPMethod.GET, fields=["sys_id", "name"], display_value=True) # Fetch display value for name
        return [{"sys_id": self._safe_extract_field(loc.get("sys_id"),'value'), "name": self._safe_extract_field(loc.get("name"),'display_value',loc.get("name"))} for loc in locations_data] if locations_data else []

    # --- Approval Operations --- #
    async def _update_approval_state(self, approval_sys_id: str, state: ApprovalState, comments: Optional[str] = None) -> Dict[str, Any]:
        log.info(f"Setting approval {approval_sys_id} state to '{state.value}'")
        data={"state":state.value};
        if comments: data["comments"]=comments
        elif state == ApprovalState.REJECTED and not comments: log.warning(f"Rejecting {approval_sys_id} without comments.")
        # Use the table_handler's PUT method, assuming query = sys_id for PUT
        return await self.table_handler.table_request(ServiceNowTable.SYS_APPROVAL,HTTPMethod.PUT,approval_sys_id,data)
    @tracer.start_as_current_span("sparc_service_now.approve_request")
    async def approve_request(self, approval_sys_id: str) -> Dict[str, Any]:
        """Approves a request given the approval sys_id."""
        log.info(f"Approving request {approval_sys_id}")
        try: updated_record = await self._update_approval_state(approval_sys_id, ApprovalState.APPROVED, None); return ResponseHandler.success(data=updated_record)
        except ServiceNowError as e: return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
    @tracer.start_as_current_span("sparc_service_now.reject_request")
    async def reject_request(self, approval_sys_id: str, comments: Optional[str] = None) -> Dict[str, Any]:
        """Rejects a request given the approval sys_id."""
        log.info(f"Rejecting request {approval_sys_id}")
        if not comments: log.warning(f"Rejecting {approval_sys_id} without comments based on original signature.")
        try: updated_record = await self._update_approval_state(approval_sys_id, ApprovalState.REJECTED, comments); return ResponseHandler.success(data=updated_record)
        except ServiceNowError as e: return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
    @tracer.start_as_current_span("sparc_service_now.add_comment_to_approval")
    async def add_comment_to_approval(self, approval_sys_id: str, comment: str) -> Dict[str, Any]:
        """Adds a comment to an approval request."""
        if not comment: return ResponseHandler.bad_request("Comment cannot be empty.")
        log.info(f"Adding comment to approval {approval_sys_id}")
        try:
             # PUT request to update comments field
             updated_record = await self.table_handler.table_request(ServiceNowTable.SYS_APPROVAL, HTTPMethod.PUT, approval_sys_id, {"comments": comment})
             return ResponseHandler.success(data=updated_record)
        except ServiceNowError as e: return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
    @tracer.start_as_current_span("sparc_service_now.get_approval_state")
    async def get_approval_state(self, approval_sys_id: str) -> Dict[str, Any]:
        """Gets the current state and details of an approval record."""
        log.info(f"Fetching state/details for approval {approval_sys_id}")
        # Define fields to retrieve, including related record fields using dot-walking
        fields=["sys_id","approver","state","comments","due_date","sys_created_on","sys_updated_on","sysapproval","sysapproval.sys_id","sysapproval.number","sysapproval.short_description","sysapproval.description","sysapproval.sys_class_name","sysapproval.state","sysapproval.stage","sysapproval.requested_for","sysapproval.requested_by","sysapproval.priority","sysapproval.urgency","document_id","approval_source"]
        try:
            approval_data = await self.table_handler.table_request(ServiceNowTable.SYS_APPROVAL, HTTPMethod.GET, f"sys_id={approval_sys_id}", fields=fields, display_value=True, limit=1)
            if not approval_data or (isinstance(approval_data,list) and not approval_data):
                raise ServiceNowNotFoundError(f"Approval {approval_sys_id} not found.")
            result = approval_data[0] if isinstance(approval_data,list) else approval_data

            # Flatten and format the response
            formatted_result = {}
            for key, value in result.items():
                # Store both value and display_value if available
                formatted_result[key] = self._safe_extract_field(value,'value')
                formatted_result[f"{key}_display"] = self._safe_extract_field(value,'display_value', formatted_result[key]) # Default display to value if missing

            # Add specific formatted state
            state_disp = formatted_result.get("state_display","").lower()
            if state_disp=="requested": formatted_result["approval_state_formatted"]="Pending"
            elif state_disp=="approved": formatted_result["approval_state_formatted"]="Approved"
            elif state_disp=="rejected": formatted_result["approval_state_formatted"]="Rejected"
            else: formatted_result["approval_state_formatted"]=state_disp.capitalize() if state_disp else "Unknown"

            # Generate link to the approved record
            target_table_name = formatted_result.get("sysapproval.sys_class_name_display")
            target_sys_id = formatted_result.get("sysapproval") # This should be the sys_id value
            table_map={"Requested Item":"sc_req_item","Change Request":"change_request","Incident":"incident"};
            target_table = table_map.get(target_table_name, target_table_name) # Use table name if not in map
            if target_table and target_sys_id:
                 formatted_result["record_link"] = self._generate_record_link(target_table.lower(), target_sys_id)

            return ResponseHandler.success(data=formatted_result)
        except ServiceNowNotFoundError as e: return ResponseHandler.not_found(str(e))
        except ServiceNowError as e: return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)

    async def _get_pending_approvals_impl(self, user_sys_id: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Internal implementation to fetch and FORMAT pending approvals like the original code.
        Returns a list of specifically formatted dictionaries.
        """
        log.info(f"Fetching pending approvals for user {user_sys_id}, limit: {limit}, offset: {offset}")

        fields = [
            "sys_id", # sys_id of the sysapproval_approver record itself
            "sysapproval.number", # Number of the record being approved (RITM, CHG, etc.)
            "sysapproval.short_description",
            "sysapproval.description",
            "sysapproval.sys_created_on", # Created date of the target record
            "sysapproval.priority",
            "sysapproval.urgency",
            "sysapproval.requested_for", # User the item is for
            "state", # State of the approval record (e.g., 'requested')
            "approver", # Should match user_sys_id
            "document_id", # Often contains the display value (e.g., RITM number)
            "due_date",
            "sysapproval.sys_class_name", # Table name of the record being approved
            "sysapproval.sys_id", # sys_id of the record being approved
            "sys_created_on" # Created date of the approval record itself
        ]
        # Query for approvals in 'requested' state for the specific approver, ordered by creation date descending
        query=f"state={ApprovalState.REQUESTED.value}^approver={user_sys_id}^ORDERBYDESCsys_created_on"

        # Fetch raw data using the table handler
        raw_approvals_data = await self.table_handler.table_request(
            table=ServiceNowTable.SYS_APPROVAL,
            method=HTTPMethod.GET,
            query=query,
            fields=fields,
            limit=limit,
            offset=offset,
            display_value=True # Get both value and display_value
        )
        raw_approvals_data = raw_approvals_data["records"]
        raw_results = raw_approvals_data if isinstance(raw_approvals_data, list) else []

        # --- Formatting logic inspired by the original code ---
        formatted_approvals = []
        table_map={"Requested Item":"sc_req_item","Change Request":"change_request","Incident":"incident"} # Map display name to table name

        for item in raw_results:
            # Extract necessary values safely using the helper
            approval_type_raw = item.get("sysapproval.sys_class_name")
            approval_type = self._safe_extract_field(approval_type_raw, 'display_value', "Unknown")

            # sys_id of the record being approved (RITM, CHG...)
            approval_target_sys_id = self._safe_extract_field(item.get("sysapproval"), 'value')
            # sys_id of the sysapproval_approver record itself
            approval_record_sys_id = self._safe_extract_field(item.get("sys_id"), 'value')


            link = "#"
            target_table_name = table_map.get(approval_type, approval_type) # Use raw name if not in map
            if target_table_name and approval_target_sys_id:
                # Generate link to the record being approved (RITM, CHG etc.)
                link = self._generate_record_link(target_table_name.lower(), approval_target_sys_id)


            requested_for = self._safe_extract_field(item.get("sysapproval.requested_for"), 'display_value', "N/A")
            # Use sysapproval.number as the primary display number
            number_display = self._safe_extract_field(item.get("sysapproval.number"), 'display_value', "N/A")
            document_id_display = self._safe_extract_field(item.get("document_id"), 'display_value', number_display) # Fallback to number if document_id is odd
            state_display = self._safe_extract_field(item.get("state"), 'display_value', "N/A") # State of the approval itself


            approval = {
                "number": number_display,
                "short_description": self._safe_extract_field(item.get("sysapproval.short_description"), 'display_value', "N/A"),
                "description": self._safe_extract_field(item.get("sysapproval.description"), 'display_value', "N/A"),
                # Use the creation date of the *target* record (RITM/CHG)
                "created": self._safe_extract_field(item.get("sysapproval.sys_created_on"), 'display_value', "N/A"),
                "priority": self._safe_extract_field(item.get("sysapproval.priority"), 'display_value', "N/A"),
                "urgency": self._safe_extract_field(item.get("sysapproval.urgency"), 'display_value', "N/A"),
                "requested_for": requested_for,
                "state": state_display, # State of the approval record ('Requested')
                "due_date": self._safe_extract_field(item.get("due_date"), 'display_value', "N/A"),
                "document_id": document_id_display, # Often the display value like number
                "link": link, # Link to the target record (RITM/CHG)
                "approval_type": approval_type, # Type of record being approved
                "approval_id": approval_target_sys_id, # Sys ID of the target record (RITM/CHG)
                "sys_id": approval_record_sys_id, # Sys ID of the approval record itself
            }
            formatted_approvals.append(approval)

        return formatted_approvals

    @tracer.start_as_current_span("sparc_service_now.get_latest_pending_approvals")
    async def get_latest_pending_approvals(self, user_sys_id: str, limit: int = 5) -> Dict[str, Any]:
        """
        Fetches the latest pending approvals for a user, matching the original output structure.
        """
        log.info(f"Getting latest {limit} pending approvals for {user_sys_id}")
        # Call the paginated version with offset 0
        return await self.get_latest_pending_approvals_pagination(user_sys_id, limit, 0)

    @tracer.start_as_current_span("sparc_service_now.get_latest_pending_approvals_pagination")
    async def get_latest_pending_approvals_pagination(self, user_sys_id: str, limit: int = 5, offset: int = 0) -> Dict[str, Any]:
        """
        Fetches paginated pending approvals from ServiceNow for a specific user,
        matching the original code's output structure.
        """
        log.info(f"Getting pending approvals page for {user_sys_id} (limit={limit}, offset={offset})")
        try:
            # Calls the implementation method which now returns the formatted list
            formatted_approvals_list = await self._get_pending_approvals_impl(user_sys_id=user_sys_id, limit=limit, offset=offset)

            # Construct the final dictionary matching the original structure
            # Note: We cannot reliably get the *total* count of pending approvals without
            # another query or if the API provided an X-Total-Count header (which it likely doesn't here).
            # We'll set total_count based on the items returned on *this* page, similar to original.
            return {
                "code": 200,
                "message": "Successfully retrieved approvals" if formatted_approvals_list else "No pending approvals found",
                "data": formatted_approvals_list,
                "total_count": len(formatted_approvals_list), # Count of items on the current page
                "limit": limit,
                "offset": offset
            }
        except ServiceNowError as e:
             # Return error structure similar to original if an API error occurs
             log.error(f"Error fetching pending approvals page: {e}", exc_info=True)
             # Use ResponseHandler just to format the error dict consistently
             return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
        except Exception as e_unexpected: # Catch any other unexpected errors
            log.exception("Unexpected error in get_latest_pending_approvals_pagination")
            return ResponseHandler.internal_error(f"Unexpected error: {str(e_unexpected)}")


    # --- Incident Operations --- #
    @tracer.start_as_current_span("servicenow_client.create_incident")
    async def create_incident(self, **kwargs: Any) -> Dict[str, Any]:
        """Creates an Incident record using the dedicated endpoint."""
        log.info("Creating incident via dedicated endpoint...")
        # Set defaults if not provided
        kwargs.setdefault('contact_type', self.config.default_contact_type)
        kwargs.setdefault('assignment_group', self.config.default_assignment_group)

        # Use the dedicated incident creation path from config
        incident_response = await self.request_handler.make_request(
            method=HTTPMethod.POST,
            endpoint=self.config.incident_create_path, # Use the specific path
            params=None, # Usually no specific query params for dedicated create endpoint
            data=kwargs # Pass the incident data as the JSON body
        )

        # The response is a tuple (status_code, headers, data)
        # Extract the actual incident data from the tuple
        if isinstance(incident_response, tuple) and len(incident_response) >= 3:
            _, _, incident_data = incident_response
        else:
            incident_data = incident_response  # Fallback if not a tuple

        # Add link if successful and sys_id is present
        if isinstance(incident_data, dict) and (sys_id := self._safe_extract_field(incident_data.get("sys_id"), 'value')):
            incident_data["link"] = self._generate_record_link(ServiceNowTable.INCIDENT, sys_id)


        
        # Extract fields safely from the incident_data dictionary
        inc_num_disp = self._safe_extract_field(incident_data.get('number'), 'display_value', 
                                               self._safe_extract_field(incident_data.get('number'), 'value'))
        inc_sysid_val = self._safe_extract_field(incident_data.get('sys_id'), 'value')
        log.info(f"Incident {inc_num_disp} created (sys_id: {inc_sysid_val}).")
        
        # Directly return the data dictionary (or raise ServiceNowError if failed)
        return incident_data
    @tracer.start_as_current_span("sparc_service_now.update_ticket")
    async def update_ticket(self, incident_sys_id: str, subject: Optional[str]=None, knowledge: Optional[str]=None, description: Optional[str]=None, urgency: Optional[str]=None, impact: Optional[str]=None, state: Optional[str]=None, category: Optional[str]=None, subcategory: Optional[str]=None, assignment_group: Optional[str]=None, assigned_to: Optional[str]=None, short_description: Optional[str]=None, comments: Optional[str]=None, work_notes: Optional[str]=None, cmdb_ci: Optional[str]=None) -> Dict[str, Any]:
        """ Deprecated wrapper. Use update_incident or update_incident_state. """
        log.warning("Method 'update_ticket' is deprecated. Use 'update_incident' or 'update_incident_state'.")
        log.info(f"Updating ticket (incident) {incident_sys_id}")
        # Collect non-None arguments into a dictionary
        data={k:v for k,v in locals().items() if v is not None and k not in ["self","incident_sys_id"]}

        # Check if 'state' is being updated - redirect to specific method if so
        if 'state' in data:
             log.warning("State updates should use update_incident_state method. Ignoring 'state' parameter here.")
             data.pop('state') # Remove state from the data dict for this generic update

        if not data: return ResponseHandler.success(message="No fields provided to update.")

        try:
            # Call the preferred update_incident method
            updated_record = await self.update_incident(incident_sys_id, **data)
            # update_incident now returns the raw dict or raises error
            return ResponseHandler.success(data=updated_record) # Wrap success response
        except ServiceNowError as e:
            # update_incident raises error on failure
            return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
        
    @tracer.start_as_current_span("servicenow_client.update_incident")
    async def update_incident(self, incident_sys_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Updates an incident record using the Table API PUT method."""
        log.info(f"Updating incident {incident_sys_id} via Table API PUT...");
        # Prevent accidental state updates through this generic method
        if 'state' in kwargs or 'incident_state' in kwargs:
             log.warning("State updates should use the specific 'update_incident_state' method. Ignoring state fields.")
             kwargs.pop('state',None)
             kwargs.pop('incident_state',None)

        if not kwargs:
             log.info("No fields provided for update. Returning current incident data.")
             # Fetch and return current data if no updates specified
             return await self.get_incident_by_sys_id_internal(incident_sys_id)

        # Use the table handler's PUT request (query = sys_id)
        updated_data = await self.table_handler.table_request(
            table=ServiceNowTable.INCIDENT,
            method=HTTPMethod.PUT,
            query=incident_sys_id, # Pass sys_id as the identifier for PUT
            data=kwargs # Pass update fields as data payload
        )

        # Add link if successful
        if isinstance(updated_data,dict) and (sys_id := self._safe_extract_field(updated_data.get("sys_id"),'value')):
            updated_data["link"]=self._generate_record_link(ServiceNowTable.INCIDENT,sys_id)

        log.info(f"Incident {incident_sys_id} updated.")
        # Return the raw updated record data (or raises ServiceNowError)
        return updated_data
    @tracer.start_as_current_span("sparc_service_now.update_incident_state")
    async def update_incident_state(self, incident_sys_id: str, action: str, comment: str="", work_note: str="") -> Dict[str, Any]:
        """ Updates the state of an incident using state name mapping. """
        log.info(f"Updating state of incident {incident_sys_id} via action '{action}'")
        try:
             # Try direct mapping first (using IncidentState enum names)
             state_code=self.state_mapper.get_incident_state_code(action)
        except ValueError:
             # Fallback for simple action verbs if direct mapping fails
             action_map={"close":"Closed","reopen":"Open","resolve":"Resolved"}
             mapped_state=action_map.get(action.lower())
             if not mapped_state:
                 # If neither direct map nor action map works, return Bad Request
                 valid_states = list(self.config.incident_state_mapping_str_key.keys())
                 valid_actions = list(action_map.keys())
                 err_msg = f"Invalid action/state: '{action}'. Valid states: {valid_states}. Valid actions: {valid_actions}."
                 return ResponseHandler.bad_request(err_msg)
             try:
                 state_code=self.state_mapper.get_incident_state_code(mapped_state)
             except ValueError:
                 # Should not happen if action_map keys are valid IncidentState names
                 log.error(f"Internal mapping error for action '{action}' -> '{mapped_state}'")
                 return ResponseHandler.internal_error(f"Internal state mapping error for action '{action}'.")


        # Prepare data for update_incident call
        # IMPORTANT: The API might expect 'state' or 'incident_state' or both. Check API docs.
        # Assuming both are needed based on original get_tickets_by_states logic using incident_state
        update_data={"state":state_code,"incident_state":state_code};
        if comment: update_data["comments"]=comment;
        if work_note: update_data["work_notes"]=work_note;
        if state_code == self.state_mapper.get_incident_state_code("Open"):
            update_data["assigned_to"]=""

        try:
            # Use the generic update_incident method to apply the state change and comments/notes
            # update_incident will ignore state/incident_state if passed directly,
            # but here we construct them specifically for state update. Let's call Table API directly.
            log.info(f"Updating incident {incident_sys_id} with data: {update_data}")
            updated_record = await self.table_handler.table_request(
                 table=ServiceNowTable.INCIDENT,
                 method=HTTPMethod.PUT,
                 query=incident_sys_id,
                 data=update_data
            )
            # Add link if successful
            if isinstance(updated_record,dict) and (sys_id := self._safe_extract_field(updated_record.get("sys_id"),'value')):
                updated_record["link"]=self._generate_record_link(ServiceNowTable.INCIDENT,sys_id)
            return ResponseHandler.success(data=updated_record) # Wrap success response
        except ServiceNowError as e:
            return ResponseHandler.error(e.status_code or 500, e.args[0], e.details) # Wrap error response

    @tracer.start_as_current_span("servicenow_client.update_incident")
    async def update_incident_vva(self, incident_sys_id: str, action: str = None, comment: str = "", work_note: str = "", **kwargs: Any) -> Dict[str, Any]:
        """
        Updates an incident record using the Table API PUT method.
        
        Args:
            incident_sys_id: The sys_id of the incident to update
            action: Optional state action (e.g., 'close', 'reopen', 'resolve', or state name)
            comment: Optional comment to add
            work_note: Optional work note to add
            **kwargs: Additional fields to update
        
        Returns:
            Dict containing the updated incident data or error response
        """
        log.info(f"Updating incident {incident_sys_id} via Table API PUT...")
        
        # Handle state updates if action is provided
        if action:
            log.info(f"Processing state update action: '{action}'")
            try:
                # Try direct mapping first (using IncidentState enum names)
                state_code = self.state_mapper.get_incident_state_code(action)
            except ValueError:
                # Fallback for simple action verbs if direct mapping fails
                action_map = {"close": "Closed", "reopen": "Open", "resolve": "Resolved"}
                mapped_state = action_map.get(action.lower())
                if not mapped_state:
                    # If neither direct map nor action map works, return Bad Request
                    valid_states = list(self.config.incident_state_mapping_str_key.keys())
                    valid_actions = list(action_map.keys())
                    err_msg = f"Invalid action/state: '{action}'. Valid states: {valid_states}. Valid actions: {valid_actions}."
                    return ResponseHandler.bad_request(err_msg)
                try:
                    state_code = self.state_mapper.get_incident_state_code(mapped_state)
                except ValueError:
                    # Should not happen if action_map keys are valid IncidentState names
                    log.error(f"Internal mapping error for action '{action}' -> '{mapped_state}'")
                    return ResponseHandler.internal_error(f"Internal state mapping error for action '{action}'.")
            
            # Add state fields to kwargs
            kwargs["state"] = state_code
            kwargs["incident_state"] = state_code
            
            # Clear assigned_to if reopening
            if state_code == self.state_mapper.get_incident_state_code("Open"):
                kwargs["assigned_to"] = ""
        else:
            # For non-state updates, prevent accidental state updates
            if 'state' in kwargs or 'incident_state' in kwargs:
                log.warning("State updates should use the 'action' parameter. Ignoring state fields.")
                kwargs.pop('state', None)
                kwargs.pop('incident_state', None)
        
        # Add comments and work notes if provided
        if comment:
            kwargs["comments"] = comment
        if work_note:
            kwargs["work_notes"] = work_note
        
        # Check if there are any fields to update
        if not kwargs:
            log.info("No fields provided for update. Returning current incident data.")
            # Fetch and return current data if no updates specified
            return await self.get_incident_by_sys_id_internal(incident_sys_id)
        
        try:
            # Use the table handler's PUT request
            log.info(f"Updating incident {incident_sys_id} with data: {kwargs}")
            updated_data = await self.table_handler.table_request(
                table=ServiceNowTable.INCIDENT,
                method=HTTPMethod.PUT,
                query=incident_sys_id,  # Pass sys_id as the identifier for PUT
                data=kwargs  # Pass update fields as data payload
            )
            
            # Add link if successful
            if isinstance(updated_data, dict) and (sys_id := self._safe_extract_field(updated_data.get("sys_id"), 'value')):
                updated_data["link"] = self._generate_record_link(ServiceNowTable.INCIDENT, sys_id)
            
            log.info(f"Incident {incident_sys_id} updated.")
            
            # Return wrapped response for state updates, raw data for general updates
            if action:
                return ResponseHandler.success(data=updated_data)
            else:
                return updated_data
                
        except ServiceNowError as e:
            if action:
                return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
            else:
                raise  # Re-raise for general updates to maintain original behavior

    @tracer.start_as_current_span("sparc_service_now.get_recent_tickets_by_creator")
    async def get_recent_tickets_by_creator(
        self, 
        user_sys_id: str,
        limit: int = 3
    ) -> Dict[str, Any]:
        """
        Gets the most recent incident tickets created by a specific user.
        
        Args:
            user_sys_id: The sys_id of the user who created the incidents
            limit: Maximum number of incidents to return (default: 3)
            
        Returns:
            A ResponseHandler dict containing a list of the most recent incidents
            created by the specified user, ordered by creation date (newest first).
        """
        log.info(f"Fetching {limit} most recent incidents created by user {user_sys_id}")
        
        try:
            # Build query for incidents where opened_by = user_sys_id, ordered by creation date descending
            query = f"opened_by={user_sys_id}^ORDERBYDESCsys_created_on"
            
            # Define fields to retrieve
            fields = [
                "sys_id", "number", "short_description", "description", 
                "state", "incident_state", "sys_created_on", "sys_updated_on",
                "priority", "urgency", "assigned_to", "caller_id", 
                "category", "subcategory", "close_code"
            ]
            
            # Use table_handler to get the incidents
            table_response = await self.table_handler.table_request(
                table=ServiceNowTable.INCIDENT,
                method=HTTPMethod.GET,
                query=query,
                limit=limit,
                fields=fields,
                display_value=True
            )
            
            incidents = table_response.get("records", [])
            total_count = table_response.get("total_count")
            
            # Add links to each incident
            for inc in incidents:
                if sys_id := self._safe_extract_field(inc.get("sys_id"), 'value'):
                    inc["link"] = self._generate_record_link(ServiceNowTable.INCIDENT, sys_id)
            
            # Return formatted response
            success_response = ResponseHandler.success(
                data=incidents,
                message=f"Retrieved {len(incidents)} recent incidents created by user."
            )
            
            # Add total count if available from the API
            if total_count is not None:
                success_response["total_count"] = total_count
                
            return success_response
            
        except ServiceNowError as e:
            log.error(f"Error fetching recent incidents: {e}", exc_info=True)
            return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
        except Exception as e:
            log.exception(f"Unexpected error in get_recent_tickets_by_creator")
            return ResponseHandler.internal_error(f"Unexpected error: {str(e)}")

    # Internal method to fetch raw incident data, raises NotFoundError
    async def get_incident_by_sys_id_internal(self, incident_sys_id: str, fields: Optional[List[str]]=None, display_value: bool=True) -> Dict[str, Any]:
        log.debug(f"Fetching incident by sys_id: {incident_sys_id}")
        incident_data = await self.table_handler.table_request(
            ServiceNowTable.INCIDENT,
            HTTPMethod.GET,
            f"sys_id={incident_sys_id}", # Query by sys_id
            fields=fields,
            display_value=display_value,
            limit=1
        )
        if not incident_data or (isinstance(incident_data,list) and not incident_data):
            raise ServiceNowNotFoundError(f"Incident {incident_sys_id} not found.")
        # Assuming table_request returns list for GET even with limit=1
        result = incident_data[0] if isinstance(incident_data,list) else incident_data
        # Add link directly to the result dictionary
        result["link"]=self._generate_record_link(ServiceNowTable.INCIDENT,self._safe_extract_field(result.get("sys_id"),'value'));
        return result
    @tracer.start_as_current_span("sparc_service_now.get_incident_by_sys_id")
    async def get_incident_by_sys_id(self, incident_sys_id: str) -> Dict[str, Any]:
        """Retrieves incident details by sys_id, wrapped in ResponseHandler format."""
        try:
            data = await self.get_incident_by_sys_id_internal(incident_sys_id)
            return ResponseHandler.success(data=data)
        except ServiceNowNotFoundError as e: return ResponseHandler.not_found(str(e))
        except ServiceNowError as e: return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)

    # Internal method to fetch raw incident data by number, raises NotFoundError
    async def get_incident_by_number_internal(self, incident_number: str, fields: Optional[List[str]]=None, display_value: bool=True) -> Dict[str, Any]:
        log.debug(f"Fetching incident by number: {incident_number}")
        incident_data = await self.table_handler.table_request(
            ServiceNowTable.INCIDENT,
            HTTPMethod.GET,
            f"number={incident_number}", # Query by number
            fields=fields,
            display_value=display_value,
            limit=1
        )
        if not incident_data or (isinstance(incident_data,list) and not incident_data):
            raise ServiceNowNotFoundError(f"Incident {incident_number} not found.")
        result = incident_data[0] if isinstance(incident_data,list) else incident_data
        result["link"]=self._generate_record_link(ServiceNowTable.INCIDENT,self._safe_extract_field(result.get("sys_id"),'value'));
        return result
    @tracer.start_as_current_span("sparc_service_now.get_incident_by_number")
    async def get_incident_by_number(self, incident_number: str) -> Dict[str, Any]:
        """Retrieves incident details by number, wrapped in ResponseHandler format."""
        try:
            data = await self.get_incident_by_number_internal(incident_number)
            return ResponseHandler.success(data=data)
        except ServiceNowNotFoundError as e: return ResponseHandler.not_found(str(e))
        except ServiceNowError as e: return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)

    @tracer.start_as_current_span("sparc_service_now.get_incident_by_number_and_caller")
    async def get_incident_by_number_and_caller(self, incident_number: str, caller_id: str) -> Dict[str, Any]:
        """ Retrieves an incident only if it matches the number AND caller sys_id. """
        log.info(f"Fetching incident {incident_number} for caller {caller_id}")
        query=f"number={incident_number}^caller_id={caller_id}"
        try:
            # Use the internal method which raises NotFoundError
            incident_data = await self.table_handler.table_request(
                ServiceNowTable.INCIDENT,
                HTTPMethod.GET,
                query,
                display_value=True,
                limit=1
            )
            if not incident_data or (isinstance(incident_data,list) and not incident_data):
                 raise ServiceNowNotFoundError(f"No incident found with number {incident_number} and caller {caller_id}")

            result = incident_data[0] if isinstance(incident_data,list) else incident_data
            result["link"]=self._generate_record_link(ServiceNowTable.INCIDENT,self._safe_extract_field(result.get("sys_id"),'value'))
            return ResponseHandler.success(data=result)
        except ServiceNowNotFoundError as e:
            # Catch the specific error for this case
            return ResponseHandler.not_found(str(e))
        except ServiceNowError as e:
            # Handle other potential SN errors
            return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)

    @tracer.start_as_current_span("sparc_service_now.get_tickets_by_states_and_caller_id") # Original name
    async def get_tickets_by_states_and_caller_id(
        self,
        states: Union[str, List[str]], # Can be "All", a single state name, or a list of state names
        user_if_field_key: str, # e.g., "caller_id" or "opened_by"
        user_if_field_value: str, # The sys_id of the user
        limit: int=5,
        exclude_states: Optional[List[str]]=None, # List of state names to exclude
        offset: Optional[int]=None,
        fields: Optional[List[str]]=None,
        display_value: bool=True
    ) -> Dict[str, Any]: # Return type is the ResponseHandler dict
        """
        Get incidents, including total count if available via X-Total-Count header.
        Returns a ResponseHandler dict potentially containing 'total_count'.
        """
        log.info(f"Fetching incidents for {user_if_field_key}={user_if_field_value} [States: {states}, Exclude: {exclude_states}, Limit: {limit}, Offset: {offset}]")
        # (Keep existing logic for building query_parts and final_query)
        query_parts=[f"{user_if_field_key}={user_if_field_value}"]
        state_codes_include=set(); state_codes_exclude=set()
        # ... (rest of query building logic is unchanged) ...
        if states != "All":
            state_list=states.split(",") if isinstance(states,str) else states
            if not isinstance(state_list, list): return ResponseHandler.bad_request("Invalid 'states' parameter type.")
            for state_name in state_list:
                if not isinstance(state_name, str): log.warning(f"Ignoring non-string state in 'states': {state_name}"); continue
                try: state_codes_include.add(self.state_mapper.get_incident_state_code(state_name.strip()))
                except ValueError as e: log.warning(f"Ignoring invalid state name in 'states': {e}")
            if state_codes_include:
                # Use the IN operator instead of multiple OR conditions
                if len(state_codes_include) > 1:
                    state_query = f"incident_stateIN{','.join(state_codes_include)}"
                else:
                    state_query = f"incident_state={next(iter(state_codes_include))}"
                query_parts.append(state_query)
            elif state_list: log.warning("Empty or invalid state list provided for inclusion.")
            else: return ResponseHandler.bad_request("Invalid 'states' parameter provided.")
        if exclude_states:
            if not isinstance(exclude_states, list): return ResponseHandler.bad_request("Invalid 'exclude_states' parameter type.")
            for state_name in exclude_states:
                if not isinstance(state_name, str): log.warning(f"Ignoring non-string state in 'exclude_states': {state_name}"); continue
                try: state_codes_exclude.add(self.state_mapper.get_incident_state_code(state_name.strip()))
                except ValueError as e: log.warning(f"Ignoring invalid state name in 'exclude_states': {e}")
            for code in state_codes_exclude: query_parts.append(f"incident_state!={code}")
        query_parts.append("ORDERBYDESCsys_updated_on")
        final_query="^".join(query_parts)
        # --- End of query building ---
        log.debug(f"Constructed incident query: {final_query}")

        if fields is None: # Define default fields
            fields=["sys_id","number","short_description","description","state","incident_state",
                    "sys_created_on","sys_updated_on","priority","urgency","assigned_to",
                    "caller_id","category","subcategory","close_code"]
        try:
            # table_request now returns {"records": [], "total_count": num_or_none}
            table_response = await self.table_handler.table_request(
                table=ServiceNowTable.INCIDENT, method=HTTPMethod.GET, query=final_query,
                limit=limit, offset=offset, fields=fields, display_value=display_value
            )

            incidents = table_response.get("records", [])
            total_count = table_response.get("total_count") # This can be None

            # Add links to each incident
            for inc in incidents:
                 if sys_id := self._safe_extract_field(inc.get("sys_id"),'value'):
                       inc["link"]=self._generate_record_link(ServiceNowTable.INCIDENT,sys_id)

            # Return ResponseHandler format, adding total_count if available
            success_response = ResponseHandler.success(data=incidents)
            if total_count is not None:
                success_response["total_count"] = total_count
            return success_response

        except ServiceNowError as e:
            log.exception(f"Error fetching incidents: {e}")
            return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
        
    @tracer.start_as_current_span("sparc_service_now.get_tickets_by_states_and_caller_id_pagination")
    async def get_tickets_by_states_and_caller_id_pagination(
        self,
        states: Union[str, List[str]],
        user_if_field_key: str,
        user_if_field_value: str,
        limit: int=5,
        offset: int=0,
        exclude_states: Optional[List[str]]=None
    ) -> Dict[str, Any]:
        """Paginated version. Returns ResponseHandler dict with total_count if available."""
        fields=["sys_id","number","short_description","description","state","incident_state","sys_created_on","sys_updated_on",
                "priority","urgency","assigned_to","caller_id","category","subcategory","close_code"]

        # Call the main method which now might return total_count
        response = await self.get_tickets_by_states_and_caller_id(
            states=states, user_if_field_key=user_if_field_key, user_if_field_value=user_if_field_value,
            limit=limit, offset=offset, exclude_states=exclude_states, fields=fields, display_value=True
        )

        # Add pagination metadata. The total_count should already be in 'response' if found.
        if response.get("code")==200:
            response["limit"] = limit
            response["offset"] = offset
            # No need to calculate total_count here, it comes from the underlying call.
            # If total_count wasn't found, the key will just be missing or None.
            if "total_count" not in response:
                 log.warning("Total count not available from API response (X-Total-Count header likely missing or not passed by gateway).")
                 response["total_count"] = -1 # Indicate unavailable total count

        return response

    # --- Attachment Operations --- #
    @tracer.start_as_current_span("sparc_service_now.get_incident_attachments")
    async def get_incident_attachments(self, incident_number: str) -> Dict[str, Any]:
        """Gets all attachments for a given incident number. Returns ResponseHandler dict."""
        log.info(f"Getting attachments for incident {incident_number}")
        try:
             # First, get the incident sys_id from its number
             # Use the internal method that returns raw data or raises NotFoundError
             incident_details = await self.get_incident_by_number_internal(incident_number)
             
             # Parse the incident details properly
             if isinstance(incident_details, dict) and 'records' in incident_details:
                 # Extract the first record if available
                 if incident_details['records'] and len(incident_details['records']) > 0:
                     incident_record = incident_details['records'][0]
                     incident_sys_id = incident_record.get('sys_id')
                     
                     if not incident_sys_id:
                         raise ServiceNowError("Could not find sys_id in incident record.", details=incident_details)
                 else:
                     raise ServiceNowError("No incident records found in response.", details=incident_details)
             else:
                 # Fall back to the original extraction method if structure is different
                 incident_sys_id = self._safe_extract_field(incident_details.get("sys_id"), 'value')
                 
                 if not incident_sys_id:
                     raise ServiceNowError("Could not extract sys_id from incident details.", details=incident_details)

             # Use the attachment handler to get attachments
             attachments = await self.attachment_handler.get_attachments(
                 ServiceNowTable.INCIDENT, incident_sys_id
             )
             
             # Add incident number for context (optional)
             for att in attachments: 
                 att["incident_number"] = incident_number
                 
             return ResponseHandler.success(data=attachments, message=f"Retrieved {len(attachments)} attachments")
        except ServiceNowNotFoundError as e:
             # Incident number not found
             return ResponseHandler.not_found(f"Incident {incident_number} not found.")
        except ServiceNowError as e:
             # Other SN errors (e.g., fetching attachments failed)
             return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
    @tracer.start_as_current_span("sparc_service_now.get_incident_attachments_pagination")
    async def get_incident_attachments_pagination(self, incident_number: str, limit: int=5, offset: int=0) -> Dict[str, Any]:
        """Gets paginated attachments for an incident. Returns ResponseHandler dict with pagination info."""
        log.info(f"Getting attachments page for incident {incident_number} (limit={limit}, offset={offset})")
        try:
            # First, get the incident sys_id from its number
            # Use the internal method that returns raw data or raises NotFoundError
            incident_details = await self.get_incident_by_number_internal(incident_number)
            
            # Parse the incident details properly
            if isinstance(incident_details, dict) and 'records' in incident_details:
                # Extract the first record if available
                if incident_details['records'] and len(incident_details['records']) > 0:
                    incident_record = incident_details['records'][0]
                    incident_sys_id = incident_record.get('sys_id')
                    
                    if not incident_sys_id:
                        raise ServiceNowError("Could not find sys_id in incident record.", details=incident_details)
                else:
                    raise ServiceNowError("No incident records found in response.", details=incident_details)
            else:
                # Fall back to the original extraction method if structure is different
                incident_sys_id = self._safe_extract_field(incident_details.get("sys_id"), 'value')
                
                if not incident_sys_id:
                    raise ServiceNowError("Could not extract sys_id from incident details.", details=incident_details)

            # Fetch the specific page of attachments
            paginated_attachments = await self.attachment_handler.get_attachments(
                ServiceNowTable.INCIDENT, incident_sys_id, limit=limit, offset=offset
            )

            # Add incident number for context
            for att in paginated_attachments:
                att["incident_number"] = incident_number

            # Prepare response with pagination metadata
            return ResponseHandler.success(
                data=paginated_attachments, 
                message=f"Retrieved {len(paginated_attachments)} attachments (page {offset//limit + 1})"
            )
        except ServiceNowNotFoundError as e:
            # Incident number not found
            return ResponseHandler.not_found(f"Incident {incident_number} not found.")
        except ServiceNowError as e:
            # Other SN errors (e.g., fetching attachments failed)
            return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
    @tracer.start_as_current_span("sparc_service_now.get_ritm_attachments")
    async def get_ritm_attachments(self, ritm_number: str) -> Dict[str, Any]:
        """Gets all attachments for a given RITM number. Returns ResponseHandler dict."""
        log.info(f"Getting attachments for RITM {ritm_number}")
        try:
            # Get RITM sys_id
            # get_ritm_details now returns raw data or raises NotFoundError
            ritm_details = await self.get_ritm_details(ritm_number)
            ritm_sys_id = self._safe_extract_field(ritm_details.get("sys_id"),'value')
            if not ritm_sys_id:
                raise ServiceNowError("Could not get sys_id from RITM response.", details=ritm_details)

            # Get attachments
            attachments = await self.attachment_handler.get_attachments(ServiceNowTable.REQ_ITEM, ritm_sys_id)
            # Add RITM number for context
            for att in attachments: att["ritm_number"] = ritm_number
            return ResponseHandler.success(data=attachments, message=f"Retrieved {len(attachments)} attachments")
        except ServiceNowNotFoundError as e:
            return ResponseHandler.not_found(f"RITM {ritm_number} not found.")
        except ServiceNowError as e:
            return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)

    # --- RITM / Request / CHG / Catalog Operations --- #
    @tracer.start_as_current_span("servicenow_client.get_ritm_details")
    async def get_ritm_details(self, ritm_number: str, fields: Optional[List[str]]=None, display_value: bool=True) -> Dict[str, Any]:
        """Gets details for a RITM by number. Returns raw data dict or raises NotFoundError."""
        log.info(f"Getting details for RITM {ritm_number}")
        ritm_data = await self.table_handler.table_request(
            ServiceNowTable.REQ_ITEM,
            HTTPMethod.GET,
            f"number={ritm_number}",
            fields=fields,
            display_value=display_value,
            limit=1
        )
        if not ritm_data or (isinstance(ritm_data,list) and not ritm_data):
            raise ServiceNowNotFoundError(f"RITM {ritm_number} not found.")
        result = ritm_data[0] if isinstance(ritm_data,list) else ritm_data;
        # Add link
        if sys_id := self._safe_extract_field(result.get("sys_id"),'value'):
            result["link"]=self._generate_record_link(ServiceNowTable.REQ_ITEM,sys_id)
        return result
    @tracer.start_as_current_span("sparc_service_now.get_change_request_by_number")
    async def get_change_request_by_number(self, change_request_number: str) -> Dict[str, Any]:
        """Gets details for a Change Request by number. Returns ResponseHandler dict."""
        log.info(f"Getting details for Change Request {change_request_number}")
        try:
            chg_data = await self.table_handler.table_request(
                ServiceNowTable.CHANGE_REQUEST,
                HTTPMethod.GET,
                f"number={change_request_number}",
                display_value=True,
                limit=1
            )
            if not chg_data or (isinstance(chg_data, list) and not chg_data):
                 raise ServiceNowNotFoundError(f"Change Request {change_request_number} not found.")
            result = chg_data[0] if isinstance(chg_data, list) else chg_data
            # Add link
            if sys_id := self._safe_extract_field(result.get("sys_id"),'value'):
                 result["link"] = self._generate_record_link(ServiceNowTable.CHANGE_REQUEST, sys_id)
            return ResponseHandler.success(data=result)
        except ServiceNowNotFoundError as e: return ResponseHandler.not_found(str(e))
        except ServiceNowError as e: return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)

    @tracer.start_as_current_span("servicenow_client.get_catalog_item_variables")
    async def get_catalog_item_variables(self, ritm_number: str) -> List[Dict[str, Any]]:
        """Gets the variables and their values for a specific RITM. Returns a list of variable dicts."""
        log.info(f"Getting variables for RITM {ritm_number}")
        # 1. Get RITM sys_id and catalog item sys_id
        # get_ritm_details raises NotFoundError if RITM doesn't exist
        ritm=await self.get_ritm_details(ritm_number,fields=["sys_id","cat_item"], display_value=False) # Need raw values
        ritm_sys_id=self._safe_extract_field(ritm.get("sys_id"),'value');
        cat_item_sys_id=self._safe_extract_field(ritm.get("cat_item"),'value')
        if not ritm_sys_id: raise ServiceNowError(f"Could not get RITM sys_id for {ritm_number}")
        if not cat_item_sys_id: raise ServiceNowError(f"Could not get Catalog Item sys_id for {ritm_number}")

        # 2. Get variable definitions (item_option_new) for the catalog item
        var_defs_data=await self.table_handler.table_request(
            ServiceNowTable.ITEM_OPTION, HTTPMethod.GET,
            f"cat_item={cat_item_sys_id}^active=true^ORDERBYorder", # Query by cat item, active only, order
            fields=["sys_id","name","type","question_text","label","help_text","mandatory","order"],
            display_value=False # Need raw values for mapping keys
        )
        # Create a map of variable definition sys_id -> definition details
        var_defs={self._safe_extract_field(vd.get('sys_id'),'value'):vd for vd in var_defs_data} if var_defs_data else {}

        # 3. Get submitted variable values (sc_item_option) for the specific RITM
        var_values_data=await self.table_handler.table_request(
            ServiceNowTable.SC_ITEM_OPTION, HTTPMethod.GET,
            f"request_item={ritm_sys_id}", # Query by RITM sys_id
            fields=["item_option_new","value"], # Link to definition and the value
            display_value=True # Get display value for the 'value' field
        )
        # Create a map of variable definition sys_id -> submitted value (prefer display value)
        values_map = {}
        if var_values_data:
            for v in var_values_data:
                 var_def_link = self._safe_extract_field(v.get("item_option_new"),'value')
                 if var_def_link:
                      # Prioritize display_value, fallback to value, then None
                      value_field = v.get("value")
                      values_map[var_def_link] = self._safe_extract_field(value_field, 'display_value', self._safe_extract_field(value_field, 'value'))


        # 4. Combine definition and value
        formatted_vars=[]
        for var_sys_id, var_def in var_defs.items():
             # Extract definition details (use raw values where appropriate)
             q_text = self._safe_extract_field(var_def.get("question_text"),'value')
             label = self._safe_extract_field(var_def.get("label"),'value')
             formatted_vars.append({
                 "sys_id":var_sys_id,
                 "name":self._safe_extract_field(var_def.get("name"),'value'),
                 "label": q_text or label, # Prefer question_text over label
                 "type":self._safe_extract_field(var_def.get("type"),'value'),
                 "mandatory":self._safe_extract_field(var_def.get("mandatory"),'value')=='true', # Convert 'true'/'false' string
                 "help_text":self._safe_extract_field(var_def.get("help_text"),'value'),
                 "order":int(self._safe_extract_field(var_def.get("order"),'value',0)), # Convert order string to int
                 "value":values_map.get(var_sys_id) # Get the submitted value from the map
             })

        # 5. Sort by order (already fetched in order, but belt-and-suspenders)
        formatted_vars.sort(key=lambda x:x["order"]);
        return formatted_vars

    @tracer.start_as_current_span("sparc_service_now.get_ritm_requests_for_user") # Original name
    async def get_ritm_requests_for_user(
        self,
        user_sys_id: str,
        limit: int=50, # Default limit from original
        states: str="All", # Accepts "All", single string, or comma-separated string of state *names*
        exclude_states: Optional[List[str]]=None # List of state *names* to exclude
    ) -> Dict[str, Any]: # Original signature returning wrapper dict
        """
        Fetches RITM requests where user is requester or approver, matching original format.
        Uses state code mapping for filtering based on input state names.
        Returns a ResponseHandler dict.
        """
        log.info(f"Fetching RITM requests for user {user_sys_id} [States: {states}, Exclude: {exclude_states}, Limit: {limit}]")
        try:
            ritm_map:Dict[str,Dict]={} # Store RITMs by sys_id to merge roles
            ritm_fields = ["sys_id","number","short_description","state","stage","sys_updated_on","requested_for", "cat_item"] # Added cat_item for context

            # 1. Fetch RITMs where user is the 'requested_for'
            # Use _fetch_all_pages to handle potential pagination internally if needed for large results before filtering/limiting
            req_ritms=await self._fetch_all_pages(
                ServiceNowTable.REQ_ITEM,
                f"requested_for={user_sys_id}^ORDERBYDESCsys_updated_on",
                ritm_fields
            )
            for ritm in req_ritms:
                if sys_id := self._safe_extract_field(ritm.get("sys_id"),'value'):
                    ritm['user_role']='requester'
                    # Store display values for later use
                    ritm['state_display'] = self._safe_extract_field(ritm.get("state"),'display_value')
                    ritm['state_value'] = self._safe_extract_field(ritm.get("state"),'value')
                    ritm_map[sys_id]=ritm

            # 2. Fetch Approvals for the user linked to RITMs
            approval_fields = ["sysapproval", "state"] # Only need link to RITM and approval state
            # Query approvals for the user where the target record is a RITM
            approvals=await self._fetch_all_pages(
                 ServiceNowTable.SYS_APPROVAL,
                 f"approver={user_sys_id}^sysapproval.sys_class_name=sc_req_item^ORDERBYDESCsys_created_on", # Filter for RITM approvals
                 approval_fields
            )

            approval_states={}; # Map RITM sys_id -> User's approval state display value for that RITM
            ritm_ids_via_approval=set()
            for app in approvals:
                if ritm_sys_id := self._safe_extract_field(app.get("sysapproval"),'value'):
                    ritm_ids_via_approval.add(ritm_sys_id)
                    # Store the approval state display value for this user/RITM pair
                    approval_states[ritm_sys_id]=self._safe_extract_field(app.get("state"),'display_value',self._safe_extract_field(app.get("state"),'value'))

            # 3. Fetch RITM details if user is only approver (not requester) and RITM not already fetched
            ids_to_fetch = ritm_ids_via_approval - set(ritm_map.keys())
            if ids_to_fetch:
                # Fetch details for RITMs found only via approvals
                approved_ritms=await self._fetch_all_pages(
                    ServiceNowTable.REQ_ITEM,
                    f"sys_idIN{','.join(ids_to_fetch)}", # Query by list of sys_ids
                    ritm_fields
                )
                for ritm in approved_ritms:
                    if sys_id := self._safe_extract_field(ritm.get("sys_id"),'value'):
                        ritm['user_role']='approver'
                         # Store display values for later use
                        ritm['state_display'] = self._safe_extract_field(ritm.get("state"),'display_value')
                        ritm['state_value'] = self._safe_extract_field(ritm.get("state"),'value')
                        ritm_map[sys_id]=ritm

            # 4. Combine roles & add user's approval state if they are an approver
            for sys_id, state in approval_states.items():
                if sys_id in ritm_map:
                    # Update role if user is both requester and approver
                    ritm_map[sys_id]['user_role'] = 'requester+approver' if ritm_map[sys_id]['user_role'] == 'requester' else 'approver'
                    # Add the user's specific approval state for this RITM
                    ritm_map[sys_id]['user_approval_state'] = state

            # 5. Filter by state (using RITM state *codes*)
            all_user_ritms=list(ritm_map.values())
            final_list = all_user_ritms # Start with all RITMs associated with the user

            if states != "All" or exclude_states:
                filtered_list=[]; include_codes=set(); exclude_codes=set()

                # Parse 'states' input (string of names, comma-separated or single)
                state_list_names = []
                if isinstance(states, str) and states != "All":
                     state_list_names = [s.strip() for s in states.split(",") if s.strip()] # Handle potential empty strings

                # Get codes for states to include
                if states!="All":
                    if not state_list_names: # states was not 'All' but empty after split/strip
                         log.warning("Empty state list provided for inclusion. Interpreting as 'fetch all non-excluded'.")
                         # include_codes remains empty, so inclusion check passes unless excluded
                    else:
                        for s_name in state_list_names:
                            try: include_codes.add(self.state_mapper.get_ritm_state_code(s_name))
                            except ValueError as e: log.warning(f"Ignoring invalid RITM state name for inclusion: {e}")
                        if not include_codes:
                             log.warning("No valid RITM state names provided for inclusion. Fetching all non-excluded.")
                             # include_codes remains empty

                # Get codes for states to exclude
                if exclude_states:
                    for s_name in exclude_states:
                        try: exclude_codes.add(self.state_mapper.get_ritm_state_code(s_name))
                        except ValueError as e: log.warning(f"Ignoring invalid RITM state name for exclusion: {e}")

                # Perform the filtering
                for ritm in all_user_ritms:
                    # Compare against the RITM's state *code* (value)
                    code = ritm.get('state_value') # Use pre-fetched value
                    display = ritm.get('state_display', code) # Use pre-fetched display

                    # Determine if it should be included based on include_codes (pass if include_codes is empty)
                    include_check = (not include_codes) or (code in include_codes)
                    # Determine if it should be excluded based on exclude_codes
                    exclude_check = code in exclude_codes

                    # Keep the RITM if it passes the include check AND fails the exclude check
                    if include_check and not exclude_check:
                        filtered_list.append(ritm)
                    else:
                         log.debug(f"Filtering out RITM: Num={self._safe_extract_field(ritm.get('number'),'display_value')}, StateCode={code}, StateDisplay='{display}', Include?={include_check}, Exclude?={exclude_check}")

                final_list = filtered_list # Update final_list with filtered results

            # 6. Sort the final list by update date (descending)
            # Sorting is done *after* filtering to ensure the limit applies to the correct set
            final_list.sort(key=lambda x: self._safe_extract_field(x.get("sys_updated_on"), 'value', ""), reverse=True)

            # 7. Apply the limit
            final_list = final_list[:limit]

            # 8. Add Links & Placeholder additional details (matching original structure)
            for ritm in final_list:
                if sys_id := self._safe_extract_field(ritm.get("sys_id"),'value'):
                    ritm["link"]=self._generate_record_link(ServiceNowTable.REQ_ITEM,sys_id)
                # Keep structure but don't populate fully here for performance (matching original)
                # These would require additional queries per RITM if needed
                ritm["additional_details"] = {"approvers": [], "approval_info": []}

            # Return original wrapper format
            base_response = ResponseHandler.success(data=final_list, message=f"Retrieved {len(final_list)} RITMs.")
            # Use dictionary unpacking for merging pagination info
            return {**base_response, "total_count":len(final_list),"limit":limit} # total_count is count after filtering/limit
        except ServiceNowError as e:
            # Handle known SN errors during fetching
            return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
        except Exception as e: # Catch unexpected errors during processing/filtering
            log.exception("Unexpected error in get_ritm_requests_for_user")
            return ResponseHandler.internal_error(f"Unexpected error: {str(e)}")
    
    @tracer.start_as_current_span("sparc_service_now.get_ritm_requests_for_user_pagination")
    async def get_ritm_requests_for_user_pagination(
        self,
        user_sys_id: str,
        limit: int = 5,
        offset: int = 0,
        states: Union[str, List[str]] = "All", # Accepts "All", single string, or list of state *names*
        exclude_states: Optional[List[str]] = None # List of state *names* to exclude
    ) -> Dict[str, Any]:
        """
        Fetches a paginated list of RITM requests where the user is the 'requested_for'.
        Includes the list of all approvers for each RITM on the page.
        Attempts to get the total count for this specific query via X-Total-Count header.
        It also checks if the user is an approver for the RITMs on the current page.

        Args:
            user_sys_id: The sys_id of the user.
            limit: Number of records per page.
            offset: Starting record index.
            states: State names to include ("All", "Open", ["Open", "Pending"]).
            exclude_states: State names to exclude (e.g., ["Closed Complete"]).

        Returns:
            A ResponseHandler dictionary containing 'data' (list of RITMs for the page,
            each including its approvers) and potentially 'total_count'
            (int, count for the primary 'requested_for' query).
        """
        log.info(f"Fetching paginated RITM requests for user {user_sys_id} [States: {states}, Exclude: {exclude_states}, Limit: {limit}, Offset: {offset}]")

        # Helper function within the method for robust sys_id extraction from reference field
        def get_reference_sys_id(ref_field_data):
            # Input 'ref_field_data' is expected to be the dictionary for a reference field,
            # e.g., {'display_value': 'Some User', 'link': '.../sys_id'}
            if not isinstance(ref_field_data, dict):
                # log.debug(f"get_reference_sys_id: Input is not a dict: {ref_field_data}")
                return None
            # 1. Prefer explicit 'value' if present (might contain sys_id)
            sys_id_val = self._safe_extract_field(ref_field_data, 'value')
            if sys_id_val and isinstance(sys_id_val, str) and len(sys_id_val) == 32:
                # log.debug(f"get_reference_sys_id: Extracted from 'value': {sys_id_val}")
                return sys_id_val
            # 2. Fallback: Try parsing sys_id from 'link'
            link = ref_field_data.get('link')
            if link and isinstance(link, str):
                 try:
                     # Assumes sys_id is the last part of the URL path after the last '/'
                     extracted_id = link.strip('/').split('/')[-1]
                     if len(extracted_id) == 32:
                          # log.debug(f"get_reference_sys_id: Extracted from 'link': {extracted_id}")
                          return extracted_id
                     # else:
                          # log.debug(f"get_reference_sys_id: Parsed segment from link is not 32 chars: {extracted_id}")
                 except Exception as e:
                      log.debug(f"get_reference_sys_id: Error parsing link '{link}': {e}")
                      pass # Ignore parsing errors
            # 3. If neither worked
            # log.warning(f"Could not reliably extract sys_id from reference field data: {ref_field_data}")
            return None
        # --- End Helper ---

        try:
            # --- 1. Build Query for sc_req_item table ---
            # ... (Query building logic - unchanged) ...
            query_parts = [f"requested_for={user_sys_id}"]
            state_codes_include = set()
            state_codes_exclude = set()

            if states != "All":
                state_list_names = states.split(",") if isinstance(states, str) else states
                if not isinstance(state_list_names, list):
                    return ResponseHandler.bad_request("Invalid 'states' parameter type.")
                valid_state_names = [s.strip() for s in state_list_names if isinstance(s, str) and s.strip()]
                if not valid_state_names: log.warning("Empty or invalid state list provided for RITM inclusion.")
                else:
                    for s_name in valid_state_names:
                        try: state_codes_include.add(self.state_mapper.get_ritm_state_code(s_name))
                        except ValueError as e: log.warning(f"Ignoring invalid RITM state name for inclusion: {e}")
                    if state_codes_include:
                        state_query = f"stateIN{','.join(state_codes_include)}" if len(state_codes_include) > 1 else f"state={next(iter(state_codes_include))}"
                        query_parts.append(state_query)
                    else: log.warning("No valid RITM state codes found for inclusion query.")

            if exclude_states:
                if not isinstance(exclude_states, list): return ResponseHandler.bad_request("Invalid 'exclude_states' parameter type.")
                valid_exclude_names = [s.strip() for s in exclude_states if isinstance(s, str) and s.strip()]
                for s_name in valid_exclude_names:
                     try: state_codes_exclude.add(self.state_mapper.get_ritm_state_code(s_name))
                     except ValueError as e: log.warning(f"Ignoring invalid RITM state name for exclusion: {e}")
                for code in state_codes_exclude: query_parts.append(f"state!={code}")

            query_parts.append("ORDERBYDESCsys_updated_on")
            final_ritm_query = "^".join(query_parts)
            log.debug(f"Constructed RITM query: {final_ritm_query}")

            ritm_fields = ["sys_id", "number", "short_description", "state", "stage",
                           "sys_updated_on", "requested_for", "cat_item", "price", "quantity",
                           "sys_created_on", "opened_by", "approval"]

            # --- 2. Fetch Paginated RITM Data (Requested For) ---
            table_response = await self.table_handler.table_request(
                table=ServiceNowTable.REQ_ITEM, method=HTTPMethod.GET, query=final_ritm_query,
                fields=ritm_fields, limit=limit, offset=offset, display_value=True
            )
            ritms_on_page = table_response.get("records", [])
            requester_total_count = table_response.get("total_count")
            log.info(f"Fetched {len(ritms_on_page)} RITMs for page. Total count for query: {requester_total_count}")
            # log.critical(f"DEBUG - First RITM data sample: {ritms_on_page[0] if ritms_on_page else 'No RITMs found'}") # Optional debug

            if not ritms_on_page:
                 success_response = ResponseHandler.success(data=[], message="No RITMs found for this page.")
                 success_response["total_count"] = requester_total_count if requester_total_count is not None else 0
                 success_response["limit"] = limit
                 success_response["offset"] = offset
                 return success_response

            # --- 3a. Fetch Approval Information (Current User Check) ---
            ritm_ids_on_page = {self._safe_extract_field(r.get("sys_id"), 'value') for r in ritms_on_page if self._safe_extract_field(r.get("sys_id"), 'value')}
            current_user_approval_map = {}
            if ritm_ids_on_page:
                # ... (Logic for checking current user's approval status - unchanged) ...
                approval_fields_user_check = ["sysapproval", "state"]
                approval_query_user_check = f"approver={user_sys_id}^sysapprovalIN{','.join(ritm_ids_on_page)}"
                # log.debug(f"Checking current user's approval status query: {approval_query_user_check}") # Optional debug
                try:
                    approval_response_user_check = await self.table_handler.table_request(
                        table=ServiceNowTable.SYS_APPROVAL, method=HTTPMethod.GET, query=approval_query_user_check,
                        fields=approval_fields_user_check, limit=len(ritm_ids_on_page), display_value=True
                    )
                    approvals_found_user_check = approval_response_user_check.get("records", [])
                    # log.debug(f"Current user approval check response: Found {len(approvals_found_user_check)} records") # Optional debug
                    for app in approvals_found_user_check:
                         ritm_sys_id_from_approval = get_reference_sys_id(app.get("sysapproval")) # Use helper
                         if ritm_sys_id_from_approval:
                             current_user_approval_map[ritm_sys_id_from_approval] = self._safe_extract_field(app.get("state"), 'display_value', 'Unknown State')
                         # else: log.warning(f"Could not get RITM sys_id from user approval check record: {app}") # Optional debug
                    log.debug(f"Found {len(current_user_approval_map)} approvals involving current user on this page.")
                except ServiceNowNotFoundError: log.debug("Current user is not an approver for any RITMs on this page.")
                except ServiceNowError as e: log.warning(f"Could not fetch current user's approval details for page RITMs: {e}.")

            # --- 3b. Fetch ALL Approvers (for RITMs on current page) ---
            all_approvers_by_ritm_id = collections.defaultdict(list)
            if ritm_ids_on_page:
                approver_fields_all = [
                    "sys_id", "approver", "state", "comments",
                    "sysapproval", "sysapproval.sys_id", # Request dot-walked sys_id
                    "sys_created_on", "sys_updated_on", "source_table"
                ]
                all_approvers_query = f"sysapprovalIN{','.join(ritm_ids_on_page)}"
                log.debug(f"Fetching all approvers for RITMs on page query: {all_approvers_query}")
                try:
                    all_approvers_response = await self.table_handler.table_request(
                        table=ServiceNowTable.SYS_APPROVAL, method=HTTPMethod.GET, query=all_approvers_query,
                        fields=approver_fields_all, display_value=True
                    )
                    all_approvers_found = all_approvers_response.get("records", [])
                    log.info(f"Fetched {len(all_approvers_found)} total approval records for RITMs on this page.")
                    # log.critical(f"DEBUG - First raw approver sample: {all_approvers_found[0] if all_approvers_found else 'None'}") # Optional debug

                    # Group approvers by the RITM they belong to
                    grouped_count = 0
                    skipped_count = 0
                    for app_idx, app in enumerate(all_approvers_found):
                        # Robustly get the sys_id of the record being approved (the RITM)
                        # Try dot-walked field first (if API returned it)
                        ritm_sys_id_key = self._safe_extract_field(app.get("sysapproval.sys_id"), 'value')

                        if not ritm_sys_id_key or len(ritm_sys_id_key) != 32:
                             # Fallback if dot-walking didn't work or wasn't returned
                             ritm_sys_id_key = get_reference_sys_id(app.get("sysapproval"))

                        if ritm_sys_id_key:
                             # Ensure the key is one of the RITMs we are processing
                             if ritm_sys_id_key in ritm_ids_on_page:
                                 all_approvers_by_ritm_id[ritm_sys_id_key].append(app)
                                 grouped_count += 1
                             # else: log.warning(f"Approver record {self._safe_extract_field(app.get('sys_id'), 'value')} references RITM {ritm_sys_id_key} which is not on the current page. Ignoring.") # Should not happen with IN query
                        else:
                             skipped_count += 1
                             log.warning(f"Could not determine RITM sys_id for approval record index {app_idx} (sys_id: {self._safe_extract_field(app.get('sys_id'), 'value')}). Skipping grouping.")
                    log.info(f"Successfully grouped {grouped_count} approval records by RITM sys_id. Skipped {skipped_count}.")

                except ServiceNowNotFoundError: log.debug("No approval records found at all for any RITMs on this page.")
                except ServiceNowError as e: log.warning(f"Could not fetch all approver details for page RITMs: {e}. Approver lists may be empty.")

            # --- 4. Process RITMs for the Current Page ---
            processed_ritms = []
            for ritm_idx, ritm in enumerate(ritms_on_page):
                ritm_sys_id = self._safe_extract_field(ritm.get("sys_id"), 'value')
                if not ritm_sys_id:
                    log.warning(f"Skipping RITM at index {ritm_idx} due to missing sys_id.")
                    continue

                ritm['user_role'] = 'requester'
                ritm['state_display'] = self._safe_extract_field(ritm.get("state"),'display_value')
                ritm['state_value'] = self._safe_extract_field(ritm.get("state"),'value')
                ritm['approval_display'] = self._safe_extract_field(ritm.get("approval"), 'display_value')
                ritm['user_approval_state'] = current_user_approval_map.get(ritm_sys_id) # Get from map or None
                if ritm['user_approval_state']:
                    ritm['user_role'] = 'requester+approver'

                ritm["link"] = self._generate_record_link(ServiceNowTable.REQ_ITEM, ritm_sys_id)

                # --- Format and Add Approvers List ---
                approvers_for_this_ritm = all_approvers_by_ritm_id.get(ritm_sys_id, []) # Get list from grouped dict
                formatted_approvers = []
                if approvers_for_this_ritm:
                     log.debug(f"Formatting {len(approvers_for_this_ritm)} approvers for RITM {ritm_sys_id}")
                     formatted_approvers = [
                         {
                             "sys_id": self._safe_extract_field(app.get("sys_id"), 'value'),
                             "approver": self._safe_extract_field(app.get("approver"), 'display_value'),
                             "approver_sys_id": get_reference_sys_id(app.get("approver")), # Use helper for approver sys_id too
                             "state": self._safe_extract_field(app.get("state"), 'display_value'),
                             "comments": self._safe_extract_field(app.get("comments"), 'value'),
                             "source_record_sys_id": self._safe_extract_field(app.get("sysapproval.sys_id"), 'value') or get_reference_sys_id(app.get("sysapproval")), # Ensure we have the source sys_id
                             "source_record_table": self._safe_extract_field(app.get("source_table"), 'value'), # Usually sc_req_item here
                             "created_on": self._safe_extract_field(app.get("sys_created_on"), 'display_value'),
                             "updated_on": self._safe_extract_field(app.get("sys_updated_on"), 'display_value'),
                         } for app in approvers_for_this_ritm
                     ]
                     formatted_approvers.sort(key=lambda x: x.get("created_on", ""), reverse=True)
                # else: log.debug(f"No approvers found in grouped data for RITM {ritm_sys_id}") # Optional debug

                ritm["additional_details"] = {
                    "approvers": formatted_approvers,
                    "approval_info": formatted_approvers
                }
                processed_ritms.append(ritm)

            # --- 5. Return Result ---
            success_response = ResponseHandler.success(
                data=processed_ritms,
                message=f"Retrieved {len(processed_ritms)} RITMs for page."
            )
            success_response["total_count"] = requester_total_count if requester_total_count is not None else -1
            success_response["limit"] = limit
            success_response["offset"] = offset

            # log.critical(f"DEBUG - Final Response Keys: {success_response.keys()}") # Optional debug
            # if processed_ritms: log.critical(f"DEBUG - First Processed RITM Approvers: {processed_ritms[0].get('additional_details', {}).get('approvers')}") # Optional debug

            return success_response

        except ServiceNowError as e:
            log.error(f"ServiceNow error fetching RITM page: {e}", exc_info=True)
            return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
        except Exception as e:
            log.exception("Unexpected error in get_ritm_requests_for_user_pagination")
            return ResponseHandler.internal_error(f"Unexpected error: {str(e)}")

    @tracer.start_as_current_span("sparc_service_now.get_request_details_for_approver")
    async def get_request_details_for_approver(self, request_number: str, user_sys_id: str) -> Dict[str, Any]:
        """ Gets details of a RITM or CHG if the user is listed as an approver. Returns ResponseHandler dict. """
        log.info(f"Getting details for {request_number} as approver {user_sys_id}")
        table:Optional[ServiceNowTable] = None;
        if request_number.upper().startswith("RITM"): table=ServiceNowTable.REQ_ITEM
        elif request_number.upper().startswith("CHG"): table=ServiceNowTable.CHANGE_REQUEST
        else: return ResponseHandler.bad_request(f"Unsupported request type prefix (expected RITM or CHG): {request_number}.")


        try:
            # 1. Get main request details
            # Assuming table_handler returns {'records': [...], 'total_count': ...}
            req_data = await self.table_handler.table_request(
                table, HTTPMethod.GET, f"number={request_number}", display_value=True, limit=1
            )

            # --- More Robust Check for Response Structure and Content ---
            if not isinstance(req_data, dict) or \
               not isinstance(req_data.get("records"), list) or \
               not req_data["records"]: # Checks if records is present, is a list, and is not empty

                total_count = req_data.get("total_count") if isinstance(req_data, dict) else None

                if total_count == 0:
                    raise ServiceNowNotFoundError(f"{request_number} not found (total_count is 0).")
                else: # Covers missing 'records', 'records' not a list, or empty list when total_count isn't 0
                    raise ServiceNowNotFoundError(f"{request_number} not found or invalid response structure received.", details=req_data)
            # --- End Robust Check ---

            # If we passed the check, req_data['records'] is a non-empty list
            main_request = req_data['records'][0] # Get the first record dict

            # --- *** CORRECTED SYS_ID EXTRACTION *** ---
            req_sys_id = main_request.get("sys_id") # Directly access the string value
            # --- ************************************* ---


            # Check if sys_id was successfully extracted (should be a string)
            if not req_sys_id or not isinstance(req_sys_id, str):
                 # Raise error if sys_id is missing or not a string after direct access attempt
                 raise ServiceNowError(f"Missing or invalid sys_id in {request_number} response.", details=main_request)

            # 2. Verify user is an approver for *this specific record*
            try:
                # Check if an approval record exists linking this user to this request sys_id
                 approval_check_response = await self.table_handler.table_request(
                    ServiceNowTable.SYS_APPROVAL, HTTPMethod.GET,
                    f"sysapproval={req_sys_id}^approver={user_sys_id}", # Query specific link
                    limit=1, fields=['sys_id'] # Only need to know if it exists
                 )

                 # Check if the 'records' list in the response is non-empty
                 if not isinstance(approval_check_response.get("records"), list) or not approval_check_response["records"]:
                      # No approval record found linking this user and request
                      raise ServiceNowNotFoundError("User is not an approver for this request.") # Use specific error

                 log.info(f"User {user_sys_id} confirmed as approver for {request_number} ({req_sys_id}).")

            except ServiceNowNotFoundError:
                 # This specifically means the approval record wasn't found
                 log.warning(f"Access denied: User {user_sys_id} is not an approver for {request_number} ({req_sys_id}).")
                 # Return a permission error (403) as the user isn't authorized *as an approver*
                 return ResponseHandler.error(403, f"Access denied: User is not an approver for {request_number}.")
            # Let other ServiceNowErrors propagate (e.g., if the query itself failed)

            # 3. Fetch additional details (approvers, attachments, related records)
            additional_details={}; all_approvers=[]; attachments=[]
            # Fetch all approvers for the record
            try:
                 all_approvers_response = await self.table_handler.table_request(
                    ServiceNowTable.SYS_APPROVAL, HTTPMethod.GET,
                    f"sysapproval={req_sys_id}", # Get all approvals for this record
                    display_value=True # Get display names etc.
                 )
                 # Ensure it's a list from the response structure
                 all_approvers_list = all_approvers_response.get("records", []) if isinstance(all_approvers_response, dict) else []

                 # Format approvers slightly (optional, but helpful)
                 additional_details["approvers"] = [
                      {
                          "sys_id": self._safe_extract_field(app.get("sys_id"), 'value'), # Approval record sys_id
                          "approver": self._safe_extract_field(app.get("approver"), 'display_value'),
                          "approver_sys_id": self._safe_extract_field(app.get("approver"), 'value'), # User sys_id
                          "state": self._safe_extract_field(app.get("state"), 'display_value'),
                          "comments": self._safe_extract_field(app.get("comments"), 'value')
                      } for app in all_approvers_list # Iterate the list
                 ]
                 additional_details["approval_info"] = additional_details["approvers"] # Match original structure naming

            except ServiceNowNotFoundError: # Should not happen if request exists, but handle defensively
                log.warning(f"No approval records found when fetching all approvers for {request_number}. List will be empty.")
                additional_details["approvers"] = []
                additional_details["approval_info"] = []
            except ServiceNowError as e:
                log.warning(f"Could not fetch all approvers for {request_number}: {e}. List may be empty.")
                additional_details["approvers"] = []
                additional_details["approval_info"] = []


            # Fetch attachments
            try:
                 # Ensure attachment_handler.get_attachments returns a list
                 attachments_list = await self.attachment_handler.get_attachments(table,req_sys_id)
                 additional_details["attachments"] = attachments_list if isinstance(attachments_list, list) else []
            except ServiceNowError as e:
                log.warning(f"Could not fetch attachments for {request_number}: {e}. List may be empty.")
                additional_details["attachments"] = []


            # Fetch related records if it's a RITM
            if table==ServiceNowTable.REQ_ITEM:
                 # Get parent Request (REQ)
                 parent_req_ref = main_request.get("request") # Might be {'link': ..., 'value': ...}
                 p_ref = self._safe_extract_field(parent_req_ref, 'value') if isinstance(parent_req_ref, dict) else None

                 if p_ref:
                      try:
                          p_data_resp=await self.table_handler.table_request(ServiceNowTable.SC_REQUEST,HTTPMethod.GET,f"sys_id={p_ref}",limit=1)
                          p_records = p_data_resp.get("records", []) if isinstance(p_data_resp, dict) else []
                          additional_details["parent_request"]=p_records[0] if p_records else {}
                      except ServiceNowError as e:
                          log.warning(f"Could not fetch parent REQ {p_ref}: {e}")
                          additional_details["parent_request"] = {} # Ensure key exists but is empty on error
                 else:
                      additional_details["parent_request"] = {}

                 # Get Catalog Item definition
                 cat_item_ref = main_request.get("cat_item") # Might be {'link': ..., 'value': ...}
                 c_ref = self._safe_extract_field(cat_item_ref, 'value') if isinstance(cat_item_ref, dict) else None

                 if c_ref:
                      try:
                           c_data_resp=await self.table_handler.table_request(ServiceNowTable.SC_CAT_ITEM,HTTPMethod.GET,f"sys_id={c_ref}",limit=1)
                           c_records = c_data_resp.get("records", []) if isinstance(c_data_resp, dict) else []
                           additional_details["catalog_item"]=c_records[0] if c_records else {}
                      except ServiceNowError as e:
                           log.warning(f"Could not fetch CatItem {c_ref}: {e}")
                           additional_details["catalog_item"] = {} # Ensure key exists but is empty on error
                 else:
                      additional_details["catalog_item"] = {}


            # Generate link for the main request

            link=self._generate_record_link(table,req_sys_id)


            # Return success with combined data
            return ResponseHandler.success(data={"request":main_request,"additional_details":additional_details,"link":link})

        except ServiceNowNotFoundError as e:
            return ResponseHandler.not_found(str(e)) # Covers main request not found OR user not approver
        except ServiceNowPermissionError as e:
            return ResponseHandler.error(403, e.args[0]) # Should be caught by approver check, but belt-and-suspenders
        except ServiceNowError as e:
            return ResponseHandler.error(e.status_code or 500, e.args[0], e.details) # Other SN errors
        except Exception as e:
            log.exception(f"STEP ERROR: Unexpected Python exception in get_request_details_for_approver") # Log traceback for unexpected
            return ResponseHandler.internal_error(f"Unexpected error: {str(e)}")
        
    @tracer.start_as_current_span("sparc_service_now.get_request_details_by_requester")
    async def get_request_details_by_requester(self, request_number: str, requester_sys_id: str) -> Dict[str, Any]:
        """
        Gets details of a RITM, REQ, or CHG if the user is the requester/requested_by.
        Includes ONLY approvers linked DIRECTLY to the primary record.
        Returns ResponseHandler dict. Uses Apigee/SPARC API endpoints via internal handlers.
        """
        log.info(f"Getting details for {request_number} requested by/for {requester_sys_id} (Direct Approvers Only) via SPARC API")
        table: Optional[ServiceNowTable] = None
        id_field: Optional[str] = None
        req_num_upper = request_number.upper()

        if req_num_upper.startswith("RITM"):
            table = ServiceNowTable.REQ_ITEM
            id_field = "requested_for"
        elif req_num_upper.startswith("REQ"):
            table = ServiceNowTable.SC_REQUEST
            id_field = "requested_for"
        elif req_num_upper.startswith("CHG"):
            table = ServiceNowTable.CHANGE_REQUEST
            id_field = "requested_by"
        else:
            return ResponseHandler.bad_request(f"Unsupported request type prefix (expected RITM, REQ, or CHG): {request_number}.")

        try:
            # 1. Get main request details, filtering by number AND requester field
            log.debug(f"Fetching main record for {request_number} ({table.value}) with {id_field}={requester_sys_id}")
            req_data_response = await self.table_handler.table_request(
                table, HTTPMethod.GET,
                f"number={request_number}^{id_field}={requester_sys_id}", # Combine number and user field in query
                display_value=True, limit=1
            )

            records = req_data_response.get('records', [])
            if not records:
                 raise ServiceNowNotFoundError(f"{request_number} for requester/user {requester_sys_id} not found.")

            main_request = records[0]

            req_sys_id = self._safe_extract_field(main_request.get("sys_id"), 'value')
            if not req_sys_id:
                log.error(f"Missing sys_id in main response for {request_number}.", extra={"response_data": main_request})
                raise ServiceNowError(f"Missing sys_id in {request_number} response.", details=main_request)
            log.debug(f"Found main record sys_id: {req_sys_id}")

            # 2. Fetch additional details
            additional_details = {}
            direct_approvers_list = [] # Only store direct approvers

            # --- Fetch DIRECT Approvers ---
            log.debug(f"Fetching direct approvers for {table.value} {req_sys_id}")
            try:
                direct_approvers_response = await self.table_handler.table_request(
                    ServiceNowTable.SYS_APPROVAL, HTTPMethod.GET,
                    f"sysapproval={req_sys_id}", # Get all approvals for THIS record
                    display_value=True
                )
                direct_approvers = direct_approvers_response.get('records', [])
                if direct_approvers:
                    log.info(f"Found {len(direct_approvers)} direct approver records.")
                    direct_approvers_list.extend(direct_approvers) # Add to our list
                else:
                     log.debug(f"No direct approvers found for {request_number} ({req_sys_id}).")
            except ServiceNowNotFoundError:
                log.debug(f"No direct approvers found (404) for {request_number} ({req_sys_id}).")
            except ServiceNowError as e:
                log.warning(f"Could not fetch direct approvers for {request_number}: {e}")

            # --- Fetch Attachments ---
            log.debug(f"Fetching attachments for {table.value} {req_sys_id}")
            try:
                additional_details["attachments"] = await self.attachment_handler.get_attachments(table, req_sys_id)
                log.info(f"Found {len(additional_details.get('attachments',[]))} attachments.")
            except ServiceNowError as e:
                log.warning(f"Could not fetch attachments for {request_number}: {e}")

            # --- Fetch Parent REQ and Catalog Item (Optional - Keep or Remove based on need) ---
            # This section does NOT fetch approvers, only the related record details.
            # NOTE: Your logs indicated these fetches failed anyway. This might be a separate issue
            # with the SPARC API or permissions for sc_request/sc_cat_item tables via sys_id query.
            if table == ServiceNowTable.REQ_ITEM:
                log.debug(f"Fetching related record details (Parent REQ, Cat Item) for RITM {request_number}")
                # Get parent Request (REQ) details
                parent_req_link_data = main_request.get("request")
                parent_request_sys_id = None
                if parent_req_link_data and isinstance(parent_req_link_data, dict):
                     parent_request_sys_id = self._safe_extract_field(parent_req_link_data, 'value')

                if parent_request_sys_id:
                    log.debug(f"Attempting to fetch parent REQ {parent_request_sys_id}")
                    try:
                        p_data_response = await self.table_handler.table_request(
                            ServiceNowTable.SC_REQUEST, HTTPMethod.GET, f"sys_id={parent_request_sys_id}", limit=1
                        )
                        parent_records = p_data_response.get('records', [])
                        additional_details["parent_request"] = parent_records[0] if parent_records else {}
                        if not additional_details["parent_request"]:
                             log.warning(f"[Check SPARC API/Permissions] Parent REQ {parent_request_sys_id} not found despite link in RITM.")
                    except ServiceNowNotFoundError:
                         log.warning(f"[Check SPARC API/Permissions] Parent REQ {parent_request_sys_id} not found for RITM {request_number} (404)")
                    except ServiceNowError as e: log.warning(f"Could not fetch parent REQ {parent_request_sys_id}: {e}")
                else:
                     log.debug("No parent request link found in RITM data.")

                # Get Catalog Item definition
                cat_item_link_data = main_request.get("cat_item")
                cat_item_sys_id = None
                if cat_item_link_data and isinstance(cat_item_link_data, dict):
                     cat_item_sys_id = self._safe_extract_field(cat_item_link_data, 'value')

                if cat_item_sys_id:
                    log.debug(f"Attempting to fetch Catalog Item {cat_item_sys_id}")
                    try:
                        c_data_response = await self.table_handler.table_request(
                            ServiceNowTable.SC_CAT_ITEM, HTTPMethod.GET, f"sys_id={cat_item_sys_id}", limit=1
                        )
                        cat_item_records = c_data_response.get('records', [])
                        additional_details["catalog_item"] = cat_item_records[0] if cat_item_records else {}
                        if not additional_details["catalog_item"]:
                             log.warning(f"[Check SPARC API/Permissions] Catalog Item {cat_item_sys_id} not found despite link in RITM.")
                    except ServiceNowNotFoundError:
                        log.warning(f"[Check SPARC API/Permissions] Catalog Item {cat_item_sys_id} not found for RITM {request_number} (404)")
                    except ServiceNowError as e: log.warning(f"Could not fetch CatItem {cat_item_sys_id}: {e}")
                else:
                     log.debug("No catalog item link found in RITM data.")
            # --- End of Optional Related Record Fetching ---


            # --- Format Only Direct Approvers ---
            # No de-duplication needed if only fetching direct approvers
            log.debug(f"Formatting {len(direct_approvers_list)} direct approver records.")
            formatted_approvers = [
                {
                    "sys_id": self._safe_extract_field(app.get("sys_id"), 'value'), # sys_id of the approval record
                    "approver": self._safe_extract_field(app.get("approver"), 'display_value'),
                    "approver_sys_id": self._safe_extract_field(app.get("approver"), 'value'),
                    "state": self._safe_extract_field(app.get("state"), 'display_value'),
                    "comments": self._safe_extract_field(app.get("comments"), 'value'),
                    "source_record_sys_id": self._safe_extract_field(app.get("sysapproval"), 'value'), # sys_id of RITM/REQ/TASK being approved
                    "source_record_table": self._safe_extract_field(app.get("sysapproval.sys_class_name"), 'display_value', table.value), # Use main table as fallback
                    "created_on": self._safe_extract_field(app.get("sys_created_on"), 'display_value'),
                    "updated_on": self._safe_extract_field(app.get("sys_updated_on"), 'display_value'),
                } for app in direct_approvers_list # Iterate only over direct approvers
            ]
            # Sort approvers (optional)
            formatted_approvers.sort(key=lambda x: x.get("created_on", ""), reverse=True)

            additional_details["approvers"] = formatted_approvers
            additional_details["approval_info"] = formatted_approvers # Keep structure consistent

            # Generate link for the main request
            link = self._generate_record_link(table, req_sys_id)

            # Return success with combined data
            log.info(f"Successfully retrieved details and {len(formatted_approvers)} direct approvers for {request_number}")
            return ResponseHandler.success(data={"request": main_request, "additional_details": additional_details, "link": link})

        except ServiceNowNotFoundError as e:
            log.warning(f"{request_number} for requester {requester_sys_id} not found: {e}")
            return ResponseHandler.not_found(str(e))
        except ServiceNowError as e:
            log.error(f"ServiceNow API error fetching details for {request_number}: {e}", exc_info=True)
            return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
        except Exception as e:
             log.exception(f"Unexpected Python error in get_request_details_by_requester for {request_number}")
             return ResponseHandler.internal_error(f"Unexpected Python error processing request details: {str(e)}")

    # --- Composite Operations --- #
    @tracer.start_as_current_span("servicenow_client.create_incident_with_attachments")
    async def create_incident_with_attachments(self, files: List[Dict], short_description: str, **kwargs) -> Dict[str, Any]:
        """ Creates an incident and then attaches files. Returns ResponseHandler dict. """
        # Validate files structure minimally
        if not isinstance(files, list) or not all(isinstance(f, dict) and 'file_name' in f and 'file_blob' in f for f in files):
            return ResponseHandler.bad_request("Invalid 'files' structure. Expected List[Dict[str, Any]] with 'file_name' and 'file_blob'.")

        log.info(f"Creating incident '{short_description}' with {len(files)} attachments.")
        incident_details: Optional[Dict] = None
        incident_sys_id: Optional[str] = None
        incident_number_disp: Optional[str] = None
        attachment_summary={"successful":0,"failed":0,"results":[]}

        try:
            # Step 1: Create the incident using the dedicated method
            # create_incident returns raw data dict or raises ServiceNowError
            incident_details = await self.create_incident(short_description=short_description, **kwargs)
            incident_sys_id = self._safe_extract_field(incident_details.get("sys_id"),'value')
            incident_number_disp = self._safe_extract_field(incident_details.get('number'),'display_value', self._safe_extract_field(incident_details.get('number'),'value', 'UNKNOWN'))

            if not incident_sys_id:
                # This shouldn't happen if create_incident succeeded without error, but check defensively
                 log.error("Incident created but sys_id missing in response. Attachments skipped.", extra={"incident_response": incident_details})
                 msg="Incident created but sys_id missing. Attachments skipped."
                 # Return error indicating partial success/failure
                 return ResponseHandler.error(500, msg, details={"incident_details": incident_details})


            # Step 2: Attach files if incident creation succeeded and sys_id was found
            log.info(f"Incident {incident_number_disp} created (sys_id: {incident_sys_id}). Attaching files...")
            s, f, r = 0, 0, []
            
            for item in files:
                try:
                    result = await self.attachment_handler.add_attachment(
                        ServiceNowTable.INCIDENT,
                        incident_sys_id,
                        item['file_blob'],  # Expecting bytes or readable stream
                        item['file_name']
                    )
                    s += 1
                    r.append({"file": item['file_name'], "status": "success", "details": result})
                    log.debug(f"Attachment succeeded for {item['file_name']} on incident {incident_number_disp}.")
                except Exception as e:
                    f += 1
                    error_details = str(e.details) if isinstance(e, ServiceNowError) else str(e)
                    r.append({"file": item['file_name'], "status": "failed", "error": str(e), "details": error_details})
                    log.error(f"Attachment failed for {item['file_name']} on incident {incident_number_disp}: {e}")

            attachment_summary = {"successful": s, "failed": f, "results": r}
            msg = f"Incident {incident_number_disp} created. Attachments: {s} succeeded, {f} failed."
            status_code = 201 if f == 0 else 207  # 201 Created (fully successful) or 207 Multi-Status

            return ResponseHandler.success(code=status_code, message=msg, data={"incident_details":incident_details,"attachment_summary":attachment_summary})

        except ServiceNowError as e:
            # Log the specific error during incident creation more clearly
            log.error(f"Failed to create incident during composite operation: Status={e.status_code}, Msg={e.args[0]}, Details={e.details}", exc_info=True) # Log exception info
            # Return the wrapper format indicating incident creation failure
            return ResponseHandler.error(e.status_code or 500, f"Incident creation failed: {e.args[0]}", e.details)
        except Exception as e:
             log.exception("Unexpected error during create_incident_with_attachments")
             return ResponseHandler.internal_error(f"Unexpected error: {str(e)}")


    @tracer.start_as_current_span("servicenow_client.update_record_with_attachments")
    async def update_record_with_attachments(self, table: ServiceNowTable, sys_id: str, files: List[Dict], **kwargs) -> Dict[str, Any]:
        """ Updates a record (e.g., Incident, RITM) and attaches files. Returns ResponseHandler dict. """
        # Validate files structure minimally
        if not isinstance(files, list) or not all(isinstance(f, dict) and 'file_name' in f and 'file_blob' in f for f in files):
             return ResponseHandler.bad_request("Invalid 'files' structure. Expected List[Dict[str, Any]] with 'file_name' and 'file_blob'.")

        log.info(f"Updating {table.value} {sys_id} with {len(files)} attachments and data: {list(kwargs.keys())}.")
        update_details: Optional[Dict] = None
        attachment_summary={"successful":0,"failed":0,"results":[]}

        try:
            # Step 1: Update the record if kwargs are provided
            if kwargs:
                log.debug(f"Attempting to update {table.value} {sys_id}...")
                # Use the appropriate update method or table handler
                if table==ServiceNowTable.INCIDENT:
                     # update_incident returns raw data or raises error
                     update_details = await self.update_incident(sys_id, **kwargs)
                else:
                     # Use generic table handler for other tables (e.g., RITM, CHG)
                     update_details = await self.table_handler.table_request(table, HTTPMethod.PUT, sys_id, kwargs)
                log.info(f"{table.value} {sys_id} updated successfully.")
            else:
                # If no update data, fetch current details for the response context
                log.debug(f"No update data provided. Fetching current details for {table.value} {sys_id}.")
                if table==ServiceNowTable.INCIDENT:
                     # Use internal getter that raises NotFoundError
                     update_details = await self.get_incident_by_sys_id_internal(sys_id)
                else:
                     # Use generic table handler GET
                     get_data = await self.table_handler.table_request(table, HTTPMethod.GET, f"sys_id={sys_id}", limit=1)
                     if not get_data or (isinstance(get_data, list) and not get_data):
                          raise ServiceNowNotFoundError(f"{table.value} {sys_id} not found.")
                     update_details = get_data[0] if isinstance(get_data, list) else get_data
                     # Add link if fetched this way
                     if isinstance(update_details,dict):
                           update_details["link"]=self._generate_record_link(table, sys_id)


            # Step 2: Attach files
            log.info(f"Attaching {len(files)} files to {table.value} {sys_id}...")
            s, f, r = 0, 0, []
            
            for item in files:
                try:
                    result = await self.attachment_handler.add_attachment(
                        table,
                        sys_id,
                        item['file_blob'],
                        item['file_name']
                    )
                    s += 1
                    r.append({"file": item['file_name'], "status": "success", "details": result})
                    log.debug(f"Attachment succeeded for {item['file_name']} on {table.value} {sys_id}.")
                except Exception as e:
                    f += 1
                    error_details = str(e.details) if isinstance(e, ServiceNowError) else str(e)
                    r.append({"file": item['file_name'], "status": "failed", "error": str(e), "details": error_details})
                    log.error(f"Attachment failed for {item['file_name']} on {table.value} {sys_id}: {e}")

            attachment_summary = {"successful": s, "failed": f, "results": r}
            update_msg = "updated" if kwargs else "processed"
            msg=f"{table.value} {sys_id} {update_msg}. Attachments: {s} succeeded, {f} failed."
            status_code = 200 if f == 0 else 207 # 200 OK (if fully successful) or 207 Multi-Status

            return ResponseHandler.success(code=status_code, message=msg, data={"record_details":update_details,"attachment_summary":attachment_summary})

        except ServiceNowNotFoundError as e:
             # Raised if the record sys_id doesn't exist (either on update attempt or fetch)
             log.error(f"Record not found during update/attachment: {e}")
             return ResponseHandler.not_found(str(e))
        except ServiceNowError as e:
             # Raised if the update fails or attachment handler fails unexpectedly
             log.error(f"ServiceNow error during update/attachment: Status={e.status_code}, Msg={e.args[0]}, Details={e.details}", exc_info=True)
             return ResponseHandler.error(e.status_code or 500, f"Update/Attachment failed: {e.args[0]}", e.details)
        except Exception as e:
             log.exception(f"Unexpected error during update_record_with_attachments for {table.value} {sys_id}")
             return ResponseHandler.internal_error(f"Unexpected error: {str(e)}")


    # --- Deprecated Wrappers (Matching Original Names) --- #
    # These now primarily log warnings and delegate to the newer composite methods

    @tracer.start_as_current_span("sparc_service_now.create_incident_and_attach_files_with_blob")
    async def create_incident_and_attach_files_with_blob(self, files: List[Dict], short_description: str="", description: str="", comments: str="", work_notes: str="", contact_type: Optional[str]=None, assignment_group: Optional[str]=None, service_now_user_sys_id: Optional[str]=None) -> Dict[str, Any]:
        log.warning("Method 'create_incident_and_attach_files_with_blob' is deprecated. Use 'create_incident_with_attachments'.")
        # Map old parameters to new **kwargs format
        kwargs={k:v for k,v in locals().items() if k not in ['self','files','short_description'] and v is not None}
        # Map 'service_now_user_sys_id' to 'caller_id' and potentially 'u_reporting_customer' if required by your instance config
        if 'service_now_user_sys_id' in kwargs:
            # kwargs['caller_id']=kwargs.pop('service_now_user_sys_id')
            # Add mapping for custom fields if needed, e.g.:
            kwargs['u_reporting_customer'] = kwargs.pop('service_now_user_sys_id')
            log.debug("Mapped service_now_user_sys_id to caller_id.")

        # Set defaults if not provided (will be overridden by explicit args if present)
        kwargs.setdefault('contact_type', self.config.default_contact_type)
        kwargs.setdefault('assignment_group', self.config.default_assignment_group)

        # Delegate to the new composite method
        return await self.create_incident_with_attachments(
            files=files,
            short_description=short_description, # Pass mandatory short_description explicitly
            **kwargs # Pass all other mapped/defaulted parameters
        )

    @tracer.start_as_current_span("sparc_service_now.update_incident_and_attach_files_with_blob")
    async def update_incident_and_attach_files_with_blob(self, incident_sys_id: str, files: List[Dict], comments: str="", work_notes: str="", service_now_user_sys_id: Optional[str]=None) -> Dict[str, Any]:
        log.warning("Method 'update_incident_and_attach_files_with_blob' is deprecated. Use 'update_record_with_attachments'.")
        # Map parameters to **kwargs
        kwargs={k:v for k,v in locals().items() if k not in ['self','incident_sys_id','files'] and v is not None}
        if 'service_now_user_sys_id' in kwargs:
            kwargs['caller_id']=kwargs.pop('service_now_user_sys_id') # Map user ID if provided
            log.debug("Mapped service_now_user_sys_id to caller_id.")

        # Delegate to the new composite method for INCIDENT table
        return await self.update_record_with_attachments(
            table=ServiceNowTable.INCIDENT,
            sys_id=incident_sys_id,
            files=files,
            **kwargs
        )

    @tracer.start_as_current_span("sparc_service_now.update_request_and_attach_files_with_blob")
    async def update_request_and_attach_files_with_blob(self, request_sys_id: str, files: List[Dict], comments: str="", work_notes: str="", service_now_user_sys_id: Optional[str]=None) -> Dict[str, Any]:
        log.warning("Method 'update_request_and_attach_files_with_blob' is deprecated. Use 'update_record_with_attachments'. Attempting to determine table type...")
        # Map parameters
        kwargs={k:v for k,v in locals().items() if k not in ['self','request_sys_id','files'] and v is not None}

        # --- Determine Table Type (Requires extra queries) ---
        # This is inefficient. The caller should ideally know the table type.
        table: Optional[ServiceNowTable]=None
        try:
             # Check if it's a RITM
             await self.table_handler.table_request(ServiceNowTable.REQ_ITEM,HTTPMethod.GET,f"sys_id={request_sys_id}",limit=1,fields=['sys_id'])
             table=ServiceNowTable.REQ_ITEM
             log.debug(f"Determined {request_sys_id} is likely a RITM.")
        except ServiceNowNotFoundError:
             log.debug(f"{request_sys_id} not found as RITM, checking CHG...")
             try:
                  # Check if it's a CHG
                  await self.table_handler.table_request(ServiceNowTable.CHANGE_REQUEST,HTTPMethod.GET,f"sys_id={request_sys_id}",limit=1,fields=['sys_id'])
                  table=ServiceNowTable.CHANGE_REQUEST
                  log.debug(f"Determined {request_sys_id} is likely a CHG.")
             except ServiceNowNotFoundError:
                  log.error(f"Cannot determine table type for request {request_sys_id}. Not found as RITM or CHG.")
                  return ResponseHandler.not_found(f"Cannot determine table type for request {request_sys_id}. Record not found as RITM or CHG.")
             except ServiceNowError as e:
                  log.error(f"Error checking CHG table for {request_sys_id}: {e}")
                  return ResponseHandler.error(e.status_code or 500, f"Error checking CHG table: {e.args[0]}", e.details)
        except ServiceNowError as e:
             log.error(f"Error checking RITM table for {request_sys_id}: {e}")
             return ResponseHandler.error(e.status_code or 500, f"Error checking RITM table: {e.args[0]}", e.details)

        if table is None: # Should have been caught above, but defensive check
             return ResponseHandler.internal_error(f"Failed to determine table type for {request_sys_id}.")

        # Map user ID based on determined table (if needed)
        if service_now_user_sys_id:
            # Determine appropriate user field based on table (this might vary by instance config)
            user_field: Optional[str] = None
            if table==ServiceNowTable.REQ_ITEM: user_field = 'opened_by' # Or maybe 'requested_for'? Check instance.
            elif table==ServiceNowTable.CHANGE_REQUEST: user_field = 'requested_by'

            if user_field:
                 kwargs[user_field] = kwargs.pop('service_now_user_sys_id')
                 log.debug(f"Mapped service_now_user_sys_id to {user_field} for {table.value}.")
            else:
                 kwargs.pop('service_now_user_sys_id', None) # Discard if no mapping defined
                 log.warning(f"No user field mapping defined for table {table.value}. Ignoring service_now_user_sys_id.")


        # Delegate to the new composite method with determined table
        return await self.update_record_with_attachments(
            table=table,
            sys_id=request_sys_id,
            files=files,
            **kwargs
        )

    @tracer.start_as_current_span("sparc_service_now.get_catalog_item_details")
    async def get_catalog_item_details(self, ritm_number: str) -> Dict[str, Any]:
        log.warning("Method 'get_catalog_item_details' is deprecated. Use 'get_catalog_item_variables' and 'get_ritm_details'.")
        try:
            # Call the new, more specific methods
            # get_ritm_details returns raw data or raises error
            ritm_details = await self.get_ritm_details(ritm_number)
            # get_catalog_item_variables returns list or raises error
            variables = await self.get_catalog_item_variables(ritm_number=ritm_number)

            # Construct the old response format using the new data
            cat_item_info = ritm_details.get("cat_item", {}) # cat_item field should contain link/value
            cat_item_sys_id = self._safe_extract_field(cat_item_info, 'value')
            cat_item_display = self._safe_extract_field(cat_item_info, 'display_value')

            return ResponseHandler.success(data={
                "ritm_details": ritm_details,
                # Provide basic catalog item info derived from RITM details
                "catalog_item": {"sys_id": cat_item_sys_id, "display_value": cat_item_display},
                "variables": variables
            })
        except ServiceNowNotFoundError as e: return ResponseHandler.not_found(str(e))
        except ServiceNowError as e: return ResponseHandler.error(e.status_code or 500, e.args[0], e.details)
        except Exception as e:
            log.exception("Unexpected error in deprecated get_catalog_item_details")
            return ResponseHandler.internal_error(f"Unexpected error: {str(e)}")

    @tracer.start_as_current_span("sparc_service_now.approve_record")
    async def approve_record(self, approval_sys_id: str, comment: str="") -> Dict[str, Any]:
        log.warning("Method 'approve_record' is deprecated. Use 'approve_request'.")
        # NOTE: The original approve_request didn't take comments. The new _update_approval_state does.
        # For backward compatibility, we call approve_request which *doesn't* pass the comment.
        # If comments are needed with approval via this deprecated method, the target approve_request
        # would need modification, or users should switch to the non-deprecated methods.
        if comment:
             log.warning("Comment provided to deprecated 'approve_record' will be ignored. Use 'approve_request' (which also ignores comments) or modify internal logic if comments are needed on approval.")
        # Call approve_request which now returns a ResponseHandler dict
        return await self.approve_request(approval_sys_id)

    @tracer.start_as_current_span("sparc_service_now.reject_record")
    async def reject_record(self, approval_sys_id: str, comment: str="") -> Dict[str, Any]:
        log.warning("Method 'reject_record' is deprecated. Use 'reject_request'.")
        # Call reject_request which takes optional comments matching original reject_request
        # Pass the comment through.
        return await self.reject_request(approval_sys_id, comments=comment if comment else None)

    @tracer.start_as_current_span("sparc_service_now.add_attachment_to_record")
    async def add_attachment_to_record(self, table: str, sys_id: str, file_content: Any, file_name: str, content_type: Optional[str]=None) -> tuple[bool, Any]:
        """ Deprecated. Use composite methods or attachment_handler directly. """
        log.warning("Method 'add_attachment_to_record' is deprecated. Use composite methods like 'update_record_with_attachments' or the internal attachment_handler directly.")
        try:
            # Attempt to convert table string to Enum
            try:
                 table_enum=ServiceNowTable(table.lower())
            except ValueError:
                 log.error(f"Invalid table name string: '{table}'. Cannot convert to ServiceNowTable enum.")
                 return (False, f"Invalid table name: {table}")

            # Delegate to the attachment handler
            result = await self.attachment_handler.add_attachment(
                 table_enum,
                 sys_id,
                 file_content,
                 file_name,
                 content_type or AttachmentHandlerImpl._get_content_type_for_file(file_name) # Determine content type if not provided
            )
            # Return tuple format expected by deprecated method
            return (True, result)
        except ValueError as e: # Catch errors from table enum conversion or file content processing
             log.error(f"Value error in add_attachment_to_record: {e}")
             return (False, str(e))
        except ServiceNowError as e:
             log.error(f"ServiceNow error in add_attachment_to_record: {e}")
             # Format error details similar to original expectation
             return (False, {"error":e.args[0],"details":e.details,"status_code":e.status_code})
        except Exception as e:
             log.exception("Unexpected error in deprecated add_attachment_to_record wrapper")
             return (False, f"Unexpected error: {str(e)}")


    def __repr__(self):
        return f"<ServiceNow(env='{self.config.env}', base_url='{self.config.base_url}')>"


# --- Example Usage --- (Unchanged - relies on the ServiceNow class structure)
async def example_usage():
    # Configure logging for the example itself if not already done by the main script
    if not logging.getLogger().handlers:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s|%(levelname)s|%(name)s|%(message)s')

    log.info("Starting ServiceNow example...")
    try:
        # Assumes environment variables SPARC_CLIENT_ID, SPARC_CLIENT_SECRET etc. are set
        # Or modify Config.from_env() to load differently
        config = Config.from_env()
        async with ServiceNow(config) as client:
            inst_name = await client.get_instance_name()
            print(f"Connected to instance: {inst_name}")

            # --- Test User Ops ---
            print("\n--- Testing User Ops ---")
            # Use a known user sys_id for testing or fetch dynamically if possible
            test_user_sys_id = os.getenv("TEST_SN_USER_SYSID", "9c8c564a1bc1d01090ee337cdc4bcb10") # Replace with a valid sys_id or get from env
            if test_user_sys_id == "replace_with_valid_sysid":
                 print("Skipping user-specific tests: TEST_SN_USER_SYSID environment variable not set.")
            else:
                 print(f"Using Test User Sys ID: {test_user_sys_id}")
                 user_details_resp = await client.get_user_by_user_sys_id(test_user_sys_id)
                 if user_details_resp.get("code") == 200:
                     user_data = user_details_resp.get('data',{})
                     print(f"  User Name: {client._safe_extract_field(user_data.get('name'),'display_value')}")
                     print(f"  User Email: {client._safe_extract_field(user_data.get('email'),'display_value')}")
                 else:
                     print(f"  Error fetching user details: {user_details_resp}")


                 # --- Test Approval Ops ---
                 print("\n--- Testing Approval Ops (using user_sys_id) ---")
                 approvals_result = await client.get_latest_pending_approvals_pagination(test_user_sys_id, limit=3, offset=0)
                 if approvals_result.get("code") == 200:
                     approvals = approvals_result.get("data", [])
                     print(f"  Found {len(approvals)} latest pending approvals (Limit 3):")
                     for app in approvals:
                         # Access the formatted fields from the dictionary returned by _get_pending_approvals_impl
                         print(f"    - Approval For: {app.get('number', 'N/A')} ({app.get('short_description', 'N/A')})")
                         print(f"      Approval Record SysID: {app.get('sys_id', 'N/A')}") # This is the sys_id of the sysapproval_approver record
                         print(f"      Target Record SysID: {app.get('approval_id', 'N/A')}") # This is the sys_id of the RITM/CHG etc.
                         print(f"      Target Type: {app.get('approval_type', 'N/A')}")
                         print(f"      State: {app.get('state', 'N/A')}")
                         print(f"      Link to Target: {app.get('link', '#')}")
                 else:
                     print(f"  Error fetching approvals: {approvals_result}")


                 # --- Test Incident Ops ---
                 print("\n--- Testing Incident Ops (using user_sys_id as caller_id) ---")
                 # Use get_tickets_by_states_and_caller_id_pagination
                 incidents_result = await client.get_tickets_by_states_and_caller_id_pagination(
                     states=["New", "Open", "Assigned", "Work in Progress"], # Example states
                     user_if_field_key="caller_id", # Query by caller
                     user_if_field_value=test_user_sys_id,
                     limit=2,
                     offset=0,
                     exclude_states=["On Hold", "Resolved", "Closed", "Canceled"] # Example excludes
                 )
                 if incidents_result.get("code") == 200:
                     incidents = incidents_result.get("data", [])
                     print(f"  Found {len(incidents)} incidents (Limit 2, Offset 0):")
                     for inc in incidents:
                         print(f"    - {client._safe_extract_field(inc.get('number'),'display_value')}: {client._safe_extract_field(inc.get('short_description'),'display_value')}")
                         print(f"      State: {client._safe_extract_field(inc.get('state'),'display_value')} ({client._safe_extract_field(inc.get('incident_state'),'display_value')})")
                         print(f"      Link: {inc.get('link')}")
                 else:
                     print(f"  Error fetching incidents: {incidents_result}")

                 # --- Test Get Incident By Num/Caller ---
                 if incidents_result.get("code") == 200 and incidents_result.get("data"):
                     test_inc_num = client._safe_extract_field(incidents_result["data"][0].get("number"), 'value') # Get raw number value
                     if test_inc_num:
                         print(f"\n--- Testing Get Incident By Num/Caller ({test_inc_num}) ---")
                         inc_detail_resp = await client.get_incident_by_number_and_caller(test_inc_num, test_user_sys_id)
                         if inc_detail_resp.get("code") == 200:
                             print(f"  Successfully fetched {test_inc_num} by number and caller.")
                         else:
                             print(f"  Error fetching {test_inc_num} by number/caller: {inc_detail_resp}")
                     else:
                          print("\n--- Skipping Get Incident By Num/Caller (No incident number found) ---")

            # --- Test Composite Ops ---
            print("\n--- Testing Composite Ops (Create Incident with Attachment) ---")
            # Use the primary composite method directly
            test_file_content = f"Test log line at {asyncio.get_event_loop().time()}\nAnother line.".encode('utf-8')
            create_resp = await client.create_incident_with_attachments(
                 short_description="Test Incident via Composite Method",
                 description="This is a test incident created with an attachment.",
                 caller_id=test_user_sys_id if test_user_sys_id != "replace_with_valid_sysid" else None, # Assign caller if known
                 # Add other incident fields as needed
                 impact="3", # Example: Low
                 urgency="3", # Example: Low
                 files=[{"file_name": "composite_test.log", "file_blob": test_file_content}]
            )
            print(f"  Create Composite Result: Code={create_resp.get('code')}, Msg={create_resp.get('message')}")
            if create_resp.get("code") in [201, 207]: # Check for Created or Multi-Status
                 inc_details = create_resp.get('data',{}).get('incident_details',{})
                 inc_num_disp = client._safe_extract_field(inc_details.get('number'),'display_value', 'N/A')
                 inc_link = inc_details.get('link', '#')
                 print(f"    Incident Created/Processed: {inc_num_disp}")
                 print(f"    Link: {inc_link}")
                 att_summary = create_resp.get('data',{}).get('attachment_summary',{})
                 print(f"    Attachment Summary: {att_summary}")

                 # --- Test Update with Attachment ---
                 if inc_details and (inc_sys_id := client._safe_extract_field(inc_details.get('sys_id'), 'value')):
                      print(f"\n--- Testing Composite Ops (Update Incident {inc_num_disp} with Attachment) ---")
                      update_file_content = b"This is content for the update attachment."
                      update_resp = await client.update_record_with_attachments(
                           table=ServiceNowTable.INCIDENT,
                           sys_id=inc_sys_id,
                           files=[{"file_name": "update_attachment.txt", "file_blob": update_file_content}],
                           # Add fields to update
                           work_notes="Adding an attachment via composite update.",
                           comments="Attaching another file for the user."
                      )
                      print(f"  Update Composite Result: Code={update_resp.get('code')}, Msg={update_resp.get('message')}")
                      if update_resp.get("code") in [200, 207]:
                          upd_att_summary = update_resp.get('data',{}).get('attachment_summary',{})
                          print(f"    Attachment Summary: {upd_att_summary}")

            else:
                 print(f"  Incident creation failed or did not return expected data: {create_resp.get('details')}")


    except ValueError as e: print(f"Configuration Error: {e}")
    except ServiceNowAuthError as e: print(f"Authentication Error: {e}")
    except ServiceNowNotFoundError as e: print(f"Resource Not Found Error: {e}")
    except ServiceNowPermissionError as e: print(f"Permission Error: {e}")
    except ServiceNowBadRequestError as e: print(f"Bad Request Error: {e.status_code} - {e.args[0]} - Details: {e.details}")
    except ServiceNowError as e: print(f"ServiceNow API Error ({e.status_code}): {e.args[0]} - Details: {e.details}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        log.exception("Unexpected error in example usage.") # Log full traceback for unexpected errors
    finally:
        log.info("ServiceNow example finished.")

if __name__ == "__main__":
    # Ensure required environment variables like SPARC_CLIENT_ID, SPARC_CLIENT_SECRET,
    # and optionally TEST_SN_USER_SYSID are set before running.
    # Example: export TEST_SN_USER_SYSID='sys_id_of_a_test_user'
    asyncio.run(example_usage())