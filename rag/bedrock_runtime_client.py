import boto3
import os
import threading
import json
from botocore.client import Config
from dotenv import load_dotenv
import logging

load_dotenv()

# Configuration
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
aws_region = os.environ.get("AWS_REGION", "us-east-1")  # Default region

logger = logging.getLogger(__name__)

class BedrockClientSingleton:
    """
    Singleton to manage a shared Boto3 Bedrock runtime client across threads.
    Ensures that only one instance of the client is created and shared.
    """
    _instance = None
    _client = None
    _lock = threading.Lock()  # Lock to ensure thread-safe singleton creation

    def __new__(cls):
        # If the instance doesn't exist, create one
        if cls._instance is None:
            with cls._lock:  # Ensure thread safety during initialization
                if cls._instance is None:  # Double check in case another thread initialized it first
                    cls._instance = super(BedrockClientSingleton, cls).__new__(cls)
                    cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        """Initialize the Bedrock client with proper error handling"""
        try:
            if not self._is_valid_aws_config():
                raise ValueError("Invalid AWS configuration. Please check your credentials and region.")
            
            # Create the client only once
            self._client = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region,
                config=Config(
                    connect_timeout=120, 
                    read_timeout=120,
                    retries={'max_attempts': 3, 'mode': 'adaptive'}  # Add retry logic
                ),
            )
            logger.info(f"Bedrock client initialized successfully for region: {aws_region}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise

    @staticmethod
    def _is_valid_aws_config():
        """Checks if the AWS configuration has the required values and they are not empty."""
        return (
            aws_access_key_id is not None and aws_access_key_id.strip() != ""
            and aws_secret_access_key is not None and aws_secret_access_key.strip() != ""
            and aws_region is not None and aws_region.strip() != ""
        )

    def __init__(self):
        # Prevent reinitialization of already created instance
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

    def get_client(self):
        """
        Returns the shared Bedrock client instance.
        """
        if self._client is None:
            raise RuntimeError("Bedrock client not initialized. Check your AWS configuration.")
        return self._client

    def test_connection(self):
        """
        Test the Bedrock connection with minimal overhead
        """
        try:
            client = self.get_client()
            # Just check if the client is properly configured
            # This doesn't make any API calls but validates the client setup
            if hasattr(client, '_service_model') and client._service_model.service_name == 'bedrock-runtime':
                logger.info("Bedrock connection test successful")
                return True
            else:
                logger.error("Bedrock connection test failed: Invalid client configuration")
                return False
                
        except Exception as e:
            logger.error(f"Bedrock connection test failed: {str(e)}")
            return False

    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance (useful for testing)
        """
        with cls._lock:
            cls._instance = None
            cls._client = None

# # Global Bedrock client singleton
# bedrock_runtime_client = BedrockClientSingleton()

# # Optional: Test connection on import (remove if you don't want this)
# try:
#     if bedrock_runtime_client.test_connection():
#         logger.info("Bedrock client ready for use")
# except Exception as e:
#     logger.warning(f"Bedrock client initialization warning: {str(e)}")