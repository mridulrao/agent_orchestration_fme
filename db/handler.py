"""
This module defines a unified database handler that provides connection management
for PostgreSQL database using Prisma ORM. It focuses on client initialization
and connection lifecycle management with improved error handling and retry logic.
"""

import logging
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from prisma import Prisma

from dotenv import load_dotenv

load_dotenv()


class DatabaseHandler:
    """
    A unified handler for database operations that shares a single Prisma client
    across different functionality with improved connection management and pooling.
    """

    _instance = None
    _client = None
    _connection_lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        """
        Implements the singleton pattern to ensure only one instance of the handler
        and thus only one Prisma client is created.
        """
        if cls._instance is None:
            cls._instance = super(DatabaseHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, logger=None):
        """
        Initializes the DatabaseHandler with a shared Prisma client instance.

        Args:
            logger: Optional logger instance
        """
        # Only initialize once
        if getattr(self, "_initialized", False):
            return

        # Initialize the shared Prisma client with optimized settings and connection pooling
        if DatabaseHandler._client is None:
            # Configure Prisma client - Python client uses DATABASE_URL from environment automatically
            DatabaseHandler._client = Prisma()

        self.client = DatabaseHandler._client
        self._connected = False
        self._connection_attempts = 0
        self._max_retries = 3  # Increased from 1 to 3
        self._retry_delay = 2.0  # Increased base delay

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info("DatabaseHandler instance initialized with connection pooling")
        self._initialized = True

    def _log_and_raise(self, message, exception):
        """Helper method to log and raise exceptions"""
        self.logger.error(f"{message}: {str(exception)}")
        raise exception

    async def connect(self, max_retries: int = None):
        """
        Connects to the PostgreSQL database using the Prisma client with retry logic and pooling.

        Args:
            max_retries: Maximum number of connection attempts (defaults to self._max_retries)
        """
        if self._connected and await self.is_connected():
            self.logger.debug("Database already connected and healthy")
            return

        max_retries = max_retries or self._max_retries

        async with self._connection_lock:
            for attempt in range(max_retries + 1):
                try:
                    self.logger.info(
                        f"Attempting database connection (attempt {attempt + 1}/{max_retries + 1})"
                    )

                    # Connect with timeout
                    await asyncio.wait_for(self.client.connect(), timeout=30.0)

                    # Verify connection with a simple test
                    # await asyncio.wait_for(self.client.$connect(), timeout=10.0)

                    self._connected = True
                    self._connection_attempts = 0
                    self.logger.info(
                        "Database connected successfully with connection pooling"
                    )
                    return

                except asyncio.TimeoutError:
                    self.logger.warning(f"Connection attempt {attempt + 1} timed out")
                    if attempt < max_retries:
                        await asyncio.sleep(self._retry_delay)
                        self._retry_delay = min(
                            self._retry_delay * 1.2, 10.0
                        )  # Gentle exponential backoff
                    else:
                        raise Exception(
                            "Database connection timed out after maximum retries"
                        )

                except Exception as e:
                    self._connection_attempts += 1
                    if attempt < max_retries:
                        self.logger.warning(
                            f"Connection attempt {attempt + 1} failed: {str(e)}. Retrying in {self._retry_delay} seconds..."
                        )
                        await asyncio.sleep(self._retry_delay)
                        self._retry_delay = min(
                            self._retry_delay * 1.2, 10.0
                        )  # Gentle exponential backoff
                    else:
                        self._log_and_raise(
                            f"Failed to connect to database after {max_retries + 1} attempts",
                            e,
                        )

    async def disconnect(self):
        """
        Disconnects from the PostgreSQL database using the Prisma client.
        """
        if not self._connected:
            self.logger.debug("Database already disconnected")
            return

        try:
            await self.client.disconnect()
            self._connected = False
            self.logger.info("Database disconnected successfully")
        except Exception as e:
            self._log_and_raise("Error disconnecting from database", e)

    async def is_connected(self):
        """Check if database is connected using both internal flag and Prisma's check"""
        try:
            return self._connected and self.client.is_connected()
        except Exception:
            return False

    async def ensure_connected(self):
        """
        Ensures the database is connected, reconnecting if necessary.
        """
        if not await self.is_connected():
            self.logger.info("Database connection lost, attempting to reconnect...")
            await self.connect()

    @asynccontextmanager
    async def get_connection(self):
        """
        Context manager for database operations with automatic connection management.
        """
        try:
            await self.ensure_connected()
            yield self.client
        except Exception as e:
            self.logger.error(f"Database operation failed: {str(e)}")
            raise

    # Async context manager support
    async def __aenter__(self):
        """Support for async context manager."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting the context."""
        await self.disconnect()

    @classmethod
    def get_instance(cls):
        """
        Get the singleton database instance.
        This is the recommended way to access the database.

        Returns:
            DatabaseHandler: The singleton database instance
        """
        if cls._instance is None:
            return cls()
        return cls._instance

    async def health_check(self) -> bool:
        """
        Perform a health check on the database connection.

        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            await self.ensure_connected()
            # Perform a simple query to test the connection
            await self.client.organizations.find_first()
            return True
        except Exception as e:
            self.logger.error(f"Database health check failed: {str(e)}")
            return False
