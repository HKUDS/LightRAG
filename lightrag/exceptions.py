from __future__ import annotations

import httpx
from typing import Literal


class APIStatusError(Exception):
    """Raised when an API response has a status code of 4xx or 5xx."""

    response: httpx.Response
    status_code: int
    request_id: str | None

    def __init__(
        self, message: str, *, response: httpx.Response, body: object | None
    ) -> None:
        super().__init__(message, response.request, body=body)
        self.response = response
        self.status_code = response.status_code
        self.request_id = response.headers.get("x-request-id")


class APIConnectionError(Exception):
    def __init__(
        self, *, message: str = "Connection error.", request: httpx.Request
    ) -> None:
        super().__init__(message, request, body=None)


class BadRequestError(APIStatusError):
    status_code: Literal[400] = 400  # pyright: ignore[reportIncompatibleVariableOverride]


class AuthenticationError(APIStatusError):
    status_code: Literal[401] = 401  # pyright: ignore[reportIncompatibleVariableOverride]


class PermissionDeniedError(APIStatusError):
    status_code: Literal[403] = 403  # pyright: ignore[reportIncompatibleVariableOverride]


class NotFoundError(APIStatusError):
    status_code: Literal[404] = 404  # pyright: ignore[reportIncompatibleVariableOverride]


class ConflictError(APIStatusError):
    status_code: Literal[409] = 409  # pyright: ignore[reportIncompatibleVariableOverride]


class UnprocessableEntityError(APIStatusError):
    status_code: Literal[422] = 422  # pyright: ignore[reportIncompatibleVariableOverride]


class RateLimitError(APIStatusError):
    status_code: Literal[429] = 429  # pyright: ignore[reportIncompatibleVariableOverride]


class APITimeoutError(APIConnectionError):
    def __init__(self, request: httpx.Request) -> None:
        super().__init__(message="Request timed out.", request=request)


class StorageNotInitializedError(RuntimeError):
    """Raised when storage operations are attempted before initialization."""

    def __init__(self, storage_type: str = "Storage"):
        super().__init__(
            f"{storage_type} not initialized. Please ensure proper initialization:\n"
            f"\n"
            f"  rag = LightRAG(...)\n"
            f"  await rag.initialize_storages()  # Required\n"
            f"  \n"
            f"  from lightrag.kg.shared_storage import initialize_pipeline_status\n"
            f"  await initialize_pipeline_status()  # Required for pipeline operations\n"
            f"\n"
            f"See: https://github.com/HKUDS/LightRAG#important-initialization-requirements"
        )


class PipelineNotInitializedError(KeyError):
    """Raised when pipeline status is accessed before initialization."""

    def __init__(self, namespace: str = ""):
        msg = (
            f"Pipeline namespace '{namespace}' not found. "
            f"This usually means pipeline status was not initialized.\n"
            f"\n"
            f"Please call 'await initialize_pipeline_status()' after initializing storages:\n"
            f"\n"
            f"  from lightrag.kg.shared_storage import initialize_pipeline_status\n"
            f"  await initialize_pipeline_status()\n"
            f"\n"
            f"Full initialization sequence:\n"
            f"  rag = LightRAG(...)\n"
            f"  await rag.initialize_storages()\n"
            f"  await initialize_pipeline_status()"
        )
        super().__init__(msg)


class PipelineCancelledException(Exception):
    """Raised when pipeline processing is cancelled by user request."""

    def __init__(self, message: str = "User cancelled"):
        super().__init__(message)
        self.message = message


class QdrantMigrationError(Exception):
    """Raised when Qdrant data migration from legacy collections fails."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


#  Database-related exceptions for improved error handling

class DatabaseConnectionError(APIConnectionError):
        """Raised when database connection fails with recovery suggestions."""

            def __init__(self, message: str, database_type: str = "", recovery_hint: str = ""):
                        self.database_type = database_type
                                self.recovery_hint = recovery_hint
                                        detailed_msg = f"Database connection error
                                                if database_type:
                                                                detailed_msg += f" [{database_type}]"
                                                                        detailed_msg += f": {message}"
                                                                                if recovery_hint:
                                                                                                detailed_msg += f"\nRecovery: {recovery_hint}"
                                                                                                        super().__init__(detailed_msg)


                                                                                                        class DatabaseTimeoutError(APITimeoutError):
                                                                                                                """Raised when database operations timeout."""

                                                                                                                    def __init__(self, message: str, operation: str = "", timeout_seconds: float = 0):
                                                                                                                                self.operation = operation
                                                                                                                                        self.timeout_seconds = timeout_seconds
                                                                                                                                                detailed_msg = f"Database timeout: {message}"
                                                                                                                                                        if operation:
                                                                                                                                                                        detailed_msg += f" during '{operation}'"
                                                                                                                                                                                if timeout_seconds > 0:
                                                                                                                                                                                                detailed_msg += f" (timeout: {timeout_seconds}s)"
                                                                                                                                                                                                        super().__init__(detailed_msg)


                                                                                                                                                                                                        class EncodingError(Exception):
                                                                                                                                                                                                                """Raised when character encoding issues occur (UTF-8, etc.)."""

                                                                                                                                                                                                                    def __init__(self, message: str, encoding: str = "UTF-8", field_name: str = ""):
                                                                                                                                                                                                                                self.encoding = encoding
                                                                                                                                                                                                                                        self.field_name = field_name
                                                                                                                                                                                                                                                detailed_msg = f"Encoding error [{encoding}]: {message}"
                                                                                                                                                                                                                                                        if field_name:
                                                                                                                                                                                                                                                                        detailed_msg += f" in field '{field_name}'"
                                                                                                                                                                                                                                                                                super().__init__(detailed_msg)


                                                                                                                                                                                                                                                                                class DataValidationError(Exception):
                                                                                                                                                                                                                                                                                        """Raised when data validation fails in RAG processing."""

                                                                                                                                                                                                                                                                                            def __init__(self, message: str, data_type: str = "", expected_format: str = ""):
                                                                                                                                                                                                                                                                                                        self.data_type = data_type
                                                                                                                                                                                                                                                                                                                self.expected_format = expected_format
                                                                                                                                                                                                                                                                                                                        detailed_msg = f"Data validation error: {message}"
                                                                                                                                                                                                                                                                                                                                if data_type:
                                                                                                                                                                                                                                                                                                                                                detailed_msg += f" for data type '{data_type}'"
                                                                                                                                                                                                                                                                                                                                                        if expected_format:
                                                                                                                                                                                                                                                                                                                                                                        detailed_msg += f". Expected format: {expected_format}"
                                                                                                                                                                                                                                                                                                                                                                                super().__init__(detailed_msg)