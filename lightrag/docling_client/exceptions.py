"""
Exceptions for Docling service client.
"""


class DoclingServiceError(Exception):
    """Base exception for Docling service errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DoclingServiceUnavailable(DoclingServiceError):
    """Raised when Docling service is unavailable."""
    pass


class DoclingServiceTimeout(DoclingServiceError):
    """Raised when Docling service request times out."""
    pass


class DoclingProcessingError(DoclingServiceError):
    """Raised when document processing fails."""
    pass


class DoclingConfigurationError(DoclingServiceError):
    """Raised when service configuration is invalid."""
    pass