"""CAIM framework exceptions."""

from typing import Optional


class CAIMException(Exception):
    """Base exception for CAIM framework."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code


class MemoryException(CAIMException):
    """Exception raised for memory-related errors."""
    pass


class RetrievalException(CAIMException):
    """Exception raised for memory retrieval errors."""
    pass


class ModelException(CAIMException):
    """Exception raised for model-related errors."""
    pass


class StorageException(CAIMException):
    """Exception raised for storage-related errors."""
    pass


class ConfigurationException(CAIMException):
    """Exception raised for configuration errors."""
    pass