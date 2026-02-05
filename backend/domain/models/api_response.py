"""Standardized API response models."""

from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiStatus(str, Enum):
    """API response status."""

    SUCCESS = "success"
    ERROR = "error"


class ApiResponse(BaseModel, Generic[T]):
    """Standardized API response envelope.

    Format: {"status": "success"|"error", "data": {...}, "message": "..."}
    """

    status: ApiStatus
    data: T | None = None
    message: str | None = None

    @classmethod
    def success(cls, data: Any = None, message: str | None = None) -> "ApiResponse":
        """Create a success response."""
        return cls(status=ApiStatus.SUCCESS, data=data, message=message)

    @classmethod
    def error(cls, message: str, data: Any = None) -> "ApiResponse":
        """Create an error response."""
        return cls(status=ApiStatus.ERROR, data=data, message=message)
