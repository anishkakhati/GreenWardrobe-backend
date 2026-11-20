"""Custom domain exceptions and centralized FastAPI exception handlers.

Ensures that errors are structured consistently across all endpoints
without exposing raw stack traces or internal secrets to clients.
"""

from typing import Any, Dict, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.core.logging import logger


class GreenWardrobeException(Exception):
    """Base domain exception for GreenWardrobe."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ImageValidationError(GreenWardrobeException):
    """Raised when an uploaded garment or label image fails validation."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, status_code=status_code, details=details)


class OCRProcessingError(GreenWardrobeException):
    """Raised when the textile composition OCR module encounters an error."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, status_code=status_code, details=details)


class VisionProcessingError(GreenWardrobeException):
    """Raised when the computer vision material identification module encounters an error."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, status_code=status_code, details=details)


class KnowledgeBaseError(GreenWardrobeException):
    """Raised when environmental benchmarks cannot be retrieved from knowledge base."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_404_NOT_FOUND,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, status_code=status_code, details=details)


class ScoringError(GreenWardrobeException):
    """Raised when the environmental impact calculation fails."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, status_code=status_code, details=details)


class ImageClassificationError(GreenWardrobeException):
    """Raised when the image-type classifier cannot determine garment vs label."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, status_code=status_code, details=details)


class GeminiServiceError(GreenWardrobeException):
    """Raised when Google Gemini API encounters an upstream failure or rate limit."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_502_BAD_GATEWAY,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message=message, status_code=status_code, details=details)


# ==============================================================================
# Centralized FastAPI Exception Handlers
# ==============================================================================

async def greenwardrobe_exception_handler(
    request: Request, exc: GreenWardrobeException
) -> JSONResponse:
    """Format custom domain exceptions into consistent client JSON."""
    logger.warning(
        "Domain error on %s %s: [%s] %s | details=%s",
        request.method,
        request.url.path,
        exc.__class__.__name__,
        exc.message,
        exc.details,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "type": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
            },
        },
    )


async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Standardize default FastAPI HTTPExceptions."""
    logger.warning(
        "HTTP %d on %s %s: %s",
        exc.status_code,
        request.method,
        request.url.path,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "type": "HTTPException",
                "message": str(exc.detail),
                "details": {},
            },
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Format request payload/query validation errors cleanly."""
    logger.warning(
        "Validation error on %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "success": False,
            "error": {
                "type": "ValidationError",
                "message": "Invalid request parameters or payload.",
                "details": exc.errors(),
            },
        },
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all for unhandled exceptions to avoid leaking tracebacks or secrets."""
    logger.exception(
        "Unhandled internal server error on %s %s: %s",
        request.method,
        request.url.path,
        str(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "type": "InternalServerError",
                "message": "An unexpected error occurred. Please try again later.",
                "details": {},
            },
        },
    )
