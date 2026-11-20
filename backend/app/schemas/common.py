"""Common enumerations, base schemas, and shared types for GreenWardrobe."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RecommendationTier(str, Enum):
    """Decision recommendation tiers as defined by GreenWardrobe framework."""

    BUY = "Buy"
    THINK_TWICE = "Think Twice"
    REPAIR = "Repair"
    REUSE = "Reuse"
    SECOND_HAND = "Second-Hand"
    UNKNOWN = "Unknown"


class ImpactLevel(str, Enum):
    """Standardized environmental impact severity ratings."""

    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"
    UNKNOWN = "Unknown"


class AnalysisSource(str, Enum):
    """Source indicating how composition was determined."""

    OCR_LABEL = "ocr_label"
    COMPUTER_VISION = "computer_vision"
    MANUAL_INPUT = "manual_input"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class ErrorDetail(BaseModel):
    """Structured error payload for failed requests."""

    type: str = Field(description="Exception or error class name")
    message: str = Field(description="Human-readable description of the error")
    details: Optional[Any] = Field(
        default=None,
        description="Detailed contextual metadata, validation errors, or field errors",
    )


class BaseResponse(BaseModel):
    """Standard API response envelope."""

    success: bool = Field(default=True, description="Whether the operation succeeded")


class ErrorResponse(BaseModel):
    """Standard API error envelope."""

    success: bool = Field(default=False, description="Always False for error responses")
    error: ErrorDetail = Field(description="Detailed error information")
