"""GreenWardrobe Schemas module."""

from app.schemas.common import (
    RecommendationTier,
    ImpactLevel,
    AnalysisSource,
    ErrorDetail,
    BaseResponse,
    ErrorResponse,
)
from app.schemas.request import (
    ImageMetadata,
    AnalysisOptions,
    DirectMaterialInput,
    DirectAnalysisRequest,
)
from app.schemas.response import (
    Material,
    Composition,
    VisionResult,
    ImpactMetric,
    EnvironmentalImpact,
    GreenWardrobeResponse,
    OCRCompositionItem,
    OCRResult,
    VisionPrediction,
    MaterialEnvironmentalProfile,
    KnowledgeBaseResult,
    IndicatorScore,
    ImpactScoreBreakdown,
    ScoringResult,
    AiExplanation,
    ProductAnalysis,
    AnalysisMetadata,
    AnalysisResponse,
    UploadResponse,
)

__all__ = [
    # Common
    "RecommendationTier",
    "ImpactLevel",
    "AnalysisSource",
    "ErrorDetail",
    "BaseResponse",
    "ErrorResponse",
    # Requests
    "ImageMetadata",
    "AnalysisOptions",
    "DirectMaterialInput",
    "DirectAnalysisRequest",
    # Legacy & Backward-compatible
    "Material",
    "Composition",
    "VisionResult",
    "ImpactMetric",
    "EnvironmentalImpact",
    "GreenWardrobeResponse",
    # Module contracts
    "OCRCompositionItem",
    "OCRResult",
    "VisionPrediction",
    "MaterialEnvironmentalProfile",
    "KnowledgeBaseResult",
    "IndicatorScore",
    "ImpactScoreBreakdown",
    "ScoringResult",
    "AiExplanation",
    # Master Unified Responses
    "ProductAnalysis",
    "AnalysisMetadata",
    "AnalysisResponse",
    "UploadResponse",
]
