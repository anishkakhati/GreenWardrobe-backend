"""Response schemas for GreenWardrobe.

Contains data models for individual module outputs (OCR, Computer Vision,
Knowledge Base, Scoring Engine, Gemini AI), the existing backward-compatible
schemas, and the master unified analysis response.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.common import RecommendationTier, ImpactLevel, AnalysisSource


# ==============================================================================
# 1. Existing Legacy Schemas (Preserved for 100% Backward Compatibility)
# ==============================================================================

class Material(BaseModel):
    """Textile fiber with name and composition percentage."""

    name: str = Field(description="Normalized fiber or material name")
    percentage: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentage of this material in garment composition",
    )


class Composition(BaseModel):
    """Extracted garment fabric composition with confidence level."""

    materials: List[Material] = Field(
        default_factory=list,
        description="List of materials comprising the garment",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for composition extraction (0.0 to 1.0)",
    )
    source: AnalysisSource = Field(
        default=AnalysisSource.UNKNOWN,
        description="Source of composition detection",
    )

    @property
    def total_percentage(self) -> float:
        """Sum of all material percentages."""
        return sum(m.percentage for m in self.materials)

    @property
    def is_complete(self) -> bool:
        """True if total percentage sums to approximately 100%."""
        return 98.0 <= self.total_percentage <= 102.0


class VisionResult(BaseModel):
    """Computer vision surface fabric prediction."""

    predicted_material: Optional[str] = Field(
        default=None,
        description="Primary predicted fabric category",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model confidence score (0.0 to 1.0)",
    )
    alternatives: List[str] = Field(
        default_factory=list,
        description="Alternative plausible material categories",
    )


class ImpactMetric(BaseModel):
    """Single environmental indicator value with unit of measure."""

    value: Optional[float] = Field(default=None, description="Numerical indicator value")
    unit: Optional[str] = Field(default=None, description="Physical unit (e.g., L/kg, kg CO2e)")
    rating: Optional[str] = Field(default=None, description="Categorical rating (Low, High, etc.)")


class EnvironmentalImpact(BaseModel):
    """Grouped environmental life-cycle impact indicators."""

    water: Optional[ImpactMetric] = None
    carbon: Optional[ImpactMetric] = None
    energy: Optional[ImpactMetric] = None
    chemicals: Optional[ImpactMetric] = None
    microfibres: Optional[ImpactMetric] = None
    waste: Optional[ImpactMetric] = None


class GreenWardrobeResponse(BaseModel):
    """Original response schema preserved for backward compatibility."""

    composition: Optional[Composition] = None
    vision_result: Optional[VisionResult] = None
    environmental_impact: Optional[EnvironmentalImpact] = None

    impact_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Overall impact index (0-100)"
    )
    recommendation: Optional[str] = None
    explanation: Optional[str] = None

    warnings: List[str] = Field(default_factory=list)


# ==============================================================================
# 2. Teammate Module Integration Schemas
# ==============================================================================

# --- PERSON 1: HARSHIT (OCR Module) ---
class OCRCompositionItem(BaseModel):
    """Single fiber item extracted by OCR from clothing label."""

    material: str = Field(description="Fiber name identified from label text")
    percentage: float = Field(ge=0.0, le=100.0, description="Extracted percentage")


class OCRResult(BaseModel):
    """Structured output returned from Harshit's OCR module."""

    composition: List[OCRCompositionItem] = Field(
        default_factory=list,
        description="Parsed textile fibers and percentages",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="OCR model extraction confidence",
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Raw unformatted text detected by OCR",
    )
    unreadable: bool = Field(
        default=False,
        description="True if label was detected but text could not be deciphered",
    )
    warnings: List[str] = Field(default_factory=list)


# --- PERSON 2: ANSHIKA THAKUR (Computer Vision Module) ---
class VisionPrediction(BaseModel):
    """Structured output returned from Anshika Thakur's Computer Vision module."""

    predicted_material: Optional[str] = Field(
        default=None,
        description="Primary fabric prediction from surface image",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Computer vision model classification confidence",
    )
    alternatives: List[str] = Field(
        default_factory=list,
        description="Top-k alternative material predictions",
    )
    probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Raw prediction probability distribution across classes",
    )
    texture_features: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detected visual features (weave type, knit, texture)",
    )
    warnings: List[str] = Field(default_factory=list)


# --- PERSON 3: ANKIT (Knowledge Base Module) ---
class MaterialEnvironmentalProfile(BaseModel):
    """Verified environmental life-cycle metrics for a specific textile material."""

    material: str = Field(description="Standardized material name")
    water_liters_per_kg: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Freshwater consumption in liters per kg of fiber produced",
    )
    co2_kg_per_kg: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Carbon footprint in kg CO2 equivalent per kg of fiber",
    )
    energy_mj_per_kg: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Embodied energy in Megajoules (MJ) per kg of fiber",
    )
    chemicals_intensity: Optional[str] = Field(
        default=None,
        description="Chemical intensity rating (e.g., Low, Moderate, High)",
    )
    microplastic_shedding_risk: Optional[str] = Field(
        default=None,
        description="Risk rating for microplastic fiber shedding (e.g., None, Severe)",
    )
    biodegradability: Optional[str] = Field(
        default=None,
        description="Biodegradability classification under standard environmental conditions",
    )
    source_reference: Optional[str] = Field(
        default=None,
        description="Traceable benchmark source citation (e.g., EEA, UNEP)",
    )
    custom_indicators: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional indicators provided by knowledge base",
    )


class KnowledgeBaseResult(BaseModel):
    """Result returned by Ankit's Knowledge Base for requested materials."""

    profiles: Dict[str, MaterialEnvironmentalProfile] = Field(
        default_factory=dict,
        description="Map of material names to their verified environmental profiles",
    )
    unrecognized_materials: List[str] = Field(
        default_factory=list,
        description="Materials queried that were not found in the knowledge base",
    )


# --- PERSON 4: MANEESH (Scoring Engine Module) ---
class IndicatorScore(BaseModel):
    """Individual indicator impact score."""

    score: float = Field(ge=0.0, le=100.0, description="Normalized score from 0 to 100")
    rating: Optional[str] = Field(
        default=None,
        description="Qualitative tier: Low, Moderate, High, Critical",
    )
    description: Optional[str] = Field(default=None)


class ImpactScoreBreakdown(BaseModel):
    """Detailed score breakdown across individual environmental dimensions."""

    water: Optional[IndicatorScore] = None
    carbon: Optional[IndicatorScore] = None
    energy: Optional[IndicatorScore] = None
    chemicals: Optional[IndicatorScore] = None
    microfibres: Optional[IndicatorScore] = None
    waste: Optional[IndicatorScore] = None


class ScoringResult(BaseModel):
    """Authoritative numerical scoring output produced by Maneesh's engine."""

    overall_impact_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Composite environmental impact score (0 to 100), or null when scoring is unavailable",
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Authoritative action tier, or null when scoring is unavailable",
    )
    breakdown: Optional[ImpactScoreBreakdown] = Field(
        default=None,
        description="Detailed impact breakdown per dimension",
    )
    key_indicators: Dict[str, str] = Field(
        default_factory=dict,
        description="Categorical summaries of top environmental flags",
    )
    calculation_basis: str = Field(
        default="rule_based",
        description="Method used for calculation",
    )


# ==============================================================================
# 3. Google Gemini AI Explanation Schema
# ==============================================================================

class AiExplanation(BaseModel):
    """Strictly evidence-grounded explanation generated by Google Gemini."""

    status: str = Field(
        default="AVAILABLE",
        description="Explanation status: 'AVAILABLE', 'UNAVAILABLE', or 'FALLBACK'",
    )
    summary: str = Field(
        description="Plain-language summary of garment fabric and environmental footprint",
    )
    material_explanation: Optional[str] = Field(
        default=None,
        description="Consumer-friendly explanation of identified fiber composition",
    )
    environmental_impact_explanation: Optional[str] = Field(
        default=None,
        description="Explanation of overall environmental impact score",
    )
    impact_category_explanations: Dict[str, str] = Field(
        default_factory=dict,
        description="Explanations of individual environmental categories (water, carbon, microfibres, etc.)",
    )
    environmental_highlights: List[str] = Field(
        default_factory=list,
        description="Key environmental impact observations directly grounded in scoring data",
    )
    practical_recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable advice for consumer decision-making",
    )
    care_and_repair_tips: List[str] = Field(
        default_factory=list,
        description="Guidelines on wash care, microfibre prevention, and maintenance",
    )
    limitations_and_uncertainty: List[str] = Field(
        default_factory=list,
        description="Explicit caveats regarding missing percentages or model confidence",
    )
    data_quality_disclaimer: str = Field(
        default="Environmental impact indicators and scores are comparative benchmarks based on published literature and demo data; they do not represent empirical certified product declarations.",
        description="Mandatory scientific data quality disclaimer",
    )
    is_fallback: bool = Field(
        default=False,
        description="True if fallback template was used due to Gemini API limits",
    )


# ==============================================================================
# 4. Master Unified Analysis Response
# ==============================================================================

class ProductAnalysis(BaseModel):
    """Synthesized product data combining OCR composition and CV predictions."""

    composition: Optional[Composition] = Field(
        default=None,
        description="Normalized fabric composition",
    )
    visual_prediction: Optional[VisionPrediction] = Field(
        default=None,
        description="Surface computer vision findings",
    )
    composition_source: AnalysisSource = Field(
        default=AnalysisSource.UNKNOWN,
        description="Primary source of composition determination",
    )


class ImageClassificationMetadata(BaseModel):
    """Metadata produced by the image-type classification stage."""

    image_type: str = Field(
        description="Classified image type: GARMENT, LABEL, or UNKNOWN",
    )
    classification_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Classifier confidence score (0.0 to 1.0)",
    )
    classifier_name: str = Field(
        description="Name of the classifier that produced this result",
    )
    analysis_source: str = Field(
        description="Pipeline branch used: VISION (garment) or OCR (label)",
    )
    classification_reasoning: List[str] = Field(
        default_factory=list,
        description="Human-readable explanation of classifier signals",
    )


class AnalysisMetadata(BaseModel):
    """Operational metadata regarding the analysis run."""

    processing_time_ms: float = Field(ge=0.0, description="Total pipeline execution time in ms")
    model_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Versions of active models and adapters",
    )
    pipeline_version: str = Field(default="1.0.0")
    gemini_model: Optional[str] = Field(
        default=None,
        description="Identifier of configured Gemini model",
    )
    explanation_status: Optional[str] = Field(
        default=None,
        description="Operational status of Gemini explanation layer",
    )


class APIErrorDetail(BaseModel):
    """Error object returned by the application's centralized exception handlers."""

    type: str = Field(description="Stable error category name")
    message: str = Field(description="Human-readable explanation of the request failure")
    details: Any = Field(
        default_factory=dict,
        description="Additional validation or domain details; may be an object, list, or empty object",
    )


class APIErrorResponse(BaseModel):
    """Error envelope returned for HTTP failures."""

    success: bool = Field(default=False, description="Always false for an error response")
    error: APIErrorDetail


class AnalysisResponse(BaseModel):
    """Production response contract returned by POST /api/v1/analyze."""

    success: bool = Field(default=True, description="Execution success flag")
    analysis_id: str = Field(description="Unique UUID for this analysis session")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp of analysis completion",
    )
    image_classification: Optional[ImageClassificationMetadata] = Field(
        default=None,
        description="Metadata from the image-type classification stage",
    )
    product: ProductAnalysis = Field(
        description="Fabric composition and visual classification data",
    )
    environmental_data: Dict[str, MaterialEnvironmentalProfile] = Field(
        default_factory=dict,
        description="Verified knowledge base benchmarks for all identified materials",
    )
    impact_assessment: ScoringResult = Field(
        description="Authoritative numerical impact score and decision recommendation",
    )
    ai_explanation: Optional[AiExplanation] = Field(
        default=None,
        description="Grounded explanation and practical guidance",
    )
    confidence: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Confidence scores from perception modules (ocr, vision)",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Explicit analytical limitations and data bounds",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings encountered during analysis",
    )
    metadata: AnalysisMetadata = Field(
        description="Execution performance and pipeline versioning metadata",
    )


class UploadResponse(BaseModel):
    """Response contract for POST /api/v1/upload."""

    message: str = Field(description="Status message")
    garment_image: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Validated garment image attributes",
    )
    label_image: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Validated label image attributes",
    )
