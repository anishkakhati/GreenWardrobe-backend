"""Request schemas for GreenWardrobe endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ImageMetadata(BaseModel):
    """Metadata describing an uploaded and validated image."""

    filename: str = Field(description="Original uploaded filename")
    format: str = Field(description="Detected image format (JPEG, PNG, WEBP)")
    width: int = Field(gt=0, description="Image width in pixels")
    height: int = Field(gt=0, description="Image height in pixels")
    size_bytes: int = Field(gt=0, description="Image size in bytes")
    mime_type: Optional[str] = Field(default=None, description="Standard MIME type")


class AnalysisOptions(BaseModel):
    """Optional settings to control analysis pipeline execution."""

    include_ai_explanation: bool = Field(
        default=True,
        description="Whether to run Google Gemini for human-readable explanation",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required to treat CV/OCR prediction as reliable",
    )
    language: str = Field(
        default="en",
        description="Target language for AI explanation (default: English)",
    )


class DirectMaterialInput(BaseModel):
    """Single material percentage entry for direct/manual analysis requests."""

    material: str = Field(min_length=1, max_length=100, description="Name of the textile fiber")
    percentage: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentage of the material (0.0 to 100.0)",
    )

    @field_validator("material")
    @classmethod
    def normalize_material_name(cls, v: str) -> str:
        """Strip whitespace and lowercase material name."""
        return v.strip().lower()


class DirectAnalysisRequest(BaseModel):
    """Request payload to analyze fabric composition directly without image upload."""

    materials: List[DirectMaterialInput] = Field(
        min_length=1,
        description="List of material components with their percentages",
    )
    garment_type: Optional[str] = Field(
        default=None,
        description="Optional garment category (e.g., 't-shirt', 'jacket', 'jeans')",
    )
    options: AnalysisOptions = Field(
        default_factory=AnalysisOptions,
        description="Analysis execution options",
    )

    @field_validator("materials")
    @classmethod
    def validate_total_percentage(
        cls, materials: List[DirectMaterialInput]
    ) -> List[DirectMaterialInput]:
        """Verify that percentages sum reasonably close to 100% when multiple materials are present."""
        total = sum(m.percentage for m in materials)
        # We allow tolerance for rounding (e.g. 98.0 to 102.0)
        if total > 100.5:
            raise ValueError(f"Total material percentage exceeds 100%: calculated {total:.1f}%")
        return materials
