"""Base abstract adapter interfaces for GreenWardrobe external and teammate modules.

Enforces clear dependency contracts, runtime availability checks,
and isolation between the core API layer and underlying model implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.schemas.response import (
    KnowledgeBaseResult,
    MaterialEnvironmentalProfile,
    OCRResult,
    ScoringResult,
    VisionPrediction,
)
from app.services.image_service import ProcessedImage


class BaseAdapter(ABC):
    """Common base class for all module adapters."""

    @property
    @abstractmethod
    def adapter_name(self) -> str:
        """Name of the adapter."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the underlying module or model is ready for execution."""
        pass

    @abstractmethod
    def status_info(self) -> Dict[str, Any]:
        """Diagnostic metadata regarding model assets, weights, and dependencies."""
        pass


class BaseOCRAdapter(BaseAdapter):
    """Abstract interface for textile care-label OCR modules."""

    @abstractmethod
    async def extract_composition(
        self, image: ProcessedImage, **kwargs: Any
    ) -> OCRResult:
        """Extract material composition from clothing label image."""
        pass


class BaseVisionAdapter(BaseAdapter):
    """Abstract interface for surface computer vision fabric identification modules."""

    @abstractmethod
    async def predict_material(
        self, image: ProcessedImage, **kwargs: Any
    ) -> VisionPrediction:
        """Predict textile fabric category from garment image."""
        pass


class BaseKnowledgeAdapter(BaseAdapter):
    """Abstract interface for environmental life-cycle knowledge base retrieval."""

    @abstractmethod
    async def get_material_data(
        self, material_name: str
    ) -> Optional[MaterialEnvironmentalProfile]:
        """Retrieve environmental benchmark data for a single material."""
        pass

    @abstractmethod
    async def get_batch_data(
        self, materials: List[str]
    ) -> KnowledgeBaseResult:
        """Retrieve environmental profiles for multiple materials."""
        pass


class BaseScoringAdapter(BaseAdapter):
    """Abstract interface for environmental impact and recommendation scoring engines."""

    @abstractmethod
    def calculate_impact(
        self,
        composition: Dict[str, float],
        *,
        garment_condition: str = "new",
        shopping: bool = False,
        second_hand_available: bool = False,
        **kwargs: Any,
    ) -> ScoringResult:
        """Calculate environmental impact score and action recommendation."""
        pass
