"""Integrations module connecting external services and teammate modules."""

from app.integrations.base import (
    BaseAdapter,
    BaseKnowledgeAdapter,
    BaseOCRAdapter,
    BaseScoringAdapter,
    BaseVisionAdapter,
)
from app.integrations.knowledge_adapter import (
    KnowledgeBaseAdapter,
    knowledge_adapter,
)
from app.integrations.ocr_adapter import (
    OCRAdapter,
    ocr_adapter,
)
from app.integrations.scoring_adapter import (
    ScoringAdapter,
    scoring_adapter,
)
from app.integrations.vision_adapter import (
    VisionAdapter,
    vision_adapter,
)

__all__ = [
    "BaseAdapter",
    "BaseOCRAdapter",
    "BaseVisionAdapter",
    "BaseKnowledgeAdapter",
    "BaseScoringAdapter",
    "OCRAdapter",
    "ocr_adapter",
    "VisionAdapter",
    "vision_adapter",
    "ScoringAdapter",
    "scoring_adapter",
    "KnowledgeBaseAdapter",
    "knowledge_adapter",
]
