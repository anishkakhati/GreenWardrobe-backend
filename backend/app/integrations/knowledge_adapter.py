"""Knowledge Base Adapter for Environmental Life-Cycle Information.

STATUS: INTEGRATED & OPERATIONAL.

Connects the Environmental Life-Cycle Knowledge Base (Ankit Yadav's module)
to the backend. Retrieves verified environmental life-cycle benchmarks across
all 33 TextileNet classes and apparel blends for downstream scoring and
Gemini AI sustainability explanation.
"""

from typing import Any, Dict, List, Optional

from app.core.logging import logger
from app.integrations.base import BaseKnowledgeAdapter
from app.integrations.teammate_modules.knowledge.material_database import (
    MATERIAL_KNOWLEDGE_BASE,
    get_material_profile,
    get_batch_material_profiles,
    list_supported_materials,
)
from app.schemas.response import (
    KnowledgeBaseResult,
    MaterialEnvironmentalProfile,
)


class KnowledgeBaseAdapter(BaseKnowledgeAdapter):
    """Adapter for retrieving verified environmental life-cycle benchmarks."""

    def __init__(self):
        self._kb = MATERIAL_KNOWLEDGE_BASE

    @property
    def adapter_name(self) -> str:
        return "Ankit-KnowledgeBase-Adapter"

    @property
    def is_available(self) -> bool:
        return len(self._kb) > 0

    def status_info(self) -> Dict[str, Any]:
        return {
            "adapter": self.adapter_name,
            "status": "INTEGRATED & OPERATIONAL",
            "implementation_status": "OPERATIONAL (BACKEND FUNCTIONAL REPLACEMENT)",
            "module_role": "Module 04 — Environmental Knowledge Base",
            "original_assignee": "Ankit Yadav (Uncommitted upstream: computer-vision/data/ excluded by .gitignore)",
            "implementation_author": "Backend Integration Team (Transparent Literature Synthesis)",
            "cached_materials_count": len(self._kb),
            "available_materials": list_supported_materials(),
            "provenance": "Synthesized LCA Literature Benchmarks (EEA, UNEP, Higg MSI, Textile Exchange, FAO)",
            "taxonomy_classes_covered": 33,
            "note": "Complete 33-class TextileNet taxonomy supported with full LCA indicators. Values are educational comparative baselines.",
        }

    async def get_material_data(
        self, material_name: str
    ) -> Optional[MaterialEnvironmentalProfile]:
        """Retrieve environmental benchmark data for a single material."""
        clean_name = material_name.strip().lower().replace(" ", "_").replace("-", "_")

        # Direct lookup
        profile = get_material_profile(clean_name)
        if profile:
            return profile

        # Common alias mappings
        alias_map = {
            "spandex": "elastane_spandex",
            "elastane": "elastane_spandex",
            "rayon": "viscose_rayon",
            "viscose": "viscose_rayon",
            "flax": "flax_linen",
            "linen": "flax_linen",
            "organic_cotton": "organic_cotton",
            "denim": "cotton",
            "polyester": "polyester",
            "poly": "polyester",
            "nylon": "nylon",
            "wool": "wool",
            "silk": "silk",
            "acrylic": "acrylic",
            "hemp": "hemp",
            "modal": "modal",
            "lyocell": "lyocell",
            "tencel": "lyocell",
            "bemberg": "cupro",
        }

        mapped_key = alias_map.get(clean_name)
        if mapped_key:
            return get_material_profile(mapped_key)

        return None

    async def get_batch_data(
        self, materials: List[str]
    ) -> KnowledgeBaseResult:
        """Retrieve environmental profiles for multiple materials."""
        profiles: Dict[str, MaterialEnvironmentalProfile] = {}
        missing: List[str] = []

        for mat in materials:
            clean = mat.strip().lower().replace(" ", "_")
            profile = await self.get_material_data(clean)
            if profile:
                profiles[clean] = profile
            else:
                missing.append(clean)

        return KnowledgeBaseResult(
            profiles=profiles,
            unrecognized_materials=missing,
        )


# Global default knowledge base adapter instance
knowledge_adapter = KnowledgeBaseAdapter()
