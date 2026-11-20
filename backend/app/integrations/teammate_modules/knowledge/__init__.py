"""Knowledge Base Module for Textile Materials and Environmental LCA Benchmarks.

Assigned Owner: Ankit Yadav (Module 04).
"""

from app.integrations.teammate_modules.knowledge.material_database import (
    MATERIAL_KNOWLEDGE_BASE,
    get_material_profile,
    get_batch_material_profiles,
    list_supported_materials,
)

__all__ = [
    "MATERIAL_KNOWLEDGE_BASE",
    "get_material_profile",
    "get_batch_material_profiles",
    "list_supported_materials",
]
