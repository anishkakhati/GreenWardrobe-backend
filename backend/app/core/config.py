"""Application configuration settings using Pydantic Settings.

Manages all environment variables, CORS settings, upload limits,
and external AI service configurations.
"""

from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core settings for GreenWardrobe backend."""

    # Project metadata
    PROJECT_NAME: str = "GreenWardrobe API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "AI-powered garment environmental analysis platform"
    API_V1_STR: str = "/api/v1"

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # CORS configuration
    # Can be provided in .env as comma-separated string or JSON list
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # File upload limits
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_IMAGE_FORMATS: List[str] = ["JPEG", "PNG", "WEBP"]
    MAX_IMAGE_PIXELS: int = 25_000_000  # 25 MP decompression bomb limit
    MAX_IMAGE_DIMENSION: int = 8192     # Max width or height in pixels

    # Google Gemini API configuration
    GEMINI_API_KEY: str = Field(
        default="",
        description="Google Gemini API key obtained from Google AI Studio"
    )
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    GEMINI_TIMEOUT_SECONDS: int = 30

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated strings or preserve list format."""
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, list):
            return v
        return []

    @property
    def is_production(self) -> bool:
        """Helper to determine whether running in production mode."""
        return self.ENVIRONMENT.lower() == "production"


# Singleton instance for application-wide access
settings = Settings()
