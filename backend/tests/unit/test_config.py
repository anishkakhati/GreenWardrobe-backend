"""Unit tests for GreenWardrobe configuration and environment parsing."""

from app.core.config import Settings


def test_default_settings():
    """Verify default application configuration values."""
    settings = Settings()
    assert settings.PROJECT_NAME == "GreenWardrobe API"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.MAX_UPLOAD_SIZE_BYTES == 10 * 1024 * 1024
    assert "JPEG" in settings.ALLOWED_IMAGE_FORMATS
    assert "http://localhost:5173" in settings.ALLOWED_ORIGINS


def test_cors_origins_parsing_from_comma_string():
    """Verify comma-separated ALLOWED_ORIGINS string parses into a clean list."""
    settings = Settings(ALLOWED_ORIGINS="http://localhost:5173, http://example.com,http://app.greenwardrobe.com")
    assert settings.ALLOWED_ORIGINS == [
        "http://localhost:5173",
        "http://example.com",
        "http://app.greenwardrobe.com",
    ]


def test_cors_origins_parsing_from_list():
    """Verify list format for ALLOWED_ORIGINS is preserved."""
    origins = ["http://localhost:5173", "http://testserver"]
    settings = Settings(ALLOWED_ORIGINS=origins)
    assert settings.ALLOWED_ORIGINS == origins


def test_is_production_flag():
    """Verify is_production property responds to environment setting."""
    dev_settings = Settings(ENVIRONMENT="development")
    assert dev_settings.is_production is False

    prod_settings = Settings(ENVIRONMENT="production")
    assert prod_settings.is_production is True
