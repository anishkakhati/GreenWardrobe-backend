"""GreenWardrobe FastAPI Application.

Main entry point configuring middleware, CORS, centralized exception handlers,
and mounting API routers.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import (
    GreenWardrobeException,
    greenwardrobe_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
)
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger.info(
        "Starting %s v%s in %s mode (debug=%s)",
        settings.PROJECT_NAME,
        settings.VERSION,
        settings.ENVIRONMENT,
        settings.DEBUG,
    )
    logger.info("Allowed CORS origins: %s", settings.ALLOWED_ORIGINS)
    yield
    logger.info("Shutting down %s", settings.PROJECT_NAME)


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS) for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register centralized exception handlers
app.add_exception_handler(GreenWardrobeException, greenwardrobe_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# Mount API routers
app.include_router(
    router,
    prefix=settings.API_V1_STR,
    tags=["Garment Upload"],
)


@app.get("/", tags=["General"])
def home():
    """Root welcome endpoint."""
    return {
        "message": "Welcome to GreenWardrobe API",
        "status": "running",
        "version": settings.VERSION,
    }


@app.get("/health", tags=["Health"])
@app.get(f"{settings.API_V1_STR}/health", tags=["Health"])
def health_check():
    """Health check endpoint accessible at /health and /api/v1/health."""
    return {
        "status": "healthy",
        "service": "GreenWardrobe Backend",
        "environment": settings.ENVIRONMENT,
        "version": settings.VERSION,
    }
