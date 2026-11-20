"""FastAPI route handlers for GreenWardrobe.

Phase 7: POST /api/v1/analyze now accepts ONE image field ('image').
         The image-type classifier determines whether it is a garment
         photograph or a care-label photograph, then routes accordingly.

POST /api/v1/upload — legacy multi-image validation endpoint (unchanged).
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.core.config import settings
from app.core.exceptions import ImageClassificationError, ImageValidationError
from app.schemas.response import AnalysisResponse, APIErrorResponse
from app.services.analysis_service import analysis_service
from app.services.image_service import image_service

router = APIRouter()

MAX_FILE_SIZE = settings.MAX_UPLOAD_SIZE_BYTES

ALLOWED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


# ---------------------------------------------------------------------------
# Shared validation helper (legacy endpoint)
# ---------------------------------------------------------------------------

async def validate_image(file: UploadFile):
    """Validate uploaded image file using central ImageService."""
    try:
        processed = await image_service.validate_and_process_upload(file)
        return processed.to_dict()
    except ImageValidationError as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message,
        )


# ---------------------------------------------------------------------------
# Legacy upload endpoint (preserved — no changes)
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_garment(
    garment_image: UploadFile = File(...),
    label_image: UploadFile | None = File(default=None),
):
    garment_data = await validate_image(garment_image)
    label_data = None
    if label_image is not None:
        label_data = await validate_image(label_image)
    return {
        "message": "Images uploaded and validated successfully.",
        "garment_image": garment_data,
        "label_image": label_data,
    }


# ---------------------------------------------------------------------------
# Phase 7: Single-image analysis endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": APIErrorResponse, "description": "Empty upload or malformed/corrupt image."},
        413: {"model": APIErrorResponse, "description": "Uploaded image exceeds the configured size limit."},
        415: {"model": APIErrorResponse, "description": "Unsupported image format or media type."},
        422: {"model": APIErrorResponse, "description": "Missing/invalid request fields, invalid image, or image classified as UNKNOWN."},
        500: {"model": APIErrorResponse, "description": "Unexpected internal server error; response does not expose a traceback."},
    },
    summary="Analyse garment environmental impact from a single image",
    description=(
        "Accepts ONE image (JPEG, PNG, or WEBP) via multipart/form-data. "
        "The backend automatically classifies the image as either a garment "
        "photograph or a clothing-label photograph, routes it through the "
        "corresponding analysis pipeline (Computer Vision or OCR), calculates "
        "the environmental impact score, and returns a Gemini-grounded "
        "sustainability explanation. "
        "\n\n"
        "If the image cannot be confidently classified (confidence below "
        "threshold), a 422 response is returned with a human-readable message "
        "asking the user to upload a clearer image."
    ),
)
async def analyze_image(
    image: UploadFile = File(
        ...,
        description=(
            "Single clothing image — either a garment fabric surface photograph "
            "or a care/composition label photograph. "
            "Accepted formats: JPEG, PNG, WEBP. Maximum size: 10 MB."
        ),
    ),
    shopping: bool = Form(
        default=True,
        description="True if analysing before purchase, False for an existing item.",
    ),
    second_hand_available: bool = Form(
        default=False,
        description="True if second-hand alternatives for this garment are available.",
    ),
    condition: str = Form(
        default="new",
        description="Garment condition: 'new', 'used', or 'damaged'.",
    ),
) -> AnalysisResponse:
    """Analyse a single uploaded garment or label image through the full pipeline."""
    try:
        return await analysis_service.analyze_image(
            image_file=image,
            shopping=shopping,
            second_hand_available=second_hand_available,
            condition=condition,
        )
    except ImageClassificationError:
        # Re-raise as domain exception so the centralized handler formats it
        raise
    except ImageValidationError:
        raise
