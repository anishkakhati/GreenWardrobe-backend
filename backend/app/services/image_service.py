"""Image validation, decoding, and in-memory stream processing service.

Provides secure image handling for GreenWardrobe:
- In-memory stream processing without temporary disk writes.
- File magic bytes and signature verification.
- Content-type / MIME consistency checks.
- Decompression bomb and dimension boundary protection.
- Safe filename sanitization (preventing path traversal).
- EXIF orientation normalization.
- Reusable, thread-safe ProcessedImage objects for perception pipelines.
"""

import io
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import DecompressionBombError

from app.core.config import settings
from app.core.exceptions import ImageValidationError
from app.core.logging import logger
from app.schemas.request import ImageMetadata

# Set Pillow global max pixels to guard against decompression bombs
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS

# Known magic bytes signatures for allowed image formats
_MAGIC_BYTES_MAP = {
    "JPEG": (b"\xff\xd8\xff",),
    "PNG": (b"\x89PNG\r\n\x1a\n",),
    "WEBP": (b"RIFF",),  # Also checked for 'WEBP' at offset 8:12
}

# Standard MIME type mappings for allowed formats
_ALLOWED_MIME_MAP = {
    "JPEG": ["image/jpeg", "image/jpg", "image/pjpeg"],
    "PNG": ["image/png"],
    "WEBP": ["image/webp"],
}

_SAFE_FILENAME_REGEX = re.compile(r"[^a-zA-Z0-9_\-\.]")


class ProcessedImage:
    """In-memory representation of a validated image.

    Allows multiple perception services (OCR, Computer Vision) to access
    clean, uncorrupted image instances and bytes without disk re-reads.
    """

    def __init__(self, raw_bytes: bytes, metadata: ImageMetadata):
        self.raw_bytes = raw_bytes
        self.metadata = metadata

    def get_stream(self) -> io.BytesIO:
        """Return a new in-memory BytesIO stream of the raw image bytes."""
        return io.BytesIO(self.raw_bytes)

    def to_pil(self) -> Image.Image:
        """Decode image into a Pillow Image instance with EXIF orientation applied."""
        img = Image.open(self.get_stream())
        return ImageOps.exif_transpose(img) or img

    def to_rgb_pil(self) -> Image.Image:
        """Decode image and normalize to standard 3-channel RGB format.

        Essential for computer vision models (ViT/CNN) expecting RGB tensors.
        """
        img = self.to_pil()
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for legacy endpoint compatibility."""
        return {
            "filename": self.metadata.filename,
            "format": self.metadata.format,
            "width": self.metadata.width,
            "height": self.metadata.height,
            "size_bytes": self.metadata.size_bytes,
        }

    def cleanup(self) -> None:
        """Clear memory references."""
        self.raw_bytes = b""


class ImageService:
    """Service encapsulating image validation, security checks, and processing."""

    @staticmethod
    def sanitize_filename(filename: Optional[str]) -> str:
        """Sanitize filename to prevent path traversal and shell injection attacks.

        Extracts the basename, removes directory separators, and filters unsafe chars.
        """
        if not filename or not filename.strip():
            raise ImageValidationError(
                message="No file selected.",
                status_code=400,
            )

        # Extract only the base name (prevents ../ path traversal)
        base_name = Path(filename).name.strip()
        if not base_name:
            raise ImageValidationError(
                message="Invalid filename provided.",
                status_code=400,
            )

        # Replace whitespace and unsafe characters with underscores
        sanitized = _SAFE_FILENAME_REGEX.sub("_", base_name)

        # Truncate to maximum 255 characters
        if len(sanitized) > 255:
            name_part, ext = os.path.splitext(sanitized)
            sanitized = name_part[: 255 - len(ext)] + ext

        return sanitized

    @staticmethod
    def detect_magic_format(data: bytes) -> Optional[str]:
        """Inspect the initial bytes of the file to determine the true format."""
        if len(data) < 12:
            return None

        # Check JPEG
        if data.startswith(_MAGIC_BYTES_MAP["JPEG"][0]):
            return "JPEG"

        # Check PNG
        if data.startswith(_MAGIC_BYTES_MAP["PNG"][0]):
            return "PNG"

        # Check WEBP: Starts with 'RIFF' and has 'WEBP' at bytes 8..12
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "WEBP"

        return None

    @classmethod
    def validate_mime_type(
        cls, content_type: Optional[str], detected_format: str
    ) -> None:
        """Verify that declared Content-Type header does not conflict with detected format."""
        if not content_type:
            return

        normalized_content_type = content_type.lower().strip()
        # Skip generic stream types
        if normalized_content_type in ["application/octet-stream", "binary/octet-stream"]:
            return

        allowed_mimes = _ALLOWED_MIME_MAP.get(detected_format, [])
        if normalized_content_type not in allowed_mimes:
            logger.warning(
                "MIME type mismatch: declared '%s' but file content is '%s'",
                content_type,
                detected_format,
            )
            raise ImageValidationError(
                message=f"MIME type mismatch: declared '{content_type}' but actual image content is {detected_format}.",
                status_code=415,
                details={
                    "declared_mime": content_type,
                    "detected_format": detected_format,
                },
            )

    @classmethod
    def inspect_and_verify(
        cls, data: bytes, filename: str
    ) -> Tuple[str, int, int]:
        """Perform comprehensive Pillow verification, dimension, and bomb checks."""
        try:
            # First pass: Pillow structure verification
            with Image.open(io.BytesIO(data)) as img_verify:
                img_verify.verify()

            # Second pass: Open to inspect header and dimensions
            with Image.open(io.BytesIO(data)) as img:
                detected_format = img.format
                width = img.width
                height = img.height

                if not detected_format:
                    raise ImageValidationError(
                        message="Invalid or corrupted image.",
                        status_code=400,
                    )

                # Ensure format is allowed (e.g. rejects GIF, BMP, TIFF)
                if detected_format not in settings.ALLOWED_IMAGE_FORMATS:
                    raise ImageValidationError(
                        message="Only JPEG, PNG and WEBP images are allowed.",
                        status_code=415,
                        details={"detected_format": detected_format},
                    )

                # Check minimum dimensions
                if width <= 0 or height <= 0:
                    raise ImageValidationError(
                        message="Image dimensions must be greater than zero.",
                        status_code=400,
                        details={"width": width, "height": height},
                    )

                # Check maximum single dimension limits
                if (
                    width > settings.MAX_IMAGE_DIMENSION
                    or height > settings.MAX_IMAGE_DIMENSION
                ):
                    raise ImageValidationError(
                        message=f"Image dimensions ({width}x{height}) exceed maximum allowed dimension of {settings.MAX_IMAGE_DIMENSION}px.",
                        status_code=400,
                        details={
                            "width": width,
                            "height": height,
                            "max_dimension": settings.MAX_IMAGE_DIMENSION,
                        },
                    )

                # Check total pixel count for decompression bomb protection
                total_pixels = width * height
                if total_pixels > settings.MAX_IMAGE_PIXELS:
                    raise ImageValidationError(
                        message="Image exceeds maximum pixel limit (decompression bomb protection).",
                        status_code=400,
                        details={
                            "total_pixels": total_pixels,
                            "max_pixels": settings.MAX_IMAGE_PIXELS,
                        },
                    )

                return detected_format, width, height

        except ImageValidationError:
            raise
        except DecompressionBombError as e:
            logger.warning("Decompression bomb detected in image '%s': %s", filename, str(e))
            raise ImageValidationError(
                message="Image exceeds safe pixel density limits (decompression bomb detected).",
                status_code=400,
            )
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            logger.warning("Corrupted or unreadable image uploaded '%s': %s", filename, str(e))
            raise ImageValidationError(
                message="Invalid or corrupted image.",
                status_code=400,
            )

    @classmethod
    def process_image(
        cls,
        data: bytes,
        filename: Optional[str],
        content_type: Optional[str] = None,
    ) -> ProcessedImage:
        """Validate and process raw image bytes entirely in memory."""
        safe_filename = cls.sanitize_filename(filename)

        # Check empty bytes
        if not data or len(data) == 0:
            raise ImageValidationError(
                message="Uploaded file is empty.",
                status_code=400,
            )

        # Check maximum file size
        if len(data) > settings.MAX_UPLOAD_SIZE_BYTES:
            raise ImageValidationError(
                message="Image size must not exceed 10 MB.",
                status_code=413,
                details={
                    "size_bytes": len(data),
                    "max_size_bytes": settings.MAX_UPLOAD_SIZE_BYTES,
                },
            )

        # Deep verification and dimension extraction (raises 400 for corrupted, 415 for unsupported formats)
        pil_format, width, height = cls.inspect_and_verify(data, safe_filename)

        # Verify magic bytes match the detected format
        magic_format = cls.detect_magic_format(data)
        if magic_format != pil_format:
            raise ImageValidationError(
                message="Invalid or corrupted image.",
                status_code=400,
                details={
                    "pil_format": pil_format,
                    "magic_format": magic_format,
                },
            )

        # Verify MIME consistency
        cls.validate_mime_type(content_type, pil_format)

        # Assemble metadata
        mime_type = _ALLOWED_MIME_MAP.get(pil_format, ["application/octet-stream"])[0]
        metadata = ImageMetadata(
            filename=safe_filename,
            format=pil_format,
            width=width,
            height=height,
            size_bytes=len(data),
            mime_type=mime_type,
        )

        return ProcessedImage(raw_bytes=data, metadata=metadata)

    @classmethod
    async def validate_and_process_upload(
        cls, file: UploadFile
    ) -> ProcessedImage:
        """Safely read and process a FastAPI UploadFile stream in memory."""
        if not file.filename:
            raise ImageValidationError(
                message="No file selected.",
                status_code=400,
            )

        # Bounded read to prevent memory exhaustion
        # Read up to MAX_UPLOAD_SIZE_BYTES + 1
        contents = await file.read(settings.MAX_UPLOAD_SIZE_BYTES + 1)

        return cls.process_image(
            data=contents,
            filename=file.filename,
            content_type=file.content_type,
        )


# Global singleton instance
image_service = ImageService()
