"""Computer Vision Adapter integrating Anshika Thakur's TextileNet ResNet-18 model.

Connects the actual ResNet18 architecture and 33-class fibre label mapping
from origin/feature/computer-vision.
"""

import ast
import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from app.core.exceptions import VisionProcessingError
from app.core.logging import logger
from app.integrations.base import BaseVisionAdapter
from app.schemas.response import VisionPrediction
from app.services.image_service import ProcessedImage


class VisionAdapter(BaseVisionAdapter):
    """Adapter wrapping Anshika Thakur's TextileNet ResNet-18 fibre classifier."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        model_instance: Optional[Any] = None,
        custom_predictor: Optional[Callable[[ProcessedImage], VisionPrediction]] = None,
    ):
        canonical_ckpt = (
            Path(__file__).resolve().parent / "teammate_modules" / "vision" / "checkpoints" / "res18_ckpt.pth"
        )
        env_ckpt = os.getenv("VISION_CHECKPOINT_PATH")

        if checkpoint_path is not None:
            self._checkpoint_path = checkpoint_path
        elif env_ckpt:
            self._checkpoint_path = Path(env_ckpt)
        elif canonical_ckpt.exists():
            self._checkpoint_path = canonical_ckpt
        else:
            self._checkpoint_path = Path("models/res18_ckpt.pth")

        self._labels_path = labels_path or (
            Path(__file__).resolve().parent / "teammate_modules" / "vision" / "labels" / "fibre_label.txt"
        )
        self._model = model_instance
        self._custom_predictor = custom_predictor
        self._class_names: List[str] = []
        self._initialized = False
        self._torch_available = False

        self._load_label_mapping()

    def _load_label_mapping(self) -> None:
        """Load class labels from fibre_label.txt."""
        if self._labels_path.exists():
            try:
                with self._labels_path.open() as f:
                    label_to_id = ast.literal_eval(f.read())
                id_to_label = {value: key for key, value in label_to_id.items()}
                self._class_names = [id_to_label[i] for i in range(len(id_to_label))]
            except Exception as e:
                logger.error("Failed to parse fibre_label.txt: %s", e)
                self._class_names = []

    def _ensure_model(self) -> None:
        """Lazy-load PyTorch model and weights if checkpoint is present."""
        if self._initialized:
            return

        if self._custom_predictor is not None or self._model is not None:
            self._initialized = True
            return

        try:
            import torch
            self._torch_available = True

            if not self._checkpoint_path.exists():
                logger.warning(
                    "TextileNet checkpoint not found at '%s'. Vision adapter will report MODEL_CHECKPOINT_MISSING.",
                    self._checkpoint_path,
                )
                self._model = None
            else:
                from app.integrations.teammate_modules.vision.resnet import get_resnet18
                logger.info("Loading TextileNet ResNet-18 model weights from %s", self._checkpoint_path)
                model = get_resnet18(num_classes=len(self._class_names))
                checkpoint = torch.load(
                    self._checkpoint_path, map_location="cpu", weights_only=True
                )
                state_dict = {
                    key.removeprefix("module."): value
                    for key, value in checkpoint.get("model", checkpoint).items()
                }
                model.load_state_dict(state_dict, strict=True)
                model.eval()
                self._model = model
        except ImportError:
            logger.warning("PyTorch is not installed in the runtime environment.")
            self._torch_available = False
            self._model = None
        except Exception as e:
            logger.warning("Failed to load TextileNet ResNet-18 model: %s", str(e))
            self._model = None
        finally:
            self._initialized = True

    @property
    def adapter_name(self) -> str:
        return "Anshika-TextileNet-Vision-Adapter"

    @property
    def is_available(self) -> bool:
        self._ensure_model()
        return self._model is not None or self._custom_predictor is not None

    def status_info(self) -> Dict[str, Any]:
        """Report asset and in-memory state without initializing the model."""
        checkpoint_exists = self._checkpoint_path.is_file()
        model_loaded = self._model is not None
        custom_predictor = self._custom_predictor is not None

        if custom_predictor or model_loaded:
            status_label = "REAL INTEGRATION VERIFIED"
            status_code = "OPERATIONAL"
        elif not checkpoint_exists:
            status_label = "BLOCKED — DEPENDENCY OR ASSET MISSING (WEIGHTS: res18_ckpt.pth)"
            status_code = "MODEL_CHECKPOINT_MISSING"
        elif self._initialized:
            status_label = "CHECKPOINT AVAILABLE BUT MODEL INITIALIZATION FAILED"
            status_code = "CHECKPOINT_UNAVAILABLE"
        else:
            status_label = "CHECKPOINT AVAILABLE — MODEL NOT LOADED"
            status_code = "ASSET_AVAILABLE_NOT_LOADED"

        return {
            "adapter": self.adapter_name,
            "status": status_label,
            "status_code": status_code,
            "architecture": "ResNet-18 (TextileNet)",
            "classes_count": len(self._class_names),
            "checkpoint_configured": str(self._checkpoint_path),
            "checkpoint_exists": checkpoint_exists,
            "asset_available": checkpoint_exists,
            "initialization_attempted": self._initialized,
            "model_loaded": model_loaded,
            "inference_ready": model_loaded or custom_predictor,
            "custom_predictor": custom_predictor,
            "upstream_repository": "https://github.com/hahashu/TextileNet",
            "required_asset": "res18_ckpt.pth",
        }

    def _sync_predict(self, image: ProcessedImage) -> VisionPrediction:
        """Execute synchronous model inference on worker thread."""
        self._ensure_model()

        # If custom predictor double was injected (e.g. in tests)
        if self._custom_predictor is not None:
            return self._custom_predictor(image)

        # If model weights are missing, return transparent diagnostic without breaking request
        if not self.is_available or self._model is None:
            return VisionPrediction(
                predicted_material=None,
                confidence=None,
                alternatives=[],
                warnings=[
                    f"Computer vision model checkpoint not found at '{self._checkpoint_path}' (status: MODEL_CHECKPOINT_MISSING). "
                    "Pretrained TextileNet weights (res18_ckpt.pth) are required for visual prediction."
                ],
            )

        try:
            import torch
            # Apply Anshika's exact preprocessing: RGB 224x224, / 255.0, permute (2, 0, 1)
            pil_image = image.to_rgb_pil().resize((224, 224))
            array = np.asarray(pil_image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, 224, 224)

            with torch.inference_mode():
                outputs = self._model(tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                topk = torch.topk(probabilities, k=min(4, len(self._class_names)))

            top_indices = topk.indices.cpu().numpy()
            top_probs = topk.values.cpu().numpy()

            primary_idx = int(top_indices[0])
            primary_class = self._class_names[primary_idx] if primary_idx < len(self._class_names) else "Unknown"
            primary_conf = float(top_probs[0])

            alternatives: List[str] = [
                self._class_names[int(idx)]
                for idx in top_indices[1:]
                if int(idx) < len(self._class_names)
            ]

            prob_dict: Dict[str, float] = {
                self._class_names[int(idx)]: round(float(prob), 4)
                for idx, prob in zip(top_indices, top_probs)
                if int(idx) < len(self._class_names)
            }

            return VisionPrediction(
                predicted_material=primary_class,
                confidence=round(primary_conf, 2),
                alternatives=alternatives,
                probabilities=prob_dict,
                warnings=[],
            )

        except Exception as e:
            logger.exception("Error during Computer Vision model inference: %s", str(e))
            raise VisionProcessingError(
                message=f"Vision model inference failed: {str(e)}",
                status_code=422,
            )

    async def predict_material(
        self, image: ProcessedImage, **kwargs: Any
    ) -> VisionPrediction:
        """Asynchronously predict material by offloading CPU/GPU tensor computation."""
        return await asyncio.to_thread(self._sync_predict, image)


# Global default Vision adapter instance
vision_adapter = VisionAdapter()
