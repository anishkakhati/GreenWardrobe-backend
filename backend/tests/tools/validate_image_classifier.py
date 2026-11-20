#!/usr/bin/env python3
"""Run the existing image classifier against a CSV validation manifest.

The manifest may reference repository images, deterministic generated inputs,
or optional images from the locally installed scikit-image sample package.
This utility never changes the production classifier or downloads data.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.integrations.image_type_classifier import (  # noqa: E402
    IMAGE_TYPE_GARMENT,
    IMAGE_TYPE_LABEL,
    IMAGE_TYPE_UNKNOWN,
    HeuristicImageTypeClassifier,
)
from app.services.image_service import ImageService  # noqa: E402

CLASSES = (IMAGE_TYPE_GARMENT, IMAGE_TYPE_LABEL, IMAGE_TYPE_UNKNOWN)
DEFAULT_MANIFEST = BACKEND_ROOT / "tests/fixtures/classifier_validation/manifest.csv"
REQUIRED_COLUMNS = {"filename", "ground_truth_category", "image_source", "notes"}


def _generated_image(filename: str) -> Image.Image:
    """Build a deterministic negative image from its manifest locator."""
    if filename == "generated_uniform_white":
        return Image.new("RGB", (256, 256), (255, 255, 255))
    if filename == "generated_uniform_black":
        return Image.new("RGB", (256, 256), (0, 0, 0))
    if filename == "generated_uniform_gray":
        return Image.new("RGB", (256, 256), (127, 127, 127))

    parts = filename.split("_")
    if filename.startswith("generated_rgb_iid_seed_"):
        seed = int(parts[-1])
        pixels = np.random.default_rng(seed).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        return Image.fromarray(pixels)

    if filename.startswith("generated_gray_iid_seed_"):
        seed = int(parts[-1])
        plane = np.random.default_rng(seed).integers(0, 256, (256, 256), dtype=np.uint8)
        return Image.fromarray(np.repeat(plane[:, :, None], 3, axis=2))

    if filename.startswith("generated_rgb_coarse_seed_"):
        seed = int(parts[-1])
        rng = np.random.default_rng(seed)
        blocks = rng.integers(50, 180, (32, 32, 3), dtype=np.uint8)
        pixels = np.repeat(np.repeat(blocks, 8, axis=0), 8, axis=1)
        noise = rng.integers(-8, 9, pixels.shape, dtype=np.int16)
        return Image.fromarray(np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8))

    raise ValueError(f"Unknown deterministic image generator: {filename}")


def _image_bytes(filename: str, manifest_dir: Path) -> tuple[bytes, str, str]:
    """Return encoded bytes, upload filename, and declared MIME type."""
    if filename.startswith("generated_"):
        image = _generated_image(filename)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), f"{filename}.png", "image/png"

    if filename.startswith("skimage_"):
        sample_name = filename.removeprefix("skimage_")
        try:
            from skimage import data
        except ImportError as exc:
            raise RuntimeError("scikit-image sample package is unavailable") from exc
        sample = getattr(data, sample_name, None)
        if not callable(sample):
            raise RuntimeError(f"scikit-image sample is unavailable: {sample_name}")
        image = Image.fromarray(sample()).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue(), f"{filename}.png", "image/png"

    path = (manifest_dir / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix)
    if mime is None:
        raise ValueError(f"Unsupported image extension in manifest: {path.name}")
    return path.read_bytes(), path.name, mime


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or [])
        combined_schema = {"relative_path", "task", "ground_truth_class", "split", "source"}.issubset(fieldnames)
        if not combined_schema:
            missing = REQUIRED_COLUMNS.difference(fieldnames)
            if missing:
                raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing))}")
        source_rows = list(reader)

    if combined_schema:
        rows = [
            {
                **row,
                "filename": row["relative_path"],
                "ground_truth_category": row["ground_truth_class"],
                "image_source": row["source"],
                "notes": row.get("annotation_notes", ""),
            }
            for row in source_rows
            if row.get("task", "").strip().lower() == "image_type"
            and row.get("split", "").strip().lower() == "test"
        ]
    else:
        rows = source_rows

    for line_number, row in enumerate(rows, start=2):
        category = row["ground_truth_category"].strip().upper()
        if category not in CLASSES:
            raise ValueError(f"Manifest line {line_number}: unsupported category {category!r}")
        if not row["filename"].strip():
            raise ValueError(f"Manifest line {line_number}: filename is empty")
        row["ground_truth_category"] = category
    return rows


def _metrics(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    confusion = {truth: {prediction: 0 for prediction in CLASSES} for truth in CLASSES}
    support = Counter(record["ground_truth_category"] for record in records)
    predicted = Counter(record["predicted_class"] for record in records)
    true_positive = Counter(
        record["ground_truth_category"]
        for record in records
        if record["ground_truth_category"] == record["predicted_class"]
    )

    for record in records:
        confusion[record["ground_truth_category"]][record["predicted_class"]] += 1

    per_class: dict[str, Any] = {}
    for category in CLASSES:
        tp = true_positive[category]
        fp = predicted[category] - tp
        fn = support[category] - tp
        if support[category] == 0:
            per_class[category] = {
                "status": "INSUFFICIENT_DATA",
                "reason": "No manifest examples have this ground-truth category.",
                "support": 0,
                "precision": None,
                "recall": None,
                "f1": None,
            }
            continue

        precision = tp / predicted[category] if predicted[category] else None
        recall = tp / support[category]
        f1 = (2 * precision * recall / (precision + recall)) if precision is not None and precision + recall else (0.0 if precision is not None else None)
        per_class[category] = {
            "status": "MEASURED" if precision is not None else "PARTIAL",
            "support": support[category],
            "interpretation": "Descriptive of this manifest only; not a production performance estimate.",
            "predicted_count": predicted[category],
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            **({"precision_note": "No predictions for this class; precision and F1 are unavailable."} if precision is None else {}),
        }

    errors = {
        "false_positives": {
            category: [
                {"filename": r["filename"], "ground_truth": r["ground_truth_category"], "confidence": r["confidence"]}
                for r in records
                if r["predicted_class"] == category and r["ground_truth_category"] != category
            ]
            for category in CLASSES
        },
        "false_negatives": {
            category: [
                {"filename": r["filename"], "predicted": r["predicted_class"], "confidence": r["confidence"]}
                for r in records
                if r["ground_truth_category"] == category and r["predicted_class"] != category
            ]
            for category in CLASSES
        },
    }
    return confusion, per_class, errors


def validate(manifest_path: Path) -> dict[str, Any]:
    rows = _load_manifest(manifest_path)
    classifier = HeuristicImageTypeClassifier()
    records: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []

    for row in rows:
        filename = row["filename"].strip()
        try:
            payload, upload_name, mime = _image_bytes(filename, manifest_path.parent)
        except (FileNotFoundError, RuntimeError) as exc:
            unavailable.append({"filename": filename, "reason": str(exc)})
            continue

        processed = ImageService.process_image(payload, upload_name, mime)
        try:
            result = classifier.classify(processed)
        finally:
            processed.cleanup()

        records.append({
            "filename": filename,
            "ground_truth_category": row["ground_truth_category"],
            "predicted_class": result.image_type,
            "confidence": result.confidence,
            "image_source": row["image_source"],
            "notes": row["notes"],
            "correct": result.image_type == row["ground_truth_category"],
        })

    confusion, per_class, errors = _metrics(records)
    manifest_counts = Counter(row["ground_truth_category"] for row in rows)
    evaluated_counts = Counter(record["ground_truth_category"] for record in records)
    supported_f1 = {
        category: metrics["f1"]
        for category, metrics in per_class.items()
        if metrics["support"] > 0 and metrics["f1"] is not None
    }
    total_supported = sum(per_class[category]["support"] for category in supported_f1)
    unknown_support = evaluated_counts[IMAGE_TYPE_UNKNOWN]
    garment_support = evaluated_counts[IMAGE_TYPE_GARMENT]
    unknown_as_garment = sum(
        record["ground_truth_category"] == IMAGE_TYPE_UNKNOWN
        and record["predicted_class"] == IMAGE_TYPE_GARMENT
        for record in records
    )
    garment_as_unknown = sum(
        record["ground_truth_category"] == IMAGE_TYPE_GARMENT
        and record["predicted_class"] == IMAGE_TYPE_UNKNOWN
        for record in records
    )

    confidence_distributions: dict[str, Any] = {}
    for category in CLASSES:
        class_records = [r for r in records if r["ground_truth_category"] == category]
        values = np.asarray([r["confidence"] for r in class_records], dtype=np.float64)
        confidence_distributions[category] = (
            {
                "status": "MEASURED",
                "count": int(values.size),
                "min": float(values.min()),
                "median": float(np.median(values)),
                "mean": float(values.mean()),
                "max": float(values.max()),
                "correct_mean": float(np.mean([r["confidence"] for r in class_records if r["correct"]])) if any(r["correct"] for r in class_records) else None,
                "incorrect_mean": float(np.mean([r["confidence"] for r in class_records if not r["correct"]])) if any(not r["correct"] for r in class_records) else None,
            }
            if values.size
            else {"status": "INSUFFICIENT_DATA", "count": 0}
        )

    return {
        "manifest": str(manifest_path),
        "classifier": classifier.classifier_name,
        "confidence_threshold": classifier.confidence_threshold,
        "examples_in_manifest": len(rows),
        "examples_evaluated": len(records),
        "ground_truth_counts_in_manifest": {category: manifest_counts[category] for category in CLASSES},
        "ground_truth_counts_evaluated": {category: evaluated_counts[category] for category in CLASSES},
        "category_metrics": per_class,
        "aggregate_metrics": {
            "macro_f1_over_supported_classes": float(np.mean(list(supported_f1.values()))) if supported_f1 else None,
            "weighted_f1_over_supported_classes": (
                float(sum(per_class[k]["f1"] * per_class[k]["support"] for k in supported_f1) / total_supported)
                if total_supported else None
            ),
            "f1_supported_classes": sorted(supported_f1),
            "f1_unsupported_classes": [category for category in CLASSES if per_class[category]["support"] == 0],
            "unknown_false_garment_acceptance_rate": unknown_as_garment / unknown_support if unknown_support else None,
            "unknown_false_garment_acceptance_support": unknown_support,
            "garment_false_unknown_rejection_rate": garment_as_unknown / garment_support if garment_support else None,
            "garment_false_unknown_rejection_support": garment_support,
            "interpretation": "Metrics describe this manifest only. Class metrics with zero ground-truth support are insufficient; aggregate F1 excludes unsupported classes.",
        },
        "confidence_distributions_by_ground_truth": confidence_distributions,
        "confusion_matrix_rows_truth_columns_prediction": confusion,
        "classification_errors": errors,
        "unavailable_examples": unavailable,
        "records": records,
        "validation_scope_note": "Synthetic and unrelated negatives do not measure garment recall. A class with no genuine examples is reported as insufficiently validated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="CSV manifest path")
    args = parser.parse_args()
    report = validate(args.manifest.resolve())
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
