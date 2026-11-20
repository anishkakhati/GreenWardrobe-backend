#!/usr/bin/env python3
"""Evaluate the local TextileNet fibre adapter on an approved held-out manifest.

This tool performs no training, downloads, threshold changes, or writes to the
manifest. Only rows with task=textilenet_fibre and split=test are evaluated.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.integrations.vision_adapter import VisionAdapter  # noqa: E402
from app.services.image_service import ImageService  # noqa: E402

REQUIRED_COLUMNS = {"relative_path", "task", "ground_truth_fibre", "split"}
MIN_CALIBRATION_SUPPORT = 100


def _load_rows(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    candidates = [
        row for row in rows
        if row.get("task", "").strip().lower() == "textilenet_fibre"
        and row.get("split", "").strip().lower() == "test"
    ]
    excluded = [
        {
            "image_id": row.get("image_id", ""),
            "reason": "Only task=textilenet_fibre and split=test rows are evaluated.",
        }
        for row in rows
        if row not in candidates
    ]
    return candidates, excluded


def _class_metrics(records: list[dict[str, Any]], labels: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = {truth: {predicted: 0 for predicted in labels} for truth in labels}
    truth_counts = Counter(row["ground_truth_fibre"] for row in records)
    prediction_counts = Counter(row["predicted_fibre"] for row in records)
    true_positive = Counter(
        row["ground_truth_fibre"] for row in records
        if row["ground_truth_fibre"] == row["predicted_fibre"]
    )
    for row in records:
        matrix[row["ground_truth_fibre"]][row["predicted_fibre"]] += 1

    per_class: dict[str, Any] = {}
    for label in labels:
        support = truth_counts[label]
        if support == 0:
            per_class[label] = {
                "status": "INSUFFICIENT_DATA",
                "support": 0,
                "precision": None,
                "recall": None,
                "f1": None,
            }
            continue
        tp = true_positive[label]
        precision = tp / prediction_counts[label] if prediction_counts[label] else 0.0
        recall = tp / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "status": "MEASURED",
            "support": support,
            "predicted_count": prediction_counts[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return matrix, per_class


def _calibration(records: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    if len(records) < MIN_CALIBRATION_SUPPORT:
        return {
            "status": "INSUFFICIENT_DATA",
            "sample_count": len(records),
            "minimum_sample_count": MIN_CALIBRATION_SUPPORT,
        }
    bin_rows = []
    weighted_gap = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            row for row in records
            if low <= row["confidence"] < high or (index == bins - 1 and row["confidence"] == 1.0)
        ]
        if not members:
            bin_rows.append({"lower": low, "upper": high, "count": 0, "mean_confidence": None, "accuracy": None})
            continue
        mean_conf = sum(row["confidence"] for row in members) / len(members)
        accuracy = sum(row["correct"] for row in members) / len(members)
        weighted_gap += len(members) / len(records) * abs(mean_conf - accuracy)
        bin_rows.append({
            "lower": low,
            "upper": high,
            "count": len(members),
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
        })
    return {
        "status": "MEASURED_EXPLORATORY",
        "sample_count": len(records),
        "expected_calibration_error": weighted_gap,
        "bins": bin_rows,
        "interpretation": "Exploratory only; calibration validity depends on representative, independent test data.",
    }


async def _run(manifest: Path) -> dict[str, Any]:
    rows, excluded = _load_rows(manifest)
    adapter = VisionAdapter()
    status = adapter.status_info()
    checkpoint_version = None
    if status["checkpoint_exists"]:
        digest = hashlib.sha256()
        with adapter._checkpoint_path.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_version = digest.hexdigest()

    if not rows:
        return {
            "status": "INSUFFICIENT_DATA",
            "task": "textilenet_fibre",
            "test_rows": 0,
            "excluded_rows": excluded,
            "checkpoint_exists": status["checkpoint_exists"],
            "checkpoint_sha256": checkpoint_version,
            "message": "No held-out TextileNet fibre rows are available; no model inference or performance metric was produced.",
        }
    if not status["checkpoint_exists"]:
        return {
            "status": "BLOCKED",
            "task": "textilenet_fibre",
            "test_rows": len(rows),
            "checkpoint_exists": False,
            "message": "Local checkpoint is missing. The tool does not download weights.",
        }

    labels = adapter._class_names
    records: list[dict[str, Any]] = []
    inference_failures: list[dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=2):
        ground_truth = row["ground_truth_fibre"].strip().lower()
        if ground_truth not in labels:
            raise ValueError(f"Manifest line {line_number}: unknown fibre label {ground_truth!r}")
        relative = row["relative_path"].strip()
        if not relative:
            raise ValueError(f"Manifest line {line_number}: relative_path is empty")
        image_path = (manifest.parent / relative).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Manifest line {line_number}: image not found: {relative}")
        processed = ImageService.process_image(image_path.read_bytes(), image_path.name)
        try:
            prediction = await adapter.predict_material(processed)
        finally:
            processed.cleanup()
        if prediction.predicted_material is None or prediction.confidence is None:
            inference_failures.append({
                "image_id": row.get("image_id", ""),
                "relative_path": relative,
                "warnings": prediction.warnings,
            })
            continue
        top_three = ([prediction.predicted_material] if prediction.predicted_material else []) + prediction.alternatives[:2]
        top_three = [value.lower() for value in top_three]
        records.append({
            "image_id": row.get("image_id", ""),
            "ground_truth_fibre": ground_truth,
            "predicted_fibre": (prediction.predicted_material or "").lower(),
            "confidence": prediction.confidence,
            "top_3_predictions": top_three,
            "checkpoint_version": checkpoint_version,
            "correct": prediction.predicted_material is not None and prediction.predicted_material.lower() == ground_truth,
            "top_3_correct": ground_truth in top_three,
            "warnings": prediction.warnings,
        })

    if not records:
        return {
            "status": "INFERENCE_UNAVAILABLE",
            "task": "textilenet_fibre",
            "test_rows": len(rows),
            "evaluated_rows": 0,
            "inference_failures": inference_failures,
            "checkpoint_sha256": checkpoint_version,
            "message": "No predictions were available; no performance metrics were calculated.",
        }

    matrix, per_class = _class_metrics(records, labels)
    supported = [value for value in per_class.values() if value["support"] > 0]
    total_support = sum(value["support"] for value in supported)
    return {
        "status": "MEASURED_DESCRIPTIVE",
        "task": "textilenet_fibre",
        "architecture": "ResNet-18 (TextileNet)",
        "class_count": len(labels),
        "test_rows": len(records),
        "excluded_rows": excluded,
        "inference_failures": inference_failures,
        "checkpoint_sha256": checkpoint_version,
        "top_1_accuracy": sum(row["correct"] for row in records) / len(records),
        "top_3_accuracy": sum(row["top_3_correct"] for row in records) / len(records),
        "macro_f1_over_supported_classes": sum(row["f1"] for row in supported) / len(supported) if supported else None,
        "weighted_f1_over_supported_classes": sum(row["f1"] * row["support"] for row in supported) / total_support if total_support else None,
        "per_class_metrics": per_class,
        "confusion_matrix_rows_truth_columns_prediction": matrix,
        "confidence_calibration": _calibration(records),
        "records": records,
        "limitations": [
            "Metrics describe only this held-out manifest and do not validate image-type classification.",
            "A TextileNet fibre prediction is not laboratory-confirmed garment composition.",
            "No model is downloaded and the source manifest is not modified.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Approved combined model-validation CSV manifest")
    args = parser.parse_args()
    report = asyncio.run(_run(args.manifest.resolve()))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
