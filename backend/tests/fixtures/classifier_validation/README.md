# Image classifier validation set

`manifest.csv` is the validation inventory. Each row has four columns:

| Column | Meaning |
| --- | --- |
| `filename` | Relative image path, or a documented `generated_...` / `skimage_...` locator |
| `ground_truth_category` | `GARMENT`, `LABEL`, or `UNKNOWN` |
| `image_source` | Provenance of the image |
| `notes` | Context; synthetic inputs must be identified as synthetic |

The five existing label fixture files are referenced in place. The manifest
also defines deterministic in-memory blank and random-texture negatives. They
are not garment examples. `skimage_astronaut` and `skimage_rocket` refer to
sample images from an already installed scikit-image package; the validator
loads them at runtime and does not copy or download them. If scikit-image is
unavailable, those rows are reported as unavailable and excluded from metric
counts.

Current manifest inventory: 5 LABEL, 12 UNKNOWN, and 0 GARMENT examples. The
GARMENT category must therefore be reported as insufficiently validated.

## Run

From the backend directory:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tests/tools/validate_image_classifier.py
```

Use a different CSV with the same columns by passing `--manifest path/to/manifest.csv`.
The JSON report includes each prediction and confidence, a truth-by-prediction
confusion matrix, per-class precision/recall/F1 where defined, unavailable
manifest entries, and false-positive/false-negative examples. Missing category
support is reported as `INSUFFICIENT_DATA`; no accuracy value is synthesized.

To add genuine validation images, place them under a suitable fixture directory
and add relative-path rows to the manifest with their source and notes. Do not
label generated textures as garments. Keep a meaningful set of genuine garment
photographs before interpreting GARMENT recall or recommending classifier
changes.
