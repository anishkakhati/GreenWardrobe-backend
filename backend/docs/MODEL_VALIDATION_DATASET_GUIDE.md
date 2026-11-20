# GreenWardrobe Model Validation Dataset Guide

This guide defines a reproducible, consented collection and evaluation process for two different models. It does not claim that either model is validated today.

## The two validation problems

| Task | Classes / target | Purpose | What it does not establish |
|---|---|---|---|
| Image-type classifier | `LABEL`, `GARMENT`, `UNKNOWN` | Choose OCR, CV, or rejection before either pipeline runs | Fibre/material accuracy |
| TextileNet fibre classifier | 33 fibre taxonomy labels in `fibre_label.txt` | Predict the visible fibre/material class from a garment image | Whether an input is a garment or label; verified blend composition |

Do not report TextileNet fibre performance as image-type performance. Do not interpret a visual fibre estimate as laboratory-confirmed composition. A garment with a blend needs an explicit annotation rule and verified label/source before it can be evaluated as one of the model's single fibre classes.

## Inventory and current evidence

The current image-type validation manifest has 5 LABEL, 12 UNKNOWN and 0 genuine GARMENT rows. Nine of those 12 UNKNOWN examples were accepted as GARMENT in the previous validation. The label fixtures are synthetic example labels suitable for pipeline regression, not evidence of garment-class performance. No TextileNet held-out images or truth annotations are currently supplied in this repository.

`backend/tests/fixtures/model_validation/manifest_template.csv` is an empty header-only template. Do not add invented records or inferred metrics. Store any approved future dataset in a separately reviewed location and add manifest rows only after permission and label review.

## Collection protocol

Obtain written consent or a license that permits local model evaluation and retention. Record the source and permission for each image. Do not scrape personal photographs or use a dataset until its license and permitted use have been checked. Avoid faces, addresses, order details, labels with personal information, and other unnecessary identifiers; crop or redact only if doing so does not change the classification target, and retain provenance of any transformations.

### LABEL collection strata

Include clear labels; folded, curved, rotated, low-resolution, poorly lit and partially obscured labels; multiple fibre percentages; mixed compositions; different fonts, languages, label materials, printing quality and camera distances. Record whether the label composition was transcribed manually from the pixels and independently checked against the original garment label. Include difficult/unreadable cases as LABEL for image-type classification, while leaving the fibre ground truth empty if its composition cannot be reliably transcribed.

### GARMENT collection strata

Seek permissioned examples spanning cotton, wool, polyester, acrylic, nylon, linen, silk and blends; varied colours, textures, knits/weaves, garment types, flat/folded presentation, lighting, distance, camera devices and backgrounds. Ground truth for textile fibres must come from a readable care label, reliable product documentation, or a verified textile sample; annotators must not infer a fibre from appearance alone. For blend inputs, record all known components and do not force them into a single-class target unless a predeclared evaluation rule (such as a certified predominant surface fibre) is applicable.

### UNKNOWN collection strata

Include blank frames, screenshots, random textures, food, furniture, landscapes, documents, unrelated objects and non-textile photos. These are negative controls for routing, not garment data. Keep synthetic examples explicitly marked `source=synthetic`; never count them as genuine garment examples or use them to estimate garment recall.

## Annotation, splits and leakage controls

1. Assign a stable random `image_id`; keep the image-to-identity lookup separately with restricted access where needed.
2. Have two people independently assign image type. Resolve disagreements before evaluation and preserve the adjudication note.
3. For fibre classification, annotate a verified fibre target from source documentation, record the source, and have a second annotator check it. Record blends as notes; do not guess a single truth label.
4. Group near-duplicate frames by garment, physical sample, source product and capture session before splitting. All related frames must remain in one split.
5. Separate training, validation and held-out test groups. Threshold/model selection may use training/validation only. Run the final metric report on the locked test split once per model/checkpoint candidate.
6. Keep class counts and missing classes visible. A class with no held-out ground-truth examples is `INSUFFICIENT_DATA`, never zero-error or implicitly passed.
7. Preserve original images read-only. Do not overwrite, rename or transform source files in place. Record preprocessing separately and use the same documented preprocessing at inference.

## Manifest schema

The header-only CSV supports both tasks. `task` is `image_type` or `textilenet_fibre`; fill the matching ground-truth column and leave the other blank. Keep `relative_path` relative to the manifest's approved dataset root. `consent_or_license` should identify the permission/license record, not contain private contact details. `split` must be `train`, `validation`, or `test`.

The output-only fields `predicted_fibre`, `confidence`, `top_3_predictions`, and `checkpoint_version` are reserved for a generated evaluation report; do not prefill them with guesses and do not make the validator overwrite the source manifest.

## Metrics and release interpretation

For image-type classification report the 3×3 confusion matrix, per-class precision/recall/F1, macro and support-weighted F1 over supported classes, UNKNOWN-to-GARMENT false acceptance rate, GARMENT-to-UNKNOWN rejection rate, confidence distributions split by correct/incorrect predictions, and counts. Include false examples by stable ID. Confidence is a score, not a calibrated probability; use calibration plots/ECE only with sufficient held-out support.

For TextileNet report top-1 and top-3 accuracy, a 33-class confusion matrix, per-class precision/recall/F1, macro/weighted metrics, confidence distributions and checkpoint hash/version. Calibration is exploratory until there is adequate independent test volume; report bin counts and uncertainty. No class without test samples gets a metric. The tools do not download checkpoints or modify thresholds.

Metrics describe this collection only. Report class counts, provenance, split method, exclusions, confidence limitations, checkpoint hash and protocol version next to every result. Model validation does not validate environmental scoring data or its scientific basis.

## Existing TextileNet source and unresolved provenance

The public [TextileNet repository](https://github.com/hahashu/TextileNet) describes fibre and fabric taxonomies, a dataset download process, training code, a fibre label map and baseline results. Its README reports baseline results for ResNet50 and ViT, while the source includes a ResNet-18 training script and an architecture module. The public README states the collection code is MIT-licensed and describes a CC BY license for an Author Accepted Manuscript; that statement does not by itself establish that every raw dataset image/checkpoint can be redistributed or used in this application. Obtain precise dataset/checkpoint licenses and artifact provenance before redistribution or deployment.

The GreenWardrobe local adapter, local vendored `resnet.py`, 33-label map, and ignored 44,853,197-byte checkpoint are distinct artifacts. Local initialization and a forward pass only verify that these files load together. They do not prove that the local checkpoint was trained by Anshika, on the upstream split, with matching preprocessing, or achieved any reported baseline. The current local inference preprocessing resizes to 224×224 RGB and scales to [0,1]; the upstream training script's test transform also resizes to 224×224 and converts to a tensor without ImageNet normalization, but checkpoint-specific provenance still needs confirmation.

## Validation commands

Image-type classifier, current manifest:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tests/tools/validate_image_classifier.py
```

TextileNet, an approved manifest with held-out fibre labels:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python tests/tools/validate_textilenet.py --manifest path/to/approved_manifest.csv
```

The TextileNet tool requires an already available local checkpoint and only evaluates rows with `task=textilenet_fibre` and `split=test`. It must never download weights. Do not run it on the empty template and interpret an empty result as performance.

## Requested Computer Vision deliverables

Before model claims or result sharing, request from Anshika the complete module, original training/evaluation scripts, dataset and permission/license record, split details, preprocessing, checkpoint/class-map provenance, per-class evaluation and limitations, plus whether she created a separate GARMENT/LABEL/UNKNOWN model. A respectful message draft is included in the Phase 1.1 completion report; it has not been sent.
