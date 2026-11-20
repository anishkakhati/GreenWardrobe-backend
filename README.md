# GreenWardrobe

GreenWardrobe explores clothing material composition and comparative environmental benchmarks from exactly one uploaded image. The backend classifies it as a care label, garment, or unknown input and selects OCR, computer vision, or rejection.

## Repositories

- Backend: this repository, branch `feature/backend`.
- Frontend: [GreenWardrobe-frontend](https://github.com/anishkakhati/GreenWardrobe/tree/feature/frontend), branch `feature/frontend`.

## Technology

- Backend: Python, FastAPI, Pydantic, PyTorch, TextileNet ResNet-18, EasyOCR, OpenCV, Pillow, NumPy, pytest.
- Frontend: React 19 and Vite.
- Optional explanation service: Google Gemini API, with deterministic local fallback.

## Backend architecture

```text
Single image upload → validation → image-type classification
                            ├─ LABEL → OCR ─────┐
                            ├─ GARMENT → CV ────┤
                            └─ UNKNOWN → reject │
                                                ↓
                        normalization → knowledge base → scoring
                                                ↓
                                  API response → frontend
```

`UNKNOWN` is rejected without OCR or CV inference. TextileNet predicts visual fibre classes; those predictions are not laboratory-confirmed composition. Environmental scores use comparative material benchmarks and are not certified garment-specific life-cycle assessments.

The modular backend connects image validation and classification, OCR and computer vision adapters, normalization, material knowledge lookup, scoring, recommendation, and explanation services. Gemini explains structured results and does not replace the underlying composition or score; a deterministic fallback is available when the optional service is unavailable.

## API and image handling

- Analysis: `POST /api/v1/analyze`, one multipart field named `image`.
- Health: `GET /health`.
- Interactive API documentation: `http://localhost:8000/docs`.
- Optional analysis fields: `shopping` (default `true`), `second_hand_available` (default `false`), and `condition` (default `new`).
- Supported formats: JPEG, PNG, WEBP.
- Current upload limits: 10 MB, 8192 pixels per edge, and 25 megapixels total.
- The classifier chooses the pipeline automatically; there is no manual image-type selection.

## Analysis modules

| Module | Role |
|---|---|
| Image service and classifier | Validate/decode the upload and route GARMENT, LABEL, or UNKNOWN. |
| OCR adapter | Extract visible care-label composition. |
| Vision adapter | Run TextileNet fibre prediction for garment images. |
| Normalization service | Normalize material names and reconcile module outputs. |
| Knowledge base | Look up available comparative material indicators. |
| Scoring and recommendation | Calculate a comparative score only when composition/data requirements are met. |
| Gemini explanation | Explain structured analysis; optional, with deterministic fallback. |

The scoring categories are water, carbon emissions, energy, chemical intensity, microfibre shedding, and waste/end-of-life. Recommendations are rule-based options such as Buy, Think Twice, Repair, Reuse, or Second-hand when the analysis supports one; unavailable scoring must not be presented as a result.

## Repository layout

```text
backend/
├── app/            # API, configuration, schemas, services, adapters
├── docs/           # Integration and model-validation guidance
├── tests/          # Unit/integration tests, fixtures, validation tools
├── requirements.txt
└── run.py
```

## Run locally

From `backend/`, activate/use the existing virtual environment and start FastAPI:

```sh
venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

For a new backend environment, from `backend/`:

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

The backend dependencies are listed in `backend/requirements.txt`. The Gemini API key is optional; keep local environment files out of Git. GARMENT inference also depends on the locally available TextileNet checkpoint, whose provenance and redistribution permissions must be verified before deployment.

From the separate frontend worktree:

```sh
npm ci
npm run dev
```

Vite normally serves at `http://localhost:5173`. The frontend backend URL is controlled by its existing configuration. Gemini is optional; do not commit local `.env` files or credentials. GARMENT inference requires the locally provided TextileNet checkpoint; do not assume it is included in Git or download it without confirming provenance and license.

## Verify

From `backend/`:

```sh
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest tests -q -p no:cacheprovider
```

The image-type evaluation utility is `tests/tools/validate_image_classifier.py`. TextileNet held-out evaluation uses `tests/tools/validate_textilenet.py` with an approved validation manifest and an already available checkpoint; neither tool downloads model weights.

From the frontend worktree:

```sh
npm test
npm run lint
npm run build
```

The handover verification run recorded 218 backend tests passing (115 warnings), 19 frontend tests passing, lint passing, and a successful production build. Rerun these commands after changes; passing tests alone do not establish production readiness.

## Current limitations

- No genuine GARMENT validation samples are available. An earlier classifier check misclassified 9 of 12 UNKNOWN examples as GARMENT.
- TextileNet held-out accuracy and checkpoint/training provenance remain unverified.
- OCR accuracy across varied real label conditions requires broader validation.
- Environmental values are comparative/demo benchmarks, not certified product-specific measurements.
- Production security and deployment controls are incomplete.
- `backend/docs/MODULE_INTEGRATION_GUIDE.md` and `backend/docs/PHASE_6_API_GUIDE.md` are legacy references; confirm behavior against current source and OpenAPI.
- Review [backend/docs/MODEL_VALIDATION_DATASET_GUIDE.md](backend/docs/MODEL_VALIDATION_DATASET_GUIDE.md) before collecting evaluation images.

## References and attribution

- [FastAPI](https://fastapi.tiangolo.com/)
- [Pydantic](https://docs.pydantic.dev/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [TextileNet](https://github.com/hahashu/TextileNet) and its [research paper](https://arxiv.org/abs/2301.06160)
- [PyTorch](https://pytorch.org/)

Check the license and permitted use for each third-party model, dataset, library, and service before redistribution or deployment.
