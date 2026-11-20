# GreenWardrobe: Phase 6 Analysis API & Frontend Integration Guide

This guide provides the complete production API contract, request formats, response schemas, and integration instructions for **Lakshya Nailwal** (Frontend Lead) to integrate the GreenWardrobe React/Tailwind frontend with the backend.

---

## 1. Overview & Base URLs

- **Backend Development Server:** `http://localhost:8000`
- **Frontend Vite Client Origin:** `http://localhost:5173` (pre-configured in CORS)
- **API Version Prefix:** `/api/v1`
- **Main Analysis Endpoint:** `POST /api/v1/analyze`
- **Backward-Compatible Upload:** `POST /api/v1/upload`
- **Health Check Endpoints:** `GET /health` and `GET /api/v1/health`

---

## 2. Main Analysis API Specification

### Endpoint: `POST /api/v1/analyze`

Executes the full end-to-end sustainable fashion analysis pipeline from a single image:
1. **Input Sanitization & Processing:** Pillow-based decoding, dimension bounds, and decompression-bomb protection.
2. **Automatic Image-Type Classification:** Distinguishes whether the image is a `GARMENT` fabric surface photo or a `LABEL` care/composition tag.
3. **Pipeline Routing:**
   - **GARMENT:** Routed to Computer Vision surface texture model (33 fabric classes) to predict material and query Knowledge Base benchmarks.
   - **LABEL:** Routed to EasyOCR + OpenCV pipeline to extract exact fiber composition and percentages.
   - **UNKNOWN:** Returns a structured 422 error asking for a clearer photo if confidence is below threshold.
4. **Knowledge Base Retrieval:** Life-cycle assessment indicators (water, carbon, energy, microplastics, biodegradability).
5. **Environmental Scoring:** Weighted composite index across 6 categories (`water`, `carbon`, `energy`, `chemicals`, `microfibres`, `waste`) and 5 actionable recommendation tiers (`Buy`, `Think Twice`, `Repair`, `Reuse`, `Second-hand`).
6. **Google Gemini Sustainability Explanation:** Grounded plain-language summary, care & repair guidelines, and purchasing suggestions.

---

## 3. Request Format

- **Content-Type:** `multipart/form-data`
- **HTTP Method:** `POST`

### Form Fields:

| Field Name | Type | Requirement | Description |
| :--- | :--- | :--- | :--- |
| `image` | Binary File | **Required** | Clear photo of either the garment fabric surface OR the care/composition label tag. |
| `shopping` | Boolean | Optional (Default: `true`) | Set to `true` if evaluating an item prior to purchase; `false` if analyzing an existing wardrobe item. |
| `second_hand_available` | Boolean | Optional (Default: `false`) | Set to `true` if thrifted or second-hand options exist for this item. |
| `condition` | String | Optional (Default: `"new"`) | Garment condition: `"new"`, `"used"`, or `"damaged"`. |

### Accepted File Constraints:
- **Formats:** JPEG (`.jpg`, `.jpeg`), PNG (`.png`), WEBP (`.webp`).
- **Maximum File Size:** `10 MB` (`10,485,760 bytes`).
- **Decompression Limit:** Maximum 25 Megapixels (rejects decompression bombs).
- **Dimension Limit:** Maximum 8,192 pixels along any edge.

---

## 4. Example cURL Request

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "image=@/path/to/clothing_image.jpg;type=image/jpeg" \
  -F "shopping=true" \
  -F "second_hand_available=false" \
  -F "condition=new"
```

---

## 5. Response Schemas

### A. Success Response (`HTTP 200 OK`)

```json
{
  "success": true,
  "analysis_id": "c1f7a40b-71a2-4a57-8979-99e5a1b3cd34",
  "timestamp": "2026-11-14T21:20:00.000000+00:00",
  "product": {
    "composition": {
      "materials": [
        {"name": "nylon", "percentage": 68.0},
        {"name": "wool", "percentage": 32.0}
      ],
      "confidence": 0.87,
      "source": "ocr_label"
    },
    "visual_prediction": {
      "predicted_material": "wool",
      "confidence": 0.58,
      "alternatives": ["nylon", "acrylic", "cashmere"],
      "probabilities": {
        "wool": 0.5812,
        "nylon": 0.2104,
        "acrylic": 0.1142,
        "cashmere": 0.0942
      },
      "texture_features": null,
      "warnings": []
    },
    "composition_source": "hybrid"
  },
  "environmental_data": {
    "nylon": {
      "material": "nylon",
      "water_liters_per_kg": 1800.0,
      "co2_kg_per_kg": 7.8,
      "energy_mj_per_kg": 150.0,
      "chemicals_intensity": "High",
      "microplastic_shedding_risk": "Severe",
      "biodegradability": "Non-biodegradable",
      "source_reference": "EEA / UNEP Sustainable Textiles: Synthetic polyamide benchmark."
    },
    "wool": {
      "material": "wool",
      "water_liters_per_kg": 3500.0,
      "co2_kg_per_kg": 13.5,
      "energy_mj_per_kg": 65.0,
      "chemicals_intensity": "Moderate",
      "microplastic_shedding_risk": "None",
      "biodegradability": "High",
      "source_reference": "Textile Exchange: Natural protein fiber life-cycle overview."
    }
  },
  "impact_assessment": {
    "overall_impact_score": 68.3,
    "recommendation": "Think Twice",
    "breakdown": {
      "water": {"score": 65.2, "rating": null, "description": "Water consumption and depletion impact"},
      "carbon": {"score": 78.6, "rating": null, "description": "Greenhouse gas emissions impact"},
      "energy": {"score": 72.0, "rating": null, "description": "Embodied manufacturing and dyeing energy"},
      "chemicals": {"score": 55.2, "rating": null, "description": "Agricultural pesticides and synthetic processing chemicals"},
      "microfibres": {"score": 69.2, "rating": null, "description": "Microplastic shedding risk per wash cycle"},
      "waste": {"score": 63.8, "rating": null, "description": "End-of-life disposal and biodegradability burden"}
    },
    "key_indicators": {
      "impact_classification": "High Impact",
      "data_coverage": "100%",
      "recommendation_reason": "High impact score (68.3) suggests reconsidering purchase."
    },
    "calculation_basis": "rule_based_lca_weighted_average"
  },
  "ai_explanation": {
    "status": "AVAILABLE",
    "summary": "This garment is a blend of 68% Nylon and 32% Wool, scoring 68.3/100 (Think Twice). It balances durable wool insulation with high synthetic energy and microfiber footprints from nylon.",
    "material_explanation": "Nylon provides weather resistance and tensile durability, while wool offers natural moisture-wicking and warmth.",
    "environmental_impact_explanation": "The score reflects elevated carbon emissions from nylon chemical synthesis and high microfiber shedding risks.",
    "impact_category_explanations": {
      "water": "Moderate water demand primarily associated with raw wool scouring.",
      "carbon": "Nylon synthesis generates substantial greenhouse gas emissions (7.8 kg CO2e/kg).",
      "energy": "High embodied energy required for synthetic petrochemical polymerization.",
      "chemicals": "Requires moderate scouring and dye auxiliaries.",
      "microfibres": "Nylon component poses critical non-biodegradable shedding risk during machine laundering.",
      "waste": "Mixed synthetic/natural blend complicates commercial closed-loop recycling."
    },
    "environmental_highlights": [
      "68% Nylon poses significant synthetic microplastic shedding risks during wash cycles.",
      "32% Wool is naturally renewable and biodegradable, balancing garment longevity."
    ],
    "practical_recommendations": [
      "Recommended decision: Think Twice.",
      "Consider whether synthetic weatherproofing is essential for your wardrobe needs before purchasing."
    ],
    "care_and_repair_tips": [
      "Wash in cold water (30°C / 85°F) on gentle cycle to preserve fibers and save energy.",
      "Use a microfibre-catching wash bag (e.g., Guppyfriend) to capture shedding synthetic filaments.",
      "Line dry flat to prevent shrinkage and avoid energy-intensive tumble drying."
    ],
    "limitations_and_uncertainty": [
      "Material impact scores are derived from rule-based calculations using comparative benchmark data; they do not represent empirical certified product declarations."
    ],
    "data_quality_disclaimer": "Environmental impact indicators and scores are comparative benchmarks based on published literature and demo data; they do not represent empirical certified product declarations.",
    "is_fallback": false
  },
  "confidence": {
    "ocr": 0.87,
    "vision": 0.58
  },
  "limitations": [
    "Material impact scores are derived from rule-based calculations using comparative benchmark data; they do not represent empirical garment-specific LCA measurements.",
    "Computer vision predictions reflect surface visual texture classification and may confuse visually similar fibers."
  ],
  "warnings": [],
  "metadata": {
    "processing_time_ms": 245.8,
    "model_versions": {
      "ocr": "Harshit-ClothingLabelOCR-Adapter",
      "vision": "Anshika-TextileNet-Vision-Adapter",
      "scoring": "Maneesh-ScoringEngine-Adapter",
      "knowledge_base": "Ankit-KnowledgeBase-Adapter"
    },
    "pipeline_version": "1.0.0",
    "gemini_model": "gemini-3.5-flash-lite",
    "explanation_status": "AVAILABLE"
  }
}
```

---

### B. Response When Gemini is Unavailable, Rate-Limited, or Timed Out (`HTTP 200 OK`)

The backend never fails or drops analytical data when the Gemini API key is missing, rate-limited (HTTP 429), experiencing timeouts, or facing service outages. It executes bounded exponential backoff retries for transient errors and returns a fully populated response with a scientifically grounded deterministic fallback:

```json
{
  "success": true,
  "analysis_id": "8f3e2912-4011-4b13-9134-8c83a1b3cd99",
  "product": { ... },
  "environmental_data": { ... },
  "impact_assessment": { ... },
  "ai_explanation": {
    "status": "UNAVAILABLE",
    "summary": "This garment is composed of 68% Nylon, 32% Wool, receiving an overall comparative environmental impact score of 68.3/100 with a recommended action tier of 'Think Twice'.",
    "material_explanation": "Identified textile composition from perception analysis: 68% Nylon, 32% Wool. Environmental evaluations are referenced against standard industry comparative lifecycle benchmarks.",
    "environmental_impact_explanation": "The garment achieved an overall comparative impact score of 68.3/100 with a recommendation of 'Think Twice'. This score represents a comparative index relative to published benchmark baselines, rather than an empirical laboratory measurement.",
    "impact_category_explanations": {
      "water": "Scored 65.2/100 based on comparative lifecycle benchmarks.",
      "carbon": "Scored 78.6/100 based on comparative lifecycle benchmarks.",
      "energy": "Scored 72.0/100 based on comparative lifecycle benchmarks.",
      "chemicals": "Scored 55.2/100 based on comparative lifecycle benchmarks.",
      "microfibres": "Scored 69.2/100 based on comparative lifecycle benchmarks.",
      "waste": "Scored 63.8/100 based on comparative lifecycle benchmarks."
    },
    "environmental_highlights": [
      "Identified fiber composition: 68% Nylon, 32% Wool.",
      "Comparative Carbon indicator: Scored 78.6/100 (Elevated) in comparative reference database.",
      "Literature benchmark for Nylon: ~1,800 L/kg water, ~7.8 kg CO2e/kg (EEA / UNEP Sustainable Textiles). (Comparative baseline, not direct measurement)."
    ],
    "practical_recommendations": [
      "Recommended decision tier: Think Twice.",
      "Garment exhibits elevated resource intensity in benchmark comparisons. Consider whether this item fits long-term wardrobe needs."
    ],
    "care_and_repair_tips": [
      "General textile-care guidance (not garment-specific): Wash in cold water (30°C / 85°F) on gentle cycle to preserve fibers and save energy.",
      "General textile-care guidance (not garment-specific): Line dry instead of machine tumble drying to prevent fiber breakdown and reduce emissions.",
      "General microfiber mitigation: Consider using a microfibre-catching wash bag to prevent synthetic microplastics entering waterways."
    ],
    "limitations_and_uncertainty": [
      "Gemini Explanation Layer note: GEMINI_API_KEY is not configured in the backend environment. Standard deterministic explanation generated.",
      "Material impact scores are derived from a rule-based comparative scoring engine using benchmark and demonstration data; they do not represent empirical certified product declarations."
    ],
    "data_quality_disclaimer": "Environmental impact indicators and scores are comparative benchmarks based on published literature and demo data; they do not represent empirical certified product declarations.",
    "is_fallback": true
  },
  "metadata": {
    "processing_time_ms": 182.4,
    "pipeline_version": "1.0.0",
    "gemini_model": "gemini-3.5-flash-lite",
    "explanation_status": "UNAVAILABLE"
  }
}
```

---

### C. Error Responses

#### 1. Missing File (`HTTP 422 Unprocessable Content`)
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "label_image"],
      "msg": "Field required"
    }
  ]
}
```

#### 2. Corrupted Image File (`HTTP 400 Bad Request`)
```json
{
  "error": {
    "type": "ImageValidationError",
    "message": "Corrupted or unreadable image file: cannot identify image file <_io.BytesIO object>",
    "status_code": 400
  }
}
```

#### 3. Oversized Image (`HTTP 413 Payload Too Large`)
```json
{
  "error": {
    "type": "ImageValidationError",
    "message": "Image file size (11.00 MB) exceeds maximum upload limit of 10.0 MB.",
    "status_code": 413
  }
}
```

---

## 6. HTTP Status Code Reference

| Status Code | Description | Scenario |
| :--- | :--- | :--- |
| `200 OK` | Success | Analysis completed successfully (including when Gemini fallback is used). |
| `400 Bad Request` | Client Error | Unsupported file format, corrupted image, or invalid parameters. |
| `413 Payload Too Large` | Limit Exceeded | File exceeds the 10 MB maximum upload restriction. |
| `422 Unprocessable` | Validation Error | Missing required `label_image` or `garment_image` multipart fields. |
| `500 Server Error` | Unexpected | Internal unhandled error (rare; protected by centralized handlers). |

---

## 7. Frontend React Integration Code (Example)

Here is a ready-to-use Axios / Fetch implementation for Lakshya's React client:

```javascript
// src/services/analysisApi.js
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export async function analyzeGarment({
  labelFile,
  garmentFile,
  shopping = true,
  secondHandAvailable = false,
  condition = 'new',
}) {
  const formData = new FormData();
  formData.append('label_image', labelFile);
  formData.append('garment_image', garmentFile);
  formData.append('shopping', shopping ? 'true' : 'false');
  formData.append('second_hand_available', secondHandAvailable ? 'true' : 'false');
  formData.append('condition', condition);

  try {
    const response = await axios.post(`${API_BASE_URL}/analyze`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data; // Type: AnalysisResponse
  } catch (error) {
    if (error.response) {
      // Backend returned structured error response
      throw new Error(error.response.data?.error?.message || error.response.data?.detail?.[0]?.msg || 'Analysis failed.');
    }
    throw new Error('Network error: Could not reach GreenWardrobe backend.');
  }
}
```

---

## 8. Backend Configuration & Startup Instructions

### 1. Configure Environment Variables
Copy `.env.example` to `.env`:

```bash
cd backend
cp .env.example .env
```

To enable live Google Gemini AI explanations, supply your API key:

```dotenv
GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>
GEMINI_MODEL=gemini-3.5-flash-lite
```

*(Note: If `GEMINI_API_KEY` is left blank, the backend runs normally and produces deterministic grounded fallbacks without errors).*

### 2. Start the Backend Server

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Verify Server Status
- API Docs (Swagger): `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/v1/health`

---

## 9. Running Tests

Execute the automated test suite from the `backend/` directory:

```bash
cd backend
./venv/bin/pytest
```

To run API integration tests exclusively:

```bash
./venv/bin/pytest tests/integration/test_analyze_api.py -v
```

---

## 10. Gemini Reliability Architecture & Known Analytical Limitations

### 1. Gemini Timeout Enforcement & Asynchronous Protection
- **Request Timeout:** `GEMINI_TIMEOUT_SECONDS` (default: `30` seconds) is configured directly into the official `google-genai` SDK using `types.HttpOptions(timeout=30000)` (in milliseconds) on both the client and content generation calls.
- **Fail-Safe Asynchronous Timeout:** Handled asynchronously via `asyncio.to_thread` wrapped in `asyncio.wait_for`, guaranteeing that a stalled or hanging external API connection can never indefinitely block the analysis endpoint. When a timeout occurs, the system immediately returns a grounded fallback explanation with `status="TIMEOUT"`.

### 2. Bounded Exponential Backoff Retry Policy
- **Configurable Bounded Retries:** The Gemini adapter implements bounded retries (`max_retries=2`, initial backoff `0.5s`, multiplier `2.0x`, maximum backoff `2.0s`).
- **Transient Errors:** Retries apply strictly to transient faults: HTTP `429` (Quota / Rate limits), `500`/`502`/`503`/`504` (Server unavailable), and temporary network connection resets.
- **Non-Retryable Failures:** Authentication errors (`401`, `403`, invalid API keys) and client validation errors (`400`) fail fast on the first attempt without unnecessary retries.
- **Retry Exhaustion:** Once retries are exhausted, the endpoint automatically returns HTTP 200 with `status="RATE_LIMITED"` or `"UNAVAILABLE"` and a grounded deterministic fallback.

### 3. Scientifically Grounded Deterministic Fallback
- **Data Grounding:** Fallback explanations strictly use actual Phase 5 composition, material profiles, overall impact scores, recommendation tiers, and category breakdowns.
- **Missing Data Handling:** If environmental category data is missing, the system never invents numbers or assigns fabricated category scores; it explicitly discloses that comparative data is unavailable for that category.
- **General Guidance Distinction:** General textile care tips are explicitly distinguished as general textile guidance rather than empirical claims about the specific garment.
- **Disclaimers Preserved:** All explanations maintain the mandatory benchmark data quality disclaimer.

### 4. Known Analytical Limitations & Disclaimers
1. **Vision Classification Baseline:** The TextileNet ResNet-18 model was trained as a research baseline (~49.7% top-1 accuracy). In accordance with Anshika's design, surface visual classification is calibrated as secondary evidence to label OCR.
2. **Benchmark Nature of Indicators:** Knowledge Base metrics and scoring data are comparative educational benchmarks derived from published LCA literature and demo datasets; they are rule-based comparative indices, not empirical laboratory measurements or certified product declarations.
3. **No Model Weight Uploads:** Model weights (`res18_ckpt.pth`, EasyOCR weights) are kept locally and are excluded from Git via `.gitignore`.
