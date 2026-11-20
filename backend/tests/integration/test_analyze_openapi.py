"""OpenAPI documentation checks for the single-image analysis endpoint."""

from fastapi.testclient import TestClient

from app.main import app


def test_analyze_openapi_documents_runtime_error_responses():
    schema = app.openapi()
    operation = schema["paths"]["/api/v1/analyze"]["post"]

    expected_descriptions = {
        "400": "Empty upload or malformed/corrupt image.",
        "413": "Uploaded image exceeds the configured size limit.",
        "415": "Unsupported image format or media type.",
        "422": "Missing/invalid request fields, invalid image, or image classified as UNKNOWN.",
        "500": "Unexpected internal server error; response does not expose a traceback.",
    }
    for status, description in expected_descriptions.items():
        response = operation["responses"][status]
        assert response["description"] == description
        assert response["content"]["application/json"]["schema"]["$ref"] == (
            "#/components/schemas/APIErrorResponse"
        )

    assert "image" in schema["components"]["schemas"][
        "Body_analyze_image_api_v1_analyze_post"
    ]["required"]


def test_docs_page_is_served_with_swagger_ui():
    response = TestClient(app).get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "swagger-ui" in response.text.lower()


def test_error_response_schema_accepts_object_or_validation_list_details():
    from app.schemas.response import APIErrorResponse

    domain_error = APIErrorResponse.model_validate({
        "success": False,
        "error": {"type": "ImageValidationError", "message": "Invalid image.", "details": {"format": "GIF"}},
    })
    request_error = APIErrorResponse.model_validate({
        "success": False,
        "error": {"type": "ValidationError", "message": "Invalid request.", "details": [{"loc": ["body", "image"]}]},
    })

    assert domain_error.error.details == {"format": "GIF"}
    assert request_error.error.details == [{"loc": ["body", "image"]}]
