"""Integration tests for core API endpoints, CORS, and exception handling."""


def test_root_endpoint(client):
    """Verify that root GET / returns healthy status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Welcome to GreenWardrobe API"
    assert data["status"] == "running"


def test_health_endpoints(client):
    """Verify that both /health and /api/v1/health report healthy status."""
    for path in ["/health", "/api/v1/health"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "GreenWardrobe Backend"


def test_cors_headers_on_preflight(client):
    """Verify that CORS middleware headers allow the React frontend."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/api/v1/upload", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_upload_valid_single_image(client, valid_jpeg_bytes):
    """Verify that upload of a valid garment image succeeds with 200."""
    files = {
        "garment_image": ("garment.jpg", valid_jpeg_bytes, "image/jpeg")
    }
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Images uploaded and validated successfully."
    assert data["garment_image"]["format"] == "JPEG"
    assert data["garment_image"]["width"] == 100
    assert data["garment_image"]["height"] == 100
    assert data["label_image"] is None


def test_upload_valid_garment_and_label(client, valid_jpeg_bytes, valid_png_bytes):
    """Verify that upload of both garment and label images succeeds."""
    files = {
        "garment_image": ("garment.jpg", valid_jpeg_bytes, "image/jpeg"),
        "label_image": ("label.png", valid_png_bytes, "image/png"),
    }
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["garment_image"]["format"] == "JPEG"
    assert data["label_image"]["format"] == "PNG"


def test_upload_empty_file_returns_structured_error(client):
    """Verify that an empty file produces our standardized error schema."""
    files = {
        "garment_image": ("empty.jpg", b"", "image/jpeg")
    }
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["type"] == "HTTPException"
    assert "empty" in data["error"]["message"].lower()


def test_upload_corrupt_file_returns_structured_error(client, corrupt_image_bytes):
    """Verify that a corrupted image file returns structured 400 error."""
    files = {
        "garment_image": ("fake.jpg", corrupt_image_bytes, "image/jpeg")
    }
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["type"] == "HTTPException"
    assert "invalid or corrupted" in data["error"]["message"].lower()
