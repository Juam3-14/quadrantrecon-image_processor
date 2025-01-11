from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)

def test_upload_image():
    file_path = "tests/test_image.jpg"
    with open(file_path, "rb") as f:
        response = client.post("/upload/", files={"file": f})
    assert response.status_code == 200
    assert "processed_image" in response.json()
