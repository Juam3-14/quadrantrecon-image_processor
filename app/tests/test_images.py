from app.services.image_utils import process_image
from pathlib import Path

def test_process_image():
    test_image = Path("tests/test_image.jpg")
    output_image = process_image(test_image)
    assert output_image.exists()
