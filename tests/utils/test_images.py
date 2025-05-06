import base64
from pathlib import Path

import pytest
from PIL import Image

from llmbatch.utils.images import encode_image, get_base64_image


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a small test image."""
    img = Image.new("RGB", (100, 100), color="red")
    return img


@pytest.fixture
def temp_image_file(sample_image, tmp_path) -> Path:
    """Save the sample image as a temporary file."""
    # Create a PNG version
    png_path = tmp_path / "test_image.png"
    sample_image.save(png_path)

    # Create a JPEG version
    jpg_path = tmp_path / "test_image.jpg"
    sample_image.convert("RGB").save(jpg_path)

    return tmp_path


def test_get_base64_image(sample_image):
    """Test the get_base64_image function."""
    base64_str = get_base64_image(sample_image)

    # Check that it's a string and has the expected prefix
    assert isinstance(base64_str, str)
    assert base64_str.startswith("data:image/png;base64,")

    # Check that it's valid base64 by attempting to decode it
    try:
        base64_data = base64_str.split(",")[1]
        base64.b64decode(base64_data)
    except Exception as e:
        pytest.fail(f"Failed to decode base64 string: {e}")


def test_encode_image_png(temp_image_file):
    """Test encoding a PNG image."""
    png_path = temp_image_file / "test_image.png"

    media_type, base64_str = encode_image(png_path)

    assert media_type == "image/png"
    assert isinstance(base64_str, str)

    # Verify it's valid base64
    try:
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0
    except Exception as e:
        pytest.fail(f"Failed to decode base64 string: {e}")


def test_encode_image_jpeg(temp_image_file):
    """Test encoding a JPEG image."""
    jpg_path = temp_image_file / "test_image.jpg"

    media_type, base64_str = encode_image(jpg_path)

    assert media_type == "image/jpeg"
    assert isinstance(base64_str, str)

    # Verify it's valid base64
    try:
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0
    except Exception as e:
        pytest.fail(f"Failed to decode base64 string: {e}")


def test_encode_image_with_custom_params(temp_image_file):
    """Test encoding with custom size and quality parameters."""
    png_path = temp_image_file / "test_image.png"

    media_type, base64_str = encode_image(png_path, max_size=(50, 50), quality=50)

    assert media_type == "image/png"
    assert isinstance(base64_str, str)


def test_encode_image_unsupported_format(tmp_path):
    """Test encoding an unsupported image format."""
    # Create a dummy text file with .txt extension
    unsupported_path = tmp_path / "test.txt"
    unsupported_path.write_text("This is not an image")

    with pytest.raises(ValueError) as excinfo:
        encode_image(unsupported_path)

    assert "Unsupported image format" in str(excinfo.value)
