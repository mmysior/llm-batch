from unittest.mock import patch

import pytest

from llmbatch.models.schemas import Body
from llmbatch.utils.messages import create_anthropic_body, create_openai_body


@pytest.fixture
def mock_encode_image():
    with patch("llmbatch.utils.messages.encode_image") as mock:
        mock.return_value = ("image/png", "mock_base64_data")
        yield mock


def test_create_openai_body_text_only():
    """Test creating OpenAI request body with text only."""
    text = "Hello, world!"

    body = create_openai_body(text, model="gpt-4", temperature=0.7, max_tokens=1000)

    assert isinstance(body, Body)
    assert len(body.messages) == 1
    assert body.messages[0]["role"] == "user"
    assert body.messages[0]["content"] == text
    assert body.model == "gpt-4"
    assert body.temperature == 0.7
    assert body.max_tokens == 1000


def test_create_openai_body_with_system_message():
    """Test creating OpenAI request body with a system message."""
    text = "Hello, world!"
    system_message = "You are a helpful assistant."

    body = create_openai_body(
        text,
        system_message=system_message,
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
    )

    assert isinstance(body, Body)
    assert len(body.messages) == 2
    assert body.messages[0]["role"] == "system"
    assert body.messages[0]["content"] == system_message
    assert body.messages[1]["role"] == "user"
    assert body.messages[1]["content"] == text


def test_create_openai_body_with_image(mock_encode_image, tmp_path):
    """Test creating OpenAI request body with an image."""
    text = "What's in this image?"
    image_path = tmp_path / "test.png"
    image_path.touch()  # Create an empty file

    body = create_openai_body(
        text,
        image_path=image_path,
        model="gpt-4-vision",
        temperature=0.7,
        max_tokens=1000,
    )

    mock_encode_image.assert_called_once_with(image_path)

    assert isinstance(body, Body)
    assert len(body.messages) == 1
    assert body.messages[0]["role"] == "user"
    assert isinstance(body.messages[0]["content"], list)
    assert len(body.messages[0]["content"]) == 2
    assert body.messages[0]["content"][0]["type"] == "text"
    assert body.messages[0]["content"][0]["text"] == text
    assert body.messages[0]["content"][1]["type"] == "image_url"
    assert (
        body.messages[0]["content"][1]["image_url"]["url"]
        == "data:image/png;base64,mock_base64_data"
    )


def test_create_anthropic_body_text_only():
    """Test creating Anthropic request body with text only."""
    text = "Hello, world!"

    body = create_anthropic_body(
        text, model="claude-3", temperature=0.7, max_tokens=1000
    )

    assert isinstance(body, Body)
    assert len(body.messages) == 1
    assert body.messages[0]["role"] == "user"
    assert body.messages[0]["content"] == text
    assert body.model == "claude-3"
    assert body.temperature == 0.7
    assert body.max_tokens == 1000
    assert not hasattr(body, "system")


def test_create_anthropic_body_with_system_message():
    """Test creating Anthropic request body with a system message."""
    text = "Hello, world!"
    system_message = "You are a helpful assistant."

    body = create_anthropic_body(
        text,
        system_message=system_message,
        model="claude-3",
        temperature=0.7,
        max_tokens=1000,
    )

    assert isinstance(body, Body)
    assert len(body.messages) == 1
    assert body.messages[0]["role"] == "user"
    assert body.messages[0]["content"] == text
    assert body.model_dump().get("system") == system_message


def test_create_anthropic_body_with_image(mock_encode_image, tmp_path):
    """Test creating Anthropic request body with an image."""
    text = "What's in this image?"
    image_path = tmp_path / "test.png"
    image_path.touch()  # Create an empty file

    body = create_anthropic_body(
        text,
        image_path=image_path,
        model="claude-3-vision",
        temperature=0.7,
        max_tokens=1000,
    )

    mock_encode_image.assert_called_once_with(image_path)

    assert isinstance(body, Body)
    assert len(body.messages) == 1
    assert body.messages[0]["role"] == "user"
    assert isinstance(body.messages[0]["content"], list)
    assert len(body.messages[0]["content"]) == 2
    assert body.messages[0]["content"][0]["type"] == "text"
    assert body.messages[0]["content"][0]["text"] == text
    assert body.messages[0]["content"][1]["type"] == "image"
    assert body.messages[0]["content"][1]["source"]["type"] == "base64"
    assert body.messages[0]["content"][1]["source"]["media_type"] == "image/png"
    assert body.messages[0]["content"][1]["source"]["data"] == "mock_base64_data"
