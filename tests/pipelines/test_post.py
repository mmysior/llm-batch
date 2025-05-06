import json
import os
import tempfile
from typing import List

import pytest

from llmbatch.models.schemas import OutputModel
from llmbatch.pipelines.post import parse_batch_jsonl


@pytest.fixture
def anthropic_sample_data() -> List[dict]:
    return [
        {
            "custom_id": "test1_suffix",
            "result": {
                "type": "succeeded",
                "message": {
                    "model": "claude-3-opus-20240229",
                    "content": [{"text": "This is a test response"}],
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            },
        },
        {
            "custom_id": "test2",
            "result": {
                "type": "failed",
                "message": {
                    "model": "claude-3-sonnet-20240229",
                    "content": [],
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            },
        },
    ]


@pytest.fixture
def openai_sample_data() -> List[dict]:
    return [
        {
            "custom_id": "test3_suffix",
            "response": {
                "status_code": 200,
                "body": {
                    "model": "gpt-4-turbo",
                    "choices": [{"message": {"content": "This is an OpenAI response"}}],
                    "usage": {"prompt_tokens": 15, "completion_tokens": 25},
                },
            },
        },
        {
            "custom_id": "test4",
            "response": {
                "status_code": 400,
                "body": {
                    "model": "gpt-3.5-turbo",
                    "choices": [{"message": {"content": ""}}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 0},
                },
            },
        },
    ]


@pytest.fixture
def anthropic_jsonl_file(anthropic_sample_data):
    """Create a temporary JSONL file with Anthropic data."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        for item in anthropic_sample_data:
            tmp.write((json.dumps(item) + "\n").encode("utf-8"))
        tmp.flush()
        file_path = tmp.name

    yield file_path

    # Cleanup after test
    os.unlink(file_path)


@pytest.fixture
def openai_jsonl_file(openai_sample_data):
    """Create a temporary JSONL file with OpenAI data."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        for item in openai_sample_data:
            tmp.write((json.dumps(item) + "\n").encode("utf-8"))
        tmp.flush()
        file_path = tmp.name

    yield file_path

    # Cleanup after test
    os.unlink(file_path)


def test_parse_anthropic_jsonl(anthropic_jsonl_file):
    # Act
    results = parse_batch_jsonl(anthropic_jsonl_file)

    # Assert
    assert len(results) == 2
    assert isinstance(results[0], OutputModel)

    # Check first result
    assert results[0].custom_id == "test1"  # Should remove the suffix
    assert results[0].type == "succeeded"
    assert results[0].model == "claude-3-opus-20240229"
    assert results[0].response == "This is a test response"
    assert results[0].input_tokens == 10
    assert results[0].output_tokens == 20

    # Check second result
    assert results[1].custom_id == "test2"
    assert results[1].type == "failed"
    assert results[1].model == "claude-3-sonnet-20240229"
    assert results[1].response == ""
    assert results[1].input_tokens == 5
    assert results[1].output_tokens == 0


def test_parse_openai_jsonl(openai_jsonl_file):
    # Act
    results = parse_batch_jsonl(openai_jsonl_file)

    # Assert
    assert len(results) == 2
    assert isinstance(results[0], OutputModel)

    # Check first result
    assert results[0].custom_id == "test3"  # Should remove the suffix
    assert results[0].type == "succeeded"
    assert results[0].model == "gpt-4-turbo"
    assert results[0].response == "This is an OpenAI response"
    assert results[0].input_tokens == 15
    assert results[0].output_tokens == 25

    # Check second result
    assert results[1].custom_id == "test4"
    assert results[1].type == "failed"  # Since status code was 400
    assert results[1].model == "gpt-3.5-turbo"
    assert results[1].response == ""
    assert results[1].input_tokens == 8
    assert results[1].output_tokens == 0


def test_parse_batch_jsonl_empty_file():
    # Create empty temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        file_path = tmp.name

    try:
        # Act & Assert
        with pytest.raises(ValueError, match="Empty or invalid JSONL file"):
            parse_batch_jsonl(file_path)
    finally:
        os.unlink(file_path)


def test_parse_batch_jsonl_unknown_format():
    # Create temp file with unrecognized format
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        data = [{"unknown_key": "value"}]
        for item in data:
            tmp.write((json.dumps(item) + "\n").encode("utf-8"))
        tmp.flush()
        file_path = tmp.name

    try:
        # Act & Assert
        with pytest.raises(ValueError, match="Failed to determine provider"):
            parse_batch_jsonl(file_path)
    finally:
        os.unlink(file_path)
