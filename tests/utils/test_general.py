import json
from pathlib import Path
from typing import List

import pytest
from pydantic import BaseModel

from llmbatch.models.schemas import Config
from llmbatch.utils.general import (
    append_to_jsonl,
    load_config,
    load_jsonl,
    load_jsonl_generator,
)


class TestModel(BaseModel):
    name: str
    value: int


@pytest.fixture
def sample_jsonl_file(tmp_path) -> Path:
    """Create a sample JSONL file for testing."""
    file_path = tmp_path / "test.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write('{"name": "item1", "value": 42}\n')
        f.write('{"name": "item2", "value": 99}\n')
    return file_path


@pytest.fixture
def sample_yaml_file(tmp_path) -> Path:
    """Create a sample YAML config file for testing."""
    file_path = tmp_path / "test_config.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("""
format: openai
params:
  model: gpt-4
  temperature: 0.7
  max_tokens: 1000
n_answers: 1
system_message: You are a helpful assistant.
""")
    return file_path


def test_load_jsonl(sample_jsonl_file):
    """Test loading JSONL file into a list of dictionaries."""
    result = load_jsonl(str(sample_jsonl_file))

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["name"] == "item1"
    assert result[0]["value"] == 42
    assert result[1]["name"] == "item2"
    assert result[1]["value"] == 99


def test_load_jsonl_generator(sample_jsonl_file):
    """Test loading JSONL file as a generator of dictionaries."""
    gen = load_jsonl_generator(str(sample_jsonl_file))

    # Verify it's a generator
    assert hasattr(gen, "__next__")

    # Get first item
    first_item = next(gen)
    assert first_item["name"] == "item1"
    assert first_item["value"] == 42

    # Get second item
    second_item = next(gen)
    assert second_item["name"] == "item2"
    assert second_item["value"] == 99

    # Should be no more items
    with pytest.raises(StopIteration):
        next(gen)


def test_append_to_jsonl(tmp_path):
    """Test appending models to a JSONL file."""
    output_path = tmp_path / "output.jsonl"

    # Create models to append
    models: List[TestModel] = [
        TestModel(name="test1", value=123),
        TestModel(name="test2", value=456),
    ]

    # Append to file
    append_to_jsonl(models, str(output_path))  # type: ignore

    # Verify file contents
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {"name": "test1", "value": 123}
    assert json.loads(lines[1]) == {"name": "test2", "value": 456}


def test_load_config(sample_yaml_file):
    """Test loading a YAML config file into a Config model."""
    config = load_config(str(sample_yaml_file))

    assert isinstance(config, Config)
    assert config.format == "openai"
    assert config.params.model == "gpt-4"
    assert config.params.temperature == 0.7
    assert config.params.max_tokens == 1000
    assert config.n_answers == 1
    assert config.system_message == "You are a helpful assistant."
