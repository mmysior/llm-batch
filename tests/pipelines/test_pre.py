from pathlib import Path
from unittest.mock import patch

import pytest

from llmbatch.models.schemas import AnthropicBatch, OpenAIBatch, Question
from llmbatch.pipelines.pre import create_batch


@pytest.fixture
def sample_questions():
    return [
        Question(question_id="q1", question="What is machine learning?"),
        Question(
            question_id="q2",
            question="Describe this image",
            image_path="tests/fixtures/sample.jpg",
        ),
    ]


@pytest.fixture
def mock_create_body_funcs():
    """Mock both message creation functions to return predictable values"""
    with (
        patch("llmbatch.pipelines.pre.create_openai_body") as mock_openai,
        patch("llmbatch.pipelines.pre.create_anthropic_body") as mock_anthropic,
    ):
        # Setup mock return values with all required fields
        mock_openai.return_value = {
            "messages": [{"role": "user", "content": "mocked"}],
            "model": "mock-model",
            "temperature": 0.7,
            "max_tokens": 100,
        }
        mock_anthropic.return_value = {
            "messages": [{"role": "user", "content": "mocked"}],
            "model": "mock-model",
            "temperature": 0.7,
            "max_tokens": 100,
        }
        yield mock_openai, mock_anthropic


def test_create_batch_openai(sample_questions, mock_create_body_funcs):
    # Arrange
    mock_openai, _ = mock_create_body_funcs

    # Act
    result = create_batch(
        questions=sample_questions,
        format="openai",
        n_answers=2,
        system_message="You are a helpful assistant",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100,
    )

    # Assert
    assert len(result) == 4  # 2 questions * 2 answers
    assert all(isinstance(item, OpenAIBatch) for item in result)

    # Check first question, first answer
    first_item = result[0]
    assert isinstance(first_item, OpenAIBatch)
    assert first_item.custom_id == "q1_rep00"

    # Check that body has expected values instead of direct comparison
    assert first_item.body.model == "mock-model"
    assert first_item.body.temperature == 0.7
    assert first_item.body.max_tokens == 100
    assert first_item.body.messages[0]["role"] == "user"
    assert first_item.body.messages[0]["content"] == "mocked"

    # Check first question, second answer
    assert result[1].custom_id == "q1_rep01"

    # Check second question (with image)
    assert result[2].custom_id == "q2_rep00"

    # Verify the calls to create_openai_body
    assert mock_openai.call_count == 4

    # Check correct arguments were passed for first question
    first_call_args = mock_openai.call_args_list[0]
    assert first_call_args[0][0] == "What is machine learning?"  # text
    assert first_call_args[0][1] is None  # image_path
    assert first_call_args[0][2] == "You are a helpful assistant"  # system_message
    assert first_call_args[1]["model"] == "gpt-4"
    assert first_call_args[1]["temperature"] == 0.7
    assert first_call_args[1]["max_tokens"] == 100

    # Check image path was correctly passed for second question
    third_call_args = mock_openai.call_args_list[2]
    assert isinstance(third_call_args[0][1], Path)
    assert str(third_call_args[0][1]).endswith("tests/fixtures/sample.jpg")


def test_create_batch_anthropic(sample_questions, mock_create_body_funcs):
    # Arrange
    _, mock_anthropic = mock_create_body_funcs

    # Act
    result = create_batch(
        questions=sample_questions,
        format="anthropic",
        n_answers=1,
        system_message="You are Claude",
        model="claude-3-opus-20240229",
        temperature=0.5,
        max_tokens=500,
    )

    # Assert
    assert len(result) == 2  # 2 questions * 1 answer
    assert all(isinstance(item, AnthropicBatch) for item in result)

    # Check first question
    first_item = result[0]
    assert isinstance(first_item, AnthropicBatch)
    assert first_item.custom_id == "q1_rep00"

    # Check that params has expected values instead of direct comparison
    assert first_item.params.model == "mock-model"
    assert first_item.params.temperature == 0.7
    assert first_item.params.max_tokens == 100
    assert first_item.params.messages[0]["role"] == "user"
    assert first_item.params.messages[0]["content"] == "mocked"

    # Check second question (with image)
    assert result[1].custom_id == "q2_rep00"

    # Verify the calls to create_anthropic_body
    assert mock_anthropic.call_count == 2

    # Check correct arguments were passed for second question (with image)
    second_call_args = mock_anthropic.call_args_list[1]
    assert second_call_args[0][0] == "Describe this image"  # text
    assert isinstance(second_call_args[0][1], Path)  # image_path
    assert second_call_args[0][2] == "You are Claude"  # system_message
    assert second_call_args[1]["model"] == "claude-3-opus-20240229"
    assert second_call_args[1]["temperature"] == 0.5
    assert second_call_args[1]["max_tokens"] == 500


def test_create_batch_invalid_format(sample_questions):
    # Act & Assert
    with pytest.raises(ValueError, match="Invalid format: unknown"):
        create_batch(questions=sample_questions, format="unknown")
