from unittest.mock import Mock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from llmbatch.models.schemas import Body, OpenAIBatch
from llmbatch.pipelines.inference import process_request


@pytest.fixture
def mock_uuid():
    """Mock uuid4 to return a predictable value"""
    fixed_uuid = "test-uuid-12345"
    with patch("llmbatch.pipelines.inference.uuid4") as mock_uuid:
        mock_uuid.return_value.hex = fixed_uuid
        yield fixed_uuid


@pytest.fixture
def sample_openai_batch():
    """Create a sample OpenAIBatch object for testing"""
    return OpenAIBatch(
        custom_id="test-custom-id",
        body=Body(
            messages=[{"role": "user", "content": "Hello, world!"}],
            model="gemma2:2b",
            temperature=0.7,
            max_tokens=100,
        ),
    )


@pytest.fixture
def successful_api_response():
    """Create a sample successful API response"""
    return ChatCompletion(
        id="chatcmpl-123",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="Hello! How can I help you today?", role="assistant"
                ),
                logprobs=None,
            )
        ],
        created=1677858242,
        model="gemma2:2b",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=10, prompt_tokens=8, total_tokens=18),
    )


@patch("llmbatch.pipelines.inference.OpenAIService")
def test_process_request_success(
    mock_openai_service_cls, sample_openai_batch, successful_api_response, mock_uuid
):
    """Test successful processing of a request"""
    # Arrange
    mock_service = Mock()
    mock_openai_service_cls.return_value = mock_service
    mock_service.create_completion.return_value = successful_api_response
    batch_id = "test-batch-id"

    # Act
    result = process_request(sample_openai_batch, batch_id)

    # Assert
    mock_service.create_completion.assert_called_once_with(
        **sample_openai_batch.body.model_dump()
    )
    assert result.id == batch_id
    assert result.custom_id == sample_openai_batch.custom_id
    assert result.error is None
    assert result.response is not None
    assert result.response.status_code == 200
    assert result.response.request_id == mock_uuid
    assert result.response.body == successful_api_response


@patch("llmbatch.pipelines.inference.OpenAIService")
def test_process_request_with_model_override(
    mock_openai_service_cls, sample_openai_batch, successful_api_response, mock_uuid
):
    """Test processing with model override in kwargs"""
    # Arrange
    mock_service = Mock()
    mock_openai_service_cls.return_value = mock_service
    mock_service.create_completion.return_value = successful_api_response
    batch_id = "test-batch-id"
    override_model = "llama3:8b"

    # Act
    result = process_request(sample_openai_batch, batch_id, model=override_model)

    # Assert
    # Check that the model was overridden in the call
    called_args = mock_service.create_completion.call_args[1]
    assert called_args["model"] == override_model
    assert result.response is not None
    assert result.response.status_code == 200


@patch("llmbatch.pipelines.inference.OpenAIService")
def test_process_request_error(mock_openai_service_cls, sample_openai_batch, mock_uuid):
    """Test handling of an error during processing"""
    # Arrange
    mock_service = Mock()
    mock_openai_service_cls.return_value = mock_service
    error_message = "API connection error"
    mock_service.create_completion.side_effect = Exception(error_message)
    batch_id = "test-batch-id"

    # Act
    result = process_request(sample_openai_batch, batch_id)

    # Assert
    assert result.id == batch_id
    assert result.custom_id == sample_openai_batch.custom_id
    assert result.error == error_message
    assert result.response is not None
    assert result.response.status_code == 500
    assert result.response.request_id == mock_uuid
    assert result.response.body is None


@patch("llmbatch.pipelines.inference.OpenAIService")
def test_process_request_non_stop_finish_reason(
    mock_openai_service_cls, sample_openai_batch, successful_api_response, mock_uuid
):
    """Test handling of a response with a finish reason other than 'stop'"""
    # Arrange
    mock_service = Mock()
    mock_openai_service_cls.return_value = mock_service

    # Modify the successful_api_response to have a different finish reason
    modified_response = successful_api_response.copy()
    modified_response.choices[0].finish_reason = "length"
    mock_service.create_completion.return_value = modified_response

    batch_id = "test-batch-id"

    # Act
    result = process_request(sample_openai_batch, batch_id)

    # Assert
    assert (
        result.response is not None and result.response.status_code == 500
    )  # Should be 500 if finish_reason is not "stop"
