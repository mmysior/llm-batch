from pathlib import Path
from typing import List, Optional, Union

from batch_cli.models.schemas import AnthropicBatch, OpenAIBatch, Question
from batch_cli.utils.messages import create_anthropic_messages, create_openai_messages

BatchFile = List[Union[OpenAIBatch, AnthropicBatch]]


def create_batch(
    questions: List[Question],
    format: str,
    n_answers: int = 1,
    system_message: Optional[str] = None,
    **kwargs,
) -> BatchFile:
    batch: BatchFile = []
    if format == "openai":
        message_func = create_openai_messages
    elif format == "anthropic":
        message_func = create_anthropic_messages
    else:
        raise ValueError(f"Invalid format: {format}")

    for question in questions:
        for i in range(n_answers):
            custom_id = f"{question.question_id}_rep{i:02d}"
            image_path = Path(question.image_path) if question.image_path else None
            messages = message_func(question.question, image_path, system_message)
            body = {"messages": messages, **kwargs}
            if format == "openai":
                batch.append(OpenAIBatch(custom_id=custom_id, body=body))
            else:
                batch.append(AnthropicBatch(custom_id=custom_id, params=body))
    return batch
