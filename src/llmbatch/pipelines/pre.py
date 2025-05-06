from pathlib import Path
from typing import List, Optional, Union

from llmbatch.models.schemas import AnthropicBatch, OpenAIBatch, Question
from llmbatch.utils.messages import create_anthropic_body, create_openai_body

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
        message_func = create_openai_body
    elif format == "anthropic":
        message_func = create_anthropic_body
    else:
        raise ValueError(f"Invalid format: {format}")

    for question in questions:
        for i in range(n_answers):
            custom_id = f"{question.question_id}_rep{i:02d}"
            image_path = Path(question.image_path) if question.image_path else None
            body = message_func(question.question, image_path, system_message, **kwargs)
            if format == "openai":
                batch.append(OpenAIBatch(custom_id=custom_id, body=body))
            else:
                batch.append(AnthropicBatch(custom_id=custom_id, params=body))
    return batch
