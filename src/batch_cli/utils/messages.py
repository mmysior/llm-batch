from pathlib import Path
from typing import List, Optional

from batch_cli.models.schemas import MessageModel
from batch_cli.utils.images import encode_image


def create_messages(
    user_message: str, system_message: Optional[str] = None
) -> List[MessageModel]:
    messages: List[MessageModel] = []
    if system_message:
        messages.append(MessageModel(role="system", content=system_message))
    messages.append(MessageModel(role="user", content=user_message))
    return messages


def create_openai_messages(
    question: str,
    image_path: Optional[Path] = None,
    system_message: Optional[str] = None,
) -> List[MessageModel]:
    messages: List[MessageModel] = []
    if system_message:
        messages.append(MessageModel(role="system", content=system_message))

    if image_path is None:
        messages.append(MessageModel(role="user", content=question))
        return messages

    media_type, base64_image = encode_image(image_path)
    messages.append(
        MessageModel(
            role="user",
            content=[
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{base64_image}"},
                },
            ],
        )
    )
    return messages


def create_anthropic_messages(
    question: str,
    image_path: Optional[Path] = None,
    system_message: Optional[str] = None,
) -> List[MessageModel]:
    messages: List[MessageModel] = []
    if system_message:
        messages.append(MessageModel(role="system", content=system_message))

    if image_path is None:
        messages.append(MessageModel(role="user", content=question))
        return messages

    media_type, base64_image = encode_image(image_path)
    messages.append(
        MessageModel(
            role="user",
            content=[
                {"type": "text", "text": question},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image,
                    },
                },
            ],
        )
    )
    return messages
