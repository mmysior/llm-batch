from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from batch_cli.utils.images import encode_image

type MessageRole = Literal["system", "user", "assistant", "developer"]
type MessageContent = Union[str, Dict[str, Any], List[Dict[str, Any]]]
type Message = Dict[str, Union[MessageRole, MessageContent]]


def create_message(question: str) -> List[Message]:
    return [{"role": "user", "content": question}]


def create_openai_message(question: str, image_path: Optional[Path] = None) -> List[Message]:
    if image_path is None:
        return create_message(question)

    media_type, base64_image = encode_image(image_path)

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{base64_image}"},
                },
            ],
        }
    ]


def create_anthropic_message(question: str, image_path: Optional[Path] = None) -> List[Message]:
    if image_path is None:
        return create_message(question)

    media_type, base64_image = encode_image(image_path)

    return [
        {
            "role": "user",
            "content": [
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
        }
    ]
