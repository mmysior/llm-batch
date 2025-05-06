from pathlib import Path
from typing import Optional

from llmbatch.models.schemas import Body
from llmbatch.utils.images import encode_image


def create_openai_body(
    text: str,
    image_path: Optional[Path] = None,
    system_message: Optional[str] = None,
    **kwargs,
) -> Body:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    if image_path:
        media_type, base64_image = encode_image(image_path)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}"
                        },
                    },
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": text})

    return Body(
        messages=messages,
        **kwargs,
    )


def create_anthropic_body(
    text: str,
    image_path: Optional[Path] = None,
    system_message: Optional[str] = None,
    **kwargs,
) -> Body:
    if image_path:
        media_type, base64_image = encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
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
    else:
        messages = [{"role": "user", "content": text}]

    body_kwargs = {
        "messages": messages,
        **kwargs,
    }
    if system_message:
        body_kwargs["system"] = system_message

    return Body(**body_kwargs)
