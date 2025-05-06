from uuid import uuid4

from llmbatch.models.schemas import BatchResponse, OpenAIBatch, Response
from llmbatch.services.openai_service import OpenAIService


def process_request(input: OpenAIBatch, batch_id: str, **kwargs) -> BatchResponse:
    openai_service = OpenAIService()
    response: Response | None = None
    error: str | None = None
    if "model" in kwargs:
        input.body.model = kwargs["model"]
    try:
        api_response = openai_service.create_completion(**input.body.model_dump())
        status_code = 200 if api_response.choices[0].finish_reason == "stop" else 500
        response = Response(
            status_code=status_code,
            request_id=str(uuid4().hex),
            body=api_response,
        )
    except Exception as e:
        error = str(e)
        response = Response(
            status_code=500,
            request_id=str(uuid4().hex),
            body=None,
        )

    return BatchResponse(
        id=batch_id,
        custom_id=input.custom_id,
        response=response,
        error=error,
    )
