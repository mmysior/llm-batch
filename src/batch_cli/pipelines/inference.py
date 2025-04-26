from uuid import uuid4

from openai import OpenAI

from batch_cli.models.schemas import BatchResponse, OpenAIBatch, Response


def get_ollama_client() -> OpenAI:
    return OpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama",
    )


def process_request(input: OpenAIBatch, batch_id: str, **kwargs) -> BatchResponse:
    client: OpenAI = get_ollama_client()
    response: Response | None = None
    error: str | None = None
    model: str = kwargs["model"] if "model" in kwargs else input.body["model"]
    try:
        api_response = client.chat.completions.create(
            messages=input.body["messages"],
            model=model,
        )
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
