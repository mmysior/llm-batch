from llmbatch.models.schemas import OutputModel
from llmbatch.utils.general import load_jsonl


def parse_anthropic_jsonl(file_path: str) -> list[OutputModel]:
    results: list[OutputModel] = []
    for line in load_jsonl(file_path):
        custom_id: str = line.get("custom_id", "")
        if "_" in custom_id:
            custom_id = custom_id.split("_")[0]

        result_data = line.get("result", {})
        model = result_data.get("message", {}).get("model", "")
        content_list = result_data.get("message", {}).get("content", [{}])
        content = content_list[0].get("text", "") if content_list else ""
        usage = result_data.get("message", {}).get("usage", {})

        result = OutputModel(
            custom_id=custom_id,
            type=result_data.get("type"),
            model=model,
            response=content,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
        results.append(result)
    return results


def parse_openai_jsonl(file_path: str) -> list[OutputModel]:
    results: list[OutputModel] = []
    for line in load_jsonl(file_path):
        custom_id = line.get("custom_id", "")
        if "_" in custom_id:
            custom_id = custom_id.split("_")[0]
        status_code = line.get("response", {}).get("status_code")
        result_type = "succeeded" if status_code == 200 else "failed"
        model = line.get("response", {}).get("body", {}).get("model", "")
        content = (
            line.get("response", {})
            .get("body", {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        input_tokens = (
            line.get("response", {})
            .get("body", {})
            .get("usage", {})
            .get("prompt_tokens")
        )
        output_tokens = (
            line.get("response", {})
            .get("body", {})
            .get("usage", {})
            .get("completion_tokens")
        )
        result = OutputModel(
            custom_id=custom_id,
            type=result_type,
            model=model,
            response=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        results.append(result)
    return results


def parse_batch_jsonl(path: str) -> list[OutputModel]:
    lines = list(load_jsonl(path))
    if not lines:
        raise ValueError(f"Empty or invalid JSONL file: {path}")

    first = lines[0]
    # Anthropic: has "result" key with "message" or "type"
    if "result" in first:
        return parse_anthropic_jsonl(path)
    # OpenAI: has "response" key with "body"
    if "response" in first and "body" in first["response"]:
        return parse_openai_jsonl(path)
    raise ValueError(f"Failed to determine provider for {path}")
