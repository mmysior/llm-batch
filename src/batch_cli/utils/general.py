import json
from typing import List

from pydantic import BaseModel


def load_jsonl(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_jsonl_generator(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def append_to_jsonl(responses: List[BaseModel], output_path: str) -> None:
    with open(output_path, "a", encoding="utf-8") as f:
        for response in responses:
            response_dict = response.model_dump()
            f.write(json.dumps(response_dict) + "\n")
