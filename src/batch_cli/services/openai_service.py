from typing import Any, Dict, List

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from batch_cli.models.schemas import Body

load_dotenv()

type LLMClient = OpenAI | instructor.Instructor


class OpenAIService:
    def __init__(self, patched: bool = False):
        self.patched: bool = patched
        self.client: LLMClient = self._get_client()

    def _get_client(self) -> LLMClient:
        client: OpenAI = OpenAI(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
        )
        if self.patched:
            return instructor.from_openai(client)
        return client

    def create_completion(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> ChatCompletion:
        completion_params = Body(messages=messages, **kwargs)
        return self.client.chat.completions.create(**completion_params.model_dump())
