"""Provider-agnostic LLM client interfaces and concrete adapters (HF/OpenAI/Mock)."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class LLMProvider(ABC):
    @abstractmethod
    def generate_json(self, prompt: str, schema_hint: str | None = None) -> dict[str, Any]:
        raise NotImplementedError


class HuggingFaceProvider(LLMProvider):
    def __init__(self, model: str | None = None, provider: str | None = None, token: str | None = None):
        from huggingface_hub import InferenceClient

        self.model = model or os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.provider = provider or os.getenv("HF_PROVIDER", "hf-inference")
        self.token = token or os.getenv("HF_TOKEN")
        self.client = InferenceClient(model=self.model, provider=self.provider, token=self.token)

    def generate_json(self, prompt: str, schema_hint: str | None = None) -> dict[str, Any]:
        schema_text = f"\nJSON schema hint:\n{schema_hint}\n" if schema_hint else ""
        user_prompt = f"Return only valid JSON.\n{schema_text}\n{prompt}"
        out = self.client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a JSON generator."},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1800,
            temperature=0.2,
        )
        text = out.choices[0].message.content.strip()
        return _parse_json(text)


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def generate_json(self, prompt: str, schema_hint: str | None = None) -> dict[str, Any]:
        schema_text = f"\nJSON schema hint:\n{schema_hint}\n" if schema_hint else ""
        user_prompt = f"Return only valid JSON.\n{schema_text}\n{prompt}"

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "You are a JSON generator."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        text = getattr(resp, "output_text", "") or ""
        return _parse_json(text)


class MockProvider(LLMProvider):
    def generate_json(self, prompt: str, schema_hint: str | None = None) -> dict[str, Any]:
        return {
            "summary": "Mock response",
            "prompt_preview": prompt[:120],
            "schema_hint": schema_hint or "",
        }


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)
