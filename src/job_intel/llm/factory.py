"""Factory for selecting an LLM provider implementation from config/env."""

from .providers import HuggingFaceProvider, LLMProvider, MockProvider, OpenAIProvider


def create_provider(name: str) -> LLMProvider:
    normalized = (name or "").strip().lower()
    if normalized in {"hf", "huggingface"}:
        return HuggingFaceProvider()
    if normalized in {"openai", "oai"}:
        return OpenAIProvider()
    if normalized in {"mock", "test"}:
        return MockProvider()
    raise ValueError(f"Unsupported provider: {name}")
