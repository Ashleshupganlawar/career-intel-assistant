import json
import sys
import types

from job_intel.llm.providers import HuggingFaceProvider, OpenAIProvider


class _HFResp:
    def __init__(self, text: str):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]


class _HFClient:
    def __init__(self, **kwargs):
        pass

    def chat_completion(self, **kwargs):
        return _HFResp('{"ok": true, "provider": "hf"}')


class _OAIResponses:
    def create(self, **kwargs):
        return types.SimpleNamespace(output_text='{"ok": true, "provider": "openai"}')


class _OAIClient:
    def __init__(self, **kwargs):
        self.responses = _OAIResponses()


def test_provider_adapter_contract(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(InferenceClient=_HFClient))
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OAIClient))

    hf = HuggingFaceProvider(model="x", provider="y", token="z")
    oai = OpenAIProvider(model="x", api_key="k")

    hf_out = hf.generate_json("test", schema_hint="{ok:bool}")
    oai_out = oai.generate_json("test", schema_hint="{ok:bool}")

    assert isinstance(hf_out, dict)
    assert isinstance(oai_out, dict)
    assert "ok" in hf_out and "ok" in oai_out
