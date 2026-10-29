"""Tests for the local chat client configuration and request contract."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import requests


from conceptgraph.utils import llm_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_TEMPLATE_PATH = PROJECT_ROOT / "env_vars.bash.template"

DEFAULT_URL = "http://127.0.0.1:20261/v1"
DEFAULT_MODEL = "qwen3.8:27b-mtp4-ud_q4_K_XL-128k"


@pytest.fixture(autouse=True)
def isolated_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in ("LLM", "VLM"):
        for key in (
            "BASE_URL",
            "MODEL",
            "API_KEY",
            "TIMEOUT",
            "MAX_TOKENS",
            "REASONING_EFFORT",
            "JSON_MODE",
            "TEMPERATURE",
            "TOP_P",
        ):
            monkeypatch.delenv(f"{prefix}_{key}", raising=False)


class FakeResponse:
    def __init__(self, status_code: int = 200, body=None, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_both_factories_use_shared_local_defaults_and_compatibility_alias() -> None:
    text = llm_client.get_llm_client()
    vision = llm_client.get_vlm_client()
    legacy = llm_client.get_openai_client()

    assert isinstance(text, llm_client.LocalLLMClient)
    assert isinstance(vision, type(text))
    assert isinstance(legacy, type(text))
    assert [
        (client.base_url, client.model, client.reasoning_effort, client.timeout)
        for client in (text, vision, legacy)
    ] == [(DEFAULT_URL, DEFAULT_MODEL, "xhigh", 600.0)] * 3


def test_factory_env_values_apply_and_explicit_values_take_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "env-text-model")
    monkeypatch.setenv("VLM_BASE_URL", "http://vlm.example/v1")
    monkeypatch.setenv("VLM_MODEL", "env-vision-model")

    text = llm_client.get_llm_client()
    vision = llm_client.get_vlm_client()
    explicit = llm_client.get_vlm_client(
        base_url="http://override.example/v1", model="explicit-model"
    )

    assert (text.base_url, text.model) == ("http://llm.example/v1", "env-text-model")
    assert (vision.base_url, vision.model) == ("http://vlm.example/v1", "env-vision-model")
    assert (explicit.base_url, explicit.model) == (
        "http://override.example/v1",
        "explicit-model",
    )


def test_factory_reasoning_and_timeout_environment_precedence_for_both_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    monkeypatch.setenv("LLM_TIMEOUT", "45")
    monkeypatch.setenv("VLM_REASONING_EFFORT", "medium")
    monkeypatch.setenv("VLM_TIMEOUT", "90")

    text = llm_client.get_llm_client()
    vision = llm_client.get_vlm_client()
    explicit_text = llm_client.get_llm_client(reasoning_effort="high", timeout=12)
    explicit_vision = llm_client.get_vlm_client(reasoning_effort="xhigh", timeout=24)

    assert (text.reasoning_effort, text.timeout) == ("low", 45.0)
    assert (vision.reasoning_effort, vision.timeout) == ("medium", 90.0)
    assert (explicit_text.reasoning_effort, explicit_text.timeout) == ("high", 12.0)
    assert (explicit_vision.reasoning_effort, explicit_vision.timeout) == ("xhigh", 24.0)


def test_environment_template_selects_primary_ollama_route_for_both_factories() -> None:
    exports = {}
    for line in ENV_TEMPLATE_PATH.read_text().splitlines():
        match = re.match(
            r"^\s*export\s+(LLM|VLM)_(BASE_URL|MODEL|REASONING_EFFORT|TIMEOUT)=(\S+)\s*$",
            line,
        )
        if match:
            prefix, key, value = match.groups()
            exports[(prefix, key)] = value

    expected = {
        (prefix, "BASE_URL"): DEFAULT_URL
        for prefix in ("LLM", "VLM")
    } | {
        (prefix, "MODEL"): DEFAULT_MODEL
        for prefix in ("LLM", "VLM")
    } | {
        (prefix, "REASONING_EFFORT"): "xhigh"
        for prefix in ("LLM", "VLM")
    } | {
        (prefix, "TIMEOUT"): "600"
        for prefix in ("LLM", "VLM")
    }
    assert exports == expected
    assert all("Qwen3-VL" not in value and ":8000" not in value for value in exports.values())


def test_request_posts_model_messages_and_xhigh_reasoning_to_configured_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def post(url, *, headers, data, timeout):
        calls.append((url, headers, json.loads(data), timeout))
        return FakeResponse(body={"id": "c1", "model": DEFAULT_MODEL, "choices": []})

    monkeypatch.setattr(llm_client.requests, "post", post)
    client = llm_client.get_llm_client(reasoning_effort="xhigh", json_mode=False)
    client.chat.completions.create(
        messages=[{"role": "user", "content": "describe this scene"}]
    )

    assert len(calls) == 1
    url, headers, payload, timeout = calls[0]
    assert url == f"{DEFAULT_URL}/chat/completions"
    assert headers["Authorization"] == "Bearer EMPTY"
    assert timeout == 600.0
    assert payload["model"] == DEFAULT_MODEL
    assert payload["messages"] == [{"role": "user", "content": "describe this scene"}]
    assert payload["reasoning_effort"] == "xhigh"
    assert payload["stream"] is False


def test_image_message_string_url_is_normalized_without_changing_other_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def post(url, **kwargs):
        calls.append(json.loads(kwargs["data"]))
        return FakeResponse(body={"choices": []})

    monkeypatch.setattr(llm_client.requests, "post", post)
    client = llm_client.get_vlm_client(json_mode=False)
    original = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is here?"},
                {"type": "image_url", "image_url": "data:image/png;base64,abc"},
                {"type": "image_url", "image_url": {"url": "https://img.test/a.png"}},
            ],
        }
    ]
    client.chat.completions.create(messages=original)

    assert calls[0]["messages"][0]["content"] == [
        {"type": "text", "text": "What is here?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "image_url", "image_url": {"url": "https://img.test/a.png"}},
    ]
    assert original[0]["content"][1]["image_url"] == "data:image/png;base64,abc"


def test_json_mode_default_and_per_call_response_format_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = []

    def post(url, **kwargs):
        payloads.append(json.loads(kwargs["data"]))
        return FakeResponse(body={"choices": []})

    monkeypatch.setattr(llm_client.requests, "post", post)
    client = llm_client.get_llm_client()
    client.chat.completions.create(messages=[])
    client.chat.completions.create(
        messages=[], response_format={"type": "text"}
    )
    plain = llm_client.get_llm_client(json_mode=False)
    plain.chat.completions.create(messages=[])

    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert payloads[1]["response_format"] == {"type": "text"}
    assert "response_format" not in payloads[2]


def test_parsed_response_supports_attribute_and_dictionary_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "id": "completion-7",
        "model": DEFAULT_MODEL,
        "created": 123,
        "usage": {"total_tokens": 8},
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "{\"objects\": []}",
                    "reasoning": "checked the frame",
                },
            }
        ],
    }
    monkeypatch.setattr(
        llm_client.requests, "post", lambda *args, **kwargs: FakeResponse(body=response)
    )
    result = llm_client.get_llm_client().chat.completions.create(messages=[])

    assert result.choices[0].message.content == '{"objects": []}'
    assert result["choices"][0]["message"]["content"] == '{"objects": []}'
    assert result.choices[0].message.reasoning_content == "checked the frame"
    assert result["usage"]["total_tokens"] == 8


def test_http_error_is_actionable_and_does_not_retry_without_effort_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return FakeResponse(status_code=503, text="model server unavailable")

    monkeypatch.setattr(llm_client.requests, "post", post)
    with pytest.raises(llm_client.LocalLLMError, match="HTTP 503") as error:
        llm_client.get_llm_client().chat.completions.create(messages=[])

    assert "model server unavailable" in str(error.value)
    assert calls == [f"{DEFAULT_URL}/chat/completions"]


def test_xhigh_rejection_is_surfaced_without_reasoning_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def post(url, **kwargs):
        calls.append(json.loads(kwargs["data"]))
        return FakeResponse(status_code=400, text="unsupported reasoning_effort: xhigh")

    monkeypatch.setattr(llm_client.requests, "post", post)
    client = llm_client.get_llm_client(reasoning_effort="xhigh", json_mode=False)
    with pytest.raises(llm_client.LocalLLMError, match="HTTP 400") as error:
        client.chat.completions.create(messages=[])

    assert "xhigh" in str(error.value)
    assert len(calls) == 1
    assert calls[0]["reasoning_effort"] == "xhigh"


def test_timeout_is_wrapped_with_server_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def post(*args, **kwargs):
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(llm_client.requests, "post", post)
    with pytest.raises(llm_client.LocalLLMError, match="local model server") as error:
        llm_client.get_llm_client().chat.completions.create(messages=[])
    assert "read timed out" in str(error.value)


def test_non_json_http_body_is_reported_as_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(body=ValueError("bad json"), text="<html>oops</html>"),
    )
    with pytest.raises(llm_client.LocalLLMError, match="non-JSON response") as error:
        llm_client.get_llm_client().chat.completions.create(messages=[])
    assert "<html>oops</html>" in str(error.value)
