"""Local, OpenAI-compatible chat client (no ``openai`` SDK required).

This replaces the cloud OpenAI dependency of ConceptGraphs with a model that
runs **on this machine**, served by `Ollama <https://ollama.com>`_:

    qwen3.8:27b-mtp4-ud_q4_K_XL-128k
        * Qwen3.8-27B, Unsloth-Dynamic Q4_K_XL quantisation ("UD")
        * MTP4 speculative decoding (``draft_num_predict = 4``)
        * 128k context window (``num_ctx = 131072``)
        * native vision (it consumes the annotated frames of the pipeline)
        * thinking mode with the levels ``false | low | medium | xhigh``

The client talks to Ollama's OpenAI-compatible endpoint
(``http://127.0.0.1:20261/v1`` by default) and therefore needs nothing but
``requests`` - no cloud SDK, no API key.

The response objects mimic the parts of the OpenAI SDK this repository uses and
support *both* access styles, so existing call sites keep working::

    response.choices[0].message.content              # attribute style
    response["choices"][0]["message"]["content"]      # dict style

Ollama returns the thinking trace in ``message.reasoning`` (vLLM and OpenAI
use ``reasoning_content``); both are parsed into ``message.reasoning_content``.

Configuration (all optional, via environment variables):

===========================  =========================================  ==============================================
variable                     meaning                                    default
===========================  =========================================  ==============================================
``LLM_BASE_URL``             base URL of the OpenAI-compatible API        ``http://127.0.0.1:20261/v1`` (Ollama)
``LLM_MODEL``                model tag as served by the server          ``qwen3.8:27b-mtp4-ud_q4_K_XL-128k``
``LLM_API_KEY``              bearer token (local servers ignore it)       ``EMPTY``
``LLM_TIMEOUT``              request timeout in seconds                  ``600``
``LLM_MAX_TOKENS``           ``max_tokens`` cap (0 = unlimited)          ``0``
``LLM_REASONING_EFFORT``     ``none|minimal|low|medium|high|xhigh|max``  ``xhigh``
``LLM_JSON_MODE``            request strict JSON output                 ``1``
``LLM_TEMPERATURE``          sampling temperature                       ``1.0``
``LLM_TOP_P``                nucleus sampling                           ``0.95``
===========================  =========================================  ==============================================

Any other OpenAI-compatible server works as well - point ``LLM_BASE_URL`` at a
vLLM instance (e.g. serving ``Qwen/Qwen3-VL-8B-Instruct`` or the scene-graph
model ``JosephZ/R1-SGG-7B``) and set ``LLM_MODEL`` accordingly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import requests

__all__ = [
    "LocalLLMClient",
    "LocalLLMError",
    "ChatCompletion",
    "Choice",
    "Message",
    "get_llm_client",
    "get_vlm_client",
    "get_openai_client",
]

DEFAULT_BASE_URL = "http://127.0.0.1:20261/v1"
DEFAULT_MODEL = "qwen3.8:27b-mtp4-ud_q4_K_XL-128k"
DEFAULT_REASONING_EFFORT = "xhigh"
DEFAULT_TIMEOUT = 600.0
DEFAULT_MAX_TOKENS = 0  # 0 -> do not send max_tokens (unlimited generation)

# Compatibility defaults for the legacy VLM factory. Both factories use the
# primary local model; image requests retain their existing output-token cap.
DEFAULT_VLM_BASE_URL = DEFAULT_BASE_URL
DEFAULT_VLM_MODEL = DEFAULT_MODEL
DEFAULT_VLM_MAX_TOKENS = 4096
DEFAULT_VLM_TIMEOUT = DEFAULT_TIMEOUT

# Reasoning effort values understood by Ollama's OpenAI-compatible layer.
# Ollama maps them onto the model's thinking levels
# (false | low | medium | xhigh for Qwen3.8).
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class LocalLLMError(RuntimeError):
    """Raised when the local model server cannot fulfil a request."""


_UNSET = object()  # sentinel: "not provided" (different from an explicit None)


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Message:
    """A single chat message (mirrors ``openai.types.chat.ChatMessage``)."""

    role: str
    content: Any = None
    reasoning_content: str | None = None
    refusal: str | None = None

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: str | None = None

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class ChatCompletion:
    """The response of ``POST /v1/chat/completions``."""

    id: str
    model: str
    choices: list[Choice] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    created: int = 0

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    @property
    def content(self) -> str:
        """Convenience accessor for the first choice's message content."""
        if not self.choices:
            return ""
        return self.choices[0].message.content or ""

    @property
    def reasoning(self) -> str:
        """Convenience accessor for the first choice's thinking trace."""
        if not self.choices:
            return ""
        return self.choices[0].message.reasoning_content or ""


class _Completions:
    """``client.chat.completions`` namespace."""

    def __init__(self, client: "LocalLLMClient") -> None:
        self._client = client

    def create(
        self,
        model: str | None = None,
        messages: list[dict] | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion:
        if stream:
            raise NotImplementedError(
                "streaming responses are not supported by the local client"
            )
        return self._client._create(model=model, messages=messages, **kwargs)


class _Chat:
    """``client.chat`` namespace."""

    def __init__(self, client: "LocalLLMClient") -> None:
        self.completions = _Completions(client)


class LocalLLMClient:
    """Client for a local OpenAI-compatible chat completions server.

    The configuration is read from environment variables prefixed with
    ``env_prefix`` (``LLM_`` for the text model, ``VLM_`` for the
    vision-language model), so both backends can be configured independently.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = _UNSET,
        json_mode: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        extra_body: dict | None = None,
        env_prefix: str = "LLM",
    ) -> None:
        self.env_prefix = env_prefix

        def env(name: str, default=None):
            value = os.getenv(f"{env_prefix}_{name}")
            return default if value is None else value

        self.base_url = (
            base_url or env("BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or env("MODEL") or DEFAULT_MODEL
        self.api_key = api_key or env("API_KEY") or "EMPTY"
        self.timeout = float(
            timeout if timeout is not None else env("TIMEOUT", DEFAULT_TIMEOUT)
        )
        self.max_tokens = int(
            max_tokens
            if max_tokens is not None
            else env("MAX_TOKENS", DEFAULT_MAX_TOKENS)
        )
        effort = (
            env("REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
            if reasoning_effort is _UNSET
            else reasoning_effort
        )
        if effort is not None and effort != "" and effort not in REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {REASONING_EFFORTS}, got {effort!r}"
            )
        self.reasoning_effort = effort or None
        self.json_mode = (
            json_mode if json_mode is not None else _env_flag(f"{env_prefix}_JSON_MODE", True)
        )
        self.temperature = float(
            temperature if temperature is not None else env("TEMPERATURE", 1.0)
        )
        self.top_p = float(top_p if top_p is not None else env("TOP_P", 0.95))
        self.extra_body = extra_body or {}
        self.chat = _Chat(self)
        # Frame-boundary release of the Ollama runner (resolved by the factory).
        self.unload_after_frame = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _normalise_messages(messages: list[dict]) -> list[dict]:
        """Accept both the dict and the string form of ``image_url``."""
        normalised = []
        for message in messages or []:
            message = dict(message)
            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "image_url"
                        and isinstance(part.get("image_url"), str)
                    ):
                        part = dict(part)
                        part["image_url"] = {"url": part["image_url"]}
                    parts.append(part)
                message["content"] = parts
            normalised.append(message)
        return normalised

    def _build_payload(
        self,
        model: str | None,
        messages: list[dict] | None,
        kwargs: dict,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": self._normalise_messages(messages or []),
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": kwargs.pop("top_p", self.top_p),
            "stream": False,
        }

        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)

        # Thinking mode: Ollama maps `reasoning_effort` onto the model's
        # thinking level (false | low | medium | xhigh for Qwen3.8).
        effort = kwargs.pop("reasoning_effort", self.reasoning_effort)
        if effort:
            payload["reasoning_effort"] = effort

        # Structured output (the repository asks for JSON objects).
        response_format = kwargs.pop("response_format", None)
        if response_format is None and self.json_mode:
            response_format = {"type": "json_object"}
        if response_format:
            payload["response_format"] = response_format

        # Anything else the caller passed wins, then the user supplied extras.
        payload.update(kwargs)
        payload.update(self.extra_body)
        return payload

    @staticmethod
    def _parse(data: dict) -> ChatCompletion:
        choices = []
        for i, choice in enumerate(data.get("choices", [])):
            message = choice.get("message", {}) or {}
            choices.append(
                Choice(
                    index=choice.get("index", i),
                    message=Message(
                        role=message.get("role", "assistant"),
                        content=message.get("content"),
                        reasoning_content=message.get("reasoning_content")
                        or message.get("reasoning"),
                        refusal=message.get("refusal"),
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
            )
        return ChatCompletion(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=choices,
            usage=data.get("usage", {}) or {},
            created=data.get("created", 0),
        )

    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        try:
            response = requests.post(
                url,
                headers=self._headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LocalLLMError(
                f"cannot reach the local model server at {self.base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise LocalLLMError(
                f"{url} returned HTTP {response.status_code}: {response.text[:2000]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise LocalLLMError(
                f"{url} returned a non-JSON response: {response.text[:500]}"
            ) from exc

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def _create(
        self,
        model: str | None = None,
        messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        payload = self._build_payload(model, messages, dict(kwargs))
        data = self._post(payload)
        return self._parse(data)

    def chat_completion(self, **kwargs: Any) -> ChatCompletion:
        """Convenience wrapper around ``client.chat.completions.create``."""
        return self._create(**kwargs)

    def list_models(self) -> list[str]:
        """Return the model names served by the local server."""
        url = f"{self.base_url}/models"
        try:
            response = requests.get(url, headers=self._headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise LocalLLMError(f"cannot reach the local model server: {exc}") from exc
        return [entry.get("id", "") for entry in data.get("data", [])]

    def unload(self) -> None:
        """Release the runner of a local Ollama server.

        Posts ``keep_alive: 0`` to the native ``/api/generate`` endpoint so
        the model is unloaded as soon as the client is done with it. The
        native origin is derived from the configured OpenAI-compatible base
        URL (``http://127.0.0.1:20261/v1`` -> ``http://127.0.0.1:20261``).
        """
        origin = self.base_url
        if origin.endswith("/v1"):
            origin = origin[: -len("/v1")]
        url = f"{origin}/api/generate"
        payload = {"model": self.model, "prompt": "", "keep_alive": 0}
        try:
            response = requests.post(
                url,
                headers=self._headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LocalLLMError(
                f"cannot reach the local model server at {self.base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise LocalLLMError(
                f"{url} returned HTTP {response.status_code}: {response.text[:2000]}"
            )
        try:
            response.json()
        except ValueError as exc:
            raise LocalLLMError(
                f"{url} returned a non-JSON response: {response.text[:500]}"
            ) from exc


def get_llm_client(**kwargs: Any) -> LocalLLMClient:
    """Client for the local **text** model (Qwen3.8 via Ollama).

    Used for the caption consolidation and the scene-graph bookkeeping calls.
    Configure with ``LLM_BASE_URL``, ``LLM_MODEL``, ``LLM_REASONING_EFFORT``.
    """
    kwargs.setdefault("env_prefix", "LLM")
    client = LocalLLMClient(**kwargs)
    # The text client never releases the runner at a frame boundary.
    client.unload_after_frame = False
    return client


def get_vlm_client(**kwargs: Any) -> LocalLLMClient:
    """Client for the pipeline's local vision-language requests.

    Uses the primary local model by default. Configure with ``VLM_BASE_URL``,
    ``VLM_MODEL``, ``VLM_REASONING_EFFORT`` and ``VLM_TIMEOUT``. Set
    ``VLM_UNLOAD_AFTER_FRAME=1`` to release the Ollama runner after each
    frame's relation and caption work.
    """
    kwargs.setdefault("env_prefix", "VLM")
    # The scene-graph prompts ask for a list of tuples / a list of dicts, not
    # for a JSON object, so the strict JSON mode is disabled by default.
    kwargs.setdefault("json_mode", False)
    # A cap for the generated tokens: without it the model can run into a very
    # long (sometimes repetitive) answer for the frames with many objects.
    if "max_tokens" not in kwargs and not os.getenv("VLM_MAX_TOKENS"):
        kwargs["max_tokens"] = DEFAULT_VLM_MAX_TOKENS
    # Use the shared request deadline unless the VLM-specific environment
    # variable or an explicit argument supplies another value.
    if "timeout" not in kwargs and not os.getenv("VLM_TIMEOUT"):
        kwargs["timeout"] = DEFAULT_TIMEOUT
    client = LocalLLMClient(**kwargs)
    # Opt-in release of the Ollama runner after each frame's VLM work (off by
    # default for compatible non-Ollama endpoints).
    client.unload_after_frame = _env_flag("VLM_UNLOAD_AFTER_FRAME", False)
    return client


def get_openai_client(**kwargs: Any) -> LocalLLMClient:
    """Backwards compatible alias for :func:`get_vlm_client`.

    ConceptGraphs used to query the OpenAI cloud API here; the name is kept so
    that existing call sites do not have to change.  It now points at the
    locally served vision-language model.
    """
    return get_vlm_client(**kwargs)
