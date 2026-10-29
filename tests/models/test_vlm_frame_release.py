"""Mock-only acceptance tests for the optional per-frame Ollama release."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from conceptgraph.utils import general_utils, llm_client


ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "env_vars.bash.template").is_file()
)
ENV_TEMPLATE = ROOT / "env_vars.bash.template"
MODEL = "qwen3.8:27b-mtp4-ud_q4_K_XL-128k"


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


@pytest.fixture(autouse=True)
def clear_unload_environment(monkeypatch):
    for key in (
        "VLM_UNLOAD_AFTER_FRAME",
        "LLM_UNLOAD_AFTER_FRAME",
        "VLM_BASE_URL",
        "VLM_MODEL",
        "VLM_TIMEOUT",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_unload_posts_exact_native_ollama_request_with_selected_model_and_timeout(
    monkeypatch,
):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(body={"done": True})

    monkeypatch.setattr(llm_client.requests, "post", post)
    client = llm_client.LocalLLMClient(
        base_url="https://ollama.example:11434/v1",
        model="qwen-local:custom-tag",
        timeout=37.5,
    )

    client.unload()

    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "https://ollama.example:11434/api/generate"
    assert json.loads(kwargs["data"]) == {
        "model": "qwen-local:custom-tag",
        "prompt": "",
        "keep_alive": 0,
    }
    assert kwargs["timeout"] == 37.5


@pytest.mark.parametrize(
    ("response", "error_text"),
    [
        (FakeResponse(status_code=503, text="runner unavailable"), "503"),
        (FakeResponse(body=ValueError("bad json"), text="not json"), "api/generate"),
    ],
)
def test_unload_http_and_malformed_response_fail_as_local_llm_error(
    monkeypatch, response, error_text
):
    monkeypatch.setattr(llm_client.requests, "post", lambda *args, **kwargs: response)
    client = llm_client.LocalLLMClient(base_url="http://127.0.0.1:20261/v1")

    with pytest.raises(llm_client.LocalLLMError, match=error_text):
        client.unload()


@pytest.mark.parametrize(
    "transport_error",
    [requests.Timeout("deadline"), requests.ConnectionError("refused")],
)
def test_unload_timeout_and_connection_failures_include_endpoint_context(
    monkeypatch, transport_error
):
    monkeypatch.setattr(
        llm_client.requests,
        "post",
        Mock(side_effect=transport_error),
    )
    client = llm_client.LocalLLMClient(base_url="http://ollama.local:11434/v1")

    with pytest.raises(llm_client.LocalLLMError, match="ollama.local:11434"):
        client.unload()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("yes", True), ("on", True),
     ("0", False), ("false", False), ("no", False), ("off", False)],
)
def test_vlm_factory_parses_frame_release_flag_using_existing_env_semantics(
    monkeypatch, value, expected
):
    monkeypatch.setenv("VLM_UNLOAD_AFTER_FRAME", value)

    client = llm_client.get_vlm_client()

    assert client.unload_after_frame is expected


def test_vlm_flag_defaults_off_but_llm_factory_ignores_vlm_flag(monkeypatch):
    assert llm_client.get_vlm_client().unload_after_frame is False
    monkeypatch.setenv("VLM_UNLOAD_AFTER_FRAME", "1")

    assert llm_client.get_vlm_client().unload_after_frame is True
    assert llm_client.get_llm_client().unload_after_frame is False


def test_environment_template_enables_release_for_project_vlm_route():
    exports = dict(
        re.findall(
            r"^\s*export\s+(VLM_UNLOAD_AFTER_FRAME)=(\S+)\s*$",
            ENV_TEMPLATE.read_text(),
            flags=re.MULTILINE,
        )
    )

    assert exports == {"VLM_UNLOAD_AFTER_FRAME": "1"}


def _prepare_frame_helper(monkeypatch, client, events):
    monkeypatch.setattr(
        general_utils,
        "filter_detections",
        lambda **kwargs: ("filtered", ["chair 1"]),
    )
    monkeypatch.setattr(
        general_utils,
        "get_vlm_annotated_image_path",
        lambda *args, **kwargs: "annotated.jpg",
    )
    monkeypatch.setattr(
        general_utils,
        "annotate_for_vlm",
        lambda *args, **kwargs: ("annotated-image", [0]),
    )
    monkeypatch.setattr(general_utils.cv2, "imwrite", lambda *args: True)
    monkeypatch.setattr(
        general_utils,
        "get_obj_rel_from_image_gpt4v",
        lambda *args: events.append("relations") or [(1, "left of", 2)],
    )
    monkeypatch.setattr(
        general_utils,
        "get_obj_captions_from_image_gpt4v",
        lambda *args: events.append("captions") or ["a chair"],
    )
    monkeypatch.setattr(
        general_utils,
        "plot_edges_from_vlm",
        lambda *args, **kwargs: "edge-image",
    )
    client.unload = Mock(side_effect=lambda: events.append("unload"))


def _run_frame_helper(client, make_edges=True):
    return general_utils.make_vlm_edges_and_captions(
        image="image",
        curr_det="detections",
        obj_classes=["chair"],
        detection_class_labels=["chair 1"],
        det_exp_vis_path=Path("."),
        color_path=Path("frame.png"),
        make_edges_flag=make_edges,
        vlm_client=client,
    )


def test_frame_helper_unloads_once_after_relations_and_captions(monkeypatch):
    client = SimpleNamespace(unload_after_frame=True)
    events = []
    _prepare_frame_helper(monkeypatch, client, events)

    result = _run_frame_helper(client)

    assert events == ["relations", "captions", "unload"]
    assert client.unload.call_count == 1
    assert result == (["chair 1"], [(1, "left of", 2)], "edge-image", ["a chair"])


def test_frame_helper_skips_unload_when_flag_is_disabled(monkeypatch):
    client = SimpleNamespace(unload_after_frame=False)
    events = []
    _prepare_frame_helper(monkeypatch, client, events)

    _run_frame_helper(client)

    assert events == ["relations", "captions"]
    client.unload.assert_not_called()


def test_frame_helper_skips_unload_when_edge_generation_is_disabled(monkeypatch):
    client = SimpleNamespace(unload_after_frame=True)
    events = []
    _prepare_frame_helper(monkeypatch, client, events)

    result = _run_frame_helper(client, make_edges=False)

    assert events == []
    client.unload.assert_not_called()
    assert result == (["chair 1"], [], None, None)


def test_frame_helper_reaches_unload_boundary_if_relation_extraction_raises(
    monkeypatch,
):
    client = SimpleNamespace(unload_after_frame=True)
    events = []
    _prepare_frame_helper(monkeypatch, client, events)
    def extraction_failure(*args):
        events.append("relations")
        raise ValueError("extract failed")

    monkeypatch.setattr(general_utils, "get_obj_rel_from_image_gpt4v", extraction_failure)

    with pytest.raises(ValueError, match="extract failed"):
        _run_frame_helper(client)

    assert events == ["relations", "unload"]
    client.unload.assert_called_once_with()


def test_frame_helper_unloads_after_extraction_wrapper_returns_empty_on_swallowed_error(
    monkeypatch,
):
    client = SimpleNamespace(unload_after_frame=True)
    events = []
    _prepare_frame_helper(monkeypatch, client, events)

    # The VLM extraction wrappers intentionally swallow request errors and
    # return empty results; frame cleanup still follows both wrapper calls.
    monkeypatch.setattr(
        general_utils,
        "get_obj_rel_from_image_gpt4v",
        lambda *args: events.append("relations") or [],
    )
    monkeypatch.setattr(
        general_utils,
        "get_obj_captions_from_image_gpt4v",
        lambda *args: events.append("captions") or [],
    )

    result = _run_frame_helper(client)

    assert events == ["relations", "captions", "unload"]
    assert result == (["chair 1"], [], "edge-image", [])
    client.unload.assert_called_once_with()


def test_frame_helper_propagates_unload_failure_after_vlm_work(monkeypatch):
    client = SimpleNamespace(unload_after_frame=True)
    events = []
    _prepare_frame_helper(monkeypatch, client, events)

    def unload_failure():
        events.append("unload")
        raise llm_client.LocalLLMError("cannot unload Ollama runner")

    client.unload.side_effect = unload_failure

    with pytest.raises(llm_client.LocalLLMError, match="cannot unload Ollama runner"):
        _run_frame_helper(client)

    assert events == ["relations", "captions", "unload"]
    client.unload.assert_called_once_with()
