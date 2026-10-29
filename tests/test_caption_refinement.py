"""Regression coverage for optional, resumable local caption refinement."""

from __future__ import annotations

import gzip
import importlib
import json
import pickle
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_MODULE = "conceptgraph.scripts.consolidate_captions_local"


def _write_map(exp_out_path: Path, suffix: str, objects: list[dict]) -> tuple[Path, Path]:
    exp_out_path.mkdir(parents=True, exist_ok=True)
    pcd_path = exp_out_path / f"pcd_{suffix}.pkl.gz"
    object_json_path = exp_out_path / f"obj_json_{suffix}.json"
    with gzip.open(pcd_path, "wb") as stream:
        pickle.dump({"objects": objects}, stream)
    object_json_path.write_text(
        json.dumps(
            {
                f"object_{index}": {
                    "object_caption": obj.get("consolidated_caption", "")
                }
                for index, obj in enumerate(objects, start=1)
            }
        ),
        encoding="utf-8",
    )
    return pcd_path, object_json_path


def _read_map(path: Path) -> dict:
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


class _FakeCompletionClient:
    def __init__(self, on_call=None):
        self.calls = []
        self.on_call = on_call
        self.model = "fake-qwen"
        self.base_url = "http://127.0.0.1:20261/v1"
        self.reasoning_effort = "low"
        self.max_tokens = 256
        self.unload_calls = 0
        self.chat = self
        self.completions = self

    def unload(self):
        self.unload_calls += 1

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_call:
            self.on_call(len(self.calls))

        class Message:
            content = json.dumps({"consolidated_caption": f"refined-{len(self.calls)}"})

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        return Response()


def _run_cli(monkeypatch, exp_out_path: Path, suffix: str, *extra_args: str):
    module = importlib.import_module(SCRIPT_MODULE)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "consolidate_captions_local.py",
            "--exp_out_path",
            str(exp_out_path),
            "--exp_suffix",
            suffix,
            *extra_args,
        ],
    )
    return module


def test_mapping_defaults_to_returning_without_caption_model_refinement():
    config_path = PROJECT_ROOT / "conceptgraph" / "hydra_configs" / "base_mapping.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["consolidate_captions"] is False


def test_standalone_cli_uses_fast_bounded_defaults_and_allows_overrides(
    tmp_path, monkeypatch
):
    exp_out_path = tmp_path / "default"
    _write_map(
        exp_out_path,
        "tiny",
        [{"class_name": "chair", "captions": [{"caption": "chair"}], "consolidated_caption": ""}],
    )
    module = _run_cli(monkeypatch, exp_out_path, "tiny")
    defaults = _FakeCompletionClient()
    factory_calls = []

    def default_factory(**kwargs):
        factory_calls.append(kwargs)
        return defaults

    monkeypatch.setattr(module, "get_llm_client", default_factory)
    module.main()

    assert factory_calls[-1]["reasoning_effort"] == "low"
    assert factory_calls[-1]["max_tokens"] == 256
    assert defaults.unload_calls == 1

    override_path = tmp_path / "override"
    _write_map(
        override_path,
        "tiny",
        [{"class_name": "chair", "captions": [{"caption": "chair"}], "consolidated_caption": ""}],
    )
    override_module = _run_cli(
        monkeypatch,
        override_path,
        "tiny",
        "--reasoning-effort",
        "medium",
        "--max-tokens",
        "96",
    )
    overridden = _FakeCompletionClient()
    monkeypatch.setattr(
        override_module,
        "get_llm_client",
        lambda **kwargs: (factory_calls.append(kwargs) or overridden),
    )
    override_module.main()

    assert factory_calls[-1]["reasoning_effort"] == "medium"
    assert factory_calls[-1]["max_tokens"] == 96
    assert overridden.unload_calls == 1


def test_local_refinement_resumes_skips_empty_captions_and_checkpoints_both_outputs(
    tmp_path, monkeypatch
):
    exp_out_path = tmp_path / "resume"
    objects = [
        {
            "class_name": "already done",
            "captions": [{"caption": "old"}],
            "consolidated_caption": "keep this",
        },
        {
            "class_name": "first",
            "captions": [{"caption": "first raw caption"}],
            "consolidated_caption": "",
        },
        {"class_name": "no captions", "captions": [], "consolidated_caption": ""},
        {
            "class_name": "second",
            "captions": [{"caption": "second raw caption"}],
            "consolidated_caption": "",
        },
    ]
    pcd_path, object_json_path = _write_map(exp_out_path, "tiny", objects)

    module = _run_cli(
        monkeypatch,
        exp_out_path,
        "tiny",
        "--checkpoint-every",
        "2",
    )
    fake_client = _FakeCompletionClient()
    monkeypatch.setattr(module, "get_llm_client", lambda **kwargs: fake_client)

    import os

    replace_calls = []
    checkpoints = []
    original_replace = os.replace

    def checked_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        result = original_replace(source, destination)
        if Path(destination) == object_json_path:
            pcd_objects = _read_map(pcd_path)["objects"]
            json_objects = json.loads(object_json_path.read_text(encoding="utf-8"))
            checkpoints.append((pcd_objects, json_objects))
        return result

    monkeypatch.setattr(os, "replace", checked_replace)
    module.main()

    assert len(fake_client.calls) == 2
    assert fake_client.unload_calls == 1
    output = _read_map(pcd_path)["objects"]
    assert output[0]["consolidated_caption"] == "keep this"
    assert output[1]["consolidated_caption"] == "refined-1"
    assert output[2]["consolidated_caption"] == ""
    assert output[3]["consolidated_caption"] == "refined-2"
    assert sum(destination == pcd_path for _, destination in replace_calls) >= 2
    assert sum(destination == object_json_path for _, destination in replace_calls) >= 2
    assert checkpoints
    checkpoint_objects, checkpoint_json = checkpoints[0]
    assert checkpoint_objects[1]["consolidated_caption"] == "refined-1"
    assert checkpoint_objects[2]["consolidated_caption"] == ""
    assert "refined-1" in checkpoint_json["object_2"]["object_caption"]
    assert checkpoint_json["object_3"]["object_caption"] == ""
    final_json = json.loads(object_json_path.read_text(encoding="utf-8"))
    assert "refined-2" in final_json["object_4"]["object_caption"]
