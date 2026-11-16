"""Real local-model acceptance coverage for selected Replica caption groups.

Set ``CONCEPTGRAPHS_CAPTION_TEST_MAP`` to the gzip-pickled mapping file from
the room0 stride-50 modernized pipeline run. This test sends real requests to
the configured local LLM and requires its Ollama runner to be visible on GPU.
Set ``CONCEPTGRAPHS_CAPTION_TEST_INDICES`` to a comma-separated object-index
list to run a broader paired scene sample instead of the four failed groups.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path
import tempfile
import time

import pytest


FAILED_ROOM0_OBJECT_INDICES = (100, 278, 348, 419)


def _selected_object_indices() -> tuple[int, ...]:
    raw = os.getenv("CONCEPTGRAPHS_CAPTION_TEST_INDICES")
    if not raw:
        return FAILED_ROOM0_OBJECT_INDICES
    try:
        indices = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        pytest.fail(f"Caption test indices must be comma-separated integers: {exc}")
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        pytest.fail("Caption test indices must be unique non-negative integers")
    return indices


def _caption_map_path() -> Path:
    raw_path = os.getenv("CONCEPTGRAPHS_CAPTION_TEST_MAP")
    if not raw_path:
        pytest.skip(
            "Set CONCEPTGRAPHS_CAPTION_TEST_MAP to the real room0 stride-50 "
            "gzip-pickled map to run local-model acceptance."
        )
    path = Path(raw_path).expanduser()
    if not path.is_file():
        pytest.fail(f"Caption acceptance map does not exist: {path}")
    return path


def _save_baseline_results(
    map_path: Path,
    model: str,
    selected: dict,
    outputs: dict,
    response_seconds: dict,
) -> None:
    output_path = os.getenv("CONCEPTGRAPHS_CAPTION_TEST_OUTPUT")
    if not output_path:
        return
    output_path = Path(output_path).expanduser()
    if output_path.resolve() == map_path.resolve() or (
        output_path.exists() and os.path.samefile(output_path, map_path)
    ):
        pytest.fail("CONCEPTGRAPHS_CAPTION_TEST_OUTPUT must not overwrite the source map")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with map_path.open("rb") as map_file:
        for chunk in iter(lambda: map_file.read(1024 * 1024), b""):
            digest.update(chunk)
    report = {
        "map_path": str(map_path.resolve()),
        "map_sha256": digest.hexdigest(),
        "model": model,
        "reasoning_effort": "none",
        "max_tokens": 256,
        "groups": [
            {
                "object_index": index,
                "source_captions": [row["caption"] for row in rows],
                "caption": outputs[index],
                "response_seconds": response_seconds[index],
            }
            for index, rows in selected.items()
        ],
    }
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def caption_map_data():
    map_path = _caption_map_path()
    with gzip.open(map_path, "rb") as map_file:
        results = pickle.load(map_file)
    assert isinstance(results, dict), f"Map {map_path} is not a mapping object"
    return map_path, results


@pytest.fixture(scope="module")
def real_local_llm_client(caption_map_data):
    """Load the configured text model through model-control and always unload."""
    from conceptgraph.utils.llm_client import get_llm_client
    from conceptgraph.utils.model_control import (
        load_ollama_gpu_model,
        require_idle_model_control,
        unload_all_models,
    )

    del caption_map_data  # Require the real map before taking model residency.
    client = get_llm_client(reasoning_effort="none", max_tokens=256)
    load_started = False
    try:
        # Do not unload an active model or interrupt an unrelated GPU job.
        # The fixture owns only the model it loads below.
        require_idle_model_control()
        load_started = True
        runner_pids = load_ollama_gpu_model(
            client.model,
            base_url=client.base_url.removesuffix("/v1"),
            keep_alive=-1,
        )
        assert runner_pids, f"No GPU runner reported for configured model {client.model!r}"
        yield client
    finally:
        if load_started:
            unload_all_models()


def test_failed_room0_caption_groups_consolidate_on_configured_gpu_model(
    caption_map_data, real_local_llm_client,
) -> None:
    """Each previously blank group must produce a concise caption via real inference."""
    map_path, results = caption_map_data
    objects = results.get("objects") if isinstance(results, dict) else None
    assert isinstance(objects, list), f"Map {map_path} has no objects list"

    selected_indices = _selected_object_indices()
    require_blank_existing = not os.getenv("CONCEPTGRAPHS_CAPTION_TEST_INDICES")
    selected = {}
    for object_index in selected_indices:
        assert object_index < len(objects), (
            f"Map {map_path} has {len(objects)} objects; expected index {object_index}"
        )
        obj = objects[object_index]
        existing_caption = obj.get("consolidated_caption")
        if require_blank_existing:
            assert not (isinstance(existing_caption, str) and existing_caption.strip()), (
                f"Object {object_index} already has a consolidated caption; expected "
                "one of the previously failed groups"
            )
        captions = [
            row
            for row in (obj.get("captions") or [])
            if isinstance(row, dict)
            and row.get("caption") is not None
        ]
        assert any(
            isinstance(row.get("caption"), str) and row["caption"].strip()
            for row in captions[:20]
        ), f"Object {object_index} has no non-empty source caption text"
        selected[object_index] = captions[:20]

    from conceptgraph.utils.vlm import consolidate_captions

    outputs = {}
    response_seconds = {}
    for index, captions in selected.items():
        started = time.perf_counter()
        outputs[index] = consolidate_captions(real_local_llm_client, captions)
        response_seconds[index] = round(time.perf_counter() - started, 3)

    for index, caption in outputs.items():
        assert isinstance(caption, str) and caption.strip(), (
            f"Configured model {real_local_llm_client.model!r} returned an empty "
            f"consolidated caption for object {index}"
        )
        assert len(caption.strip()) <= 500, (
            f"Consolidated caption for object {index} is not concise "
            f"({len(caption.strip())} characters)"
        )
    _save_baseline_results(
        map_path,
        real_local_llm_client.model,
        selected,
        outputs,
        response_seconds,
    )
