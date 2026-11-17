"""Run the NInfer Qwen3.8 27B NVFP4 route on the fixed Replica room0 set.

Standalone counterpart of ``scripts/benchmark_vlm_shortlist_transformers.py``
for the pending held-out NInfer route. It queries only the local NInfer
OpenAI-compatible endpoint (loopback; the managed ``qwen3.8-27b-nvfp4``
profile must already be active, started by the model-control lifecycle) and
reuses the same 40-frame stride-50 Replica ``room0`` detections, cached
annotated images, object IDs, legacy relation/caption prompts, strict
parsers, and cached Qwen3.8 reference metrics as the Transformers runner.

This runner starts no model, contacts no model-control service, and performs
no GPU probe. It only issues the 80 chat-completion requests (one relation
and one caption per frame) and records raw outputs, schema checks, agreement
with the cached Qwen3.8 annotations (model-output agreement only, not
ground-truth accuracy), per-request latency, token usage, and whole-device
GPU memory samples.

Fail-closed storage and identity contract:

- Output must live inside the repository's Git-ignored
  ``models/experiments/vlm/data/`` subtree; every other path (including
  ``/tmp`` and ``/mnt/HDD1/tmp``) is refused, and existing outputs are never
  overwritten.
- The frame cohort is the exact ordered stride-50 list ``frame000000`` ...
  ``frame001950``; missing frames or extra ``frame*.pkl.gz`` files fail the
  run.
- Every response must carry a ``model`` field equal to the requested model;
  an omission or mismatch fails that request (its text is preserved as
  evidence only and can never score) and the run finishes incomplete.
- Profile identity and launch details come from the required
  ``--profile-manifest`` JSON captured by root before the run; the runner
  reads that file locally and never contacts model-control.
- The route is pinned: ``--model`` must be exactly ``qwen3.8-27b-nvfp4``
  and ``--endpoint`` the NInfer loopback base URL on port 20301 with path
  ``/v1`` (loopback aliases ``localhost``/``::1`` only, no userinfo,
  query, or fragment); every other value fails before any endpoint request
  or output creation, as does a manifest whose ``controller_load`` does not
  report ``job_status == "done"`` and ``ready is True``.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RUNNER_VERSION = "1.0.0"
MODEL_DEFAULT = "qwen3.8-27b-nvfp4"
ENDPOINT_PORT = 20301
ENDPOINT_PATH = "/v1"
ENDPOINT_DEFAULT = f"http://127.0.0.1:{ENDPOINT_PORT}{ENDPOINT_PATH}"
EXPECTED_FRAME_COUNT = 40
ALLOWED_RELATIONS = {"on top of", "under"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
# The held-out cohort is the fixed ordered stride-50 list frame000000 ...
# frame001950 (40 names). Not merely any 40 files: missing or extra frame
# files in the detections directory fail the run so the cohort cannot be
# silently altered.
FRAME_STRIDE = 50
EXPECTED_FRAME_NAMES = [
    f"frame{index:06d}"
    for index in range(0, EXPECTED_FRAME_COUNT * FRAME_STRIDE, FRAME_STRIDE)
]

REPLICA = Path("/mnt/HDD1/Project_Data/public_datasets/Replica/room0")
DETECTION_SUFFIX = "s_modernized_yoloe26l_stride50"


def _find_repo_root() -> Path:
    """Locate the repository root (the directory containing ``conceptgraph``).

    The candidate is staged under
    ``<repo>/.agentic_workflow/workspace/.../candidate/scripts/``, so walking
    up from this file reaches the checkout.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "conceptgraph").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the conceptgraph package by walking up from "
        f"{here}; the staged candidate must stay inside the repository checkout."
    )


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Benchmark output is confined to the repository's Git-ignored data subtree.
# Every other path -- including /tmp, /mnt/HDD1/tmp, and any other location
# on the project HDD -- is refused.
ALLOWED_OUTPUT_ROOT = REPO_ROOT / "models" / "experiments" / "vlm" / "data"

from conceptgraph.scripts.benchmark_edge_compatible import (
    caption_by_id,
    edge_metrics,
    load_pickle_gzip,
    pipeline_label_list,
    word_f1,
)
from conceptgraph.utils.vlm import (
    extract_list_of_tuples,
    system_prompt_captions,
    system_prompt_only_top,
    vlm_extract_object_captions,
)


RELATION_USER_PROMPT_TEMPLATE = (
    "Here is the list of labels for the annotations of the objects in the image: "
    "{labels}. Please describe the spatial relationships between the objects in the image."
)
CAPTION_USER_PROMPT_TEMPLATE = (
    "Here is the list of labels for the annotations of the objects in the image: "
    "{labels}. Please accurately caption the objects in the image."
)

PROMPT_PROTOCOL_NOTE = (
    "legacy profile: OpenAI-style messages [system text; image-only user turn "
    "(image_url data URL); text-only user turn] built from "
    "conceptgraph.utils.vlm system_prompt_only_top / system_prompt_captions; "
    "same labels and annotated_for_vlm.jpg cache as the Transformers runner; "
    "the NInfer server applies its own chat template server-side"
)

SERIALIZATION_NOTE = (
    "Request serialization differs from the Transformers routes: the "
    "Transformers runner applies the processor chat template client-side and "
    "feeds tokenized tensors to a local model, while this runner sends "
    "OpenAI-compatible chat-completion JSON over HTTP and the NInfer server "
    "applies its own chat template and image preprocessing. The model-visible "
    "prompt token stream, image tokenization, and token accounting may "
    "therefore differ even though the raw text prompts are identical. Token "
    "counts are server-reported. No retry or alternate prompt is used."
)

DECODE_NOTE = (
    "Deterministic decoding: the Transformers runner used do_sample=False "
    "(greedy). The NInfer OpenAI-compatible API exposes temperature/top_p; "
    "temperature=0.0 is the closest supported deterministic setting and is a "
    "route-specific setting that cannot be matched exactly. max_tokens caps "
    "completion length (the runner counterpart of max_new_tokens)."
)

METRIC_NOTE = (
    "Agreement with cached Qwen3.8 outputs is model-output agreement/schema "
    "reliability only, not independent semantic ground truth."
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class RunnerRequestError(RuntimeError):
    """A single request failed or returned a malformed response."""

    def __init__(self, message: str, *, http_status: int | None = None,
                 body_preview: str | None = None, raw_text: str | None = None,
                 response_model: str | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.body_preview = body_preview
        # Evidence carried by failures that did receive a body; never scored.
        self.raw_text = raw_text
        self.response_model = response_model


class ResponseModelMismatchError(RunnerRequestError):
    """The endpoint answered for a different (or no) model ID than requested."""


def _endpoint_is_pinned_ninfer(endpoint: str) -> bool:
    """True only for the pinned NInfer loopback base URL (route pin).

    The Qwen3.8 NInfer route is fixed to ``http://127.0.0.1:20301/v1``. The
    documented loopback aliases (``localhost``, ``::1``) resolve to the same
    local endpoint and are accepted only with the same port and path; a
    different scheme, port, or path, or any userinfo, query, or fragment, is
    refused.
    """
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and parsed.hostname in LOOPBACK_HOSTS
        and port == ENDPOINT_PORT
        and parsed.path == ENDPOINT_PATH
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def _http_json(method: str, url: str, payload: dict | None, timeout: float) -> tuple[int, dict, dict]:
    """Perform one HTTP request and decode a JSON object body. Fail closed."""
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            header_map = {key.lower(): value for key, value in response.headers.items()}
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RunnerRequestError(
            f"HTTP {exc.code} from {url}", http_status=exc.code, body_preview=raw[:2000]
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RunnerRequestError(
            f"Request to {url} failed: {type(exc).__name__}: {exc}"
        ) from None
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RunnerRequestError(
            f"Malformed JSON body from {url}: {exc}", http_status=status,
            body_preview=body[:2000],
        ) from None
    if not isinstance(decoded, dict):
        raise RunnerRequestError(
            f"Malformed JSON body from {url}: expected object, got {type(decoded).__name__}",
            http_status=status,
        )
    return status, header_map, decoded


def _preflight_models(endpoint: str, model: str, timeout: float) -> dict:
    """Verify the loopback endpoint is up and serves the exact model ID."""
    status, headers, body = _http_json("GET", endpoint.rstrip("/") + "/models", None, timeout)
    data = body.get("data")
    if not isinstance(data, list):
        raise RuntimeError("Malformed /models response: missing 'data' list")
    model_ids = [
        item.get("id") for item in data if isinstance(item, dict)
    ]
    if model not in model_ids:
        raise RuntimeError(
            f"Model mismatch: {endpoint!r} does not serve {model!r}; "
            f"available: {model_ids}"
        )
    version_headers = {
        key: value for key, value in headers.items()
        if key.startswith("x-ninfer") or key == "server"
    }
    return {
        "models_url": endpoint.rstrip("/") + "/models",
        "http_status": status,
        "version_headers": version_headers,
        "models_response": body,
    }


def _gpu_memory_sample(samples: list[int], errors: list[str], stop: threading.Event) -> None:
    """Sample whole-device ``memory.used`` (MiB) once per second via nvidia-smi.

    Whole-device scope: includes the NInfer process plus any other consumer
    and desktop baseline. The process-only scope is not observable from this
    runner because it does not own the NInfer process.
    """
    while not stop.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                rows = [int(row.strip()) for row in result.stdout.splitlines() if row.strip()]
                if rows:
                    samples.append(max(rows))
            else:
                errors.append(f"nvidia-smi failed: {result.stderr.strip()[:200]}")
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        stop.wait(1)


def _strict_list(raw: str, *, relation: bool) -> bool:
    """Ported verbatim from the Transformers runner: legacy strict-list check."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].rstrip()
    start = raw.find("[")
    if start < 0:
        return False
    depth = 0
    quote = None
    escaped = False
    end = None
    for index in range(start, len(raw)):
        char = raw[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        return False
    if raw[:start].strip() or raw[end:].strip():
        return False
    try:
        value = ast.literal_eval(raw[start:end])
    except (ValueError, SyntaxError):
        try:
            value = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return False
    if not isinstance(value, list):
        return False
    if relation:
        return all(
            isinstance(item, (list, tuple)) and len(item) == 3
            and all(isinstance(field, str) for field in item)
            for item in value
        )
    return all(
        isinstance(item, dict)
        and {"id", "name", "caption"} <= set(item)
        and all(isinstance(item.get(key), str) and item[key].strip()
                for key in ("id", "name", "caption"))
        for item in value
    )


def _parse_model_list(raw: str, *, relation: bool, prefer_json: bool):
    """Ported verbatim from the Transformers runner.

    The NInfer route only uses the legacy path (``prefer_json=False``); the
    JSON branch is kept so both runners stay behaviorally identical.
    """
    if prefer_json:
        candidate = raw.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[-1]
            if candidate.rstrip().endswith("```"):
                candidate = candidate.rstrip()[:-3].rstrip()
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            value = None
        if isinstance(value, list):
            return value
        return None
    value = extract_list_of_tuples(raw) if relation else vlm_extract_object_captions(raw)
    if value is not None:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            return None
    return value


def _strict_json_list(raw: str, *, relation: bool, expected_ids: set[str]) -> dict:
    """Ported verbatim from the Transformers runner: JSON schema/ID checks."""
    result = {"json_valid": False, "schema_valid": False, "ids_valid": False}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return result
    result["json_valid"] = isinstance(value, list)
    if not isinstance(value, list):
        return result
    if relation:
        schema = all(
            isinstance(item, list) and len(item) == 3
            and all(isinstance(field, str) and field.strip() for field in item)
            for item in value
        )
        result["schema_valid"] = schema
        result["ids_valid"] = schema and all(
            item[0] in expected_ids and item[2] in expected_ids
            and " ".join(item[1].lower().split()) in ALLOWED_RELATIONS
            for item in value
        )
        result["endpoint_count"] = 2 * len(value)
        result["endpoint_valid_count"] = sum(
            int(item[0] in expected_ids) + int(item[2] in expected_ids)
            for item in value if isinstance(item, list) and len(item) == 3
            and isinstance(item[0], str) and isinstance(item[2], str)
        )
        result["predicate_valid_count"] = sum(
            isinstance(item, list) and len(item) == 3 and isinstance(item[1], str)
            and " ".join(item[1].lower().split()) in ALLOWED_RELATIONS
            for item in value
        )
        result["predicate_total_count"] = len(value)
    else:
        schema = all(
            isinstance(item, dict)
            and all(isinstance(item.get(key), str) and item[key].strip()
                    for key in ("id", "name", "caption"))
            for item in value
        )
        result["schema_valid"] = schema
        ids = [item["id"] for item in value if isinstance(item, dict)
               and isinstance(item.get("id"), str)]
        result.update({
            "expected_id_count": len(expected_ids),
            "matched_id_count": len(set(ids) & expected_ids),
            "missing_ids": sorted(expected_ids - set(ids)),
            "extra_ids": sorted(set(ids) - expected_ids),
            "duplicate_id_count": len(ids) - len(set(ids)),
        })
        result["ids_valid"] = (
            schema and len(ids) == len(expected_ids)
            and set(ids) == expected_ids and len(ids) == len(set(ids))
        )
    return result


def _aggregate_json_validation(records: list[dict], task: str) -> dict:
    """Ported verbatim from the Transformers runner: per-task schema rollup.

    Only entries carrying the full ``strict_json`` block are aggregated, so
    failed requests (which are recorded with ``api_error`` only) count as
    not completed rather than raising.
    """
    entries = [
        record[task] for record in records
        if task in record and "strict_json" in record[task]
    ]
    strict = [entry["strict_json"] for entry in entries]
    expected = 40
    if task == "relation_request":
        json_frames = sum(bool(item["json_valid"]) for item in strict)
        schema_frames = sum(bool(item["schema_valid"]) for item in strict)
        valid_frames = sum(bool(item["ids_valid"]) for item in strict)
        return {
            "expected": expected,
            "completed": len(entries),
            "failed_or_not_completed": expected - len(entries),
            "json_array_frames": json_frames,
            "json_array_rate_over_expected": json_frames / expected,
            "schema_valid_frames": schema_frames,
            "schema_valid_rate_over_expected": schema_frames / expected,
            "ids_and_predicates_valid_frames": valid_frames,
            "ids_and_predicates_valid_rate_over_expected": valid_frames / expected,
            "relation_endpoints_valid": sum(item.get("endpoint_valid_count", 0) for item in strict),
            "relation_endpoints_total": sum(item.get("endpoint_count", 0) for item in strict),
            "relation_predicates_valid": sum(item.get("predicate_valid_count", 0) for item in strict),
            "relation_predicates_total": sum(
                item.get("predicate_total_count", 0) for item in strict
            ),
        }
    json_frames = sum(bool(item["json_valid"]) for item in strict)
    schema_frames = sum(bool(item["schema_valid"]) for item in strict)
    valid_frames = sum(bool(item["ids_valid"]) for item in strict)
    return {
        "expected": expected,
        "completed": len(entries),
        "failed_or_not_completed": expected - len(entries),
        "json_array_frames": json_frames,
        "json_array_rate_over_expected": json_frames / expected,
        "schema_valid_frames": schema_frames,
        "schema_valid_rate_over_expected": schema_frames / expected,
        "exact_id_coverage_frames": valid_frames,
        "exact_id_coverage_rate_over_expected": valid_frames / expected,
        "ids_matched": sum(item.get("matched_id_count", 0) for item in strict),
        "ids_expected": sum(item.get("expected_id_count", 0) for item in strict),
        "ids_missing": sum(len(item.get("missing_ids", [])) for item in strict),
        "ids_extra": sum(len(item.get("extra_ids", [])) for item in strict),
        "duplicate_ids": sum(item.get("duplicate_id_count", 0) for item in strict),
    }


def _chat_request(endpoint: str, model: str, messages: list[dict], max_tokens: int,
                  temperature: float, top_p: float, timeout: float) -> dict:
    """Send one legacy-profile chat completion and validate the response.

    Raises ``RunnerRequestError`` (fail closed) for transport failures,
    non-2xx status, or malformed response structure, and
    ``ResponseModelMismatchError`` (a request failure) when the response
    ``model`` field is missing or differs from the requested model.
    """
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    started = time.perf_counter()
    status, _headers, body = _http_json("POST", url, payload, timeout)
    elapsed = time.perf_counter() - started
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RunnerRequestError(
            "Malformed response: missing or empty 'choices'", http_status=status
        )
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RunnerRequestError(
            "Malformed response: choices[0].message.content is not a string",
            http_status=status,
        )
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise RunnerRequestError(
            "Malformed response: finish_reason is not a string", http_status=status
        )
    usage_raw = body.get("usage")
    if usage_raw is not None and not isinstance(usage_raw, dict):
        raise RunnerRequestError(
            "Malformed response: 'usage' is not an object", http_status=status
        )
    usage = usage_raw or {}

    def _usage_int(key: str) -> int | None:
        value = usage.get(key)
        return value if isinstance(value, int) else None

    # The response model field must be present and equal the requested model;
    # an omission or mismatch fails this request (its text can never score).
    response_model = body.get("model")
    if not isinstance(response_model, str) or response_model != model:
        raise ResponseModelMismatchError(
            f"Response model mismatch: requested {model!r}, got "
            f"{response_model!r} (missing or not the exact requested model); "
            "request failed, its output is preserved as evidence only and "
            "cannot score",
            http_status=status,
            raw_text=message["content"],
            response_model=response_model if isinstance(response_model, str) else None,
        )
    return {
        "text": message["content"],
        "finish_reason": finish_reason,
        "usage": {key: value for key, value in usage.items()
                  if isinstance(value, (int, str))},
        "prompt_tokens": _usage_int("prompt_tokens"),
        "completion_tokens": _usage_int("completion_tokens"),
        "total_tokens": _usage_int("total_tokens"),
        "seconds": elapsed,
        "http_status": status,
        "response_id": body.get("id") if isinstance(body.get("id"), str) else None,
        "response_model": response_model if isinstance(response_model, str) else None,
    }


def _format_request_error(exc: BaseException) -> str:
    if isinstance(exc, RunnerRequestError):
        detail = f"{exc}"
        if exc.http_status is not None:
            detail += f" (http_status={exc.http_status})"
        if exc.body_preview:
            detail += f" (body_preview={exc.body_preview!r})"
        return f"RunnerRequestError: {detail}"
    return f"{type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True,
        help=(
            "Explicit report path inside the Git-ignored "
            "models/experiments/vlm/data/ subtree of the repository; refused "
            "if it exists or outside that subtree"
        ),
    )
    parser.add_argument(
        "--profile-manifest", type=Path, required=True,
        help=(
            "Immutable NInfer profile manifest (JSON) captured by root before "
            "the run; read locally only -- the runner never contacts "
            "model-control"
        ),
    )
    parser.add_argument(
        "--endpoint", default=ENDPOINT_DEFAULT,
        help=(
            f"Pinned NInfer loopback base URL (default {ENDPOINT_DEFAULT}): "
            f"scheme http, port {ENDPOINT_PORT}, path {ENDPOINT_PATH}, no "
            "userinfo, query, or fragment; the loopback aliases 127.0.0.1/"
            "localhost/::1 are accepted only with the same port and path"
        ),
    )
    parser.add_argument(
        "--model", default=MODEL_DEFAULT,
        help=(
            f"Pinned NInfer model ID (default {MODEL_DEFAULT}); this benchmark "
            "is the qwen3.8-27b-nvfp4 route only and every other value is "
            "refused before any endpoint request or output creation"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Completion token cap per request (counterpart of max_new_tokens)")
    parser.add_argument("--request-timeout", type=float, default=300.0,
                        help="Client-side timeout in seconds per request")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Deterministic-decoding temperature (0.0 = greedy approximation)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling probability (1.0 keeps the full vocabulary at temperature 0)")
    parser.add_argument("--gpu-sampling", choices=("on", "off"), default="on",
                        help="Sample whole-device nvidia-smi memory.used while running")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> Path:
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.model != MODEL_DEFAULT:
        raise ValueError(
            f"Model pin refused: this benchmark is fixed to the NInfer "
            f"{MODEL_DEFAULT!r} route; --model must be exactly {MODEL_DEFAULT!r} "
            f"(got {args.model!r}); refused before any endpoint request or "
            "output creation"
        )
    if not _endpoint_is_pinned_ninfer(args.endpoint):
        raise ValueError(
            f"Endpoint pin refused: must be the NInfer loopback base URL on "
            f"port {ENDPOINT_PORT} with path {ENDPOINT_PATH} and no userinfo, "
            f"query, or fragment (loopback aliases {sorted(LOOPBACK_HOSTS)} "
            f"only, same port and path; e.g. {ENDPOINT_DEFAULT!r}); got "
            f"{args.endpoint!r}"
        )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output: {output}")
    if output.is_dir():
        raise FileExistsError(f"Output path is a directory: {output}")
    try:
        output.relative_to(ALLOWED_OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            "Output must live inside the Git-ignored project data subtree "
            f"{ALLOWED_OUTPUT_ROOT.resolve()}; every other path is refused, "
            f"including /tmp and /mnt/HDD1/tmp: got {output}"
        ) from exc
    return output


def _load_profile_manifest(path: Path, requested_model: str) -> dict:
    """Load and validate the immutable NInfer profile manifest (local read).

    Root captures this manifest from the model-control app before the run.
    The runner only reads the file; it never contacts model-control. The
    manifest must identify the requested profile/model and carry the
    artifact identity, complete launch arguments, controller load
    status/duration (or unavailability reason), and GPU identity/size.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read --profile-manifest {path}: {exc}") from exc
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--profile-manifest {path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"--profile-manifest {path} must be a JSON object")

    def _required_string(key: str) -> str:
        value = manifest.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"--profile-manifest {path} is missing required string field {key!r}"
            )
        return value

    controller_profile_id = _required_string("controller_profile_id")
    model_id = _required_string("model_id")
    if controller_profile_id != requested_model or model_id != requested_model:
        raise ValueError(
            f"--profile-manifest does not match --model: manifest "
            f"controller_profile_id={controller_profile_id!r} / "
            f"model_id={model_id!r}, requested model={requested_model!r}"
        )

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError(f"--profile-manifest {path} is missing the 'artifact' object")
    artifact_path = artifact.get("path")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise ValueError("--profile-manifest 'artifact.path' must be a nonempty string")
    artifact_bytes = artifact.get("bytes")
    if not isinstance(artifact_bytes, int) or isinstance(artifact_bytes, bool) \
            or artifact_bytes <= 0:
        raise ValueError("--profile-manifest 'artifact.bytes' must be a positive integer")
    artifact_sha256 = artifact.get("sha256")
    if not isinstance(artifact_sha256, str) \
            or not re.fullmatch(r"[0-9a-fA-F]{64}", artifact_sha256):
        raise ValueError(
            "--profile-manifest 'artifact.sha256' must be a 64-hex SHA-256 digest"
        )

    profile_arguments = manifest.get("profile_arguments")
    if not isinstance(profile_arguments, list) or not profile_arguments \
            or not all(isinstance(item, str) for item in profile_arguments):
        raise ValueError(
            "--profile-manifest 'profile_arguments' must be a nonempty list of strings"
        )

    controller_load = manifest.get("controller_load")
    if not isinstance(controller_load, dict) \
            or "job_status" not in controller_load or "ready" not in controller_load:
        raise ValueError(
            "--profile-manifest 'controller_load' must be an object reporting "
            "'job_status' and 'ready'"
        )
    # Readiness gate: the managed profile must be fully loaded before the
    # runner proceeds. Stale, failed, or not-ready profiles are rejected
    # before any endpoint request or output creation.
    job_status = controller_load.get("job_status")
    if job_status != "done":
        raise ValueError(
            f"--profile-manifest 'controller_load.job_status' must be 'done' "
            f"before the runner proceeds (got {job_status!r}); "
            "stale/failed/not-ready profiles are rejected before network or "
            "output creation"
        )
    if controller_load.get("ready") is not True:
        raise ValueError(
            f"--profile-manifest 'controller_load.ready' must be true (boolean) "
            f"before the runner proceeds (got {controller_load.get('ready')!r}); "
            "stale/failed/not-ready profiles are rejected before network or "
            "output creation"
        )
    duration = controller_load.get("load_duration_seconds")
    if duration is not None and (
        isinstance(duration, bool) or not isinstance(duration, (int, float))
        or duration < 0
    ):
        raise ValueError(
            "--profile-manifest 'controller_load.load_duration_seconds' must be "
            "null or a nonnegative number"
        )
    if duration is None and not isinstance(controller_load.get("load_duration_note"), str):
        raise ValueError(
            "--profile-manifest 'controller_load' must carry "
            "load_duration_seconds or a load_duration_note explaining its "
            "unavailability"
        )

    gpu = manifest.get("gpu")
    if not isinstance(gpu, dict):
        raise ValueError(f"--profile-manifest {path} is missing the 'gpu' object")
    gpu_name = gpu.get("name")
    if not isinstance(gpu_name, str) or not gpu_name:
        raise ValueError("--profile-manifest 'gpu.name' must be a nonempty string")
    memory_total_mib = gpu.get("memory_total_mib")
    if not isinstance(memory_total_mib, int) or isinstance(memory_total_mib, bool) \
            or memory_total_mib <= 0:
        raise ValueError("--profile-manifest 'gpu.memory_total_mib' must be a positive integer")

    return manifest


def _collect_frames() -> tuple[list[Path], list[Path], list[str], int]:
    """Return (detection files, image paths, frame names, expected caption observations).

    The cohort is the exact ordered stride-50 list ``frame000000`` ...
    ``frame001950``. Every required file must exist, and any extra
    ``frame*.pkl.gz`` in the detections directory fails closed so missing or
    extra files cannot silently alter the held-out cohort.
    """
    detection_root = REPLICA / "exps" / DETECTION_SUFFIX
    detection_dir = detection_root / "detections"
    detection_files = [detection_dir / f"{name}.pkl.gz" for name in EXPECTED_FRAME_NAMES]
    missing = [str(path) for path in detection_files if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Missing required room0 detection frames (fixed ordered cohort "
            f"{EXPECTED_FRAME_NAMES[0]} .. {EXPECTED_FRAME_NAMES[-1]}): {missing}"
        )
    expected_names = set(EXPECTED_FRAME_NAMES)
    extra = sorted(
        str(path) for path in detection_dir.glob("frame*.pkl.gz")
        if path.name.removesuffix(".pkl.gz") not in expected_names
    )
    if extra:
        raise RuntimeError(
            f"Unexpected extra detection frame files fail the fixed cohort: {extra}"
        )
    image_root = detection_root / "vis"
    image_paths = [
        image_root / f"{path.name.removesuffix('.pkl.gz')}annotated_for_vlm.jpg"
        for path in detection_files
    ]
    missing_images = [str(path) for path in image_paths if not path.is_file()]
    if missing_images:
        raise RuntimeError(f"Missing annotated images: {missing_images}")
    frame_names = list(EXPECTED_FRAME_NAMES)
    expected_caption_observations = sum(
        len(pipeline_label_list(load_pickle_gzip(path).get("labels") or []))
        for path in detection_files
    )
    return detection_files, image_paths, frame_names, expected_caption_observations


def _execute_task(
    context: dict,
    *,
    task: str,
    frame: str,
    record: dict,
    system_prompt: str,
    user_prompt: str,
    image_part: dict,
    valid_ids: set[str],
    expected_id_names: dict[str, str],
    baseline_relations: list,
    baseline_captions: list,
) -> None:
    """Send one task request (relations or captions) and fill the frame record.

    On transport or structural failure the error is recorded on the frame
    record and in ``context["failures"]``; the run continues (fail closed on
    the single request, not on the whole benchmark).
    """
    is_relation = task == "relations"
    task_key = "relation_request" if is_relation else "caption_request"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [image_part]},
        {"role": "user", "content": user_prompt},
    ]
    sent_at_offset = time.perf_counter() - context["wall_started"]
    try:
        response = _chat_request(
            context["endpoint"], context["model"], messages,
            context["max_tokens"], context["temperature"], context["top_p"],
            context["timeout"],
        )
    except Exception as exc:
        error = _format_request_error(exc)
        record["task_errors"][task] = error
        context["failures"].append({"frame": frame, "task": task, "error": error})
        if isinstance(exc, ResponseModelMismatchError):
            context["model_mismatch_count"] += 1
        # A response-model mismatch or omission is a failed request: the text
        # is preserved as evidence only and can never score (every scoring
        # path below is gated on api_error being None / the strict_json block).
        record[task_key] = {
            "seconds": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "usage": None,
            "finish_reason": None,
            "raw_response": getattr(exc, "raw_text", None),
            "response_model": getattr(exc, "response_model", None),
            "sent_at_offset_seconds": sent_at_offset,
            "api_error": error,
        }
        return

    raw = response["text"]
    record[task_key] = {
        "seconds": response["seconds"],
        "prompt_tokens": response["prompt_tokens"],
        "completion_tokens": response["completion_tokens"],
        "total_tokens": response["total_tokens"],
        "usage": response["usage"],
        "finish_reason": response["finish_reason"],
        "raw_response": raw,
        "http_status": response["http_status"],
        "response_id": response["response_id"],
        "response_model": response["response_model"],
        "sent_at_offset_seconds": sent_at_offset,
        "api_error": None,
    }
    parsed = _parse_model_list(raw, relation=is_relation, prefer_json=False)
    strict_list_valid = _strict_list(raw, relation=is_relation)
    strict_json = _strict_json_list(raw, relation=is_relation, expected_ids=valid_ids)
    record[task_key]["parsed"] = parsed
    record[task_key]["strict_list_valid"] = strict_list_valid
    record[task_key]["strict_json"] = strict_json
    if is_relation:
        relation_items = [
            edge for edge in (parsed or [])
            if isinstance(edge, (list, tuple)) and len(edge) == 3
        ]
        rel_ids = [str(edge[i]).strip() for edge in relation_items for i in (0, 2)]
        record[task_key]["endpoint_valid_count"] = sum(value in valid_ids for value in rel_ids)
        record[task_key]["endpoint_count"] = len(rel_ids)
        record[task_key]["relation_types_valid"] = strict_list_valid and all(
            " ".join(str(edge[1]).lower().split()) in ALLOWED_RELATIONS
            for edge in relation_items
        )
        record["candidate_relations"] = parsed
        record["baseline_relations"] = baseline_relations
        if strict_json["json_valid"] and strict_json["schema_valid"]:
            record["relation_agreement"] = edge_metrics(parsed, baseline_relations)
        else:
            record["relation_agreement"] = None
    else:
        caption_ids = [
            str(item.get("id")).strip()
            for item in (parsed or [])
            if isinstance(item, dict) and item.get("id") is not None
        ]
        record[task_key]["id_coverage"] = {
            "candidate": len(set(caption_ids)),
            "expected": len(valid_ids),
            "matched": len(set(caption_ids) & valid_ids),
            "missing": sorted(valid_ids - set(caption_ids)),
            "extra": sorted(set(caption_ids) - valid_ids),
            "duplicates": len(caption_ids) - len(set(caption_ids)),
        }
        record["candidate_captions"] = parsed
        record["baseline_captions"] = baseline_captions
        usable_caption_ids = {
            str(item["id"]).strip() for item in (parsed or [])
            if isinstance(item, dict)
            and item.get("id") is not None
            and str(item.get("id")).strip() in valid_ids
            and isinstance(item.get("name"), str)
            and item["name"].strip() == expected_id_names[str(item["id"]).strip()]
            and isinstance(item.get("caption"), str) and item["caption"].strip()
        } if (strict_json.get("json_valid") and strict_json.get("schema_valid")) else set()
        record[task_key]["usable_caption_observation_count"] = len(usable_caption_ids)
        record["usable_caption_ids"] = sorted(usable_caption_ids)


def main() -> int:
    args = parse_args()
    output = _validate_args(args)
    manifest_path = args.profile_manifest.expanduser().resolve()
    manifest = _load_profile_manifest(manifest_path, args.model)
    manifest_controller_load = manifest["controller_load"]
    detection_files, image_paths, frame_names, expected_caption_observations = _collect_frames()
    preflight_info = _preflight_models(args.endpoint, args.model, args.request_timeout)

    started_utc = datetime.now(timezone.utc).isoformat()
    wall_started = time.perf_counter()
    samples: list[int] = []
    sampler_errors: list[str] = []
    stop_sample = threading.Event()
    sampler = None
    if args.gpu_sampling == "on":
        sampler = threading.Thread(
            target=_gpu_memory_sample, args=(samples, sampler_errors, stop_sample),
            daemon=True,
        )
        sampler.start()

    report = {
        "created_utc": started_utc,
        "status": "running",
        "candidate": {
            "id": "qwen3.8-27b-nvfp4-ninfer",
            "model": args.model,
            "backend": "NInfer OpenAI-compatible HTTP endpoint (local managed profile)",
            "ninfer_profile": {
                "id": args.model,
                "managed_by": (
                    "model-control lifecycle at 127.0.0.1:20268; root starts "
                    "the managed profile before this run. The runner never "
                    "contacts model-control: all profile details below come "
                    "from the immutable preflight manifest."
                ),
                "manifest_path": str(manifest_path),
                "manifest_captured_utc": manifest.get("captured_utc"),
                "manifest_source": manifest.get("source"),
                "controller_profile_id": manifest["controller_profile_id"],
                "model_id": manifest["model_id"],
                "artifact": {
                    "path": manifest["artifact"]["path"],
                    "bytes": manifest["artifact"]["bytes"],
                    "sha256": manifest["artifact"]["sha256"],
                },
                "profile_arguments": manifest["profile_arguments"],
                "controller_load": manifest_controller_load,
                "gpu": {
                    "name": manifest["gpu"]["name"],
                    "memory_total_mib": manifest["gpu"]["memory_total_mib"],
                },
            },
            "endpoint": args.endpoint,
            "endpoint_scope": (
                "pinned NInfer loopback base URL (loopback aliases "
                "127.0.0.1/localhost/::1, port 20301, path /v1, no "
                "userinfo/query/fragment); every other endpoint is refused "
                "before any request"
            ),
            "model_pin": MODEL_DEFAULT,
            "request_path": "/chat/completions",
            "runner": {
                "script": "benchmark_vlm_shortlist_ninfer.py",
                "version": RUNNER_VERSION,
            },
            "packages": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "http_client": "Python stdlib urllib.request",
                "openai_sdk": "not used",
            },
            "repo_root": str(REPO_ROOT),
            "decode_parameters": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "n": 1,
                "do_sample_equivalent": "temperature=0.0 greedy approximation",
            },
            "decode_note": DECODE_NOTE,
            "request_timeout_seconds": args.request_timeout,
            "retry_policy": (
                "none; no retry that changes model input; per-request failures are "
                "recorded and the run is marked incomplete"
            ),
        },
        "preflight": preflight_info,
        "dataset": {
            "scene": "Replica room0",
            "frames": EXPECTED_FRAME_COUNT,
            "frame_stride": 50,
            "detection_suffix": DETECTION_SUFFIX,
            "detection_root": str(REPLICA / "exps" / DETECTION_SUFFIX),
            "object_observations": expected_caption_observations,
        },
        "frame_order": frame_names,
        "prompt_protocol": {
            "profile": "legacy",
            "serialization": PROMPT_PROTOCOL_NOTE,
            "system_prompts": {
                "relations": system_prompt_only_top,
                "captions": system_prompt_captions,
            },
            "user_prompt_templates": {
                "relations": RELATION_USER_PROMPT_TEMPLATE,
                "captions": CAPTION_USER_PROMPT_TEMPLATE,
            },
        },
        "serialization_note": SERIALIZATION_NOTE,
        "baseline": {
            "model": "qwen3.8:27b-mtp4-ud_q4_K_XL-128k",
            "source": "cached annotations from the same 40-frame modernized detection cache",
        },
        "metric_note": METRIC_NOTE,
        "model_load_seconds": manifest_controller_load.get("load_duration_seconds"),
        "model_load_note": (
            "Controller-reported, via the preflight profile manifest: "
            f"job_status={manifest_controller_load.get('job_status')!r}, "
            f"ready={manifest_controller_load.get('ready')!r}, "
            f"finished_epoch_seconds="
            f"{manifest_controller_load.get('finished_epoch_seconds')!r}. "
            f"{manifest_controller_load.get('load_duration_note') or 'Duration reported by the controller load job.'}"
        ),
        "failures": [],
        "frames": [],
    }
    _write_json(output, report)

    context = {
        "endpoint": args.endpoint,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timeout": args.request_timeout,
        "wall_started": wall_started,
        "request_latencies": [],
        "first_request": None,
        "model_mismatch_count": 0,
        "failures": report["failures"],
    }

    fatal_error = None
    try:
        for detection_path, image_path in zip(detection_files, image_paths):
            cached = load_pickle_gzip(detection_path)
            labels = pipeline_label_list(cached.get("labels") or [])
            frame = detection_path.name.removesuffix(".pkl.gz")
            relation_prompt = RELATION_USER_PROMPT_TEMPLATE.format(labels=labels)
            caption_prompt = CAPTION_USER_PROMPT_TEMPLATE.format(labels=labels)
            record = {
                "frame": frame,
                "labels": labels,
                "user_prompts": {"relations": relation_prompt, "captions": caption_prompt},
                "system_prompts": {
                    "relations": system_prompt_only_top,
                    "captions": system_prompt_captions,
                },
                "task_errors": {},
            }
            report["frames"].append(record)
            expected_id_names = {
                item.split(":", 1)[0].strip(): item.split(":", 1)[1].strip()
                for item in labels if ":" in item
            }
            valid_ids = set(expected_id_names)
            image_part = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
                },
            }
            baseline_relations = cached.get("edges") or []
            baseline_captions = cached.get("captions") or []
            _execute_task(
                context, task="relations", frame=frame, record=record,
                system_prompt=system_prompt_only_top, user_prompt=relation_prompt,
                image_part=image_part, valid_ids=valid_ids,
                expected_id_names=expected_id_names,
                baseline_relations=baseline_relations,
                baseline_captions=baseline_captions,
            )
            _write_json(output, report)
            print(
                f"Frame {frame}: relations {record['task_errors'].get('relations', 'done')}",
                flush=True,
            )
            _execute_task(
                context, task="captions", frame=frame, record=record,
                system_prompt=system_prompt_captions, user_prompt=caption_prompt,
                image_part=image_part, valid_ids=valid_ids,
                expected_id_names=expected_id_names,
                baseline_relations=baseline_relations,
                baseline_captions=baseline_captions,
            )
            _write_json(output, report)
            print(
                f"Frame {frame}: captions {record['task_errors'].get('captions', 'done')}",
                flush=True,
            )

            # Cached-Qwen3.8 caption agreement for usable captions only.
            if (
                "caption_request" in record
                and record["caption_request"].get("api_error") is None
                and "parsed" in record["caption_request"]
            ):
                usable_ids = set(record.get("usable_caption_ids") or [])
                candidate_caption_map = caption_by_id(record["candidate_captions"])
                candidate_caption_map = {
                    key: value for key, value in candidate_caption_map.items()
                    if key in usable_ids
                }
                baseline_caption_map = caption_by_id(baseline_captions)
                shared_ids = sorted(candidate_caption_map.keys() & baseline_caption_map.keys())
                record["caption_exact_agreement"] = (
                    sum(candidate_caption_map[i] == baseline_caption_map[i] for i in shared_ids)
                    / len(shared_ids)
                    if shared_ids else None
                )
                record["caption_word_f1_mean"] = (
                    float(np.mean([
                        word_f1(candidate_caption_map[i], baseline_caption_map[i])
                        for i in shared_ids
                    ]))
                    if shared_ids else None
                )
                record["caption_agreement_scored_id_count"] = len(shared_ids)

            for task, task_key in (
                ("relations", "relation_request"),
                ("captions", "caption_request"),
            ):
                entry = record.get(task_key)
                if entry is None or entry.get("seconds") is None:
                    continue
                record.setdefault("latency_seconds", {})[task] = entry["seconds"]
                record.setdefault("token_counts", {})[task] = {
                    "prompt": entry["prompt_tokens"],
                    "output": entry["completion_tokens"],
                }
                record.setdefault("raw_responses", {})[task] = entry["raw_response"]
                context["request_latencies"].append(entry["seconds"])
                if context["first_request"] is None:
                    context["first_request"] = {
                        "frame": frame,
                        "task": task,
                        "seconds": entry["seconds"],
                        "sent_at_offset_seconds": entry["sent_at_offset_seconds"],
                        "finish_reason": entry.get("finish_reason"),
                    }
            if record["task_errors"]:
                record["error"] = "; ".join(
                    f"{task}: {error}" for task, error in record["task_errors"].items()
                )
            _write_json(output, report)

    except Exception as exc:
        fatal_error = exc
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"

    finally:
        stop_sample.set()
        if sampler is not None:
            sampler.join(timeout=7)
            if sampler.is_alive():
                report["sampler_join_timeout"] = True

        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["wall_seconds"] = time.perf_counter() - wall_started
        report["gpu_memory"] = {
            "process_scope": {
                "peak_mib": None,
                "note": (
                    "Unavailable from this runner: the NInfer process is owned by "
                    "the model-control lifecycle, not by this process."
                ),
            },
            "whole_device_scope": {
                "peak_used_mib": max(samples) if samples else None,
                "sample_count": len(samples),
                "sample_interval_seconds": 1,
                "source": (
                    "nvidia-smi --query-gpu=memory.used "
                    "(maximum across visible GPUs per sample)"
                ),
                "note": (
                    "Includes other processes and the desktop baseline; this is a "
                    "whole-device peak, not a process-only peak."
                ),
            },
            "sampler_errors": sampler_errors[:10] if sampler_errors else [],
        }

        records = report["frames"]

        def _completed(task_key: str) -> set[int]:
            return {
                id(item) for item in records
                if task_key in item
                and item[task_key].get("api_error") is None
                and item[task_key].get("raw_response") is not None
            }

        relation_ok = _completed("relation_request")
        caption_ok = _completed("caption_request")
        complete_records = [
            item for item in records if id(item) in relation_ok and id(item) in caption_ok
        ]
        relation_agreements = [
            item["relation_agreement"]["agreement_f1"]
            for item in records
            if id(item) in relation_ok and isinstance(item.get("relation_agreement"), dict)
        ]
        caption_agreements = [
            item["caption_word_f1_mean"]
            for item in records
            if id(item) in caption_ok and item.get("caption_word_f1_mean") is not None
        ]
        latencies = context["request_latencies"]
        usable_caption_count = sum(
            item.get("caption_request", {}).get("usable_caption_observation_count", 0)
            or 0
            for item in records
        )

        report["summary"] = {
            "selected_frames": EXPECTED_FRAME_COUNT,
            "records_written": len(records),
            "frame_records_missing": max(0, EXPECTED_FRAME_COUNT - len(records)),
            "relation_requests_completed": len(relation_ok),
            "caption_requests_completed": len(caption_ok),
            "complete_frame_pairs": len(complete_records),
            "relation_legacy_parse_valid_frames": sum(
                bool(item["relation_request"].get("strict_list_valid"))
                for item in records if id(item) in relation_ok
            ),
            "relation_strict_json": _aggregate_json_validation(records, "relation_request"),
            "caption_legacy_parse_valid_frames": sum(
                bool(item["caption_request"].get("strict_list_valid"))
                for item in records if id(item) in caption_ok
            ),
            "caption_strict_json": _aggregate_json_validation(records, "caption_request"),
            "usable_caption_observations": {
                "count": usable_caption_count,
                "expected": expected_caption_observations,
                "rate": (
                    usable_caption_count / expected_caption_observations
                    if expected_caption_observations else 0.0
                ),
                "definition": (
                    "Detection observation with a nonempty caption, resolvable ID "
                    "and exact cached detector name; also requires valid JSON array "
                    "schema (legacy batch mode)."
                ),
            },
            "relation_error_count": sum("relations" in item.get("task_errors", {}) for item in records),
            "caption_error_count": sum("captions" in item.get("task_errors", {}) for item in records),
            "relation_agreement_f1_mean": statistics.mean(relation_agreements) if relation_agreements else None,
            "relation_agreement_scored_frames": len(relation_agreements),
            "relation_agreement_empty_reference_frames": sum(
                not item.get("baseline_relations") for item in records
                if id(item) in relation_ok
            ),
            "caption_word_f1_mean": statistics.mean(caption_agreements) if caption_agreements else None,
            "caption_word_f1_scored_frames": len(caption_agreements),
            "caption_word_f1_scored_ids": sum(
                item.get("caption_agreement_scored_id_count", 0) for item in records
            ),
            "first_request": context["first_request"],
            "first_request_latency_seconds": (
                context["first_request"]["seconds"] if context["first_request"] else None
            ),
            "request_latency": {
                "count": len(latencies),
                "mean_seconds": statistics.fmean(latencies) if latencies else None,
                "median_seconds": statistics.median(latencies) if latencies else None,
                "p95_seconds": statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 2 else None,
                "min_seconds": min(latencies) if latencies else None,
                "max_seconds": max(latencies) if latencies else None,
                "scope": "successful chat-completion requests only, both tasks",
            },
            "effective_frames_per_second": (
                len(complete_records) / report["wall_seconds"]
                if report["wall_seconds"] > 0 else None
            ),
            "response_model_mismatch_count": context["model_mismatch_count"],
        }
        report["status"] = (
            "complete"
            if (
                fatal_error is None
                and not report["failures"]
                and len(records) == EXPECTED_FRAME_COUNT
                and len(complete_records) == EXPECTED_FRAME_COUNT
            )
            else "incomplete"
        )
        _write_json(output, report)
        print(json.dumps(
            {
                "status": report["status"],
                "summary": report["summary"],
                "wall_seconds": report["wall_seconds"],
                "gpu_memory": report["gpu_memory"],
            },
            indent=2,
        ))

    if fatal_error is not None:
        raise fatal_error
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
