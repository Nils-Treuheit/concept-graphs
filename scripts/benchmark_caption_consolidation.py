"""Compare one edge-compatible GGUF model on a paired set of caption groups.

Start the selected CUDA llama.cpp server with
``scripts/edge-compatible/serve_model.sh`` first. This script makes only text
consolidation requests to that server; it does not load or unload a model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
import time
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conceptgraph.utils.llm_client import LocalLLMClient, LocalLLMError
from conceptgraph.utils.vlm import consolidate_captions


EDGE_CANDIDATES = {
    "qwen35-9b-q4": "Qwen3.5 9B UD-Q4_K_XL",
    "qwen35-9b-q8": "Qwen3.5 9B UD-Q8_K_XL",
    "gemma4-12b": "Gemma 4 12B QAT UD-Q4_K_XL",
}
DEFAULT_BASE_URL = "http://127.0.0.1:20262/v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".agentic_workflow/workspace/caption-consolidation-benchmark"


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as map_file:
        for chunk in iter(lambda: map_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_groups(map_path: Path, object_indices: list[int]) -> list[dict]:
    with gzip.open(map_path, "rb") as map_file:
        results = pickle.load(map_file)
    objects = results.get("objects") if isinstance(results, dict) else None
    if not isinstance(objects, list):
        raise ValueError(f"Map {map_path} does not contain an objects list")

    groups = []
    for object_index in object_indices:
        if object_index >= len(objects):
            raise ValueError(
                f"Map has {len(objects)} objects; expected index {object_index}"
            )
        obj = objects[object_index]
        captions = [
            row
            for row in (obj.get("captions") or [])
            if isinstance(row, dict) and row.get("caption") is not None
        ][:20]
        if not any(
            isinstance(row.get("caption"), str) and row["caption"].strip()
            for row in captions
        ):
            raise ValueError(f"Object {object_index} has no nonempty source captions")
        groups.append(
            {
                "object_index": object_index,
                "object_id": obj.get("curr_obj_num", object_index + 1),
                "class_name": obj.get("class_name", ""),
                "source_captions": [row["caption"] for row in captions],
            }
        )
    return groups


def _word_f1(left: str, right: str) -> float:
    left_words = Counter(re.findall(r"[\w']+", left.lower()))
    right_words = Counter(re.findall(r"[\w']+", right.lower()))
    if not left_words or not right_words:
        return float(left_words == right_words)
    overlap = sum((left_words & right_words).values())
    precision = overlap / sum(left_words.values())
    recall = overlap / sum(right_words.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _load_baseline(
    path: Path, map_path: Path, source_groups: list[dict]
) -> tuple[dict, dict[int, dict]]:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if baseline.get("map_sha256") != _file_sha256(map_path):
        raise ValueError("Qwen baseline results were produced from a different map")
    baseline_rows = baseline.get("groups", [])
    baseline_groups = {int(row["object_index"]): row for row in baseline_rows}
    if len(baseline_groups) != len(baseline_rows) or not baseline_groups:
        raise ValueError("Qwen baseline must contain unique caption groups")
    if set(baseline_groups) != {group["object_index"] for group in source_groups}:
        raise ValueError("Qwen baseline groups differ from the selected source objects")
    for group in source_groups:
        baseline_group = baseline_groups[group["object_index"]]
        if baseline_group.get("source_captions") != group["source_captions"]:
            raise ValueError(
                f"Qwen baseline captions differ for object {group['object_index']}"
            )
        if not str(baseline_group.get("caption", "")).strip():
            raise ValueError(
                f"Qwen baseline output is empty for object {group['object_index']}"
            )
    return baseline, baseline_groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path, help="Real room0 stride-50 map")
    parser.add_argument(
        "--baseline-results",
        required=True,
        type=Path,
        help="Qwen baseline results for the selected source groups",
    )
    parser.add_argument("--candidate-id", required=True, choices=EDGE_CANDIDATES)
    parser.add_argument(
        "--base-url",
        default=os.getenv("EDGE_COMPATIBLE_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--server-reasoning-mode",
        choices=("off", "on", "auto"),
        default="off",
        help="Reasoning mode used when starting the edge-compatible server",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("--max-tokens and --timeout must be positive")
    if not args.map.is_file() or not args.baseline_results.is_file():
        parser.error("--map and --baseline-results must both exist")
    parsed_base_url = urlparse(args.base_url)
    if parsed_base_url.scheme != "http" or parsed_base_url.hostname not in {
        "127.0.0.1", "localhost", "::1",
    }:
        parser.error("--base-url must use the local loopback model server")
    model_alias = f"edge-compatible-{args.candidate_id}"

    output_path = args.output or (
        DEFAULT_OUTPUT_DIR / f"room0-{args.candidate_id}-caption-consolidation.json"
    )
    protected_inputs = (args.map.resolve(), args.baseline_results.resolve())
    if output_path.resolve() in protected_inputs or (
        output_path.exists()
        and any(os.path.samefile(output_path, path) for path in (args.map, args.baseline_results))
    ):
        parser.error("--output must not overwrite the source map or Qwen baseline")

    try:
        baseline_preview = json.loads(args.baseline_results.read_text(encoding="utf-8"))
        selected_indices = [int(row["object_index"]) for row in baseline_preview["groups"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"Invalid Qwen baseline group list: {exc}")
    source_groups = _load_source_groups(args.map, selected_indices)
    baseline, baseline_groups = _load_baseline(
        args.baseline_results, args.map, source_groups
    )
    client = LocalLLMClient(
        base_url=args.base_url,
        model=model_alias,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        reasoning_effort=None,
        json_mode=False,
        env_prefix="EDGE_COMPATIBLE",
    )
    try:
        available_models = client.list_models()
    except LocalLLMError as exc:
        parser.error(f"Cannot reach edge-compatible server at {args.base_url}: {exc}")
    if available_models != [model_alias]:
        parser.error(
            f"Expected only candidate alias {model_alias!r} on the server; "
            f"found {available_models}. Start the matching edge-compatible model."
        )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "map_path": str(args.map.resolve()),
        "map_sha256": _file_sha256(args.map),
        "baseline": {
            "model": baseline.get("model", "configured Qwen"),
            "source": str(args.baseline_results.resolve()),
            "provenance": baseline.get("baseline_provenance"),
        },
        "candidate": {
            "id": args.candidate_id,
            "name": EDGE_CANDIDATES[args.candidate_id],
            "model_alias": model_alias,
            "base_url": args.base_url,
            "max_tokens": args.max_tokens,
            "reasoning_effort": None,
            "server_reasoning_mode": args.server_reasoning_mode,
        },
        "metric_note": (
        f"Paired {len(source_groups)}-group scene pilot. Word F1 and exact match measure "
            "agreement with Qwen, not ground-truth accuracy. Single-caption groups "
            "measure output behavior but do not exercise multi-caption merging."
        ),
        "sample_summary": {
            "case_count": len(source_groups),
            "multi_caption_groups": sum(
                len(group["source_captions"]) > 1 for group in source_groups
            ),
            "single_caption_groups": sum(
                len(group["source_captions"]) == 1 for group in source_groups
            ),
        },
        "groups": [],
    }
    _atomic_write_json(output_path, report)
    for group in source_groups:
        baseline_caption = str(baseline_groups[group["object_index"]]["caption"])
        record = {
            **group,
            "baseline_caption": baseline_caption,
            "status": "running",
        }
        report["groups"].append(record)
        _atomic_write_json(output_path, report)
        started = time.perf_counter()
        try:
            caption = consolidate_captions(
                client,
                [{"caption": text} for text in group["source_captions"]],
            )
            record.update(
                {
                    "candidate_caption": caption,
                    "response_seconds": round(time.perf_counter() - started, 3),
                    "valid_output": bool(caption.strip()) and len(caption.strip()) <= 500,
                    "word_f1_to_qwen": _word_f1(caption, baseline_caption),
                    "exact_agreement_to_qwen": " ".join(caption.lower().split())
                    == " ".join(baseline_caption.lower().split()),
                    "quality_review": {
                        "faithful_to_source_captions": None,
                        "unsupported_claims": None,
                        "concise": None,
                        "notes": "",
                    },
                    "status": "complete",
                }
            )
        except Exception as exc:
            record.update(
                {
                    "response_seconds": round(time.perf_counter() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": "failed",
                }
            )
        _atomic_write_json(output_path, report)

    report["summary"] = {
        "selected_groups": len(report["groups"]),
        "valid_outputs": sum(bool(row.get("valid_output")) for row in report["groups"]),
        "mean_response_seconds": (
            sum(row["response_seconds"] for row in report["groups"] if "response_seconds" in row)
            / len([row for row in report["groups"] if "response_seconds" in row])
            if any("response_seconds" in row for row in report["groups"])
            else None
        ),
        "mean_word_f1_to_qwen": (
            sum(row["word_f1_to_qwen"] for row in report["groups"] if "word_f1_to_qwen" in row)
            / len([row for row in report["groups"] if "word_f1_to_qwen" in row])
            if any("word_f1_to_qwen" in row for row in report["groups"])
            else None
        ),
    }
    _atomic_write_json(output_path, report)
    print(f"Caption-consolidation comparison saved to {output_path}")
    print(json.dumps(report["summary"], indent=2))
    return (
        0
        if report["summary"]["valid_outputs"] == report["summary"]["selected_groups"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
