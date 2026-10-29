"""Paired Replica benchmark for the edge-compatible local VLM/LLM route.

The reference outputs are the Qwen3.8 relation and caption results already
stored in the detection cache, plus consolidated captions in a saved map. The
candidate server is called with the pipeline's existing prompts. This is an
agreement and integration benchmark, not a ground-truth accuracy benchmark.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from conceptgraph.utils.llm_client import LocalLLMClient, LocalLLMError
from conceptgraph.utils.vlm import (
    encode_image_for_openai,
    extract_list_of_tuples,
    system_prompt_captions,
    system_prompt_consolidate_captions,
    system_prompt_only_top,
    vlm_extract_object_captions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLICA_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/Replica")
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".agentic_workflow/workspace/edge-compatible-benchmark/results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_REPLICA_ROOT / "room0")
    parser.add_argument("--detections-exp-suffix", default="s_detections_stride50")
    parser.add_argument("--mapping-exp-suffix", default="r_mapping_stride50")
    parser.add_argument("--base-url", default=os.getenv("EDGE_COMPATIBLE_BASE_URL", "http://127.0.0.1:20262/v1"))
    parser.add_argument("--model", default=os.getenv("EDGE_COMPATIBLE_MODEL_ALIAS", "edge-compatible"))
    parser.add_argument(
        "--candidate-id",
        default=os.getenv("EDGE_COMPATIBLE_CANDIDATE_ID", "candidate"),
        help="label for this downloaded candidate; used to keep result files separate",
    )
    parser.add_argument("--baseline-model", default="qwen3.8:27b-mtp4-ud_q4_K_XL-128k")
    parser.add_argument("--frames", type=int, default=5, help="evenly sampled cached Replica frames")
    parser.add_argument("--objects", type=int, default=5, help="saved objects with multiple captions")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-vlm-tokens", type=int, default=4096)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_pickle_gzip(path: Path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def dump_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def pipeline_label_list(labels: list[str]) -> list[str]:
    formatted = []
    for label in labels:
        parts = str(label).rsplit(" ", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        formatted.append(f"{parts[1]}: {parts[0]}")
    return formatted


def _call(client: LocalLLMClient, messages: list[dict], max_tokens: int, json_mode: bool = False) -> dict:
    started = time.perf_counter()
    kwargs = {"max_tokens": max_tokens}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        model=client.model, messages=messages, **kwargs
    )
    elapsed = time.perf_counter() - started
    usage = getattr(response, "usage", {}) or {}
    return {
        "text": response.choices[0].message.content or "",
        "seconds": elapsed,
        "usage": usage,
    }


def run_frame_requests(client: LocalLLMClient, image_path: Path, labels: list[str], max_tokens: int) -> dict:
    image_data = encode_image_for_openai(str(image_path))
    image_part = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
    }
    label_prompt = (
        f"Here is the list of labels for the annotations of the objects in the image: "
        f"{labels}. Please describe the spatial relationships between the objects in the image."
    )
    relation = _call(
        client,
        [
            {"role": "system", "content": system_prompt_only_top},
            {"role": "user", "content": [image_part]},
            {"role": "user", "content": label_prompt},
        ],
        max_tokens,
    )
    caption_prompt = (
        f"Here is the list of labels for the annotations of the objects in the image: "
        f"{labels}. Please accurately caption the objects in the image."
    )
    captions = _call(
        client,
        [
            {"role": "system", "content": system_prompt_captions},
            {"role": "user", "content": [image_part]},
            {"role": "user", "content": caption_prompt},
        ],
        max_tokens,
    )
    relation["parsed"] = extract_list_of_tuples(relation["text"])
    captions["parsed"] = vlm_extract_object_captions(captions["text"])
    return {"relations": relation, "captions": captions}


def run_consolidation(client: LocalLLMClient, captions: list[dict], max_tokens: int) -> dict:
    caption_text = "\n".join(
        str(item["caption"]) for item in captions
        if isinstance(item, dict) and item.get("caption") is not None
    )
    query = (
        "Here are several captions for the same object:\n"
        f"{caption_text}\n\n"
        "Please consolidate these into a single, clear caption that accurately describes the object."
    )
    result = _call(
        client,
        [
            {"role": "system", "content": system_prompt_consolidate_captions},
            {"role": "user", "content": query},
        ],
        max_tokens,
        json_mode=True,
    )
    try:
        parsed = json.loads(result["text"])
        if isinstance(parsed, dict):
            result["parsed"] = (
                parsed.get("consolidated_caption")
                or parsed.get("caption")
                or next((v for v in parsed.values() if isinstance(v, str)), "")
            )
        else:
            result["parsed"] = ""
    except (TypeError, ValueError):
        result["parsed"] = ""
    return result


def normalize_edges(edges) -> set[tuple[str, str, str]]:
    output = set()
    for edge in edges or []:
        if isinstance(edge, (list, tuple)) and len(edge) == 3:
            output.add(tuple(" ".join(str(value).lower().split()) for value in edge))
    return output


def caption_by_id(captions) -> dict[str, str]:
    output = {}
    for item in captions or []:
        if isinstance(item, dict) and item.get("id") is not None and item.get("caption"):
            output[str(item["id"])] = " ".join(str(item["caption"]).lower().split())
    return output


def word_f1(left: str, right: str) -> float:
    left_words, right_words = Counter(re.findall(r"[\w']+", left.lower())), Counter(re.findall(r"[\w']+", right.lower()))
    if not left_words or not right_words:
        return float(left_words == right_words)
    overlap = sum((left_words & right_words).values())
    precision = overlap / sum(left_words.values())
    recall = overlap / sum(right_words.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def edge_metrics(candidate, baseline) -> dict:
    candidate_set, baseline_set = normalize_edges(candidate), normalize_edges(baseline)
    true_positive = len(candidate_set & baseline_set)
    precision = true_positive / len(candidate_set) if candidate_set else float(not baseline_set)
    recall = true_positive / len(baseline_set) if baseline_set else float(not candidate_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "candidate_count": len(candidate_set),
        "baseline_count": len(baseline_set),
        "agreement_precision": precision,
        "agreement_recall": recall,
        "agreement_f1": f1,
    }


def select_evenly(items: list[Path], count: int) -> list[Path]:
    if count <= 0:
        return []
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, num=count, dtype=int)
    return [items[index] for index in indices]


def main() -> int:
    args = parse_args()
    scene_root = args.scene_root.expanduser().resolve()
    det_root = scene_root / "exps" / args.detections_exp_suffix
    detection_dir = det_root / "detections"
    vis_dir = det_root / "vis"
    map_file = scene_root / "exps" / args.mapping_exp_suffix / f"pcd_{args.mapping_exp_suffix}.pkl.gz"
    detection_files = sorted(detection_dir.glob("*.pkl.gz"))
    if not detection_files:
        print(f"No cached detection records found in {detection_dir}", file=sys.stderr)
        return 2
    if not map_file.is_file():
        print(f"Saved baseline map not found: {map_file}", file=sys.stderr)
        return 2

    client = LocalLLMClient(
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_vlm_tokens,
        reasoning_effort=None,
        json_mode=False,
        env_prefix="EDGE_COMPATIBLE",
    )
    try:
        available_models = client.list_models()
    except LocalLLMError as exc:
        print(f"Cannot reach candidate server: {exc}", file=sys.stderr)
        return 2
    if available_models and args.model not in available_models:
        print(f"Server does not expose model {args.model!r}; available: {available_models}", file=sys.stderr)
        return 2

    output_path = args.output or (
        DEFAULT_OUTPUT_DIR / f"{scene_root.name}-{args.detections_exp_suffix}-{args.candidate_id}.json"
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene": str(scene_root),
        "candidate": {"id": args.candidate_id, "model": args.model, "base_url": args.base_url},
        "baseline": {
            "model": args.baseline_model,
            "source": "saved Qwen3.8 detections and map outputs from this Replica run",
        },
        "metric_note": "Agreement with the cached Qwen3.8 outputs is not ground-truth accuracy.",
        "frames": [],
        "consolidations": [],
    }

    for detection_path in select_evenly(detection_files, args.frames):
        cached = load_pickle_gzip(detection_path)
        image_path = vis_dir / f"{detection_path.name.removesuffix('.pkl.gz')}annotated_for_vlm.jpg"
        if not image_path.is_file():
            report["frames"].append({"frame": detection_path.stem, "error": f"missing {image_path}"})
            continue
        raw_labels = cached.get("labels") or []
        labels = pipeline_label_list(raw_labels)
        record = {"frame": detection_path.stem, "labels": labels}
        try:
            candidate = run_frame_requests(client, image_path, labels, args.max_vlm_tokens)
            baseline_edges = cached.get("edges") or []
            baseline_captions = cached.get("captions") or []
            candidate_caption_map = caption_by_id(candidate["captions"]["parsed"])
            baseline_caption_map = caption_by_id(baseline_captions)
            shared_ids = sorted(candidate_caption_map.keys() & baseline_caption_map.keys())
            record.update(
                {
                    "candidate_relations": candidate["relations"]["parsed"],
                    "baseline_relations": baseline_edges,
                    "relation_agreement": edge_metrics(candidate["relations"]["parsed"], baseline_edges),
                    "candidate_captions": candidate["captions"]["parsed"],
                    "baseline_captions": baseline_captions,
                    "caption_id_coverage": {
                        "candidate": len(candidate_caption_map),
                        "baseline": len(baseline_caption_map),
                        "shared": len(shared_ids),
                    },
                    "caption_exact_agreement": (
                        sum(candidate_caption_map[i] == baseline_caption_map[i] for i in shared_ids) / len(shared_ids)
                        if shared_ids else None
                    ),
                    "caption_word_f1_mean": (
                        float(np.mean([word_f1(candidate_caption_map[i], baseline_caption_map[i]) for i in shared_ids]))
                        if shared_ids else None
                    ),
                    "latency_seconds": {
                        "relations": candidate["relations"]["seconds"],
                        "captions": candidate["captions"]["seconds"],
                    },
                    "token_usage": {
                        "relations": candidate["relations"]["usage"],
                        "captions": candidate["captions"]["usage"],
                    },
                    "raw_responses": {
                        "relations": candidate["relations"]["text"],
                        "captions": candidate["captions"]["text"],
                    },
                }
            )
        except Exception as exc:  # preserve failure evidence per selected frame
            record["error"] = f"{type(exc).__name__}: {exc}"
        report["frames"].append(record)
        dump_json_atomic(output_path, report)
        print(f"Frame {detection_path.stem}: {record.get('error', 'done')}", flush=True)

    baseline_map = load_pickle_gzip(map_file)
    consolidatable = []
    for index, obj in enumerate(baseline_map.get("objects", [])):
        captions = [
            item for item in (obj.get("captions") or [])
            if isinstance(item, dict) and item.get("caption")
        ]
        baseline_caption = obj.get("consolidated_caption")
        if len(captions) >= 2 and isinstance(baseline_caption, str) and baseline_caption.strip():
            consolidatable.append((index, obj, captions, baseline_caption))
    for index, obj, captions, baseline_caption in consolidatable[: args.objects]:
        record = {"object_index": index, "class_name": obj.get("class_name", "")}
        try:
            result = run_consolidation(client, captions[:20], args.max_text_tokens)
            candidate_caption = result["parsed"]
            record.update(
                {
                    "candidate_caption": candidate_caption,
                    "baseline_caption": baseline_caption,
                    "word_f1": word_f1(candidate_caption, baseline_caption),
                    "exact_agreement": " ".join(candidate_caption.lower().split())
                    == " ".join(baseline_caption.lower().split()),
                    "seconds": result["seconds"],
                    "usage": result["usage"],
                    "raw_response": result["text"],
                    "valid_output": bool(candidate_caption),
                }
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        report["consolidations"].append(record)
        dump_json_atomic(output_path, report)
        print(f"Consolidation {index}: {record.get('error', 'done')}", flush=True)

    successful_frames = [item for item in report["frames"] if "relation_agreement" in item]
    report["summary"] = {
        "selected_frames": len(report["frames"]),
        "successful_frames": len(successful_frames),
        "relation_agreement_f1_mean": (
            float(np.mean([item["relation_agreement"]["agreement_f1"] for item in successful_frames]))
            if successful_frames else None
        ),
        "caption_word_f1_mean": (
            float(np.mean([item["caption_word_f1_mean"] for item in successful_frames if item["caption_word_f1_mean"] is not None]))
            if any(item["caption_word_f1_mean"] is not None for item in successful_frames) else None
        ),
        "consolidations_completed": sum(bool(item.get("valid_output")) for item in report["consolidations"]),
        "consolidations_selected": len(report["consolidations"]),
    }
    dump_json_atomic(output_path, report)
    print(f"Benchmark written to {output_path}")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["successful_frames"] or report["consolidations"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
