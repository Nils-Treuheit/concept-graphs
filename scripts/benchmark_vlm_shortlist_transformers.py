"""Run one cached Hugging Face VLM on the fixed Replica room0 set.

The benchmark supports the original ConceptGraphs prompts, a LLaVA model-card
single-turn structured-output profile, and targeted object captions with
deterministic cached-ID stitching. It records raw outputs, parsing and ID
checks, agreement with saved Qwen3.8 annotations, latency, and GPU memory.
Agreement is not ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


REPLICA = Path("/mnt/HDD1/Project_Data/public_datasets/Replica/room0")
DETECTION_SUFFIX = "s_modernized_yoloe26l_stride50"
MAPPING_SUFFIX = "r_modernized_yoloe26l_stride50"
HUB = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface/hub")
MODEL_PATHS = {
    "llava-onevision-qwen2-7b": (
        HUB / "models--llava-hf--llava-onevision-qwen2-7b-ov-hf/snapshots/0d50680527681998e456c7b78950205bedd8a068",
        "float16",
    ),
    "qwen2.5-vl-7b": (
        HUB / "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "bfloat16",
    ),
}
ALLOWED_RELATIONS = {"on top of", "under"}
LLAVA_STRUCTURED_RELATION_TEMPLATE = """Image contains numbered, outlined detections. Listed IDs are the only allowed endpoints. Detection labels: {labels}
Task: report visible physical support relations only. Return exactly one JSON array and nothing else: no Markdown, prose, headings, or code fences. Each item must be ["subject_id","relation","object_id"]. The relation string must be exactly "on top of" or "under". Copy IDs exactly. Use [] when no confident relation is visible. Example: [["3","on top of","8"]]."""
LLAVA_STRUCTURED_CAPTION_TEMPLATE = """Image contains numbered, outlined detections. Listed IDs are the only allowed objects. Detection labels: {labels}
Task: caption each listed visible object separately. Do not write a scene summary. Preserve each detector name as the name field; use a concise visual description for caption. Include every listed ID exactly once and no other IDs. Return exactly one JSON array and nothing else: no Markdown, prose, headings, or code fences. Each item must be {{"id":"ID","name":"detector name","caption":"visual description"}}. Example: [{{"id":"3","name":"chair","caption":"A wooden chair beside the table."}}]."""
LLAVA_TARGETED_CAPTION_TEMPLATE = """This image has outlined detections with numeric IDs. Focus only on the object outlined with ID {target_id}, whose detector label is {target_name}. Other outlined objects are context only. If the target is identifiable, give one concise factual sentence about its visible appearance and condition. Mention an action only if it clearly applies to this object. Do not describe the whole room or other objects. If the detector label seems wrong, describe the visible object faithfully. If the target cannot be identified clearly, respond with exactly NOT_VISIBLE and do not guess. Return plain text only, with no ID, JSON, Markdown, or explanation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_PATHS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument(
        "--prompt-profile",
        choices=("legacy", "llava-structured"),
        default="legacy",
        help="Use the original shared prompt, or the LLaVA-only single-turn JSON retry.",
    )
    parser.add_argument(
        "--caption-profile",
        choices=("batch-json", "targeted-object"),
        default="batch-json",
        help="Caption all detections in one JSON response or query and stitch one response per object ID.",
    )
    parser.add_argument("--targeted-max-new-tokens", type=int, default=256)
    parser.add_argument("--targeted-max-wall-seconds", type=float, default=3600)
    return parser.parse_args()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _gpu_memory_sample(samples: list[int], stop: threading.Event) -> None:
    while not stop.is_set():
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            try:
                samples.append(max(int(row.strip()) for row in result.stdout.splitlines()))
            except ValueError:
                pass
        stop.wait(1)


def _chat_inputs(
    processor,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    *,
    single_user_turn: bool = False,
) -> dict:
    if single_user_turn:
        # Follow the model-card pure-Transformers example: a single user turn,
        # task text first, then the image placeholder, with no system role.
        content = [{"type": "text", "text": user_prompt}, {"type": "image"}]
        messages = [
            {"role": "user", "content": content},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image"}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return inputs, text


def _generate(model, processor, image_path: Path, system_prompt: str, user_prompt: str,
              max_new_tokens: int, timeout: float, *, single_user_turn: bool = False) -> dict:
    image = Image.open(image_path).convert("RGB")
    inputs, serialized_prompt = _chat_inputs(
        processor, image, system_prompt, user_prompt,
        single_user_turn=single_user_turn,
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    torch = __import__("torch")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    text = processor.batch_decode(
        generated[:, prompt_tokens:], skip_special_tokens=True
    )[0].strip()
    return {"text": text, "serialized_prompt": serialized_prompt,
            "seconds": elapsed, "prompt_tokens": prompt_tokens,
            "output_tokens": int(generated.shape[-1] - prompt_tokens),
            "posthoc_timeout_exceeded": elapsed > timeout}


def _strict_list(raw: str, *, relation: bool) -> bool:
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
    """Parse contracted JSON arrays without the legacy first-bracket regex."""
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
    if relation:
        return extract_list_of_tuples(raw)
    return vlm_extract_object_captions(raw)


def _strict_json_list(raw: str, *, relation: bool, expected_ids: set[str]) -> dict:
    """Validate a JSON-only response and the frame-level ID contract."""
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
    entries = [record[task] for record in records if task in record]
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


def main() -> int:
    args = parse_args()
    if args.targeted_max_new_tokens <= 0 or args.targeted_max_wall_seconds <= 0:
        raise ValueError("targeted token and wall-time limits must be positive")
    model_path, dtype_name = MODEL_PATHS[args.model]
    if not model_path.is_dir():
        raise FileNotFoundError(f"Cached model snapshot is missing: {model_path}")

    detection_root = REPLICA / "exps" / DETECTION_SUFFIX
    detection_files = sorted((detection_root / "detections").glob("frame*.pkl.gz"))
    if len(detection_files) != 40:
        raise RuntimeError(f"Expected exactly 40 saved room0 frames, found {len(detection_files)}")
    expected_caption_observations = sum(
        len(pipeline_label_list(load_pickle_gzip(path).get("labels") or []))
        for path in detection_files
    )
    image_root = detection_root / "vis"
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output: {output}")
    target_stream_path = output.with_suffix(output.suffix + ".targets.jsonl")
    if args.caption_profile == "targeted-object" and target_stream_path.exists():
        raise FileExistsError(f"Refusing to overwrite targeted-call stream: {target_stream_path}")

    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark has no CPU fallback")
    if args.model.startswith("llava"):
        model_class = LlavaOnevisionForConditionalGeneration
    else:
        model_class = Qwen2_5_VLForConditionalGeneration
    dtype = getattr(torch, dtype_name)

    started_utc = datetime.now(timezone.utc).isoformat()
    wall_started = time.perf_counter()
    structured = args.prompt_profile == "llava-structured"
    if structured and not args.model.startswith("llava"):
        raise ValueError("llava-structured is only valid with the LLaVA candidate")
    if args.caption_profile == "targeted-object" and not args.model.startswith("llava"):
        raise ValueError("targeted-object is only valid with the LLaVA candidate")
    if args.caption_profile == "targeted-object" and not structured:
        raise ValueError("targeted-object requires --prompt-profile llava-structured")
    report_system_prompts = (
        {"relations": "", "captions": ""}
        if structured else
        {"relations": system_prompt_only_top, "captions": system_prompt_captions}
    )
    prompt_templates = (
        {
            "relations": LLAVA_STRUCTURED_RELATION_TEMPLATE,
            "captions": LLAVA_STRUCTURED_CAPTION_TEMPLATE,
            "targeted_captions": LLAVA_TARGETED_CAPTION_TEMPLATE,
        }
        if structured else
        {
            "relations": "Here is the list of labels for the annotations of the objects in the image: {labels}. Please describe the spatial relationships between the objects in the image.",
            "captions": "Here is the list of labels for the annotations of the objects in the image: {labels}. Please accurately caption the objects in the image.",
        }
    )

    report = {
        "created_utc": started_utc,
        "candidate": {
            "id": args.model,
            "model": str(model_path),
            "revision": model_path.name,
            "backend": "Transformers local-files-only",
            "dtype": dtype_name,
            "device": "cuda:0",
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "request_timeout_seconds_posthoc_only": args.request_timeout,
            "request_timeout_enforced_during_generation": False,
        },
        "dataset": {"scene": "Replica room0", "frames": 40,
                    "frame_stride": 50, "detection_suffix": DETECTION_SUFFIX,
                    "object_observations": expected_caption_observations},
        "prompt_source": (
            "LLaVA model-card format: one user turn with task text before image, no system role; explicit JSON schema and example"
            if structured else
            "conceptgraph.utils.vlm system_prompt_only_top/system_prompt_captions; same labels and annotated_for_vlm.jpg cache; system then image-only user turn then text-only user turn"
        ),
        "prompt_profile": args.prompt_profile,
        "caption_profile": args.caption_profile,
        "targeted_caption_max_new_tokens": (
            args.targeted_max_new_tokens if args.caption_profile == "targeted-object" else None
        ),
        "targeted_max_wall_seconds_between_calls": (
            args.targeted_max_wall_seconds if args.caption_profile == "targeted-object" else None
        ),
        "targeted_stream_path": (
            str(target_stream_path) if args.caption_profile == "targeted-object" else None
        ),
        "caption_id_provenance": (
            "IDs and detector names are attached by the harness from the frozen detection cache; LLaVA returns description text only"
            if args.caption_profile == "targeted-object" else
            "IDs and names are required in the model response"
        ),
        "system_prompts": report_system_prompts,
        "user_prompt_templates": prompt_templates,
        "baseline": {"model": "qwen3.8:27b-mtp4-ud_q4_K_XL-128k",
                     "source": "cached annotations from the same 40-frame modernized detection cache"},
        "metric_note": "Agreement with cached Qwen3.8 outputs is not ground-truth accuracy.",
        "model_load_seconds": None,
        "frames": [],
    }
    _write_json(output, report)
    target_stream = target_stream_path.open("x", encoding="utf-8") if args.caption_profile == "targeted-object" else None

    samples: list[int] = []
    stop_sample = threading.Event()
    sampler = threading.Thread(target=_gpu_memory_sample, args=(samples, stop_sample), daemon=True)
    sampler.start()
    processor = None
    model = None
    load_started = time.perf_counter()
    fatal_error = None
    try:
        torch.cuda.reset_peak_memory_stats()
        processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="cuda:0",
            local_files_only=True,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        ).eval()
        torch.cuda.synchronize()
        report["model_load_seconds"] = time.perf_counter() - load_started
        _write_json(output, report)
        for detection_path in detection_files:
            cached = load_pickle_gzip(detection_path)
            image_path = image_root / f"{detection_path.name.removesuffix('.pkl.gz')}annotated_for_vlm.jpg"
            labels = pipeline_label_list(cached.get("labels") or [])
            record = {"frame": detection_path.name.removesuffix(".pkl.gz"), "labels": labels}
            if not image_path.is_file():
                record["error"] = f"missing annotated image {image_path}"
                report["frames"].append(record)
                _write_json(output, report)
                continue
            if args.prompt_profile == "llava-structured":
                if not args.model.startswith("llava"):
                    raise ValueError("llava-structured is only valid with the LLaVA candidate")
                llava_system_prompt = ""
                relation_prompt = LLAVA_STRUCTURED_RELATION_TEMPLATE.format(labels=labels)
                caption_prompt = LLAVA_STRUCTURED_CAPTION_TEMPLATE.format(labels=labels)
                single_user_turn = True
            else:
                relation_prompt = (
                    f"Here is the list of labels for the annotations of the objects in the image: "
                    f"{labels}. Please describe the spatial relationships between the objects in the image."
                )
                caption_prompt = (
                    f"Here is the list of labels for the annotations of the objects in the image: "
                    f"{labels}. Please accurately caption the objects in the image."
                )
                single_user_turn = False
            relation_system_prompt = "" if single_user_turn else system_prompt_only_top
            caption_system_prompt = "" if single_user_turn else system_prompt_captions
            record["user_prompts"] = {"relations": relation_prompt, "captions": caption_prompt}
            record["system_prompts"] = {
                "relations": relation_system_prompt,
                "captions": caption_system_prompt,
            }
            record["task_errors"] = {}
            expected_id_names = {
                item.split(":", 1)[0].strip(): item.split(":", 1)[1].strip()
                for item in labels if ":" in item
            }
            valid_ids = set(expected_id_names)
            relation = None
            captions = None
            usable_caption_ids: set[str] = set()
            try:
                relation = _generate(
                    model, processor, image_path,
                    relation_system_prompt,
                    relation_prompt, args.max_new_tokens, args.request_timeout,
                    single_user_turn=single_user_turn,
                )
                record["relation_request"] = {
                    "seconds": relation["seconds"],
                    "prompt_tokens": relation["prompt_tokens"],
                    "output_tokens": relation["output_tokens"],
                    "raw_response": relation["text"],
                    "serialized_prompt": relation["serialized_prompt"],
                    "strict_list_valid": _strict_list(relation["text"], relation=True),
                    "strict_json": _strict_json_list(
                        relation["text"], relation=True, expected_ids=valid_ids,
                    ),
                }
                relation["parsed"] = _parse_model_list(
                    relation["text"], relation=True,
                    prefer_json=args.prompt_profile == "llava-structured",
                )
                relation["strict_list_valid"] = _strict_list(relation["text"], relation=True)
                relation["strict_json"] = _strict_json_list(
                    relation["text"], relation=True, expected_ids=valid_ids,
                )
                relation_items = [
                    edge for edge in (relation["parsed"] or [])
                    if isinstance(edge, (list, tuple)) and len(edge) == 3
                ]
                rel_ids = [str(edge[i]).strip() for edge in relation_items for i in (0, 2)]
                relation.update({
                    "endpoint_valid_count": sum(value in valid_ids for value in rel_ids),
                    "endpoint_count": len(rel_ids),
                    "relation_types_valid": relation["strict_list_valid"] and all(
                        " ".join(str(edge[1]).lower().split()) in ALLOWED_RELATIONS
                        for edge in relation_items
                    ),
                })
                record["relation_request"].update({
                    "seconds": relation["seconds"],
                    "prompt_tokens": relation["prompt_tokens"],
                    "output_tokens": relation["output_tokens"],
                    "raw_response": relation["text"],
                    "parsed": relation["parsed"],
                    "strict_list_valid": relation["strict_list_valid"],
                    "strict_json": relation["strict_json"],
                    "endpoint_valid_count": relation["endpoint_valid_count"],
                    "endpoint_count": relation["endpoint_count"],
                    "relation_types_valid": relation["relation_types_valid"],
                })
                record["candidate_relations"] = relation["parsed"]
                record["baseline_relations"] = cached.get("edges") or []
                if relation["strict_json"]["json_valid"] and relation["strict_json"]["schema_valid"]:
                    record["relation_agreement"] = edge_metrics(
                        relation["parsed"], record["baseline_relations"],
                    )
                else:
                    record["relation_agreement"] = None
            except Exception as exc:
                record["task_errors"]["relations"] = f"{type(exc).__name__}: {exc}"

            try:
                if args.caption_profile == "targeted-object":
                    target_captions = []
                    target_requests = []
                    target_budget_hit = False
                    for target_index, label in enumerate(labels):
                        if time.perf_counter() - wall_started >= args.targeted_max_wall_seconds:
                            target_budget_hit = True
                            report["targeted_wall_budget_hit"] = True
                            break
                        target_id, separator, target_name = label.partition(":")
                        if not separator:
                            raise ValueError(f"Malformed detection label; expected ID: name: {label!r}")
                        target_id = target_id.strip()
                        target_name = target_name.strip()
                        target_prompt = LLAVA_TARGETED_CAPTION_TEMPLATE.format(
                            target_id=target_id, target_name=target_name
                        )
                        try:
                            response = _generate(
                                model, processor, image_path, "", target_prompt,
                                args.targeted_max_new_tokens, args.request_timeout,
                                single_user_turn=True,
                            )
                            raw_response = response["text"]
                            description = raw_response.strip()
                            not_visible = description.strip(" .\n\t").casefold() == "not_visible"
                            if not_visible:
                                description = ""
                                target_status = "not_visible"
                            else:
                                target_status = "described" if description else "empty_response"
                            target_captions.append({
                                "id": target_id,
                                "name": target_name,
                                "caption": description,
                                "status": target_status,
                            })
                            target_requests.append({
                                "id": target_id,
                                "name": target_name,
                                "prompt": target_prompt,
                                "serialized_prompt": response["serialized_prompt"],
                                "raw_response": raw_response,
                                "seconds": response["seconds"],
                                "prompt_tokens": response["prompt_tokens"],
                                "output_tokens": response["output_tokens"],
                                "posthoc_timeout_exceeded": response["posthoc_timeout_exceeded"],
                                "error": None,
                            })
                        except Exception as exc:
                            target_captions.append({
                                "id": target_id, "name": target_name,
                                "caption": "", "status": "error",
                            })
                            target_requests.append({
                                "id": target_id, "name": target_name,
                                "prompt": target_prompt, "raw_response": None,
                                "seconds": None, "prompt_tokens": None,
                                "output_tokens": None,
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                        target_stream.write(json.dumps({
                            "frame": record["frame"],
                            **target_requests[-1],
                            "status": target_captions[-1]["status"],
                            "caption": target_captions[-1]["caption"],
                        }, ensure_ascii=False) + "\n")
                        target_stream.flush()
                        if (target_index + 1) % 10 == 0:
                            print(
                                f"Frame {record['frame']}: targeted calls {target_index + 1}/{len(labels)}",
                                flush=True,
                            )
                    visible_entries = [
                        {key: item[key] for key in ("id", "name", "caption")}
                        for item in target_captions if item["caption"].strip()
                    ]
                    captions = {
                        "parsed": target_captions,
                        "text": json.dumps(visible_entries, ensure_ascii=False),
                        "seconds": sum(item["seconds"] or 0 for item in target_requests),
                        "prompt_tokens": sum(item["prompt_tokens"] or 0 for item in target_requests),
                        "output_tokens": sum(item["output_tokens"] or 0 for item in target_requests),
                        "serialized_prompt": None,
                        "target_requests": target_requests,
                    }
                    if target_budget_hit:
                        record["targeted_wall_budget_hit"] = True
                else:
                    captions = _generate(
                        model, processor, image_path,
                        caption_system_prompt,
                        caption_prompt, args.max_new_tokens, args.request_timeout,
                        single_user_turn=single_user_turn,
                    )
                target_mode = args.caption_profile == "targeted-object"
                record["caption_request"] = {
                    "seconds": captions["seconds"],
                    "prompt_tokens": captions["prompt_tokens"],
                    "output_tokens": captions["output_tokens"],
                    "raw_response": captions["text"],
                    "serialized_prompt": captions["serialized_prompt"],
                    "strict_list_valid": None if target_mode else _strict_list(captions["text"], relation=False),
                    "strict_json": None if target_mode else _strict_json_list(
                        captions["text"], relation=False, expected_ids=valid_ids,
                    ),
                }
                if args.caption_profile != "targeted-object":
                    captions["parsed"] = _parse_model_list(
                        captions["text"], relation=False,
                        prefer_json=args.prompt_profile == "llava-structured",
                    )
                captions["strict_list_valid"] = None if target_mode else _strict_list(captions["text"], relation=False)
                captions["strict_json"] = None if target_mode else _strict_json_list(
                    captions["text"], relation=False, expected_ids=valid_ids,
                )
                caption_ids = [
                    str(item.get("id")).strip()
                    for item in (captions["parsed"] or [])
                    if isinstance(item, dict) and item.get("id") is not None
                ]
                record["caption_request"].update({
                    "seconds": captions["seconds"],
                    "prompt_tokens": captions["prompt_tokens"],
                    "output_tokens": captions["output_tokens"],
                    "raw_response": captions["text"],
                    "parsed": captions["parsed"],
                    "strict_list_valid": captions["strict_list_valid"],
                    "strict_json": captions["strict_json"],
                    "id_coverage": None if target_mode else {
                        "candidate": len(set(caption_ids)), "expected": len(valid_ids),
                        "matched": len(set(caption_ids) & valid_ids),
                        "missing": sorted(valid_ids - set(caption_ids)),
                        "extra": sorted(set(caption_ids) - valid_ids),
                        "duplicates": len(caption_ids) - len(set(caption_ids)),
                    },
                })
                if args.caption_profile == "targeted-object":
                    record["caption_request"].update({
                        "id_attachment_method": "deterministic harness join by cached detection ID and name",
                        "model_native_structured_output": False,
                        "target_request_count": len(captions["target_requests"]),
                        "target_request_error_count": sum(bool(item.get("error")) for item in captions["target_requests"]),
                        "target_nonempty_count": sum(bool(item["caption"].strip()) for item in captions["parsed"]),
                        "target_not_visible_count": sum(item["status"] == "not_visible" for item in captions["parsed"]),
                        "target_empty_response_count": sum(item["status"] == "empty_response" for item in captions["parsed"]),
                        "target_requests": captions["target_requests"],
                        "stitching_note": "The harness attaches IDs/names; no model-format or ID-generation score is computed from this serialization.",
                        "cache_id_attachment": {
                            "expected_targets_in_frame": len(valid_ids),
                            "attempted_target_ids_attached": len(set(caption_ids) & valid_ids),
                            "unattempted_or_unattached_ids": sorted(valid_ids - set(caption_ids)),
                            "unexpected_target_ids": sorted(set(caption_ids) - valid_ids),
                        },
                    })
                    usable_caption_ids = {
                        item["id"] for item in (captions["parsed"] or [])
                        if item["status"] == "described" and item["caption"].strip()
                    }
                else:
                    usable_caption_ids = {
                        str(item["id"]).strip() for item in (captions["parsed"] or [])
                        if isinstance(item, dict)
                        and item.get("id") is not None
                        and str(item.get("id")).strip() in valid_ids
                        and isinstance(item.get("name"), str)
                        and item["name"].strip() == expected_id_names[str(item["id"]).strip()]
                        and isinstance(item.get("caption"), str) and item["caption"].strip()
                    } if (captions["strict_json"].get("json_valid")
                          and captions["strict_json"].get("schema_valid")) else set()
                record["caption_request"]["usable_caption_observation_count"] = len(usable_caption_ids)
                record["candidate_captions"] = captions["parsed"]
                record["baseline_captions"] = cached.get("captions") or []
            except Exception as exc:
                record["task_errors"]["captions"] = f"{type(exc).__name__}: {exc}"

            if captions is not None and "parsed" in captions:
                candidate_caption_map = caption_by_id(captions["parsed"])
                candidate_caption_map = {
                    key: value for key, value in candidate_caption_map.items()
                    if key in usable_caption_ids
                }
                baseline_caption_map = caption_by_id(cached.get("captions") or [])
                shared_ids = sorted(candidate_caption_map.keys() & baseline_caption_map.keys())
                record["caption_exact_agreement"] = (
                    sum(candidate_caption_map[i] == baseline_caption_map[i] for i in shared_ids) / len(shared_ids)
                    if shared_ids else None
                )
                record["caption_word_f1_mean"] = (
                    float(np.mean([word_f1(candidate_caption_map[i], baseline_caption_map[i]) for i in shared_ids]))
                    if shared_ids else None
                )
                record["caption_agreement_scored_id_count"] = len(shared_ids)
            record["latency_seconds"] = {
                task: response["seconds"] for task, response in
                (("relations", relation), ("captions", captions)) if response is not None
            }
            record["token_counts"] = {
                task: {"prompt": response["prompt_tokens"], "output": response["output_tokens"]}
                for task, response in (("relations", relation), ("captions", captions))
                if response is not None
            }
            record["raw_responses"] = {
                task: response["text"] for task, response in
                (("relations", relation), ("captions", captions)) if response is not None
            }
            if record["task_errors"]:
                record["error"] = "; ".join(
                    f"{task}: {error}" for task, error in record["task_errors"].items()
                )
            report["frames"].append(record)
            _write_json(output, report)
            print(f"Frame {record['frame']}: {record.get('error', 'done')}", flush=True)
            if record.get("targeted_wall_budget_hit"):
                break
    except Exception as exc:
        fatal_error = exc
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_sample.set()
        sampler.join(timeout=7)
        if sampler.is_alive():
            report["sampler_join_timeout"] = True
        try:
            torch.cuda.synchronize()
        except Exception as exc:
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        if target_stream is not None:
            target_stream.flush()
            target_stream.close()
        target_stream_records = []
        target_stream_parse_errors = 0
        if args.caption_profile == "targeted-object" and target_stream_path.is_file():
            with target_stream_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        target_stream_parse_errors += 1
                        continue
                    if (
                        not isinstance(item, dict)
                        or not isinstance(item.get("frame"), str)
                        or not isinstance(item.get("id"), str)
                        or not isinstance(item.get("name"), str)
                        or not isinstance(item.get("prompt"), str)
                        or item.get("status") not in {"described", "not_visible", "empty_response", "error"}
                        or not isinstance(item.get("caption"), str)
                        or (item.get("error") is not None and not isinstance(item.get("error"), str))
                    ):
                        target_stream_parse_errors += 1
                        continue
                    target_stream_records.append(item)
        target_stream_totals = {
            "target_requests_with_valid_record": len(target_stream_records),
            "target_request_errors": sum(bool(item.get("error")) for item in target_stream_records),
            "ids_attached_from_detection_cache": sum(
                bool(item.get("id") is not None and item.get("name")) for item in target_stream_records
            ),
            "target_descriptions_nonempty": sum(
                item.get("status") == "described" and bool(item.get("caption", "").strip())
                for item in target_stream_records
            ),
            "targets_not_visible": sum(item.get("status") == "not_visible" for item in target_stream_records),
            "empty_target_responses": sum(item.get("status") == "empty_response" for item in target_stream_records),
            "unparsed_target_stream_lines": target_stream_parse_errors,
        } if args.caption_profile == "targeted-object" else None
        if target_stream_totals is not None:
            target_stream_totals["expected_target_requests"] = expected_caption_observations
            target_stream_totals["target_requests_without_valid_record"] = max(
                0, expected_caption_observations
                - target_stream_totals["target_requests_with_valid_record"]
                - target_stream_parse_errors
            )
            report["target_stream_totals"] = target_stream_totals
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["wall_seconds"] = time.perf_counter() - wall_started
        try:
            report["gpu_memory"] = {
                "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            }
        except Exception as exc:
            report["gpu_memory_error"] = f"{type(exc).__name__}: {exc}"
        report["gpu_memory"] = {
            **report.get("gpu_memory", {}),
            "nvidia_smi_peak_all_gpus_used_mib": max(samples) if samples else None,
            "nvidia_smi_sample_count": len(samples),
            "nvidia_smi_scope": "maximum memory.used among all visible GPUs; includes other processes and desktop baseline",
        }
        records = report["frames"]
        relation_records = [item for item in records if "relation_request" in item]
        caption_records = [item for item in records if "caption_request" in item]
        complete_records = [item for item in records
                            if "relation_request" in item and "caption_request" in item]
        relation_agreements = [
            item["relation_agreement"]["agreement_f1"] for item in relation_records
            if isinstance(item.get("relation_agreement"), dict)
        ]
        caption_agreements = [
            item["caption_word_f1_mean"] for item in caption_records
            if item.get("caption_word_f1_mean") is not None
        ]
        report["summary"] = {
            "selected_frames": 40,
            "records_written": len(records),
            "frame_records_missing": max(0, 40 - len(records)),
            "frames_missing_annotated_image": sum(
                "missing annotated image" in item.get("error", "") for item in records
            ),
            "relation_requests_completed": len(relation_records),
            "caption_requests_completed": len(caption_records),
            "caption_frame_records_completed": len(caption_records),
            "complete_frame_pairs": len(complete_records),
            "relation_legacy_parse_valid_frames": sum(
                bool(item["relation_request"].get("strict_list_valid")) for item in relation_records
            ),
            "relation_strict_json": _aggregate_json_validation(records, "relation_request"),
            "caption_legacy_parse_valid_frames": (
                None if args.caption_profile == "targeted-object" else sum(
                    bool(item["caption_request"].get("strict_list_valid")) for item in caption_records
                )
            ),
            "caption_strict_json": (
                None if args.caption_profile == "targeted-object" else
                _aggregate_json_validation(records, "caption_request")
            ),
            "caption_native_structured_output": (
                None if args.caption_profile == "targeted-object" else
                _aggregate_json_validation(records, "caption_request")
            ),
            "caption_harness_stitching_note": (
                "Targeted IDs/names come from cached detections. No model-native format or ID-generation score is computed."
                if args.caption_profile == "targeted-object" else None
            ),
            "usable_caption_observations": {
                "count": (
                    target_stream_totals["target_descriptions_nonempty"]
                    if args.caption_profile == "targeted-object" else
                    sum(item.get("caption_request", {}).get("usable_caption_observation_count", 0)
                        for item in records)
                ),
                "expected": expected_caption_observations,
                "rate": (
                    (target_stream_totals["target_descriptions_nonempty"]
                     if args.caption_profile == "targeted-object" else
                     sum(item.get("caption_request", {}).get("usable_caption_observation_count", 0)
                         for item in records)) / expected_caption_observations
                    if expected_caption_observations else 0.0
                ),
                "definition": (
                    "Detection observation with a nonempty description and resolvable ID/name; batch mode also requires valid JSON array schema; targeted IDs are prompt-linked then cache-attached."
                ),
            },
            "targeted_caption_totals": ({
                **target_stream_totals,
                "caption_word_f1_scored_ids": sum(
                    item.get("caption_agreement_scored_id_count", 0)
                    for item in records
                ),
            } if args.caption_profile == "targeted-object" else None),
            "relation_error_count": sum("relations" in item.get("task_errors", {}) for item in records),
            "caption_error_count": sum("captions" in item.get("task_errors", {}) for item in records),
            "relation_agreement_f1_mean": statistics.mean(relation_agreements) if relation_agreements else None,
            "relation_agreement_scored_frames": len(relation_agreements),
            "relation_agreement_empty_reference_frames": sum(
                not item.get("baseline_relations") for item in relation_records
            ),
            "caption_word_f1_mean": statistics.mean(caption_agreements) if caption_agreements else None,
            "caption_word_f1_scored_frames": len(caption_agreements),
            "caption_word_f1_scored_ids": sum(
                item.get("caption_agreement_scored_id_count", 0) for item in records
            ),
        }
        _write_json(output, report)
        print(json.dumps({"summary": report["summary"], "wall_seconds": report["wall_seconds"], "gpu_memory": report["gpu_memory"]}, indent=2))

    if fatal_error is not None:
        raise fatal_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
