#!/usr/bin/env python3
"""GPU-only ObjEmbed retrieval/association benchmark on frozen ReplicaSSG boxes.

This adapter uses the official ObjEmbed model and processor with oracle boxes
from a manifest frozen before inference. It is separate from the WeDetect
proposal-conditioned experiment and from the SigLIP2/FG-CLIP2 crop controls.
Run only after the workflow release/controller gate passes in host context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
REPO = ROOT / "models/experiments/objembed/repo"
DEFAULT_HF_HOME = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface")
DEFAULT_MANIFEST = ROOT / "models/experiments/objembed/data/replicassg-five-scene-v1.json"
DEFAULT_OUT_ROOT = ROOT / "models/experiments/objembed/results"
SNAPSHOTS = {
    "2b": ("fushh7/ObjEmbed-2B", "0e8aafeb53a95fa624de76de3a3c0f59615acbc2"),
    "4b": ("fushh7/ObjEmbed-4B", "18b9b26c6d93f94045d2acc4914d1e4e1ded2a0a"),
}
EXPECTED = {"scene_count": 5, "frame_view_count": 250,
            "visible_object_view_count": 2204, "category_count": 49}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path.resolve(strict=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--hf-home", type=Path, default=Path(os.getenv("HF_HOME", DEFAULT_HF_HOME)))
    p.add_argument("--models", nargs="+", choices=tuple(SNAPSHOTS), default=["2b", "4b"])
    p.add_argument("--output-dir", type=Path,
                   default=DEFAULT_OUT_ROOT / f"gt-region-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    p.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
    return p.parse_args()


def validate_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    path = require(path, "frozen ObjEmbed manifest")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "objembed-replicassg-gt-manifest/v1" or data.get("status") != "frozen_before_model_inference":
        raise ValueError("manifest schema/status is not the accepted frozen ObjEmbed input")
    counts = data.get("counts", {})
    for key, expected in EXPECTED.items():
        if counts.get(key) != expected:
            raise ValueError(f"manifest count {key}={counts.get(key)!r}, expected {expected}")
    scenes = data.get("split", {}).get("evaluated_scenes", [])
    if len(scenes) != 5 or len(set(scenes)) != 5:
        raise ValueError(f"expected five unique frozen scenes, got {scenes}")
    rows = data.get("object_views", [])
    if len(rows) != EXPECTED["visible_object_view_count"]:
        raise ValueError(f"expected 2204 object views, got {len(rows)}")
    view_ids = [row.get("view_id") for row in rows]
    if len(set(view_ids)) != len(rows):
        raise ValueError("manifest contains duplicate view IDs")
    gallery = data.get("gallery", {})
    if gallery.get("view_count") != len(rows) or set(gallery.get("view_ids", [])) != set(view_ids):
        raise ValueError("gallery membership does not exactly match object views")
    tracks = data.get("tracks", [])
    track_views = [view_id for track in tracks for view_id in track.get("view_ids", [])]
    if sorted(track_views) != sorted(view_ids):
        raise ValueError("track membership does not cover every view exactly once")
    verified_files: dict[str, tuple[int, str]] = {}
    for row in rows:
        for key in ("box_crop", "mask_crop", "source_rgb"):
            item = row.get(key, {})
            source = require(Path(item.get("path", "")), f"{key} for {row.get('view_id')}")
            identity = str(source)
            if identity not in verified_files:
                verified_files[identity] = (source.stat().st_size, sha256_file(source))
            if verified_files[identity] != (item.get("bytes"), item.get("sha256")):
                raise ValueError(f"manifest hash mismatch: {source}")
        box = row.get("box_crop", {}).get("region_box_xyxy")
        if not isinstance(box, list) or len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
            raise ValueError(f"invalid source-frame box for {row.get('view_id')}: {box}")
    categories = sorted({str(row["category"]) for row in rows})
    if len(categories) != EXPECTED["category_count"]:
        raise ValueError(f"expected 49 categories, found {len(categories)}")
    return data, rows, categories


def resolve_snapshot(hf_home: Path, model_key: str) -> tuple[Path, str, str, list[dict[str, Any]]]:
    repo_id, revision = SNAPSHOTS[model_key]
    repo_slug = "models--" + repo_id.replace("/", "--")
    snapshot = hf_home / "hub" / repo_slug / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing local HF snapshot {repo_id}@{revision}: {snapshot}")
    snapshot = snapshot.resolve(strict=True)
    files = sorted(p for p in snapshot.iterdir() if p.is_file())
    must_have = ("config.json", "model.safetensors.index.json", "tokenizer.json", "preprocessor_config.json")
    missing = [name for name in must_have if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"snapshot {repo_id}@{revision} is incomplete: {missing}")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    records = []
    for name in shards:
        weight = require(snapshot / name, f"checkpoint shard {name}")
        records.append({"name": name, "bytes": weight.stat().st_size, "sha256": sha256_file(weight)})
    return snapshot, repo_id, revision, records


def chat_messages(query: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text":
             f"Find an object that matches the given caption. {query} <local_text>"}]}]


def category_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    relevant = labels[order].astype(bool)
    positives = int(relevant.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
    return float(precision[relevant].sum() / positives)


def retrieval_metrics(features: torch.Tensor, text_features: torch.Tensor,
                      objness: torch.Tensor, rows: list[dict[str, Any]],
                      categories: list[str], model: Any) -> dict[str, Any]:
    features = F.normalize(features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    logits = (text_features @ features.T) * model.logit_log_scale.float().exp() + model.logit_bias.float()
    scores = logits.sigmoid() * objness.float().sigmoid().reshape(1, -1)
    score_array = scores.detach().cpu().numpy()
    row_categories = np.asarray([row["category"] for row in rows])
    aps, recalls = [], {1: [], 5: [], 10: []}
    for class_idx, category in enumerate(categories):
        binary = row_categories == category
        aps.append(category_ap(score_array[class_idx], binary))
        ranked = np.argsort(-score_array[class_idx], kind="stable")
        for k in recalls:
            recalls[k].append(float(binary[ranked[:k]].any()))
    predicted = np.asarray(categories)[score_array.argmax(axis=0)]
    accuracy = float(np.mean(predicted == row_categories))
    return {
        "category_text_macro_ap": float(np.nanmean(aps)),
        "category_text_recall_at_1": float(np.mean(recalls[1])),
        "category_text_recall_at_5": float(np.mean(recalls[5])),
        "category_text_recall_at_10": float(np.mean(recalls[10])),
        "zero_shot_category_accuracy": accuracy,
        "category_query_count": len(categories),
        "object_view_gallery_count": len(rows),
    }


def load_model(model_key: str, snapshot: Path, attention: str, device: torch.device):
    sys.path.insert(0, str(REPO))
    from models.qwen3vl_objembed import ObjectEmbed
    from transformers import AutoProcessor

    dtype = torch.bfloat16
    model = ObjectEmbed.from_pretrained(
        str(snapshot), torch_dtype=dtype, attn_implementation=attention,
        local_files_only=True, trust_remote_code=False,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    object_token_id = processor.tokenizer.convert_tokens_to_ids("<object>")
    local_text_id = processor.tokenizer.convert_tokens_to_ids("<local_text>")
    global_id = None
    global_text_id = None
    if model.use_global_caption:
        global_id = processor.tokenizer.convert_tokens_to_ids("<global>")
        global_text_id = processor.tokenizer.convert_tokens_to_ids("<global_text>")
    token_ids = {"<object>": object_token_id, "<local_text>": local_text_id}
    if model.use_global_caption:
        token_ids.update({"<global>": global_id, "<global_text>": global_text_id})
    if any(token_id is None or token_id == processor.tokenizer.unk_token_id
           for token_id in token_ids.values()):
        raise ValueError(f"checkpoint processor lacks ObjEmbed special tokens: {token_ids}")
    model.model.object_token_id = object_token_id
    return model, processor, object_token_id, local_text_id, global_id, global_text_id


def encode_text_queries(model: Any, processor: Any, local_text_id: int,
                        global_id: int | None, global_text_id: int | None,
                        categories: list[str], device: torch.device) -> torch.Tensor:
    messages = [processor.apply_chat_template(chat_messages(f"a photo of a {category}"), tokenize=False).strip()
                for category in categories]
    text_inputs = processor(text=messages, return_tensors="pt", padding=True, do_resize=False)
    text_inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                   for key, value in text_inputs.items()}
    with torch.inference_mode():
        outputs = model(text_processed=text_inputs, global_id=global_id,
                        local_text_id=local_text_id, global_text_id=global_text_id)
    embeddings = outputs.local_text_embeddings
    if embeddings is None or embeddings.shape[0] != len(categories):
        raise ValueError(f"expected {len(categories)} local text embeddings, got {None if embeddings is None else tuple(embeddings.shape)}")
    if not torch.isfinite(embeddings).all():
        raise ValueError("non-finite ObjEmbed text embeddings")
    return embeddings


def encode_objects(model: Any, processor: Any, object_token_id: int,
                   local_text_id: int, global_id: int | None,
                   global_text_id: int | None, rows: list[dict[str, Any]],
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    from models.vision_process import process_vision_info

    features: list[torch.Tensor] = []
    objness: list[torch.Tensor] = []
    started = time.perf_counter()
    by_image: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_image.setdefault(row["source_rgb"]["path"], []).append(row)
    for image_path, image_rows in by_image.items():
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        boxes = [row["box_crop"]["region_box_xyxy"] for row in image_rows]
        object_prompts = "".join(f"Object {idx}: <object><object>. " for idx in range(len(boxes)))
        if model.use_global_caption and model.use_two_tokens == 2:
            object_prompt = "The coarse global image is <global>. " + object_prompts \
                            + " The detailed global image is <global>. "
        elif model.use_global_caption and model.use_two_tokens == 1:
            object_prompt = "The global image is <global>. " + object_prompts \
                            + "The global image is <global>"
        elif model.use_global_caption:
            object_prompt = object_prompts + "The global image is <global>"
        else:
            object_prompt = object_prompts
        prompt = (
            "Locate the specific object being described by analyzing its unique instance-level attributes, "
            "its spatial position, and its relationship with surrounding objects. "
            + object_prompt
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt},
        ]}]
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        text = [processor.apply_chat_template(messages, tokenize=False).strip()]
        model_inputs = processor(text=text, images=image_inputs, videos=video_inputs,
                                 return_tensors="pt", padding=True, do_resize=False)
        model_inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in model_inputs.items()}
        bboxes = torch.as_tensor(boxes, dtype=torch.float32, device=device).to(model.dtype)
        with torch.inference_mode():
            outputs = model(**model_inputs, bboxes=[bboxes], ori_shapes=[image.size],
                            bboxes_id=object_token_id, global_id=global_id,
                            local_text_id=local_text_id, global_text_id=global_text_id)
        found = outputs.object_embeddings
        scores = outputs.objness
        if found is None or scores is None or found.shape[0] != len(image_rows) or scores.shape[0] != len(image_rows):
            raise ValueError(f"ObjEmbed output count mismatch for {image_path}: expected {len(image_rows)}")
        if not torch.isfinite(found).all() or not torch.isfinite(scores).all():
            raise ValueError(f"non-finite ObjEmbed outputs for {image_path}")
        features.append(found.float().cpu())
        objness.append(scores.float().cpu().reshape(-1))
        image.close()
        torch.cuda.synchronize(device)
        print(f"[objembed] encoded {len(features)}/{len(by_image)} images", flush=True)
    return torch.cat(features), torch.cat(objness), time.perf_counter() - started


def run_one(model_key: str, args: argparse.Namespace, data: dict[str, Any],
            rows: list[dict[str, Any]], categories: list[str],
            output_dir: Path) -> dict[str, Any]:
    model_started = time.perf_counter()
    snapshot, repo_id, revision, weight_records = resolve_snapshot(args.hf_home.expanduser().resolve(), model_key)
    require(REPO / "models/qwen3vl_objembed.py", "official ObjEmbed model source")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark has no CPU fallback")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model, processor, object_token_id, local_text_id, global_id, global_text_id = load_model(
        model_key, snapshot, args.attention, device)
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    resident_allocated = torch.cuda.memory_allocated(device) / 1024**3
    load_peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    load_peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    torch.cuda.reset_peak_memory_stats(device)
    query_started = time.perf_counter()
    text_features = encode_text_queries(model, processor, local_text_id, global_id,
                                        global_text_id, categories, device)
    text_seconds = time.perf_counter() - query_started
    object_features, objectness, image_seconds = encode_objects(
        model, processor, object_token_id, local_text_id, global_id, global_text_id,
        rows, device)
    torch.cuda.synchronize(device)
    feature_tensor = object_features.to(device)
    association = __import__("conceptgraph.scripts.replica_clip_benchmark", fromlist=["_association_and_graph_metrics"])._association_and_graph_metrics(
        feature_tensor, [{**row, "frame": row["frame_index"]} for row in rows],
        data["ground_truth_relationships"], data["split"]["evaluated_scenes"],
    )
    retrieval = retrieval_metrics(object_features.to(device), text_features, objectness.to(device), rows,
                                  categories, model)
    infer_peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    infer_peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    result = {
        "schema": "objembed-replicassg-gt-box-run/v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"key": model_key, "repo_id": repo_id, "revision": revision,
                  "snapshot_path": str(snapshot), "weight_shards": weight_records,
                  "attention": args.attention, "dtype": "bfloat16",
                  "use_global_caption": bool(model.use_global_caption),
                  "use_two_tokens": int(model.use_two_tokens),
                  "use_two_captions": bool(model.use_two_captions)},
        "input": {"manifest": str(args.manifest.resolve()),
                  "manifest_sha256": sha256_file(args.manifest.resolve()),
                  "split": data["split"], "counts": data["counts"],
                  "proposal_source": "ReplicaSSG semantic instance ground-truth boxes; oracle proposals",
                  "object_view_source": "full RGB frame plus corresponding GT source-frame box; official ObjEmbed object-token path",
                  "query_text": "a photo of a {category}",
                  "categories": categories, "gallery_id": data["gallery"]["gallery_id"]},
        "metrics": {**retrieval, **association},
        "timing_seconds": {"model_load": model_load_seconds, "text_encoding": text_seconds,
                           "object_image_encoding": image_seconds,
                           "total_model_run": time.perf_counter() - model_started},
        "resources": {"device_name": properties.name, "device_uuid": str(properties.uuid),
                      "total_memory_gib": properties.total_memory / 1024**3,
                      "model_resident_allocated_gib": resident_allocated,
                      "load_peak_allocated_gib": load_peak_allocated,
                      "load_peak_reserved_gib": load_peak_reserved,
                      "inference_peak_allocated_gib": infer_peak_allocated,
                      "inference_peak_reserved_gib": infer_peak_reserved,
                      "peak_allocated_gib": max(load_peak_allocated, infer_peak_allocated),
                      "peak_reserved_gib": max(load_peak_reserved, infer_peak_reserved)},
        "comparison_scope": "Oracle-box ObjEmbed encoder comparison. Do not compare as a WeDetect proposal benchmark or combine with proposal-conditioned AP.",
    }
    del model, processor, text_features, feature_tensor
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    args.manifest = args.manifest.expanduser().resolve(strict=True)
    if not REPO.is_dir():
        raise FileNotFoundError(f"official ObjEmbed repo missing: {REPO}")
    data, rows, categories = validate_manifest(args.manifest)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    (output_dir / "run-input.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "models": args.models,
        "attention": args.attention,
        "expected_counts": EXPECTED,
    }, indent=2) + "\n", encoding="utf-8")
    results = []
    try:
        for model_key in args.models:
            result = run_one(model_key, args, data, rows, categories, output_dir)
            path = output_dir / f"{model_key}-metrics.json"
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            results.append({"model": model_key, "result": str(path)})
    except BaseException as exc:
        (output_dir / "failure.json").write_text(json.dumps({
            "status": "failed", "exception_type": type(exc).__name__, "error": str(exc),
            "completed_models": results,
        }, indent=2) + "\n", encoding="utf-8")
        raise
    (output_dir / "index.json").write_text(json.dumps({"status": "complete", "results": results},
                                                          indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    raise SystemExit(main())
