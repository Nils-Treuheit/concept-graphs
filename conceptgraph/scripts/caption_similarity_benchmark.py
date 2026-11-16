"""Compare saved captions with GPT-generated visual references using FG-CLIP2."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from conceptgraph.scripts.clip_component import Encoder


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK = REPO_ROOT / ".agentic_workflow/workspace/caption-grounding-benchmark"
DEFAULT_RESULTS = REPO_ROOT / ".agentic_workflow/workspace/caption-consolidation-benchmark"
DEFAULT_HF_HOME = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface")
MODEL = "fg-clip2-so400m"
OUTPUT_MODELS = {
    "Qwen3.8 baseline": ("room0-qwen3.8-baseline-40.json", "caption"),
    "Qwen3.5 9B Q4": ("room0-qwen35-9b-q4-caption-consolidation-40.json", "candidate_caption"),
    "Qwen3.5 9B Q8": ("room0-qwen35-9b-q8-caption-consolidation-40.json", "candidate_caption"),
    "Gemma 4 12B Q4": ("room0-gemma4-12b-caption-consolidation-40.json", "candidate_caption"),
}


def _normalize(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _word_f1(left: str, right: str) -> float:
    import re

    a = Counter(re.findall(r"[\w']+", _normalize(left)))
    b = Counter(re.findall(r"[\w']+", _normalize(right)))
    if not a or not b:
        return float(a == b)
    overlap = sum((a & b).values())
    precision, recall = overlap / sum(a.values()), overlap / sum(b.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _load_caption_rows(results_dir: Path, reference_manifest: dict) -> dict[int, dict]:
    expected_hash = reference_manifest["source_map_sha256"]
    reference_by_id = {
        int(row["object_index"]): row for row in reference_manifest["crops"]
    }
    if len(reference_by_id) != 40:
        raise ValueError(f"Expected 40 references; found {len(reference_by_id)}")
    outputs: dict[str, dict[int, str]] = {}
    for model_name, (filename, caption_field) in OUTPUT_MODELS.items():
        path = results_dir / filename
        payload = json.loads(path.read_text())
        if payload.get("map_sha256") != expected_hash:
            raise ValueError(f"{model_name} caption output has a different map hash")
        rows = {int(row["object_index"]): row for row in payload["groups"]}
        if set(rows) != set(reference_by_id):
            raise ValueError(f"{model_name} caption IDs do not match the reference set")
        outputs[model_name] = {}
        for object_index, row in rows.items():
            caption = row.get(caption_field)
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"{model_name} has no valid caption for object {object_index}")
            outputs[model_name][object_index] = caption.strip()
    combined = {}
    for object_index, reference in reference_by_id.items():
        description = reference.get("reference") or {}
        ref_caption = description.get("caption")
        if not isinstance(ref_caption, str) or not ref_caption.strip():
            raise ValueError(f"GPT reference missing for object {object_index}")
        combined[object_index] = {
            "reference": ref_caption.strip(),
            **{model: captions[object_index] for model, captions in outputs.items()},
        }
    return combined


def _encode_texts(encoder: Encoder, texts: list[str], batch_size: int) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunks.append(
                encoder.encode_text(
                    texts[start : start + batch_size], prompt_template=False, fg_walk_type="short"
                ).float().cpu()
            )
    return torch.cat(chunks, dim=0)


def _summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean_cosine_similarity": sum(ordered) / len(ordered),
        "median_cosine_similarity": (
            ordered[len(ordered) // 2] if len(ordered) % 2 else
            (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
        ),
        "min_cosine_similarity": min(ordered),
        "max_cosine_similarity": max(ordered),
    }


def run(args: argparse.Namespace) -> Path:
    work = args.work.expanduser().resolve()
    manifest_path = args.references.expanduser().resolve()
    reference_manifest = json.loads(manifest_path.read_text())
    captions = _load_caption_rows(args.results.expanduser().resolve(), reference_manifest)
    model_names = list(OUTPUT_MODELS)
    item_ids = sorted(captions)
    all_texts = []
    text_index: dict[tuple[int, str], int] = {}
    for object_index in item_ids:
        row = captions[object_index]
        for key in ("reference", *model_names):
            text_index[(object_index, key)] = len(all_texts)
            all_texts.append(row[key])

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("FG-CLIP2 caption comparison requires the configured CUDA GPU")
    # Passing an explicit torch.device to these APIs raises "Invalid device
    # argument" on the installed PyTorch 2.14 CUDA build, despite cuda:0 being
    # the sole current device. Use the current device after the availability
    # check above.
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    encoder = Encoder(MODEL, args.hf_home.expanduser().resolve(), device)
    load_seconds = encoder.load_seconds
    revision = encoder.revision
    features = _encode_texts(encoder, all_texts, args.batch_size)
    encoding_seconds = time.perf_counter() - started - load_seconds
    if features.shape[0] != len(all_texts):
        raise ValueError(f"Expected {len(all_texts)} embeddings, got {features.shape[0]}")
    norms = torch.linalg.vector_norm(features, dim=-1)
    if not torch.isfinite(features).all() or not torch.allclose(
        norms, torch.ones_like(norms), rtol=2e-3, atol=2e-3
    ):
        raise ValueError("FG-CLIP2 text embeddings are non-finite or not normalized")
    similarities = features @ features.T
    per_object = []
    by_model_ref = {name: [] for name in model_names}
    by_model_qwen = {name: [] for name in model_names}
    for object_index in item_ids:
        row = captions[object_index]
        ref_ix = text_index[(object_index, "reference")]
        qwen_ix = text_index[(object_index, "Qwen3.8 baseline")]
        model_scores = {}
        for name in model_names:
            ix = text_index[(object_index, name)]
            ref_score = float(similarities[ix, ref_ix])
            qwen_score = float(similarities[ix, qwen_ix])
            model_scores[name] = {
                "cosine_to_gpt_reference": ref_score,
                "cosine_to_qwen38_baseline": qwen_score,
                "different_from_qwen38_text": _normalize(row[name]) != _normalize(row["Qwen3.8 baseline"]),
                "word_f1_to_gpt_reference": _word_f1(row[name], row["reference"]),
            }
            by_model_ref[name].append(ref_score)
            by_model_qwen[name].append(qwen_score)
        pairwise = {}
        for i, left in enumerate(model_names):
            for right in model_names[i + 1 :]:
                li, ri = text_index[(object_index, left)], text_index[(object_index, right)]
                pairwise[f"{left} vs {right}"] = float(similarities[li, ri])
        per_object.append({
            "object_index": object_index,
            "gpt_reference_caption": row["reference"],
            "model_captions": {name: row[name] for name in model_names},
            "per_model": model_scores,
            "pairwise_model_cosines": pairwise,
        })
    summary = {}
    for name in model_names:
        scores = [row["per_model"][name] for row in per_object]
        summary[name] = {
            "n": len(scores),
            "mean_cosine_to_gpt_reference": _summary(by_model_ref[name]),
            "mean_cosine_to_qwen38_baseline": _summary(by_model_qwen[name]),
            "different_from_qwen38_text": sum(row["different_from_qwen38_text"] for row in scores),
            "mean_word_f1_to_gpt_reference": sum(row["word_f1_to_gpt_reference"] for row in scores) / len(scores),
        }
    pairwise_summary = {}
    for i, left in enumerate(model_names):
        for right in model_names[i + 1 :]:
            key = f"{left} vs {right}"
            pairwise_summary[key] = _summary([
                row["pairwise_model_cosines"][key] for row in per_object
            ])
    result = {
        "schema_version": 1,
        "metric": "FG-CLIP2 So400m text embedding cosine similarity",
        "interpretation": "Similarity scores indicate text embedding proximity, not caption truth or visual grounding accuracy.",
        "scene": reference_manifest.get("scene"),
        "sample_size": len(item_ids),
        "source_map_sha256": reference_manifest.get("source_map_sha256"),
        "reference_type": reference_manifest.get("reference_type"),
        "reference_model": reference_manifest.get("reference_model"),
        "encoder": {
            "model_id": "qihoo360/fg-clip2-so400m",
            "revision": revision,
            "walk_type": "short",
            "prompt_template": False,
            "hf_home": str(args.hf_home.expanduser().resolve()),
        },
        "runtime": {
            "load_seconds": load_seconds,
            "text_encoding_seconds": encoding_seconds,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "models": summary,
        "pairwise_model_cosine_similarity": pairwise_summary,
        "per_object": per_object,
    }
    output = args.output.expanduser().resolve()
    _write_json(output, result)
    del similarities, features, encoder
    torch.cuda.empty_cache()
    print(f"Saved FG-CLIP2 caption similarity results: {output}")
    for name, values in summary.items():
        print(
            f"{name}: GPT-reference cosine {values['mean_cosine_to_gpt_reference']['mean_cosine_similarity']:.4f}; "
            f"Qwen3.8 cosine {values['mean_cosine_to_qwen38_baseline']['mean_cosine_similarity']:.4f}; "
            f"changed captions {values['different_from_qwen38_text']}/40"
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=DEFAULT_WORK / "references.json")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_WORK / "fg-clip2-similarity-40.json")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
