"""Stage 4 of the local ConceptGraphs pipeline: consolidate the captions.

The VLM (stage 2) produces one caption per object *per frame*.  This stage
merges the captions of every object into a single one with the **local text
model** (Qwen3.8 served by Ollama) and stores the result in the map of the
mapping stage.

This script sends serial requests to the local text model after mapping. Model
residency remains managed by the configured local inference service.
"""

from __future__ import annotations

import atexit
import argparse
import gzip
import json
import os
import pickle
from pathlib import Path
import tempfile

from conceptgraph.utils.llm_client import REASONING_EFFORTS, get_llm_client
from conceptgraph.utils.vlm import consolidate_captions


def _unload_llm_at_stage_exit(client) -> None:
    """Best-effort cleanup if consolidation exits before its explicit unload."""
    try:
        client.unload()
    except Exception as exc:  # keep checkpointed results usable if cleanup fails
        print(f"Warning: could not unload the text model after consolidation: {exc}")


def load_map(exp_out_path: Path, exp_suffix: str) -> dict:
    pcd_path = Path(exp_out_path) / f"pcd_{exp_suffix}.pkl.gz"
    with gzip.open(pcd_path, "rb") as f:
        return pickle.load(f)


def save_map(exp_out_path: Path, exp_suffix: str, results: dict) -> Path:
    pcd_path = Path(exp_out_path) / f"pcd_{exp_suffix}.pkl.gz"
    _atomic_write(pcd_path, lambda temp_path: _write_gzip_pickle(temp_path, results))
    return pcd_path


def _atomic_write(path: Path, write_temp) -> None:
    """Replace a file only after its complete contents have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        write_temp(temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_gzip_pickle(path: Path, results: dict) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump(results, f)


def _write_json(path: Path, contents: dict) -> None:
    with path.open("w") as f:
        json.dump(contents, f, indent=2)


def write_object_json(exp_out_path: Path, exp_suffix: str, objects: list[dict]) -> Path:
    """Write the objects of the map as JSON (same schema as ``save_obj_json``)."""
    json_obj_list = {}
    for curr_idx, curr_obj in enumerate(objects):
        bbox_np = curr_obj.get("bbox_np")
        if bbox_np is not None:
            bbox_np = [[round(float(v), 4) for v in corner] for corner in bbox_np]
            extent = [
                round(max(c[0] for c in bbox_np) - min(c[0] for c in bbox_np), 3),
                round(max(c[1] for c in bbox_np) - min(c[1] for c in bbox_np), 3),
                round(max(c[2] for c in bbox_np) - min(c[2] for c in bbox_np), 3),
            ]
            center = [
                round((max(c[0] for c in bbox_np) + min(c[0] for c in bbox_np)) / 2, 3),
                round((max(c[1] for c in bbox_np) + min(c[1] for c in bbox_np)) / 2, 3),
                round((max(c[2] for c in bbox_np) + min(c[2] for c in bbox_np)) / 2, 3),
            ]
            volume = round(extent[0] * extent[1] * extent[2], 2)
        else:
            extent, center, volume = [], [], 0.0

        json_obj_list[f"object_{curr_idx + 1}"] = {
            "id": curr_obj.get("curr_obj_num", curr_idx + 1),
            "object_tag": curr_obj.get("class_name", ""),
            "object_caption": curr_obj.get("consolidated_caption", ""),
            "bbox_extent": extent,
            "bbox_center": center,
            "bbox_volume": volume,
        }

    out_path = Path(exp_out_path) / f"obj_json_{exp_suffix}.json"
    _atomic_write(out_path, lambda temp_path: _write_json(temp_path, json_obj_list))
    print(f"Saved object JSON to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp_out_path",
        required=True,
        help="Folder of the mapping experiment (contains pcd_<exp_suffix>.pkl.gz)",
    )
    parser.add_argument("--exp_suffix", required=True, help="Suffix of the mapping experiment")
    parser.add_argument(
        "--max_captions",
        type=int,
        default=20,
        help="How many captions per object are consolidated",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save the map and object JSON after this many visited objects",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default="low",
        help="Reasoning effort for local text-model requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per object caption",
    )
    args = parser.parse_args()
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be a positive integer")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be a positive integer")

    exp_out_path = Path(args.exp_out_path)
    results = load_map(exp_out_path, args.exp_suffix)
    objects = results["objects"]
    print(f"Loaded {len(objects)} objects from the map.")

    client = get_llm_client(reasoning_effort=args.reasoning_effort, max_tokens=args.max_tokens)
    atexit.register(_unload_llm_at_stage_exit, client)
    print(f"Using the local text model '{client.model}' at {client.base_url}")

    num_consolidated = 0
    for obj_idx, curr_obj in enumerate(objects):
        existing = curr_obj.get("consolidated_caption")
        if isinstance(existing, str) and existing.strip():
            pass
        else:
            captions = [
                cap
                for cap in (curr_obj.get("captions") or [])
                if isinstance(cap, dict) and cap.get("caption") is not None
            ]
            if not captions:
                curr_obj["consolidated_caption"] = ""
            else:
                print(
                    f"[{obj_idx + 1}/{len(objects)}] Consolidating "
                    f"{curr_obj.get('class_name', '?')}...",
                    flush=True,
                )
                consolidated = consolidate_captions(client, captions[: args.max_captions])
                curr_obj["consolidated_caption"] = consolidated
                num_consolidated += 1
                print(
                    f"[{obj_idx + 1}/{len(objects)}] {curr_obj.get('class_name', '?')}: "
                    f"{consolidated}",
                    flush=True,
                )

        # Count all visited objects, including skipped and captionless objects,
        # so restart checkpoint intervals do not depend on model-call counts.
        if (obj_idx + 1) % args.checkpoint_every == 0:
            save_map(exp_out_path, args.exp_suffix, results)
            write_object_json(exp_out_path, args.exp_suffix, objects)

    save_map(exp_out_path, args.exp_suffix, results)
    write_object_json(exp_out_path, args.exp_suffix, objects)
    print(f"Consolidated the captions of {num_consolidated} objects.")
    # Do not report this stage complete until Ollama has accepted the unload.
    client.unload()
    atexit.unregister(_unload_llm_at_stage_exit)


if __name__ == "__main__":
    main()
