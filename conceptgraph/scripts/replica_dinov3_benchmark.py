"""Visual-only DINOv3 object association benchmark on ReplicaSSG ground truth.

This evaluates frozen DINOv3 ViT-L/16 CLS features for cross-view object
identity and oracle-predicate graph recovery. It deliberately does not claim
text retrieval or DINOde performance.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from conceptgraph.scripts import replica_clip_benchmark as shared

MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_HF_HOME = shared.DEFAULT_HF_HOME
DEFAULT_OUTPUT = Path(".agentic_workflow/workspace/optional-evals/dinov3-replica-association.json")
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=shared.DEFAULT_IMAGE_ROOT)
    parser.add_argument("--semantic-root", type=Path, default=shared.DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--annotations", type=Path, default=shared.DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--hf-home", type=Path, default=Path(os.getenv("HF_HOME", DEFAULT_HF_HOME)))
    parser.add_argument("--scenes", nargs="+", help="Defaults to scenes with local Replica RGB sequences")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames-per-scene", type=int, default=50)
    parser.add_argument("--crop-cache", type=Path, default=shared.DEFAULT_OUTPUT / "dinov3-rendered-crops")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-data-only", action="store_true")
    return parser.parse_args()


def checkpoint_snapshot(hf_home: Path) -> Path:
    snapshot_root = hf_home / "hub" / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots"
    for snapshot in sorted(snapshot_root.glob("*"), reverse=True):
        if (snapshot / "model.safetensors").is_file() and (snapshot / "config.json").is_file():
            return snapshot
    raise FileNotFoundError(
        f"DINOv3 checkpoint not complete under {snapshot_root}; expected model.safetensors and config.json"
    )


class DinoVisualEncoder:
    def __init__(self, snapshot: Path, device: torch.device):
        from transformers import AutoModel

        self.device = device
        self.snapshot = snapshot
        started = time.perf_counter()
        self.model = AutoModel.from_pretrained(
            str(snapshot), torch_dtype=torch.float32, local_files_only=True,
        ).to(device).eval()
        self.load_seconds = time.perf_counter() - started
        self.image_size = int(self.model.config.image_size)
        self.feature_dimension = int(self.model.config.hidden_size)

    @property
    def masked_background_rgb(self) -> tuple[int, int, int]:
        return tuple(int(round(channel * 255)) for channel in IMAGE_MEAN)

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32)[:, None, None]
        std = torch.tensor(IMAGE_STD, dtype=torch.float32)[:, None, None]
        tensors = []
        for image in images:
            resized = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC,
            )
            pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy() / 255.0)
            tensors.append((pixels.permute(2, 0, 1) - mean) / std)
            resized.close()
        batch = torch.stack(tensors).to(device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            output = self.model(pixel_values=batch)
            features = getattr(output, "pooler_output", None)
            if features is None:
                features = output.last_hidden_state[:, 0]
            features = features.float()
            if not torch.isfinite(features).all():
                raise FloatingPointError(
                    "DINOv3 emitted non-finite pre-normalization features; refusing to score associations"
                )
            normalized = F.normalize(features, dim=-1)
            if not torch.isfinite(normalized).all():
                raise FloatingPointError(
                    "DINOv3 emitted non-finite normalized features; refusing to score associations"
                )
        return normalized

    def close(self) -> None:
        del self.model


def run_model(args: argparse.Namespace, rows: list[dict], graph_data: dict,
              snapshot: Path, device: torch.device) -> dict:
    import transformers

    print(f"[dinov3-replica] loading {MODEL_ID} from {snapshot}", flush=True)
    run_started = time.perf_counter()
    encoder = DinoVisualEncoder(snapshot, device)
    resident = torch.cuda.memory_allocated(device) / (1024**3)
    torch.cuda.reset_peak_memory_stats(device)
    conditions = {}
    for condition, path_key in (("ground-truth-padded-box", "box_path"),
                                ("ground-truth-mask-isolated", "masked_path")):
        features, elapsed = shared._encode_condition(
            encoder, rows, path_key, args.batch_size, device,
        )
        metrics = shared._association_and_graph_metrics(
            features, rows, graph_data["relationship_rows"], graph_data["scenes"],
        )
        metrics["image_encode_seconds"] = elapsed
        metrics["object_views_per_second"] = len(rows) / elapsed
        conditions[condition] = metrics
        del features
    run_total_seconds = time.perf_counter() - run_started
    result = {
        "name": "dinov3-vitl16",
        "model_id": MODEL_ID,
        "model_revision": snapshot.name,
        "valid": True,
        "compute_dtype": str(torch.float32),
        "finite_feature_check": "passed for pre-normalized and normalized features in both conditions",
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "feature_source": "final-layer pooler_output, falling back to CLS token; L2-normalized",
        "model_load_seconds": encoder.load_seconds,
        "model_run_total_seconds": run_total_seconds,
        "model_resident_cuda_allocated_gib": resident,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "vram_measurement_scope": "per-process PyTorch CUDA allocated/reserved bytes; excludes other processes",
        "preprocessing": {
            "resize": f"{encoder.image_size}x{encoder.image_size} bicubic",
            "normalization_mean_rgb": list(IMAGE_MEAN),
            "normalization_std_rgb": list(IMAGE_STD),
            "metadata_limitation": "HF processor metadata was absent from local snapshot; preprocessing is explicitly recorded and should be checked against the upstream model card before interpreting absolute scores.",
        },
        "text_retrieval": "not evaluated; DINOv3 visual features have no shared text-image space; this is not DINOde",
        "feature_dimension": encoder.feature_dimension,
        "input_conditions": conditions,
    }
    encoder.close()
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.max_frames_per_scene == 0 or args.max_frames_per_scene < -1:
        raise ValueError("--batch-size must be positive; --max-frames-per-scene must be positive or -1")
    args.hf_home = args.hf_home.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    os.environ["HF_HOME"] = str(args.hf_home)
    snapshot = checkpoint_snapshot(args.hf_home)
    excluded_test_scans = []
    if args.scenes:
        scenes = args.scenes
        split_name = "explicit scene selection"
        requested_test_scenes = None
    else:
        test_split = args.annotations.expanduser().resolve() / "test_scans.txt"
        if not test_split.is_file():
            raise FileNotFoundError(f"ReplicaSSG held-out test split file is missing: {test_split}")
        requested_test_scenes = [line.strip() for line in test_split.read_text().splitlines() if line.strip()]
        image_root = args.image_root.expanduser().resolve()
        scenes = []
        for scene in requested_test_scenes:
            local_scene = image_root / scene.replace("_", "")
            if (local_scene / "results").is_dir() and (local_scene / "traj.txt").is_file():
                scenes.append(scene)
            else:
                excluded_test_scans.append({
                    "scene": scene,
                    "reason": "local RGB result directory or trajectory is missing",
                    "expected_sequence_path": str(local_scene),
                })
        if not scenes:
            raise FileNotFoundError(
                f"No ReplicaSSG test scans have local RGB and pose sequences under {image_root}"
            )
        split_name = str(test_split)
    args.scenes = scenes
    objects, relations, scenes = shared._require_dataset(args)
    if args.check_data_only:
        print(f"DINOv3 checkpoint ready: {snapshot}")
        if requested_test_scenes is not None:
            print(f"Held-out test scans requested: {len(requested_test_scenes)}")
            print(f"Held-out scans available locally: {len(scenes)} ({', '.join(scenes)})")
            print(f"Held-out scans excluded for missing local RGB/poses: {len(excluded_test_scans)}")
            for row in excluded_test_scans:
                print(f"  excluded {row['scene']}: {row['expected_sequence_path']}")
        print(f"ReplicaSSG inputs ready for {len(scenes)} scene(s): {', '.join(scenes)}")
        return 0
    args.crop_cache = args.crop_cache.expanduser().resolve()
    rows, graph_data = shared._render_crop_rows(args, objects, relations, scenes)
    if not rows:
        raise RuntimeError("No held-out ReplicaSSG GT object views were rendered")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("DINOv3 association benchmark requires CUDA; CPU fallback is disabled")
    result = run_model(args, rows, graph_data, snapshot, device)
    payload = {
        "benchmark": "ReplicaSSG visual-only DINOv3 cross-view association and graph identity impact",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "rgb_sequence_root": str(args.image_root.expanduser().resolve()),
            "semantic_mesh_root": str(args.semantic_root.expanduser().resolve()),
            "replicassg_annotations": str(args.annotations.expanduser().resolve()),
            "scenes": graph_data["scene_info"],
            "held_out_test_scans_requested": requested_test_scenes,
            "held_out_test_scans_evaluated": scenes if requested_test_scenes is not None else None,
            "held_out_test_scans_excluded": excluded_test_scans,
            "frames_used": len({(row["scene"], row["frame"]) for row in rows}),
            "object_observations": len(rows),
            "object_categories": len({row["label"] for row in rows}),
            "proposal_source": "Replica semantic-mesh face object IDs raycast at the matching RGB camera poses",
            "label_source": "ReplicaSSG objects.json",
            "relationship_source": "ReplicaSSG relationships.json; predicates held fixed",
            "split": split_name,
        },
        "measurement": {
            "primary": "leave-one-frame-out cross-view object identity retrieval top-1, MRR, Recall@1/5/10",
            "secondary": "scene-graph node identity accuracy and oracle-predicate triplet recall",
            "scope": "visual object identity association only; no text query retrieval, detector evaluation, predicate generation, or end-to-end map acceptance",
            "conditions": "GT padded-box crops and GT-mask-isolated crops, matching replica_clip_benchmark.py",
        },
        "settings": {"batch_size": args.batch_size, "max_frames_per_scene": args.max_frames_per_scene,
                     "device": str(device)},
        "results": [result],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    print(f"[dinov3-replica] wrote {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[dinov3-replica] ERROR: {exc}", file=sys.stderr)
        raise
