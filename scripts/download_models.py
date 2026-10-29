#!/usr/bin/env python3
"""Download / verify the models that the local ConceptGraphs pipeline needs.

All models are read from the **local** Hugging Face hub
(``HF_HOME=/mnt/HDD1/unsloth_and_hugging_face_models/huggingface``), nothing
is sent to a cloud API.  The script

* checks which of the models are already present in the local hub,
* downloads the missing ones (``--download``),
* downloads the ultraLytics checkpoints (YOLO-World, SAM) into ``weights/``.

Examples
--------
    # only show what is missing
    python scripts/download_models.py

    # download everything that is missing
    python scripts/download_models.py --download

    # only the scene-graph model and the CLIP weights
    python scripts/download_models.py --download --only r1_sgg clip
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_HOME = "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface"
MIN_HUB_WEIGHT_BYTES = 1024 * 1024
MIN_CLIP_WEIGHT_BYTES = 1024**3
CLIP_HUB_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# (key, hub id, description)
HF_MODELS = [
    (
        "qwen3vl",
        "Qwen/Qwen3-VL-8B-Instruct",
        "vision-language model for the scene-graph stage (served with vLLM)",
    ),
    (
        "qwen3vl_thinking",
        "Qwen/Qwen3-VL-8B-Thinking",
        "vision-language model with thinking mode (optional)",
    ),
    (
        "r1_sgg",
        "JosephZ/R1-SGG-7B",
        "scene-graph specialised model (relations between the objects)",
    ),
    (
        "r1_sgg_zero",
        "JosephZ/R1-SGG-Zero-7B",
        "scene-graph model without the scene-graph specific training",
    ),
    (
        "clip",
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "open-vocabulary CLIP features (ViT-H-14, laion2b_s32b_b79k)",
    ),
    (
        "llava",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "LLaVA for captioning the cropped objects (Hugging Face format)",
    ),
    (
        "llava_onevision",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "LLaVA OneVision (alternative to the Mistral checkpoint)",
    ),
    (
        "sam",
        "facebook/sam-vit-huge",
        "original Segment-Anything checkpoint (sam_vit_h_4b8939.pth)",
    ),
    (
        "grounding_dino",
        "IDEA-Research/grounding-dino-base",
        "Grounding DINO for the (optional) Grounded-Segment-Anything path",
    ),
]

# ultraLytics checkpoints -> weights/<name>
ULTRALYTICS_WEIGHTS = ["yolov8l-world.pt", "mobile_sam.pt", "sam_l.pt"]


def hub_dir(hub_id: str) -> Path:
    """Path of a model inside the local hub."""
    folder = "models--" + hub_id.replace("/", "--")
    return Path(os.environ["HF_HOME"]) / "hub" / folder


def is_present(hub_id: str) -> bool:
    folder = hub_dir(hub_id)
    if not folder.exists():
        return False
    minimum_size = MIN_CLIP_WEIGHT_BYTES if hub_id == CLIP_HUB_ID else MIN_HUB_WEIGHT_BYTES
    # Require a resolvable model-sized file. Broken HF cache symlinks must not
    # count as downloaded weights.
    for snapshot in (folder / "snapshots").glob("*"):
        for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
            for checkpoint in snapshot.glob(pattern):
                try:
                    resolved = checkpoint.resolve(strict=True)
                    if resolved.is_file() and resolved.stat().st_size >= minimum_size:
                        return True
                except OSError:
                    continue
    return False


def clip_checkpoint_path(hf_home: str | Path) -> Path | None:
    folder = Path(hf_home) / "hub" / f"models--{CLIP_HUB_ID.replace('/', '--')}" / "snapshots"
    candidates = sorted(
        folder.glob("*/open_clip_model.safetensors"),
        key=lambda path: path.parent.stat().st_mtime if path.parent.exists() else 0,
        reverse=True,
    )
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            if resolved.is_file() and resolved.stat().st_size >= MIN_CLIP_WEIGHT_BYTES:
                return resolved
        except OSError:
            continue
    return None


def download_hf_model(hub_id: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed - run `uv add huggingface_hub`.")
        return False

    print(f"Downloading {hub_id} ...", flush=True)
    try:
        snapshot_download(repo_id=hub_id, max_workers=4)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"  failed: {exc}")
        return False
    print("  done.")
    return True


def download_ultralytics_weights() -> None:
    weights_dir = Path(os.environ.get("CG_WEIGHTS_DIR", REPO_ROOT / "weights"))
    weights_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import SAM, YOLO

    for name in ULTRALYTICS_WEIGHTS:
        target = weights_dir / name
        if target.exists():
            print(f"{name} is already in {weights_dir}")
            continue
        print(f"Downloading {name} ...", flush=True)
        if name.endswith(".pt") and "sam" in name:
            SAM(str(target))
        else:
            YOLO(str(target))
        print("  done.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the models that are missing (otherwise only report)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=[key for key, _, _ in HF_MODELS] + ["ultralytics"],
        help="restrict the check to these models",
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", DEFAULT_HF_HOME),
        help="local Hugging Face hub",
    )
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_home
    os.environ.setdefault("HF_HUB_OFFLINE", "0")

    selected = HF_MODELS
    if args.only:
        selected = [m for m in HF_MODELS if m[0] in args.only]

    missing = []
    print(f"Local hub: {args.hf_home}\n")
    for key, hub_id, description in selected:
        present = is_present(hub_id)
        status = "present" if present else "MISSING"
        print(f"[{status:8s}] {key:16s} {hub_id}\n             {description}")
        if not present:
            missing.append((key, hub_id))

    if "ultralytics" in (args.only or [key for key, _, _ in HF_MODELS]):
        weights_dir = Path(os.environ.get("CG_WEIGHTS_DIR", REPO_ROOT / "weights"))
        for name in ULTRALYTICS_WEIGHTS:
            present = (weights_dir / name).exists()
            status = "present" if present else "MISSING"
            print(f"[{status:8s}] {'ultralytics':16s} {weights_dir / name}")

    if not args.download:
        print(f"\n{len(missing)} model(s) missing - re-run with --download to fetch them.")
        if any(key == "clip" for key, _, _ in selected) and not is_present(CLIP_HUB_ID):
            print("\nFor the local detector's ViT-H-14 checkpoint, run:")
            print("  HF_HUB_OFFLINE=0 uv run python scripts/download_models.py --download --only clip")
            print("Then either let the detector discover the HF cache automatically or set:")
            print(
                "  export CONCEPTGRAPH_CLIP_H14_CHECKPOINT=\""
                f"{args.hf_home}/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/"
                "snapshots/<revision>/open_clip_model.safetensors\""
            )
        return 0

    print()
    download_ok = True
    for key, hub_id in missing:
        downloaded = download_hf_model(hub_id)
        if not downloaded or not is_present(hub_id):
            print(f"  incomplete or unavailable model files for {hub_id}")
            download_ok = False

    if any(key == "clip" for key, _, _ in selected):
        checkpoint = clip_checkpoint_path(args.hf_home)
        if checkpoint is None:
            print("\nViT-H-14 checkpoint is not complete in the local HF cache.")
            print("  HF_HUB_OFFLINE=0 uv run python scripts/download_models.py --download --only clip")
            download_ok = False
        else:
            print("\nUse this resolved local ViT-H-14 checkpoint:")
            print(f"  export CONCEPTGRAPH_CLIP_H14_CHECKPOINT=\"{checkpoint}\"")

    if not args.only or "ultralytics" in args.only:
        download_ultralytics_weights()

    return 0 if download_ok else 1


if __name__ == "__main__":
    sys.exit(main())
