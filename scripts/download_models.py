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
import importlib.util
import os
import shutil
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
        "siglip2_base",
        "google/siglip2-base-patch16-224",
        "modernized masked-object features (default mask mode)",
    ),
    (
        "siglip2_so400m",
        "google/siglip2-so400m-patch14-384",
        "modernized masked-object image and text features",
    ),
    (
        "fg_clip2_so400m",
        "qihoo360/fg-clip2-so400m",
        "modernized padded-proposal image and text features",
    ),
    (
        "qwen2_5_vl_7b",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "optional modernized object-annotation benchmark candidate",
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
LEGACY_DEFAULT_KEYS = {
    "qwen3vl", "qwen3vl_thinking", "r1_sgg", "r1_sgg_zero", "clip",
    "llava", "llava_onevision", "sam", "grounding_dino",
}
YOLOE_CHECKPOINTS = {
    "yoloe26l_detect": "yoloe-26l.pt",
    "yoloe26l": "yoloe-26l-seg.pt",
    "yoloe26m": "yoloe-26m-seg.pt",
}
YOLOE_PROMPT_ENCODER = "mobileclip2_b.ts"
YOLOE_MIN_BYTES = 50 * 1024**2


def download_ultralytics_asset(name: str, destination: Path, minimum_bytes: int) -> bool:
    """Resolve a named Ultralytics release asset and copy it to its local target."""
    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except ImportError as exc:
        print(f"Cannot download {name}: installed Ultralytics has no asset downloader: {exc}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved = Path(attempt_download_asset(name)).expanduser().resolve()
        if not resolved.is_file() or resolved.stat().st_size < minimum_bytes:
            print(f"  incomplete Ultralytics asset {name}: {resolved}")
            return False
        if resolved != destination.resolve():
            shutil.copy2(resolved, destination)
        return destination.is_file() and destination.stat().st_size >= minimum_bytes
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"  failed to download {name}: {exc}")
        return False


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


def is_modernized_encoder_ready(key: str) -> bool:
    hub_id = {
        "siglip2_so400m": "google/siglip2-so400m-patch14-384",
        "fg_clip2_so400m": "qihoo360/fg-clip2-so400m",
    }[key]
    root = hub_dir(hub_id) / "snapshots"
    required_tokenizer = (
        ("tokenizer.json", "spiece.model", "tokenizer.model")
        if key == "siglip2_so400m" else
        ("tokenizer.json", "tokenizer_config.json", "vocab.json")
    )
    for snapshot in root.glob("*"):
        weight_files = [
            path for pattern in ("*.safetensors", "*.bin") for path in snapshot.glob(pattern)
            if path.is_file()
        ]
        has_weights = sum(path.stat().st_size for path in weight_files) >= 500 * 1024**2
        has_tokenizer = any((snapshot / name).is_file() for name in required_tokenizer)
        if (
            has_weights and has_tokenizer and
            (snapshot / "config.json").is_file() and
            (snapshot / "preprocessor_config.json").is_file()
        ):
            return True
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
        choices=[key for key, _, _ in HF_MODELS] + [
            "ultralytics", *YOLOE_CHECKPOINTS, "modernized_features"
        ],
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

    selected = [model for model in HF_MODELS if model[0] in LEGACY_DEFAULT_KEYS]
    if args.only:
        requested = set(args.only)
        if "modernized_features" in requested:
            requested.update({"siglip2_base", "fg_clip2_so400m"})
        selected = [m for m in HF_MODELS if m[0] in requested]

    missing = []
    print(f"Local hub: {args.hf_home}\n")
    for key, hub_id, description in selected:
        present = (
            is_modernized_encoder_ready(key)
            if key in {"siglip2_so400m", "fg_clip2_so400m"}
            else is_present(hub_id)
        )
        status = "present" if present else "MISSING"
        print(f"[{status:8s}] {key:16s} {hub_id}\n             {description}")
        if not present:
            missing.append((key, hub_id))

    if args.only is not None and "ultralytics" in args.only:
        weights_dir = Path(os.environ.get("CG_WEIGHTS_DIR", REPO_ROOT / "weights"))
        for name in ULTRALYTICS_WEIGHTS:
            present = (weights_dir / name).exists()
            status = "present" if present else "MISSING"
            print(f"[{status:8s}] {'ultralytics':16s} {weights_dir / name}")

    requested_yoloe = set(args.only or []) & set(YOLOE_CHECKPOINTS)
    yoloe_missing = 0
    if requested_yoloe:
        weights_dir = Path(os.environ.get("CG_WEIGHTS_DIR", REPO_ROOT / "weights"))
        for key in sorted(requested_yoloe):
            name = YOLOE_CHECKPOINTS[key]
            path = weights_dir / name
            present = path.is_file() and path.stat().st_size >= YOLOE_MIN_BYTES
            status = "present" if present else "MISSING"
            print(f"[{status:8s}] {key:16s} {name}: {path}")
            yoloe_missing += int(not present)
        prompt_path = REPO_ROOT / YOLOE_PROMPT_ENCODER
        present = prompt_path.is_file() and prompt_path.stat().st_size >= 100 * 1024**2
        status = "present" if present else "MISSING"
        print(f"[{status:8s}] {'yoloe-prompt':16s} {YOLOE_PROMPT_ENCODER}: {prompt_path}")
        yoloe_missing += int(not present)
        clip_ready = importlib.util.find_spec("clip") is not None
        print(
            f"[{('present' if clip_ready else 'MISSING'):8s}] "
            "yoloe-prompt     project CLIP prompt-tokenizer dependency"
        )
        yoloe_missing += int(not clip_ready)

    if not args.download:
        missing_total = len(missing) + yoloe_missing
        print(f"\n{missing_total} model asset(s) missing - re-run with --download to fetch them.")
        if requested_yoloe and importlib.util.find_spec("clip") is None:
            print("\nYOLOE prompting also requires the declared `clip` dependency:")
            print("  uv sync")
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
    if requested_yoloe and importlib.util.find_spec("clip") is None:
        print("YOLOE prompting requires the declared `clip` dependency; install it with `uv sync`.")
        download_ok = False
    for key, hub_id in missing:
        downloaded = download_hf_model(hub_id)
        complete = (
            is_modernized_encoder_ready(key)
            if key in {"siglip2_so400m", "fg_clip2_so400m"}
            else is_present(hub_id)
        )
        if not downloaded or not complete:
            print(f"  incomplete or unavailable model files for {hub_id}")
            download_ok = False

    if requested_yoloe:
        weights_dir = Path(os.environ.get("CG_WEIGHTS_DIR", REPO_ROOT / "weights"))
        yoloe_targets = [weights_dir / YOLOE_CHECKPOINTS[key] for key in sorted(requested_yoloe)]
        prompt_target = REPO_ROOT / YOLOE_PROMPT_ENCODER
        for yoloe_target in yoloe_targets:
            if not (yoloe_target.is_file() and yoloe_target.stat().st_size >= YOLOE_MIN_BYTES):
                print(f"Downloading {yoloe_target.name} to {yoloe_target} ...", flush=True)
                download_ok &= download_ultralytics_asset(
                    yoloe_target.name, yoloe_target, YOLOE_MIN_BYTES
                )
        if not (prompt_target.is_file() and prompt_target.stat().st_size >= 100 * 1024**2):
            print(f"Downloading {YOLOE_PROMPT_ENCODER} to {prompt_target} ...", flush=True)
            download_ok &= download_ultralytics_asset(
                YOLOE_PROMPT_ENCODER, prompt_target, 100 * 1024**2
            )
        if download_ok:
            print("\nYOLOE checkpoint(s) and offline text-prompt encoder are ready:")
            for yoloe_target in yoloe_targets:
                print(f"  {yoloe_target}")
            print(f"  {prompt_target}")

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
