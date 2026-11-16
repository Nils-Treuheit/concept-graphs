#!/usr/bin/env python3
"""Cache the missing official DINOv3 ViT-L processor config at a pinned revision.

This downloads only ``preprocessor_config.json`` into the Hugging Face cache.
It never writes model assets under this repository or under ``/tmp``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


REPOSITORY = "facebook/dinov3-vitl16-pretrain-lvd1689m"
REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
FILENAME = "preprocessor_config.json"
DEFAULT_HF_HOME = Path(
    "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", DEFAULT_HF_HOME)),
        help="Hugging Face cache root on the HDD",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch the pinned processor file even if cached",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hf_home = args.hf_home.expanduser().resolve()
    cache_dir = hf_home / "hub"
    expected_snapshot = (
        cache_dir
        / "models--facebook--dinov3-vitl16-pretrain-lvd1689m"
        / "snapshots"
        / REVISION
    )
    existing = expected_snapshot / FILENAME

    if existing.is_file() and not args.force:
        path = existing
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit(
                "huggingface_hub is required; use the project's existing Python environment"
            ) from exc
        path = Path(
            hf_hub_download(
                repo_id=REPOSITORY,
                filename=FILENAME,
                revision=REVISION,
                cache_dir=str(cache_dir),
                force_download=args.force,
            )
        )

    resolved = path.resolve(strict=True)
    if path.name != FILENAME or not path.is_file():
        raise SystemExit(f"Unexpected download result: {path}")
    if not resolved.is_relative_to(cache_dir.resolve()):
        raise SystemExit(f"Refusing a result outside the Hugging Face cache: {resolved}")
    if path.parent != expected_snapshot:
        raise SystemExit(
            f"Expected the pinned snapshot path {expected_snapshot / FILENAME}, got {path}"
        )

    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("image_processor_type") != "DINOv3ViTImageProcessorFast":
        raise SystemExit("The cached file is not the expected DINOv3 image processor config")
    expected_mean = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    if config.get("image_mean") != expected_mean or config.get("image_std") != expected_std:
        raise SystemExit("DINOv3 processor normalization does not match the official config")
    if config.get("size") != {"height": 224, "width": 224}:
        raise SystemExit("DINOv3 processor size does not match the pinned official config")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"Verified {REPOSITORY}@{REVISION}/{FILENAME}")
    print(f"Path: {path}")
    print(f"Bytes: {path.stat().st_size}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
