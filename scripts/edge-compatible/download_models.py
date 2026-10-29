#!/usr/bin/env python3
"""Download selected edge-compatible GGUF models and vision projectors.

This is a preview by default. Pass ``--download`` to fetch the selected files.
It does not change the existing Hugging Face cache or remove any files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone


DEFAULT_HF_HOME = Path(
    os.environ.get("HF_HOME", "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface")
)
DEFAULT_OUTPUT = DEFAULT_HF_HOME / "edge-compatible-gguf"

MODELS = {
    "qwen35-9b-q4": {
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "file": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        "mmproj": "mmproj-F16.gguf",
        "model_bytes": 5_970_000_000,
        "mmproj_bytes": 918_000_000,
        "note": "recommended first comparison; lower VRAM than Q8",
    },
    "qwen35-9b-q8": {
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "file": "Qwen3.5-9B-UD-Q8_K_XL.gguf",
        "mmproj": "mmproj-F16.gguf",
        "model_bytes": 13_000_000_000,
        "mmproj_bytes": 918_000_000,
        "note": "higher precision, larger download and VRAM use",
    },
    "gemma4-12b": {
        "repo": "unsloth/gemma-4-12B-it-qat-GGUF",
        "file": "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        "mmproj": "mmproj-F16.gguf",
        "model_bytes": 6_720_000_000,
        "mmproj_bytes": 175_000_000,
        "note": "compact QAT model; includes image support through projector",
    },
    "gemma4-26b-a4b": {
        "repo": "unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        "file": "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        "mmproj": "mmproj-F16.gguf",
        "model_bytes": 14_200_000_000,
        "mmproj_bytes": 1_190_000_000,
        "note": "larger MoE candidate; verify full GPU fit at runtime",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*MODELS, "all"],
        required=True,
        help="one or more model ids, or 'all' (large: about 43 GB total)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("CG_EDGE_MODEL_DIR", DEFAULT_OUTPUT)),
        help=f"download destination (default: {os.environ.get('CG_EDGE_MODEL_DIR', DEFAULT_OUTPUT)})",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="perform downloads; without this flag only show files and disk requirements",
    )
    return parser.parse_args()


def selected_models(names: list[str]) -> list[tuple[str, dict]]:
    if "all" in names:
        if len(names) != 1:
            raise ValueError("'all' cannot be combined with individual model ids")
        names = list(MODELS)
    return [(name, MODELS[name]) for name in names]


def main() -> int:
    args = parse_args()
    try:
        models = selected_models(args.models)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    destination = args.output_dir.expanduser().resolve()
    required = sum(info["model_bytes"] + info["mmproj_bytes"] for _, info in models)
    print(f"Destination: {destination}")
    for name, info in models:
        print(f"  {name}: {info['repo']} / {info['file']} + {info['mmproj']} ({info['note']})")
    print(f"Approximate total: {required / 1_000_000_000:.2f} GB")
    if not args.download:
        print("Preview only. Add --download to fetch the selected files.")
        return 0

    destination.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination).free
    missing_bytes = 0
    for name, info in models:
        model_dir = destination / name
        for filename, file_bytes in (
            (info["file"], info["model_bytes"]),
            (info["mmproj"], info["mmproj_bytes"]),
        ):
            if not (model_dir / filename).is_file():
                missing_bytes += file_bytes
    if free_bytes < missing_bytes:
        print(
            f"error: destination has {free_bytes / 1_000_000_000:.2f} GB free; "
            f"missing selected files need approximately {missing_bytes / 1_000_000_000:.2f} GB",
            file=sys.stderr,
        )
        return 2

    # The user's general environment is intentionally offline. This explicit
    # downloader opts into network access only for this command.
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        print(
            "error: huggingface_hub is unavailable; run this with the project "
            "Python environment (.venv/bin/python)",
            file=sys.stderr,
        )
        raise exc

    api = HfApi(token=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }
    for name, info in models:
        print(f"\nResolving {info['repo']}...")
        revision = api.model_info(info["repo"], revision="main").sha
        model_dir = destination / name
        model_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for filename in (info["file"], info["mmproj"]):
            print(f"Downloading {filename} at {revision}...")
            paths[filename] = str(
                hf_hub_download(
                    repo_id=info["repo"],
                    filename=filename,
                    revision=revision,
                    local_dir=model_dir,
                    token=False,
                    local_files_only=False,
                )
            )
        manifest["models"][name] = {
            "repo": info["repo"],
            "revision": revision,
            "model_file": paths[info["file"]],
            "mmproj_file": paths[info["mmproj"]],
        }
        print(f"Ready: {model_dir}")

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nDownload manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
