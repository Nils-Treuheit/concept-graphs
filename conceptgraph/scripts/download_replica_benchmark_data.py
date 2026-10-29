#!/usr/bin/env python3
"""Download Replica semantic assets and ReplicaSSG graph annotations.

By default, data are placed in
``/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark`` and the user's
existing ``Replica`` directory is left untouched. The official Replica archive
is large; downloads resume when rerun after an interruption.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_OUTPUT = Path("/mnt/HDD1/Project_Data/public_datasets/Replica_benchmark")
REPLICA_RELEASE = "https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0"
REPLICASSG_API = "https://api.github.com/repos/Howardkhh/ReplicaSSG/commits/main"
REPLICASSG_ARCHIVE = "https://codeload.github.com/Howardkhh/ReplicaSSG/zip/{ref}"
SCENES = (
    "apartment_0",
    "apartment_1",
    "apartment_2",
    "frl_apartment_0",
    "frl_apartment_1",
    "frl_apartment_2",
    "frl_apartment_3",
    "frl_apartment_4",
    "frl_apartment_5",
    "hotel_0",
    "office_0",
    "office_1",
    "office_2",
    "office_3",
    "office_4",
    "room_0",
    "room_1",
    "room_2",
)
REQUIRED_ANNOTATIONS = (
    "objects.json",
    "relationships.json",
    "replica_to_visual_genome.json",
    "trajectories.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download official Replica semantic/instance meshes and ReplicaSSG "
            "object/relation annotations for the CLIP component benchmark."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download ReplicaSSG annotations only; skip the large Replica archive.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the multipart Replica archive after successful extraction.",
    )
    return parser.parse_args()


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "conceptgraph-replica-benchmark"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download_annotations(output_root: Path) -> dict:
    annotations_dir = output_root / "ReplicaSSG" / "files"
    if all((annotations_dir / name).is_file() for name in REQUIRED_ANNOTATIONS):
        print(f"ReplicaSSG annotations already present: {annotations_dir}")
        return {"path": str(annotations_dir), "status": "already_present"}
    if annotations_dir.exists():
        raise RuntimeError(
            f"Found an incomplete annotation directory at {annotations_dir}. "
            "Move it aside or remove it, then rerun."
        )

    revision = fetch_json(REPLICASSG_API)["sha"]
    archive_url = REPLICASSG_ARCHIVE.format(ref=revision)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="replicassg-") as temp_name:
        archive_path = Path(temp_name) / "replicassg.zip"
        request = urllib.request.Request(archive_url, headers={"User-Agent": "conceptgraph-replica-benchmark"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, archive_path.open("wb") as target:
                shutil.copyfileobj(response, target)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Could not download ReplicaSSG revision {revision}: {exc}") from exc

        staging = output_root / ".ReplicaSSG-files.incomplete"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            members = [
                item
                for item in archive.infolist()
                if "/files/" in item.filename and not item.is_dir()
            ]
            for item in members:
                relative = Path(item.filename.split("/files/", 1)[1])
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)

        missing = [name for name in REQUIRED_ANNOTATIONS if not (staging / name).is_file()]
        if missing:
            shutil.rmtree(staging)
            raise RuntimeError(
                "The downloaded ReplicaSSG archive did not contain required files: "
                + ", ".join(missing)
            )
        annotations_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(annotations_dir)

    print(f"Downloaded ReplicaSSG annotations at revision {revision}: {annotations_dir}")
    return {"path": str(annotations_dir), "status": "downloaded", "revision": revision, "url": archive_url}


def validate_replica(root: Path) -> list[str]:
    missing = []
    for scene in SCENES:
        scene_root = root / scene
        for relative in ("habitat/mesh_semantic.ply", "habitat/info_semantic.json"):
            if not (scene_root / relative).is_file():
                missing.append(f"{scene}/{relative}")
    return missing


def download_replica(output_root: Path, keep_archives: bool) -> dict:
    replica_dir = output_root / "replica_v1"
    missing = validate_replica(replica_dir)
    if not missing:
        print(f"Official Replica semantic assets already present: {replica_dir}")
        return {"path": str(replica_dir), "status": "already_present"}
    if replica_dir.exists():
        raise RuntimeError(
            f"Found an incomplete Replica directory at {replica_dir}; first missing asset is "
            f"{missing[0]}. Move it aside or remove it, then rerun."
        )

    curl = shutil.which("curl")
    tar = shutil.which("tar")
    if not curl or not tar:
        raise RuntimeError("Downloading Replica requires curl and tar on PATH.")

    output_root.mkdir(parents=True, exist_ok=True)
    parts_dir = output_root / ".replica-download-parts"
    parts_dir.mkdir(exist_ok=True)
    parts = [parts_dir / f"replica_v1_0.tar.gz.parta{letter}" for letter in "abcdefghijklmnopq"]
    for part_path in parts:
        url = f"{REPLICA_RELEASE}/{part_path.name}"
        print(f"Downloading/resuming {part_path.name}")
        subprocess.run(
            [curl, "--fail", "--location", "--retry", "5", "--retry-all-errors", "--continue-at", "-", "--output", str(part_path), url],
            check=True,
        )

    staging = output_root / ".replica_v1.incomplete"
    staging.mkdir(exist_ok=True)
    cat = shutil.which("cat")
    if not cat:
        raise RuntimeError("Extracting the multipart Replica archive requires cat on PATH.")
    print("Extracting the official Replica archive; this can take a while.")
    with (staging / ".replica-extraction.log").open("wb") as log:
        cat_process = subprocess.Popen([cat, *(str(part) for part in parts)], stdout=subprocess.PIPE)
        try:
            extract_process = subprocess.run(
                [tar, "-xzf", "-", "-C", str(staging)],
                stdin=cat_process.stdout,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finally:
            if cat_process.stdout:
                cat_process.stdout.close()
        cat_status = cat_process.wait()
    if extract_process.returncode or cat_status:
        raise RuntimeError(
            f"Replica extraction failed (tar={extract_process.returncode}, cat={cat_status}); "
            f"partial files and resumable parts remain under {output_root}."
        )

    # The upstream archive may contain a wrapper directory. Find the extracted
    # scene root by checking the expected semantic assets before promoting it.
    candidate = staging
    if validate_replica(candidate):
        nested = [path for path in staging.iterdir() if path.is_dir() and not path.name.startswith(".")]
        valid_nested = [path for path in nested if not validate_replica(path)]
        if len(valid_nested) == 1:
            candidate = valid_nested[0]
    missing = validate_replica(candidate)
    if missing:
        raise RuntimeError(
            "Replica archive extracted, but expected semantic assets were not found "
            f"(first missing: {missing[0]}). Extraction log: {staging / '.replica-extraction.log'}"
        )

    if candidate == staging:
        staging.rename(replica_dir)
    else:
        candidate.rename(replica_dir)
        shutil.rmtree(staging)
    if not keep_archives:
        shutil.rmtree(parts_dir)
    else:
        print(f"Keeping compressed download parts at {parts_dir}")
    print(f"Replica semantic assets are ready: {replica_dir}")
    return {"path": str(replica_dir), "status": "downloaded", "scenes": list(SCENES)}


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    try:
        annotations = download_annotations(output_root)
        replica = None if args.annotations_only else download_replica(output_root, args.keep_archives)
    except (OSError, subprocess.CalledProcessError, urllib.error.URLError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    manifest = {
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "replica": replica,
        "replicassg": annotations,
        "note": "ReplicaSSG labels object identities and relationships; original Replica supplies semantic/instance meshes.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "benchmark_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Download manifest: {manifest_path}")
    print("Benchmark inputs are in place. No rendering or model inference was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
