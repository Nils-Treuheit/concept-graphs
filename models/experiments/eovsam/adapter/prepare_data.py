#!/usr/bin/env python3
"""Extract and verify only the frozen 500 COCO panoptic validation PNGs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image
from io import BytesIO


ROOT = Path(__file__).resolve().parents[4]
COCO_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/coco")
SAMPLE_MANIFEST = ROOT / "scripts/pipeline_model_reports/data/yoloe26/coco-val2017-500-manifest.json"
PANOPTIC_JSON = COCO_ROOT / "annotations/panoptic_val2017.json"
PANOPTIC_ZIP = COCO_ROOT / "annotations/annotations/panoptic_val2017.zip"
IMAGE_ROOT = COCO_ROOT / "val2017"
OUTPUT_ROOT = Path(__file__).resolve().parent / "data/panoptic_val2017"
OUTPUT_MANIFEST = Path(__file__).resolve().parent / "data/coco500-extracted-manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    selected = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    image_ids = [int(value) for value in selected["sample_image_ids"]]
    if len(image_ids) != 500 or len(set(image_ids)) != 500 or image_ids != sorted(image_ids):
        raise ValueError("frozen sample must contain 500 unique, sorted image IDs")
    source = json.loads(PANOPTIC_JSON.read_text(encoding="utf-8"))
    image_rows = {int(row["id"]): row for row in source["images"]}
    annotations = {int(row["image_id"]): row for row in source["annotations"]}
    if set(image_ids) - image_rows.keys() or set(image_ids) - annotations.keys():
        raise ValueError("panoptic source JSON does not cover every frozen image ID")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(PANOPTIC_ZIP) as archive:
        names = set(archive.namelist())
        for image_id in image_ids:
            annotation = annotations[image_id]
            member = f"panoptic_val2017/{annotation['file_name']}"
            if member not in names:
                raise FileNotFoundError(f"missing exact ZIP member {member}")
            data = archive.read(member)
            with Image.open(BytesIO(data)) as mask:
                mask.load()
                size = [int(mask.width), int(mask.height)]
            image = image_rows[image_id]
            expected = [int(image["width"]), int(image["height"])]
            if size != expected:
                raise ValueError(f"panoptic dimension mismatch for {image_id}: {size} != {expected}")
            image_path = IMAGE_ROOT / image["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(f"missing image for {image_id}: {image_path}")
            destination = OUTPUT_ROOT / annotation["file_name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            if sha256_bytes(destination.read_bytes()) != sha256_bytes(data):
                raise IOError(f"extracted PNG hash mismatch for {image_id}")
            extracted.append({
                "image_id": image_id,
                "zip_member": member,
                "output": str(destination.resolve()),
                "sha256": sha256_bytes(data),
                "dimensions": size,
            })

    if len(extracted) != 500:
        raise RuntimeError(f"expected 500 extracted masks, got {len(extracted)}")
    report = {
        "status": "extracted_and_verified",
        "sample_manifest": str(SAMPLE_MANIFEST.resolve()),
        "sample_manifest_sha256": sha256_bytes(SAMPLE_MANIFEST.read_bytes()),
        "panoptic_json": str(PANOPTIC_JSON.resolve()),
        "panoptic_json_sha256": sha256_bytes(PANOPTIC_JSON.read_bytes()),
        "panoptic_zip": str(PANOPTIC_ZIP.resolve()),
        "image_count": len(extracted),
        "zip_prefix": "panoptic_val2017/",
        "extracted": extracted,
    }
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"verified {len(extracted)} panoptic PNGs -> {OUTPUT_ROOT}")
    print(f"manifest: {OUTPUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
