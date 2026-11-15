#!/usr/bin/env python3
"""Register copied COCO500 runtime records with upstream EOVSAM category maps."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EOVSAM_ROOT = Path(__file__).resolve().parents[1] / "repo"
COCO_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/coco")
SAMPLE_MANIFEST = ROOT / "scripts/pipeline_model_reports/data/yoloe26/coco-val2017-500-manifest.json"
PANOPTIC_JSON = COCO_ROOT / "annotations/panoptic_val2017.json"
IMAGE_ROOT = COCO_ROOT / "val2017"
PANOPTIC_ROOT = Path(__file__).resolve().parent / "data/panoptic_val2017"
EXTRACTED_MANIFEST = Path(__file__).resolve().parent / "data/coco500-extracted-manifest.json"
DATASET_NAME = "eovsam_coco_val2017_500_seed42"


def _sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_and_validate_sources():
    if not EXTRACTED_MANIFEST.is_file():
        raise RuntimeError("run prepare_data.py first; extracted-mask manifest is missing")
    extraction = json.loads(EXTRACTED_MANIFEST.read_text(encoding="utf-8"))
    if extraction.get("status") != "extracted_and_verified" or extraction.get("image_count") != 500:
        raise RuntimeError("500-image extraction manifest is incomplete")
    sample = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    image_ids = [int(value) for value in sample["sample_image_ids"]]
    if len(image_ids) != 500 or len(set(image_ids)) != 500 or image_ids != sorted(image_ids):
        raise ValueError("frozen sample must contain 500 unique, sorted image IDs")
    source = json.loads(PANOPTIC_JSON.read_text(encoding="utf-8"))
    image_rows = {int(row["id"]): row for row in source["images"]}
    annotation_rows = {int(row["image_id"]): row for row in source["annotations"]}
    if set(image_ids) - image_rows.keys() or set(image_ids) - annotation_rows.keys():
        raise ValueError("panoptic JSON is missing frozen sample records")
    if set(int(row["image_id"]) for row in source["annotations"] if int(row["image_id"]) in set(image_ids)) != set(image_ids):
        raise ValueError("selected image-to-annotation IDs do not round-trip")
    return image_ids, image_rows, annotation_rows, source, extraction


def _load_upstream_categories_and_metadata():
    """Load the upstream category source without executing maft package hooks.

    Importing ``maft.data.datasets`` executes registrations for unrelated
    training datasets. This adapter loads the exact upstream category module
    directly, then applies the same ordered mapping construction as the
    upstream ``get_metadata`` function. It does not create or register any
    upstream dataset.
    """
    module_path = EOVSAM_ROOT / "maft/data/datasets/openseg_classes.py"
    previous_cwd = Path.cwd()
    os.chdir(EOVSAM_ROOT)
    try:
        spec = importlib.util.spec_from_file_location("eovsam_openseg_classes", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load upstream category module: {module_path}")
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        panoptic_categories = upstream.get_coco_categories_with_prompt_eng()
        semantic_categories = upstream.get_coco_stuff_categories_with_prompt_eng()
    finally:
        os.chdir(previous_cwd)

    thing_ids = [int(row["id"]) for row in panoptic_categories if int(row["isthing"]) == 1]
    metadata = {
        "thing_classes": [row["name"] for row in panoptic_categories if int(row["isthing"]) == 1],
        "thing_colors": [row["color"] for row in panoptic_categories if int(row["isthing"]) == 1],
        "stuff_classes": [row["name"] for row in panoptic_categories],
        "stuff_colors": [row["color"] for row in panoptic_categories],
        "sem_stuff_classes": [row["name"] for row in semantic_categories],
        "thing_dataset_id_to_contiguous_id": {
            int(row["id"]): index for index, row in enumerate(panoptic_categories)
            if int(row["isthing"]) == 1
        },
        # This mirrors upstream get_metadata(): every panoptic category gets
        # an index in the semantic/stuff map, including things.
        "stuff_dataset_id_to_contiguous_id": {
            int(row["id"]): index for index, row in enumerate(panoptic_categories)
        },
        "contiguous_id_to_class_name": [row["name"] for row in panoptic_categories],
    }
    return panoptic_categories, metadata


def register_dataset():
    os.environ.setdefault("DETECTRON2_DATASETS", str(COCO_ROOT.parent))
    from detectron2.data import DatasetCatalog, MetadataCatalog

    image_ids, image_rows, annotation_rows, source, extraction_manifest = _load_and_validate_sources()
    COCO_CATEGORIES_pan, metadata = _load_upstream_categories_and_metadata()
    source_categories = {int(row["id"]): int(row["isthing"]) for row in source["categories"]}
    upstream_categories = {int(row["id"]): int(row["isthing"]) for row in COCO_CATEGORIES_pan}
    if source_categories != upstream_categories:
        raise ValueError("full COCO panoptic ID/isthing taxonomy differs from upstream EOVSAM")

    thing_map = {int(key): int(value) for key, value in metadata["thing_dataset_id_to_contiguous_id"].items()}
    stuff_map = {int(key): int(value) for key, value in metadata["stuff_dataset_id_to_contiguous_id"].items()}
    thing_ids = {identifier for identifier, flag in source_categories.items() if flag == 1}
    stuff_ids = set(source_categories)
    if set(thing_map) != thing_ids:
        raise ValueError("upstream thing map domain does not match COCO thing IDs")
    # EOVSAM's upstream metadata intentionally places every COCO category in
    # its stuff map for the semantic path, including thing IDs.
    if set(stuff_map) != stuff_ids:
        raise ValueError("upstream stuff map domain does not match full COCO ID set")

    extracted_rows = {int(row["image_id"]): row for row in json.loads(EXTRACTED_MANIFEST.read_text(encoding="utf-8"))["extracted"]}
    if set(extracted_rows) != set(image_ids):
        raise ValueError("extracted PNG IDs differ from the frozen sample")
    records = []
    for image_id in image_ids:
        image = image_rows[image_id]
        annotation = annotation_rows[image_id]
        extracted_row = extracted_rows[image_id]
        image_path = IMAGE_ROOT / image["file_name"]
        mask_path = PANOPTIC_ROOT / annotation["file_name"]
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"missing image or extracted panoptic PNG for {image_id}")
        segments = copy.deepcopy(annotation["segments_info"])
        for segment in segments:
            official_id = int(segment["category_id"])
            if official_id in thing_map:
                segment["category_id"] = thing_map[official_id]
                segment["isthing"] = True
            else:
                segment["category_id"] = stuff_map[official_id]
                segment["isthing"] = False
        records.append({
            "file_name": str(image_path.resolve()),
            "image_id": image_id,
            "pan_seg_file_name": str(mask_path.resolve()),
            "segments_info": segments,
        })

    if DATASET_NAME in DatasetCatalog.list():
        raise RuntimeError(f"dataset name already registered: {DATASET_NAME}")
    DatasetCatalog.register(DATASET_NAME, lambda: copy.deepcopy(records))
    MetadataCatalog.get(DATASET_NAME).set(
        **metadata,
        image_root=str(IMAGE_ROOT.resolve()),
        panoptic_root=str(PANOPTIC_ROOT.resolve()),
        panoptic_json=str(PANOPTIC_JSON.resolve()),
        json_file=str((COCO_ROOT / "annotations/instances_val2017.json").resolve()),
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        dataset_name=DATASET_NAME,
        sample_manifest=str(SAMPLE_MANIFEST.resolve()),
        sample_manifest_sha256=extraction_manifest["sample_manifest_sha256"],
    )
    return {
        "dataset_name": DATASET_NAME,
        "image_ids": image_ids,
        "source_category_mapping_sha256": _sha256_json(source_categories),
        "thing_map": thing_map,
        "stuff_map": stuff_map,
        "thing_map_sha256": _sha256_json(thing_map),
        "stuff_map_sha256": _sha256_json(stuff_map),
        "category_id_isthing_mapping_sha256": _sha256_json(
            [[identifier, source_categories[identifier]] for identifier in sorted(source_categories)]
        ),
        "source_annotation_sha256": extraction_manifest["panoptic_json_sha256"],
        "sample_manifest_sha256": extraction_manifest["sample_manifest_sha256"],
        "extracted_png_manifest": str(EXTRACTED_MANIFEST.resolve()),
    }
