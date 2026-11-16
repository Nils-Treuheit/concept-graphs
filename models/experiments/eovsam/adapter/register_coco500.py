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
SUBSET_JSON = Path(__file__).resolve().parent / "data/coco500-panoptic-subset.json"
SUBSET_MANIFEST = Path(__file__).resolve().parent / "data/coco500-panoptic-subset-manifest.json"
DATASET_NAME = "eovsam_coco_val2017_500_seed42"
PANOPTIC_ZIP_PREFIX = "panoptic_val2017/"
_HEX_DIGITS = frozenset("0123456789abcdef")


def _sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_extracted_pngs(extraction):
    """Verify extracted-mask manifest entries against the real PNG files.

    Entries written by ``prepare_data.py`` carry ``image_id``, ``zip_member``,
    ``output``, ``sha256`` and ``dimensions``; they have no ``file_name``
    field. The mask basename is derived from ``zip_member``: the exact
    ``panoptic_val2017/`` prefix is required and the remainder must be a
    single basename. ``output`` must resolve to the expected local PNG under
    ``data/panoptic_val2017/`` with that basename, the file must exist, and
    its bytes must match the entry's ``sha256`` digest. Returns
    ``{image_id: mask_basename}``.
    """
    if not isinstance(extraction, dict) or extraction.get("zip_prefix") != PANOPTIC_ZIP_PREFIX:
        raise ValueError(f"extraction manifest zip_prefix is not {PANOPTIC_ZIP_PREFIX!r}")
    extracted = extraction.get("extracted")
    if not isinstance(extracted, list) or len(extracted) != 500:
        raise ValueError("extraction manifest must contain exactly 500 extracted entries")
    basenames = {}
    seen_basenames = set()
    for entry in extracted:
        if not isinstance(entry, dict):
            raise ValueError("extraction manifest entry is not an object")
        image_id = entry.get("image_id")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError("extracted entry image_id must be an integer")
        zip_member = entry.get("zip_member")
        if not isinstance(zip_member, str) or not zip_member.startswith(PANOPTIC_ZIP_PREFIX):
            raise ValueError(
                f"extracted entry {image_id} zip_member lacks the exact "
                f"{PANOPTIC_ZIP_PREFIX!r} prefix: {zip_member!r}"
            )
        basename = zip_member[len(PANOPTIC_ZIP_PREFIX):]
        if not basename or "/" in basename:
            raise ValueError(f"extracted entry {image_id} zip_member has no single basename: {zip_member!r}")
        output = entry.get("output")
        if not isinstance(output, str) or not output:
            raise ValueError(f"extracted entry {image_id} has no output path")
        expected_path = (PANOPTIC_ROOT / basename).resolve()
        if not Path(output).is_absolute() or Path(output).resolve() != expected_path:
            raise ValueError(
                f"extracted entry {image_id} output does not resolve to the "
                f"expected local PNG under data/panoptic_val2017: {output}"
            )
        if not expected_path.is_file():
            raise FileNotFoundError(f"missing extracted panoptic PNG for {image_id}: {expected_path}")
        entry_sha = entry.get("sha256")
        if not isinstance(entry_sha, str) or len(entry_sha) != 64 or not set(entry_sha) <= _HEX_DIGITS:
            raise ValueError(f"extracted entry {image_id} sha256 is not a 64-character lowercase hex digest")
        if _sha256_file(expected_path) != entry_sha:
            raise ValueError(f"extracted panoptic PNG bytes do not match sha256 for {image_id}")
        if image_id in basenames:
            raise ValueError(f"extraction manifest has duplicate entries for image {image_id}")
        if basename in seen_basenames:
            raise ValueError(f"extraction manifest has duplicate mask basenames: {basename}")
        basenames[image_id] = basename
        seen_basenames.add(basename)
    return basenames


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


def _validate_panoptic_subset(image_ids, image_rows, annotation_rows, source, extracted_basenames, sample_sha, source_sha, extraction_sha):
    """Fail closed unless the derived 500-image GT subset is current and intact.

    The subset manifest and JSON must match the current source hash and the
    sample/extraction manifest identities, all 500 ordered IDs must match the
    frozen sample, every selected image/annotation record must equal its
    source record, the category records must equal the full source taxonomy,
    and every subset mask filename must equal the mask basename derived from
    the extraction manifest's ``zip_member`` (whose ``output`` resolves to an
    existing extracted PNG under ``data/panoptic_val2017/`` with matching
    ``sha256`` bytes, verified by ``_verify_extracted_pngs``).
    """
    if not SUBSET_MANIFEST.is_file():
        raise RuntimeError("run prepare_panoptic_subset.py first; subset manifest is missing")
    if not SUBSET_JSON.is_file():
        raise RuntimeError("run prepare_panoptic_subset.py first; subset JSON is missing")
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "validated":
        raise ValueError("panoptic subset manifest status is not validated")
    expected_paths = {
        "source_json": PANOPTIC_JSON,
        "sample_manifest": SAMPLE_MANIFEST,
        "extracted_manifest": EXTRACTED_MANIFEST,
        "output_json": SUBSET_JSON,
    }
    for key, path in expected_paths.items():
        if manifest.get(key) != str(path.resolve()):
            raise ValueError(f"panoptic subset manifest path mismatch for {key}")
    if manifest.get("source_json_sha256") != source_sha:
        raise ValueError("current source panoptic JSON no longer matches the subset manifest")
    if manifest.get("sample_manifest_sha256") != sample_sha:
        raise ValueError("current sample manifest no longer matches the subset manifest")
    if manifest.get("extracted_manifest_sha256") != extraction_sha:
        raise ValueError("current extraction manifest no longer matches the subset manifest")
    if manifest.get("output_json_sha256") != _sha256_file(SUBSET_JSON):
        raise ValueError("subset JSON no longer matches the subset manifest")
    subset = json.loads(SUBSET_JSON.read_text(encoding="utf-8"))
    if not isinstance(subset, dict) or not isinstance(subset.get("info"), dict):
        raise ValueError("subset JSON info is not an object")
    for key in ("licenses", "categories", "images", "annotations"):
        if not isinstance(subset.get(key), list):
            raise ValueError(f"subset JSON {key} is not a list")
    if len(subset["images"]) != len(image_ids) or len(subset["annotations"]) != len(image_ids):
        raise ValueError("subset JSON must contain exactly one row per frozen sample ID")
    ordered_ids = []
    for row in subset["images"]:
        if not isinstance(row, dict) or isinstance(row.get("id"), bool) or not isinstance(row.get("id"), int):
            raise ValueError("subset image row id must be an integer")
        ordered_ids.append(row["id"])
    if ordered_ids != image_ids:
        raise ValueError("subset image IDs differ from the frozen sample order")
    if manifest.get("ordered_image_ids") != image_ids:
        raise ValueError("subset manifest ordered IDs differ from the frozen sample")
    if manifest.get("ordered_image_ids_sha256") != _sha256_json(ordered_ids):
        raise ValueError("subset manifest ordered-ID digest mismatch")
    annotation_ids = []
    for row in subset["annotations"]:
        if not isinstance(row, dict) or isinstance(row.get("image_id"), bool) or not isinstance(row.get("image_id"), int):
            raise ValueError("subset annotation row image_id must be an integer")
        annotation_ids.append(row["image_id"])
    if annotation_ids != image_ids:
        raise ValueError("subset annotation image IDs differ from the frozen sample order")
    if manifest.get("image_count") != len(image_ids) or manifest.get("annotation_count") != len(image_ids):
        raise ValueError("subset manifest row counts mismatch")
    if manifest.get("category_count") != len(source["categories"]):
        raise ValueError("subset manifest category count mismatch")
    for index, image_id in enumerate(image_ids):
        if subset["images"][index] != image_rows[image_id]:
            raise ValueError(f"subset image record differs from the source for {image_id}")
        if subset["annotations"][index] != annotation_rows[image_id]:
            raise ValueError(f"subset annotation record differs from the source for {image_id}")
    if subset["categories"] != source["categories"]:
        raise ValueError("subset categories differ from the full source taxonomy")
    if subset["info"] != source["info"] or subset["licenses"] != source["licenses"]:
        raise ValueError("subset info/licenses differ from the source")
    if _sha256_json(subset["images"]) != manifest.get("selected_image_records_sha256"):
        raise ValueError("subset selected image record digest mismatch")
    if _sha256_json(subset["annotations"]) != manifest.get("selected_annotation_records_sha256"):
        raise ValueError("subset selected annotation record digest mismatch")
    for index, image_id in enumerate(image_ids):
        mask_name = subset["annotations"][index].get("file_name")
        if not isinstance(mask_name, str) or not mask_name:
            raise ValueError(f"subset annotation for image {image_id} has no mask file_name")
        if extracted_basenames.get(image_id) != mask_name:
            raise ValueError(f"subset mask filename differs from the extraction manifest for {image_id}")
        if not (PANOPTIC_ROOT / mask_name).is_file():
            raise FileNotFoundError(f"missing extracted panoptic PNG for {image_id}")


def register_dataset():
    os.environ.setdefault("DETECTRON2_DATASETS", str(COCO_ROOT.parent))
    from detectron2.data import DatasetCatalog, MetadataCatalog

    image_ids, image_rows, annotation_rows, source, extraction_manifest = _load_and_validate_sources()
    sample_sha = _sha256_file(SAMPLE_MANIFEST)
    source_sha = _sha256_file(PANOPTIC_JSON)
    extraction_sha = _sha256_file(EXTRACTED_MANIFEST)
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

    extracted_basenames = _verify_extracted_pngs(extraction_manifest)
    if set(extracted_basenames) != set(image_ids):
        raise ValueError("extracted PNG IDs differ from the frozen sample")
    _validate_panoptic_subset(
        image_ids, image_rows, annotation_rows, source, extracted_basenames,
        sample_sha, source_sha, extraction_sha,
    )
    records = []
    for image_id in image_ids:
        image = image_rows[image_id]
        annotation = annotation_rows[image_id]
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
            "meta": {"dataname": DATASET_NAME},
        })

    if DATASET_NAME in DatasetCatalog.list():
        raise RuntimeError(f"dataset name already registered: {DATASET_NAME}")
    DatasetCatalog.register(DATASET_NAME, lambda: copy.deepcopy(records))
    MetadataCatalog.get(DATASET_NAME).set(
        **metadata,
        image_root=str(IMAGE_ROOT.resolve()),
        panoptic_root=str(PANOPTIC_ROOT.resolve()),
        # run5 fix: the evaluator receives the frozen 500-image GT subset,
        # not the full 5,000-annotation COCO panoptic JSON.
        panoptic_json=str(SUBSET_JSON.resolve()),
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
        "panoptic_subset_json": str(SUBSET_JSON.resolve()),
        "panoptic_subset_manifest": str(SUBSET_MANIFEST.resolve()),
        "panoptic_subset_json_sha256": _sha256_file(SUBSET_JSON),
        "panoptic_subset_manifest_sha256": _sha256_file(SUBSET_MANIFEST),
    }
