#!/usr/bin/env python3
"""Build or validate the frozen 500-image COCO panoptic GT subset.

Standard library only. Two modes:

- Writer mode (default, one-time): validates the sample manifest, the
  extracted-mask manifest, the full source panoptic JSON, and the extracted
  mask PNGs, then writes ``data/coco500-panoptic-subset.json`` and
  ``data/coco500-panoptic-subset-manifest.json``. Refuses to overwrite either
  derived output.
- ``--validate-only``: strictly read-only. Recomputes source digests,
  derived-output digests, ordered IDs, record equality, category equality,
  and mask filename checks, and fails closed on any mismatch. Never rewrites
  either JSON file or any manifest.

The derived JSON preserves ``info``, ``licenses``, and ``categories`` verbatim
from the source and holds exactly one image row and one annotation row per
frozen sample ID, in frozen order, copied verbatim.

Extracted-manifest entries (as written by ``prepare_data.py``) carry
``image_id``, ``zip_member``, ``output``, ``sha256`` and ``dimensions``; they
have no ``file_name`` field. The mask basename is derived from
``zip_member``: the exact ``panoptic_val2017/`` prefix is required and the
remainder must be a single basename. ``output`` must resolve to the expected
local PNG under ``data/panoptic_val2017/`` with that basename, the file must
exist, and its bytes must match the entry's ``sha256`` digest. Source and
subset annotation ``file_name`` values are then matched to that basename.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COCO_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/coco")
SAMPLE_MANIFEST = ROOT / "scripts/pipeline_model_reports/data/yoloe26/coco-val2017-500-manifest.json"
PANOPTIC_JSON = COCO_ROOT / "annotations/panoptic_val2017.json"
PANOPTIC_ROOT = Path(__file__).resolve().parent / "data/panoptic_val2017"
EXTRACTED_MANIFEST = Path(__file__).resolve().parent / "data/coco500-extracted-manifest.json"
SUBSET_JSON = Path(__file__).resolve().parent / "data/coco500-panoptic-subset.json"
SUBSET_MANIFEST = Path(__file__).resolve().parent / "data/coco500-panoptic-subset-manifest.json"
EXPECTED_IMAGE_COUNT = 500
PANOPTIC_ZIP_PREFIX = "panoptic_val2017/"
_HEX_DIGITS = frozenset("0123456789abcdef")


def _fail(message: str):
    raise RuntimeError(message)


def _sha256_json(value) -> str:
    """Canonical record digest: UTF-8 JSON, sorted keys, compact separators."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deterministic_bytes(value) -> bytes:
    """Deterministic file form: UTF-8, sorted object keys, indent 2, trailing newline."""
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _as_image_id(value, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{context} must be an integer image ID")
    return value


def _load_sample_manifest() -> list:
    if not SAMPLE_MANIFEST.is_file():
        _fail(f"sample manifest is missing: {SAMPLE_MANIFEST}")
    sample = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    raw_ids = sample.get("sample_image_ids") if isinstance(sample, dict) else None
    if not isinstance(raw_ids, list):
        _fail("sample manifest is missing the sample_image_ids list")
    image_ids = [_as_image_id(value, "sample_image_ids entry") for value in raw_ids]
    if (
        len(image_ids) != EXPECTED_IMAGE_COUNT
        or len(set(image_ids)) != EXPECTED_IMAGE_COUNT
        or image_ids != sorted(image_ids)
    ):
        _fail("frozen sample must contain 500 unique, sorted image IDs")
    return image_ids


def _mask_basename(zip_member, image_id) -> str:
    """Derive the mask basename from an exact ``panoptic_val2017/<name>`` member."""
    if not isinstance(zip_member, str) or not zip_member.startswith(PANOPTIC_ZIP_PREFIX):
        _fail(
            f"extracted entry {image_id} zip_member lacks the exact "
            f"{PANOPTIC_ZIP_PREFIX!r} prefix: {zip_member!r}"
        )
    basename = zip_member[len(PANOPTIC_ZIP_PREFIX):]
    if not basename or "/" in basename:
        _fail(f"extracted entry {image_id} zip_member has no single basename: {zip_member!r}")
    return basename


def _load_extraction_manifest(image_ids: list):
    if not EXTRACTED_MANIFEST.is_file():
        _fail(f"extracted-mask manifest is missing: {EXTRACTED_MANIFEST}")
    extraction = json.loads(EXTRACTED_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(extraction, dict):
        _fail("extraction manifest root is not an object")
    if extraction.get("status") != "extracted_and_verified":
        _fail("extraction manifest status is not extracted_and_verified")
    if extraction.get("image_count") != EXPECTED_IMAGE_COUNT:
        _fail("extraction manifest image_count is not 500")
    if extraction.get("zip_prefix") != PANOPTIC_ZIP_PREFIX:
        _fail(f"extraction manifest zip_prefix is not {PANOPTIC_ZIP_PREFIX!r}")
    entries = extraction.get("extracted")
    if not isinstance(entries, list) or len(entries) != EXPECTED_IMAGE_COUNT:
        _fail("extraction manifest must contain exactly 500 extracted entries")
    extracted_rows = {}
    seen_basenames = set()
    for entry in entries:
        if not isinstance(entry, dict):
            _fail("extraction manifest entry is not an object")
        image_id = _as_image_id(entry.get("image_id"), "extracted entry image_id")
        basename = _mask_basename(entry.get("zip_member"), image_id)
        output = entry.get("output")
        if not isinstance(output, str) or not output:
            _fail(f"extracted entry {image_id} has no output path")
        expected_path = (PANOPTIC_ROOT / basename).resolve()
        if not Path(output).is_absolute() or Path(output).resolve() != expected_path:
            _fail(
                f"extracted entry {image_id} output does not resolve to the "
                f"expected local PNG under data/panoptic_val2017: {output}"
            )
        if not expected_path.is_file():
            _fail(f"extracted panoptic PNG is missing for image {image_id}: {expected_path}")
        entry_sha = entry.get("sha256")
        if not isinstance(entry_sha, str) or len(entry_sha) != 64 or not set(entry_sha) <= _HEX_DIGITS:
            _fail(f"extracted entry {image_id} sha256 is not a 64-character lowercase hex digest")
        if _sha256_file(expected_path) != entry_sha:
            _fail(f"extracted panoptic PNG bytes do not match sha256 for image {image_id}")
        if image_id in extracted_rows:
            _fail(f"extraction manifest has duplicate entries for image {image_id}")
        if basename in seen_basenames:
            _fail(f"extraction manifest has duplicate mask basenames: {basename}")
        extracted_rows[image_id] = basename
        seen_basenames.add(basename)
    if set(extracted_rows) != set(image_ids):
        _fail("extracted manifest image IDs differ from the frozen sample")
    return extraction, extracted_rows


def _load_source_panoptic(image_ids: list):
    if not PANOPTIC_JSON.is_file():
        _fail(f"source panoptic JSON is missing: {PANOPTIC_JSON}")
    source = json.loads(PANOPTIC_JSON.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        _fail("source panoptic JSON root is not an object")
    if not isinstance(source.get("info"), dict):
        _fail("source panoptic JSON info is not an object")
    for key in ("licenses", "categories", "images", "annotations"):
        if not isinstance(source.get(key), list):
            _fail(f"source panoptic JSON {key} is not a list")
    image_rows = {}
    for row in source["images"]:
        if not isinstance(row, dict):
            _fail("source image row is not an object")
        image_id = _as_image_id(row.get("id"), "source image row id")
        if image_id in image_rows:
            _fail(f"source panoptic JSON has duplicate image rows for {image_id}")
        image_rows[image_id] = row
    annotation_rows = {}
    for row in source["annotations"]:
        if not isinstance(row, dict):
            _fail("source annotation row is not an object")
        image_id = _as_image_id(row.get("image_id"), "source annotation row image_id")
        if image_id in annotation_rows:
            _fail(f"source panoptic JSON has duplicate annotation rows for image {image_id}")
        annotation_rows[image_id] = row
    for image_id in image_ids:
        if image_id not in image_rows:
            _fail(f"source panoptic JSON is missing the image row for {image_id}")
        if image_id not in annotation_rows:
            _fail(f"source panoptic JSON is missing the annotation row for image {image_id}")
    return source, image_rows, annotation_rows


def _check_mask_filenames(image_ids: list, annotation_rows: dict, extracted_rows: dict):
    for image_id in image_ids:
        mask_name = annotation_rows[image_id].get("file_name")
        if not isinstance(mask_name, str) or not mask_name:
            _fail(f"annotation for image {image_id} has no mask file_name")
        if mask_name != extracted_rows[image_id]:
            _fail(f"mask filename for image {image_id} differs from the extracted manifest")
        if not (PANOPTIC_ROOT / mask_name).is_file():
            _fail(f"extracted panoptic PNG is missing for image {image_id}: {PANOPTIC_ROOT / mask_name}")


def _write_subset() -> dict:
    if SUBSET_JSON.exists() or SUBSET_MANIFEST.exists():
        _fail("derived output already exists; this is a one-time writer and refuses overwrite")
    image_ids = _load_sample_manifest()
    extraction, extracted_rows = _load_extraction_manifest(image_ids)
    source, image_rows, annotation_rows = _load_source_panoptic(image_ids)
    sample_sha = _sha256_file(SAMPLE_MANIFEST)
    source_sha = _sha256_file(PANOPTIC_JSON)
    extraction_sha = _sha256_file(EXTRACTED_MANIFEST)
    if extraction.get("sample_manifest_sha256") != sample_sha:
        _fail("sample manifest SHA-256 does not match the extraction manifest")
    if extraction.get("panoptic_json_sha256") != source_sha:
        _fail("source panoptic JSON SHA-256 does not match the extraction manifest")
    _check_mask_filenames(image_ids, annotation_rows, extracted_rows)
    # Selected records and categories are the parsed source objects themselves
    # (verbatim copy); --validate-only re-verifies full equality against the
    # on-disk source independently.
    subset = {
        "info": source["info"],
        "licenses": source["licenses"],
        "categories": source["categories"],
        "images": [image_rows[image_id] for image_id in image_ids],
        "annotations": [annotation_rows[image_id] for image_id in image_ids],
    }
    subset_bytes = _deterministic_bytes(subset)
    SUBSET_JSON.write_bytes(subset_bytes)
    output_sha = _sha256_file(SUBSET_JSON)
    if SUBSET_JSON.read_bytes() != subset_bytes or json.loads(SUBSET_JSON.read_text(encoding="utf-8")) != subset:
        _fail("written subset JSON does not round-trip to the selected records")
    manifest = {
        "status": "validated",
        "source_json": str(PANOPTIC_JSON.resolve()),
        "sample_manifest": str(SAMPLE_MANIFEST.resolve()),
        "extracted_manifest": str(EXTRACTED_MANIFEST.resolve()),
        "output_json": str(SUBSET_JSON.resolve()),
        "source_json_sha256": source_sha,
        "sample_manifest_sha256": sample_sha,
        "extracted_manifest_sha256": extraction_sha,
        "ordered_image_ids": image_ids,
        "ordered_image_ids_sha256": _sha256_json(image_ids),
        "image_count": len(subset["images"]),
        "annotation_count": len(subset["annotations"]),
        "category_count": len(subset["categories"]),
        "selected_image_records_sha256": _sha256_json(subset["images"]),
        "selected_annotation_records_sha256": _sha256_json(subset["annotations"]),
        "output_json_sha256": output_sha,
        "mask_filename_equality": {
            "checked": EXPECTED_IMAGE_COUNT,
            "matches_extracted_manifest": True,
            "existing_png_files": True,
            "png_sha256_matches": True,
        },
    }
    SUBSET_MANIFEST.write_bytes(_deterministic_bytes(manifest))
    return {
        "mode": "writer",
        "status": "validated",
        "image_count": len(subset["images"]),
        "annotation_count": len(subset["annotations"]),
        "category_count": len(subset["categories"]),
        "output_json": str(SUBSET_JSON.resolve()),
        "output_json_sha256": output_sha,
        "subset_manifest": str(SUBSET_MANIFEST.resolve()),
        "ordered_image_ids_sha256": _sha256_json(image_ids),
    }


def _validate_only() -> dict:
    for path in (SUBSET_JSON, SUBSET_MANIFEST):
        if not path.is_file():
            _fail(f"derived output is missing; run the writer first: {path}")
    for path in (PANOPTIC_JSON, SAMPLE_MANIFEST, EXTRACTED_MANIFEST):
        if not path.is_file():
            _fail(f"source input is missing: {path}")
    manifest = json.loads(SUBSET_MANIFEST.read_text(encoding="utf-8"))
    subset = json.loads(SUBSET_JSON.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "validated":
        _fail("subset manifest status is not validated")
    if not isinstance(subset, dict) or not isinstance(subset.get("info"), dict):
        _fail("subset JSON info is not an object")
    for key in ("licenses", "categories", "images", "annotations"):
        if not isinstance(subset.get(key), list):
            _fail(f"subset JSON {key} is not a list")
    expected_paths = {
        "source_json": PANOPTIC_JSON,
        "sample_manifest": SAMPLE_MANIFEST,
        "extracted_manifest": EXTRACTED_MANIFEST,
        "output_json": SUBSET_JSON,
    }
    for key, path in expected_paths.items():
        if manifest.get(key) != str(path.resolve()):
            _fail(f"subset manifest path mismatch for {key}")
    sample_sha = _sha256_file(SAMPLE_MANIFEST)
    source_sha = _sha256_file(PANOPTIC_JSON)
    extraction_sha = _sha256_file(EXTRACTED_MANIFEST)
    output_sha = _sha256_file(SUBSET_JSON)
    if manifest.get("sample_manifest_sha256") != sample_sha:
        _fail("sample manifest SHA-256 no longer matches the subset manifest")
    if manifest.get("source_json_sha256") != source_sha:
        _fail("source panoptic JSON SHA-256 no longer matches the subset manifest")
    if manifest.get("extracted_manifest_sha256") != extraction_sha:
        _fail("extracted manifest SHA-256 no longer matches the subset manifest")
    if manifest.get("output_json_sha256") != output_sha:
        _fail("subset JSON SHA-256 no longer matches the subset manifest")
    if SUBSET_JSON.read_bytes() != _deterministic_bytes(subset):
        _fail("subset JSON is not in the canonical deterministic serialization")
    if SUBSET_MANIFEST.read_bytes() != _deterministic_bytes(manifest):
        _fail("subset manifest is not in the canonical deterministic serialization")
    image_ids = _load_sample_manifest()
    extraction, extracted_rows = _load_extraction_manifest(image_ids)
    if extraction.get("sample_manifest_sha256") != sample_sha:
        _fail("extraction manifest sample SHA-256 does not match the current sample manifest")
    if extraction.get("panoptic_json_sha256") != source_sha:
        _fail("extraction manifest source SHA-256 does not match the current source panoptic JSON")
    source, image_rows, annotation_rows = _load_source_panoptic(image_ids)
    if len(subset["images"]) != EXPECTED_IMAGE_COUNT or len(subset["annotations"]) != EXPECTED_IMAGE_COUNT:
        _fail("subset JSON must contain exactly 500 image rows and 500 annotation rows")
    if not all(isinstance(row, dict) for row in subset["images"]):
        _fail("subset JSON image rows must be objects")
    if not all(isinstance(row, dict) for row in subset["annotations"]):
        _fail("subset JSON annotation rows must be objects")
    derived_image_ids = [_as_image_id(row.get("id"), "subset image row id") for row in subset["images"]]
    derived_annotation_ids = [_as_image_id(row.get("image_id"), "subset annotation row image_id") for row in subset["annotations"]]
    if derived_image_ids != image_ids or derived_annotation_ids != image_ids:
        _fail("subset JSON image/annotation IDs differ from the frozen sample order")
    if manifest.get("ordered_image_ids") != image_ids:
        _fail("subset manifest ordered IDs differ from the frozen sample")
    if manifest.get("ordered_image_ids_sha256") != _sha256_json(image_ids):
        _fail("subset manifest ordered-ID digest mismatch")
    if manifest.get("image_count") != EXPECTED_IMAGE_COUNT or manifest.get("annotation_count") != EXPECTED_IMAGE_COUNT:
        _fail("subset manifest row counts mismatch")
    if manifest.get("category_count") != len(source["categories"]):
        _fail("subset manifest category count mismatch")
    for index, image_id in enumerate(image_ids):
        if subset["images"][index] != image_rows[image_id]:
            _fail(f"subset image record differs from the current source for {image_id}")
        if subset["annotations"][index] != annotation_rows[image_id]:
            _fail(f"subset annotation record differs from the current source for {image_id}")
    if subset["categories"] != source["categories"]:
        _fail("subset categories differ from the full source taxonomy")
    if subset["info"] != source["info"] or subset["licenses"] != source["licenses"]:
        _fail("subset info/licenses differ from the source")
    if _sha256_json(subset["images"]) != manifest.get("selected_image_records_sha256"):
        _fail("subset selected image record digest mismatch")
    if _sha256_json(subset["annotations"]) != manifest.get("selected_annotation_records_sha256"):
        _fail("subset selected annotation record digest mismatch")
    derived_annotation_rows = {
        _as_image_id(row.get("image_id"), "subset annotation row image_id"): row for row in subset["annotations"]
    }
    _check_mask_filenames(image_ids, derived_annotation_rows, extracted_rows)
    return {
        "mode": "validate-only",
        "status": "validated",
        "image_count": len(subset["images"]),
        "annotation_count": len(subset["annotations"]),
        "category_count": len(subset["categories"]),
        "output_json": str(SUBSET_JSON.resolve()),
        "output_json_sha256": output_sha,
        "subset_manifest": str(SUBSET_MANIFEST.resolve()),
        "ordered_image_ids_sha256": _sha256_json(image_ids),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="One-time writer or read-only validator for the frozen 500-image COCO panoptic GT subset.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="read-only: recompute digests, IDs, record/category equality, and mask filenames; never write",
    )
    args = parser.parse_args(argv)
    try:
        result = _validate_only() if args.validate_only else _write_subset()
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        print(f"panoptic-subset: ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
