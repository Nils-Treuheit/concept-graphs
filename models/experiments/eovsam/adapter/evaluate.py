#!/usr/bin/env python3
"""Prepare, preflight, or run EOVSAM on the frozen COCO500 panoptic subset."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


ADAPTER = Path(__file__).resolve().parent
ROOT = ADAPTER.parents[4]
REPO = ADAPTER.parent / "repo"
OUTPUT = ADAPTER / "run-manifest.json"


def require_frozen_val_inputs():
    """Fail closed unless the adapter's frozen val2017-500 panoptic subset is fully present.

    This adapter evaluates only ``eovsam_coco_val2017_500_seed42``. Upstream COCO
    panoptic dataset registration is lazy, so val-only inference never needs COCO
    training panoptic or semantic assets. This validates the adapter's own frozen
    val inputs -- the 500 authentic images, extracted panoptic labels, panoptic
    annotations and the official ID/category mapping -- through the existing
    register_dataset() path and its manifest checks. It creates no files and
    requests no train assets, so a train dataset is never fabricated to satisfy
    the import path.
    """
    from register_coco500 import DATASET_NAME, register_dataset

    try:
        registration = register_dataset()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise RuntimeError(
            "BLOCKED before registering the frozen val subset: the adapter's own 500-image "
            "COCO val2017 panoptic inputs (images, extracted panoptic labels, annotations or "
            "ID/category mapping) are incomplete. Upstream panoptic registration is lazy, so "
            "no COCO train assets are required and no placeholders will be created. "
            f"Underlying error: {exc}"
        ) from exc

    if registration.get("dataset_name") != DATASET_NAME:
        raise RuntimeError(
            f"unexpected frozen dataset name: {registration.get('dataset_name')!r}; "
            f"expected the val-only subset {DATASET_NAME!r}"
        )

    image_ids = registration.get("image_ids", [])
    if len(image_ids) != 500 or len(set(image_ids)) != 500 or image_ids != sorted(image_ids):
        raise RuntimeError(
            f"frozen val sample must contain 500 unique, sorted image IDs; got {len(image_ids)}"
        )

    for field in (
        "sample_manifest_sha256",
        "source_annotation_sha256",
        "thing_map_sha256",
        "stuff_map_sha256",
        "category_id_isthing_mapping_sha256",
    ):
        if not registration.get(field):
            raise RuntimeError(
                f"manifest-derived field {field!r} is missing from the frozen registration"
            )

    return registration


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def evaluator_preflight(registration=None):
    from detectron2 import __version__ as detectron2_version
    from detectron2.data import MetadataCatalog
    from detectron2.evaluation import COCOPanopticEvaluator
    from register_coco500 import _load_upstream_categories_and_metadata

    if registration is None:
        from register_coco500 import register_dataset
        registration = register_dataset()
    metadata = MetadataCatalog.get(registration["dataset_name"])
    evaluator = COCOPanopticEvaluator(registration["dataset_name"], output_dir=str(ADAPTER / "preflight-output"))
    source_path = inspect.getsourcefile(COCOPanopticEvaluator)
    source_text = inspect.getsource(COCOPanopticEvaluator)
    if not source_path or not source_text:
        raise RuntimeError("could not inspect installed COCOPanopticEvaluator source")

    things = metadata.thing_dataset_id_to_contiguous_id
    stuffs = metadata.stuff_dataset_id_to_contiguous_id
    thing_reverse = getattr(evaluator, "_thing_contiguous_id_to_dataset_id", None)
    stuff_reverse = getattr(evaluator, "_stuff_contiguous_id_to_dataset_id", None)
    if not isinstance(thing_reverse, dict) or not isinstance(stuff_reverse, dict):
        raise RuntimeError("installed evaluator lacks inspectable thing/stuff reverse maps")

    def check_round_trip(source_map, reverse_map, isthing):
        official_id = min(int(key) for key in source_map)
        contiguous_id = int(source_map[official_id])
        restored_id = int(reverse_map[contiguous_id])
        if restored_id != official_id:
            raise RuntimeError(f"COCO category round-trip failed: {official_id} -> {contiguous_id} -> {restored_id}")
        return {"isthing": bool(isthing), "official_id": official_id,
                "contiguous_id": contiguous_id, "restored_id": restored_id}

    thing_result = check_round_trip(things, thing_reverse, True)
    categories, _ = _load_upstream_categories_and_metadata()
    stuff_ids = [int(row["id"]) for row in categories if int(row["isthing"]) == 0]
    stuff_result = check_round_trip({key: stuffs[key] for key in stuff_ids}, stuff_reverse, False)
    if "_thing_contiguous_id_to_dataset_id" not in source_text or "_stuff_contiguous_id_to_dataset_id" not in source_text:
        raise RuntimeError("evaluator source does not show the inspected reverse-map fields")
    return {
        **registration,
        "status": "evaluator_roundtrip_passed",
        "detectron2_version": detectron2_version,
        "evaluator_class": f"{COCOPanopticEvaluator.__module__}.{COCOPanopticEvaluator.__name__}",
        "evaluator_source": str(Path(source_path).resolve()),
        "evaluator_source_sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        "evaluator_reverse_fields": ["_thing_contiguous_id_to_dataset_id", "_stuff_contiguous_id_to_dataset_id"],
        "thing_round_trip": thing_result,
        "stuff_round_trip": stuff_result,
    }


def run_evaluation():
    os.environ.setdefault("DETECTRON2_DATASETS", "/mnt/HDD1/Project_Data/public_datasets")
    registration = require_frozen_val_inputs()
    sys.path.insert(0, str(REPO))
    from detectron2.engine import default_argument_parser, launch
    try:
        import train_net
    except FileNotFoundError as exc:
        raise RuntimeError(
            "BLOCKED: importing upstream train_net hit a missing file. If this is a COCO "
            "train asset, note that this val-only adapter intentionally does not provide "
            "train panoptic or semantic assets and will not create placeholders for them. "
            f"Underlying error: {exc}"
        ) from exc

    preflight = evaluator_preflight(registration)
    OUTPUT.write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    args = default_argument_parser().parse_args()
    if not args.eval_only:
        raise ValueError("EOVSAM COCO500 adapter is evaluation-only")
    launch(train_net.main, args.num_gpus, num_machines=args.num_machines,
           machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))


def main():
    os.environ.setdefault("DETECTRON2_DATASETS", "/mnt/HDD1/Project_Data/public_datasets")
    sys.path.insert(0, str(REPO))
    if len(sys.argv) >= 2 and sys.argv[1] == "prepare-data":
        from prepare_data import main as prepare
        return prepare()
    if len(sys.argv) >= 2 and sys.argv[1] == "assets":
        require_frozen_val_inputs()
        print("frozen COCO val2017-500 panoptic subset is present and verified")
        return 0
    if len(sys.argv) >= 2 and sys.argv[1] == "preflight":
        result = evaluator_preflight()
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"preflight passed; manifest: {OUTPUT}")
        return 0
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        del sys.argv[1]
        run_evaluation()
        return 0
    raise SystemExit("usage: evaluate.py {prepare-data|assets|preflight|run --num-gpus 1 --eval-only --config-file eovsam_coco500.yaml OUTPUT_DIR ... MODEL.WEIGHTS ...}")


if __name__ == "__main__":
    raise SystemExit(main())
