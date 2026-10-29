"""Stage 2 of the local ConceptGraphs pipeline: annotate the frames with the VLM.

This stage reads the detections that were produced by
:mod:`conceptgraph.scripts.streamlined_detections` and asks the **locally
served** vision-language model (Qwen3-VL through vLLM, see
``scripts/serve_qwen3_vl.sh``) for

* the spatial relations between the detected objects, and
* a caption for every detected object.

The answers are written back into the detection files, so that the mapping
stage (:mod:`conceptgraph.slam.rerun_realtime_mapping`) can consume them
without ever loading a model itself.

Only one model is resident on the GPU at a time: this stage talks to the VLM
over HTTP (the server owns the GPU), the text model is started afterwards for
the caption consolidation / scene-graph stages.
"""

from __future__ import annotations

import atexit
import gzip
import os
import pickle
import tempfile
from pathlib import Path

import cv2
import hydra
import supervision as sv
import torch
from omegaconf import DictConfig
from tqdm import trange

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.general_utils import (
    ObjectClasses,
    get_det_out_path,
    get_exp_out_path,
    get_vis_out_path,
    make_vlm_edges_and_captions,
)
from conceptgraph.utils.llm_client import get_vlm_client
from conceptgraph.utils.model_control import unload_all_models, verify_ollama_gpu_model
from conceptgraph.utils.gpu_memory_budget import NvmlMemoryBudget
from conceptgraph.utils.model_utils import require_cuda_device


def _unload_vlm_at_stage_exit(client=None) -> None:
    """Best-effort cleanup through the managed model-control service."""
    try:
        unload_all_models()
    except Exception as exc:  # keep completed annotations usable if cleanup fails
        print(f"Warning: could not unload the VLM after annotation: {exc}")


def _save_annotation(path: Path, data: dict) -> None:
    """Replace a detection cache only after its annotated pickle is complete."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
        with gzip.open(temp_path, "wb") as stream:
            pickle.dump(data, stream)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="vlm_annotations")
def main(cfg: DictConfig):
    # Clear any model left by a prior run before evaluating startup headroom.
    # The model-control service confirms this release before allowing the VLM
    # server to take ownership of the GPU below.
    unload_all_models()
    cuda_device = require_cuda_device("cuda")
    memory_budget = NvmlMemoryBudget(cuda_device)
    # The detections of the previous stage
    det_exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)

    # The dataset is only used to get the images of the frames (same order as
    # the detections were saved in).
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.desired_height,
        desired_width=cfg.desired_width,
        device="cpu",
        dtype=torch.float,
    )

    obj_classes = ObjectClasses(
        classes_file_path=cfg.classes_file,
        bg_classes=cfg.bg_classes,
        skip_bg=cfg.skip_bg,
    )

    vlm_client = get_vlm_client()
    # Establish a zero-runner baseline after the detector process has exited,
    # then verify the configured model is served before any image request.
    available_models = vlm_client.list_models()
    expected_model = vlm_client.model.casefold().removesuffix(":latest")
    if not any(
        name.casefold().removesuffix(":latest") == expected_model
        for name in available_models
    ):
        raise RuntimeError(
            f"Configured local VLM '{vlm_client.model}' is not available from "
            f"{vlm_client.base_url}; server models: {available_models}"
        )
    # Keep the VLM warm across frames, then release it before the launcher
    # starts the GPU mapping stage. Per-frame unloading causes needless reloads.
    vlm_client.unload_after_frame = False
    atexit.register(_unload_vlm_at_stage_exit, vlm_client)
    print(f"Using the local VLM '{vlm_client.model}' at {vlm_client.base_url}")

    num_done = 0
    for frame_idx in trange(len(dataset)):
        memory_budget.check(f"VLM before frame {frame_idx}")
        color_path = Path(dataset.color_paths[frame_idx])
        det_path = det_exp_pkl_path / f"{color_path.stem}.pkl.gz"
        if not det_path.exists():
            print(f"No detections for {color_path.name}, skipping.")
            continue

        with gzip.open(det_path, "rb") as f:
            raw_gobs = pickle.load(f)

        annotations_complete = (
            raw_gobs.get("captions") is not None
            and raw_gobs.get("edges") is not None
        )
        if annotations_complete and not cfg.overwrite:
            print(f"{color_path.name} is already annotated, skipping.")
            num_done += 1
            continue

        image = cv2.imread(str(color_path))
        if image is None:
            print(f"Could not read {color_path}, skipping.")
            continue

        curr_det = sv.Detections(
            xyxy=raw_gobs["xyxy"],
            confidence=raw_gobs["confidence"],
            class_id=raw_gobs["class_id"],
            mask=raw_gobs["mask"],
        )

        # The label of a detection is "<class name> <index in the frame>" -
        # exactly like in the mapping stage.
        detection_class_labels = [
            f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(raw_gobs["class_id"])
        ]

        labels, edges, edge_image, captions = make_vlm_edges_and_captions(
            image,
            curr_det,
            obj_classes,
            detection_class_labels,
            det_exp_vis_path,
            color_path,
            cfg.make_edges,
            vlm_client,
        )

        if len(curr_det.xyxy):
            runner_pids = verify_ollama_gpu_model(vlm_client.model)
            runner_memory = {
                pid: memory_budget.process_memory_bytes(pid, allow_missing=False)
                for pid in runner_pids
            }
            if not any(memory_bytes > 0 for memory_bytes in runner_memory.values()):
                raise RuntimeError(
                    "The local VLM handled an image request, but NVML did not "
                    f"observe GPU memory for its serving process: {runner_memory}"
                )
            runner_total = sum(runner_memory.values())
            if runner_total > memory_budget.max_used_bytes:
                raise RuntimeError(
                    "The Ollama VLM runner exceeds the configured per-stage GPU "
                    f"limit: {runner_total / (1024**3):.2f} GiB across "
                    f"PIDs {runner_memory}; limit "
                    f"{memory_budget.max_used_bytes / (1024**3):.2f} GiB"
                )
            print(
                f"[gpu-memory] VLM server PIDs {runner_memory}: "
                f"{runner_total / (1024**3):.2f} GiB"
            )
            memory_budget.check(f"VLM after frame {frame_idx} request")

        raw_gobs["detection_class_labels"] = detection_class_labels
        raw_gobs["labels"] = labels
        raw_gobs["edges"] = edges
        raw_gobs["captions"] = captions

        _save_annotation(det_path, raw_gobs)

        num_done += 1
        print(
            f"[{frame_idx}] {color_path.name}: {len(labels)} objects, "
            f"{len(edges)} relations, {len(captions)} captions"
        )

    print(f"Annotated {num_done} frames with the local VLM.")
    # Require a successful unload before the launcher starts mapping or loads
    # the text model for consolidation.
    unload_all_models()
    atexit.unregister(_unload_vlm_at_stage_exit)


if __name__ == "__main__":
    main()
