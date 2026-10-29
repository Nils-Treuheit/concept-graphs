"""Bounded, CUDA-resident YOLO -> SAM -> CLIP detection windows."""

from __future__ import annotations

import gc
import gzip
import os
import pickle
import tempfile
from pathlib import Path

import cv2
import hydra
import numpy as np
import open_clip
import supervision as sv
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from tqdm import trange
from ultralytics import SAM, YOLO
from ultralytics.models.sam.predict import Predictor as SamPredictor
from ultralytics.utils.nms import non_max_suppression

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.general_utils import (
    ObjectClasses,
    get_det_out_path,
    get_exp_out_path,
    get_vis_out_path,
    get_weights_dir,
    save_hydra_config,
)
from conceptgraph.utils.gpu_memory_budget import NvmlMemoryBudget
from conceptgraph.utils.model_utils import require_cuda_device
from conceptgraph.utils.vis import save_video_detections, vis_result_fast


WINDOW_FRAMES = 8
MAX_DETECTIONS = 300
MODEL_BATCH = 4
YOLO_SIZE = 640
SAM_SIZE = 1024
CLIP_SIZE = 224
CLIP_HUB_REPO = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
DEFAULT_HF_HOME = "/mnt/HDD1/unsloth_and_hugging_face_models/huggingface"
MIN_CLIP_CHECKPOINT_BYTES = 1 * 1024**3
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _letterbox(image: torch.Tensor, size: int, *, fill: float) -> tuple[torch.Tensor, dict]:
    """Aspect-preserving square resize and padding on CUDA."""
    _, height, width = image.shape
    gain = min(size / width, size / height)
    resized_w = max(1, min(size, round(width * gain)))
    resized_h = max(1, min(size, round(height * gain)))
    resized = F.interpolate(
        image[None].float(), size=(resized_h, resized_w), mode="bilinear",
        align_corners=False, antialias=True,
    )[0]
    left = (size - resized_w) // 2
    top = (size - resized_h) // 2
    canvas = torch.full((3, size, size), fill, device=image.device, dtype=resized.dtype)
    canvas[:, top : top + resized_h, left : left + resized_w] = resized
    return canvas, {
        "gain_x": resized_w / width, "gain_y": resized_h / height,
        "left": left, "top": top,
        "resized_h": resized_h, "resized_w": resized_w,
        "height": height, "width": width,
    }


def _restore_boxes(boxes: torch.Tensor, transform: dict) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform["left"]) / transform["gain_x"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform["top"]) / transform["gain_y"]
    boxes[:, [0, 2]].clamp_(0, transform["width"])
    boxes[:, [1, 3]].clamp_(0, transform["height"])
    return boxes


def _sam_boxes(boxes: torch.Tensor, transform: dict) -> torch.Tensor:
    out = boxes.clone()
    out[:, [0, 2]] = out[:, [0, 2]] * transform["gain_x"] + transform["left"]
    out[:, [1, 3]] = out[:, [1, 3]] * transform["gain_y"] + transform["top"]
    return out


def _clip_input(image: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Build OpenCLIP-style center-cropped 224 inputs entirely on CUDA."""
    count, _, height, width = boxes.shape[0], *image.shape
    if count == 0:
        return image.new_empty((0, 3, CLIP_SIZE, CLIP_SIZE), dtype=torch.float32)
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    x1 = (x1.floor() - 20).clamp(0, width - 1)
    y1 = (y1.floor() - 20).clamp(0, height - 1)
    x2 = (x2.ceil() + 20).clamp(1, width)
    y2 = (y2.ceil() + 20).clamp(1, height)
    crop_w = (x2 - x1).clamp_min(1)
    crop_h = (y2 - y1).clamp_min(1)
    side = torch.minimum(crop_w, crop_h)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    offsets = (torch.arange(CLIP_SIZE, device=image.device, dtype=torch.float32) + 0.5)
    offsets = (offsets / CLIP_SIZE - 0.5)[None, :] * side[:, None]
    sample_x = center_x[:, None, None] + offsets[:, None, :]
    sample_y = center_y[:, None, None] + offsets[:, :, None]
    grid_x = ((sample_x + 0.5) * (2.0 / width) - 1.0).expand(-1, CLIP_SIZE, -1)
    grid_y = ((sample_y + 0.5) * (2.0 / height) - 1.0).expand(-1, -1, CLIP_SIZE)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    crops = F.grid_sample(
        image[None].expand(count, -1, -1, -1).float(), grid,
        mode="bicubic", padding_mode="border", align_corners=False,
    ) / 255.0
    mean = crops.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = crops.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (crops - mean) / std


def _pack_frame(fields: list[tuple[str, torch.Tensor]]) -> dict:
    """Convert mixed device tensors to independent host NumPy arrays."""
    return {
        name: tensor.detach().to(device="cpu").contiguous().numpy().copy()
        for name, tensor in fields
    }


def _padded_crops(image_rgb: np.ndarray, boxes: np.ndarray) -> list[Image.Image]:
    image = Image.fromarray(image_rgb)
    width, height = image.size
    crops = []
    for x1, y1, x2, y2 in boxes:
        left, top = max(0, int(np.floor(x1)) - 20), max(0, int(np.floor(y1)) - 20)
        right, bottom = min(width, int(np.ceil(x2)) + 20), min(height, int(np.ceil(y2)) + 20)
        crops.append(image.crop((left, top, right, bottom)))
    return crops


def _atomic_save(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temp_path = Path(stream.name)
        with gzip.open(temp_path, "wb") as stream:
            pickle.dump(results, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _load_sam(device: torch.device, checkpoint: Path):
    wrapper = SAM(str(checkpoint))
    predictor = SamPredictor(overrides={
        "imgsz": SAM_SIZE, "device": str(device), "verbose": False,
    })
    predictor.setup_model(model=wrapper.model, verbose=False)
    predictor.args.imgsz = [SAM_SIZE, SAM_SIZE]
    return wrapper, predictor


def _resolve_clip_checkpoint() -> Path:
    configured = os.getenv("CONCEPTGRAPH_CLIP_H14_CHECKPOINT")
    if configured:
        checkpoint = Path(configured).expanduser()
        if checkpoint.is_file() and checkpoint.stat().st_size >= MIN_CLIP_CHECKPOINT_BYTES:
            # Keep the .safetensors filename suffix so OpenCLIP selects its
            # safetensors loader instead of torch.load on the resolved blob.
            return checkpoint.absolute()
        raise FileNotFoundError(
            f"CONCEPTGRAPH_CLIP_H14_CHECKPOINT is not a complete H14 checkpoint: {checkpoint}"
        )

    hf_home = Path(os.getenv("HF_HOME", DEFAULT_HF_HOME)).expanduser()
    snapshot_root = hf_home / "hub" / f"models--{CLIP_HUB_REPO.replace('/', '--')}" / "snapshots"
    candidates = sorted(
        snapshot_root.glob("*/open_clip_model.safetensors"),
        key=lambda path: path.parent.stat().st_mtime if path.parent.exists() else 0,
        reverse=True,
    )
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            if resolved.is_file() and resolved.stat().st_size >= MIN_CLIP_CHECKPOINT_BYTES:
                # Hugging Face cache paths are symlinks to extensionless blobs.
                # Preserve the filename that lets OpenCLIP detect safetensors.
                return candidate.absolute()
        except OSError:
            continue

    expected = snapshot_root / "<revision>" / "open_clip_model.safetensors"
    raise FileNotFoundError(
        "A complete local ViT-H-14 checkpoint is required. Download it with "
        "`uv run python scripts/download_models.py --download --only clip`. "
        "The downloader stores it in the HF cache. Then either leave "
        "CONCEPTGRAPH_CLIP_H14_CHECKPOINT unset for automatic cache discovery, "
        "or set it to the resolved file, for example:\n"
        f"  export CONCEPTGRAPH_CLIP_H14_CHECKPOINT=\"{expected}\"\n"
        f"Current HF cache location: {snapshot_root}"
    )


def _process_window(paths, frame_indices, cfg, class_names, device, budget, out_paths):
    cpu_images = []
    rgb_paths = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read RGB frame: {path}")
        if image.shape[:2] != (720, 1280):
            raise ValueError(
                f"Expected native 1280x720 RGB input, got {image.shape[1]}x{image.shape[0]} at {path}"
            )
        cpu_images.append(image)
        rgb_paths.append(path)

    # Source images cross HtoD once and remain resident through each model phase.
    source_gpu = torch.from_numpy(np.stack(cpu_images)).to(device=device, non_blocking=False)
    source_gpu = source_gpu.permute(0, 3, 1, 2).flip(1).contiguous()  # RGB, uint8
    frame_count = len(paths)
    transforms_yolo = []
    yolo_inputs = []
    for image in source_gpu:
        canvas, transform = _letterbox(image, YOLO_SIZE, fill=114.0)
        transforms_yolo.append(transform)
        yolo_inputs.append(canvas.div_(255.0))
    yolo_batch = torch.stack(yolo_inputs)
    del yolo_inputs

    budget.check(f"window {frame_indices[0]}-{frame_indices[-1]} before YOLO")
    weights_dir = get_weights_dir(cfg.repo_root)
    budget.check(f"window {frame_indices[0]}-{frame_indices[-1]} before YOLO load")
    detection_model = YOLO(str(weights_dir / "yolov8l-world.pt"))
    detection_model.model.to(device).eval()
    detection_model.set_classes(class_names)
    budget.check(f"window {frame_indices[0]}-{frame_indices[-1]} after YOLO load")
    proposals = []
    with torch.inference_mode():
        # Keep the model loaded for the window, but execute one frame at a
        # time. This avoids YOLO's large activation peak for an 8-frame batch.
        for frame_idx in range(frame_count):
            raw_prediction = detection_model.model(yolo_batch[frame_idx : frame_idx + 1])
            frame_proposals = non_max_suppression(
                raw_prediction, conf_thres=0.1, iou_thres=0.7,
                max_det=MAX_DETECTIONS, nc=len(class_names),
            )
            proposals.append(frame_proposals[0])
            del raw_prediction, frame_proposals
            budget.check(f"YOLO complete frame {frame_indices[frame_idx]}")
    proposal_boxes = []
    proposal_conf = []
    proposal_class = []
    for idx, proposal in enumerate(proposals):
        proposal = proposal[:MAX_DETECTIONS]
        proposal_boxes.append(_restore_boxes(proposal[:, :4].float(), transforms_yolo[idx]))
        proposal_conf.append(proposal[:, 4].float())
        proposal_class.append(proposal[:, 5].long())
    del proposals, detection_model, yolo_batch
    gc.collect()
    torch.cuda.empty_cache()
    budget.check(f"window {frame_indices[0]}-{frame_indices[-1]} after YOLO release")

    # Process SAM features per image; move completed masks to host memory.
    sam_checkpoint = weights_dir / "sam_l.pt"
    budget.check(f"window {frame_indices[0]}-{frame_indices[-1]} before SAM load")
    sam_wrapper, sam_predictor = _load_sam(device, sam_checkpoint)
    budget.check(f"window {frame_indices[0]}-{frame_indices[-1]} after SAM load")
    masks_cpu = []
    with torch.inference_mode():
        for frame_idx in range(frame_count):
            sam_canvas, transform = _letterbox(source_gpu[frame_idx], SAM_SIZE, fill=114.0)
            sam_input = (sam_canvas.float() - sam_predictor.mean) / sam_predictor.std
            budget.check(f"SAM image features before frame {frame_indices[frame_idx]}")
            features = sam_predictor.get_im_features(sam_input[None])
            budget.check(f"SAM image features after frame {frame_indices[frame_idx]}")
            boxes_canvas = _sam_boxes(proposal_boxes[frame_idx], transform)
            frame_mask_chunks = []
            for begin in range(0, len(boxes_canvas), MODEL_BATCH):
                budget.check(
                    f"SAM prompts {begin}:{begin + MODEL_BATCH} frame {frame_indices[frame_idx]}"
                )
                chunk, _ = sam_predictor.inference_features(
                    features=features,
                    src_shape=(SAM_SIZE, SAM_SIZE),
                    dst_shape=(SAM_SIZE, SAM_SIZE),
                    bboxes=boxes_canvas[begin : begin + MODEL_BATCH],
                    multimask_output=False,
                )
                top, left = transform["top"], transform["left"]
                square_chunk = chunk[
                    :, top : top + transform["resized_h"],
                    left : left + transform["resized_w"],
                ]
                original_chunk = F.interpolate(
                    square_chunk[:, None].float(),
                    size=(transform["height"], transform["width"]),
                    mode="nearest",
                )[:, 0].bool()
                frame_mask_chunks.append(original_chunk.cpu())
                del chunk, square_chunk, original_chunk
                budget.check(
                    f"SAM prompts complete {begin}:{begin + MODEL_BATCH} "
                    f"frame {frame_indices[frame_idx]}"
                )
            if frame_mask_chunks:
                frame_masks_out = torch.cat(frame_mask_chunks, dim=0)
            else:
                frame_masks_out = torch.empty(
                    (0, transform["height"], transform["width"]),
                    dtype=torch.bool,
                )
            # Persist each frame's full-resolution masks on the host now;
            # retaining several frames' masks on CUDA needlessly raises the
            # peak while CLIP runs later in this window.
            masks_cpu.append(frame_masks_out)
            del features, sam_input, sam_canvas, frame_mask_chunks
    del sam_wrapper, sam_predictor
    gc.collect()
    torch.cuda.empty_cache()
    budget.check(f"window {frame_indices[0]}-{frame_indices[-1]} after SAM release")

    checkpoint = _resolve_clip_checkpoint()
    budget.check(f"window {frame_indices[0]}-{frame_indices[-1]} before CLIP load")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained=str(checkpoint)
    )
    clip_model = clip_model.to(device).eval()
    budget.check(f"window {frame_indices[0]}-{frame_indices[-1]} after CLIP load")
    image_feats_cpu = []
    with torch.inference_mode():
        for frame_idx in range(frame_count):
            frame_features = []
            boxes = proposal_boxes[frame_idx]
            for begin in range(0, len(boxes), MODEL_BATCH):
                budget.check(
                    f"CLIP crops {begin}:{begin + MODEL_BATCH} frame {frame_indices[frame_idx]}"
                )
                crop_batch = _clip_input(
                    source_gpu[frame_idx], boxes[begin : begin + MODEL_BATCH]
                )
                encoded = clip_model.encode_image(crop_batch).float()
                encoded = F.normalize(encoded, dim=-1)
                frame_features.append(encoded.cpu())
                budget.check(
                    f"CLIP crops complete {begin}:{begin + MODEL_BATCH} "
                    f"frame {frame_indices[frame_idx]}"
                )
                del crop_batch, encoded
            frame_image_feats = (
                torch.cat(frame_features, dim=0)
                if frame_features else torch.empty((0, 0), dtype=torch.float32)
            )
            image_feats_cpu.append(frame_image_feats)
            del frame_image_feats
            del frame_features
    del clip_model
    gc.collect()
    torch.cuda.empty_cache()
    budget.check(f"window {frame_indices[0]}-{frame_indices[-1]} after CLIP release")

    # Pack frame outputs from host and CUDA tensors, then persist atomically.
    for frame_idx, path in enumerate(rgb_paths):
        fields = [
            ("xyxy", proposal_boxes[frame_idx].float()),
            ("confidence", proposal_conf[frame_idx].float()),
            ("class_id", proposal_class[frame_idx].long()),
            ("mask", masks_cpu[frame_idx]),
            ("image_feats", image_feats_cpu[frame_idx].float()),
        ]
        packed = _pack_frame(fields)
        image_rgb_cpu = cv2.cvtColor(cpu_images[frame_idx], cv2.COLOR_BGR2RGB)
        crops = _padded_crops(image_rgb_cpu, packed["xyxy"])
        results = {
            "xyxy": packed["xyxy"],
            "confidence": packed["confidence"],
            "class_id": packed["class_id"],
            "mask": packed["mask"],
            "classes": class_names,
            "image_crops": crops,
            "image_feats": packed["image_feats"],
            "text_feats": [],
        }
        curr_det = sv.Detections(
            xyxy=results["xyxy"], confidence=results["confidence"],
            class_id=results["class_id"], mask=results["mask"],
        )
        annotated_image, _ = vis_result_fast(cpu_images[frame_idx], curr_det, class_names)
        vis_path = (out_paths["vis"] / path.name).with_suffix(".jpg")
        if not cv2.imwrite(str(vis_path), annotated_image):
            raise OSError(f"Could not write detection visualization: {vis_path}")
        detection_path = out_paths["pkl"] / f"{path.stem}.pkl.gz"
        _atomic_save(detection_path, results)
        print(
            f"[detections] frame {frame_indices[frame_idx]} saved: "
            f"{len(results['xyxy'])} objects, {detection_path}"
        )

    del source_gpu, proposal_boxes, proposal_conf, proposal_class, masks_cpu, image_feats_cpu
    gc.collect()
    torch.cuda.empty_cache()
    budget.check(f"window {frame_indices[0]}-{frame_indices[-1]} complete")


@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="streamlined_detections")
def main(cfg: DictConfig):
    device = require_cuda_device(cfg.device)
    budget = NvmlMemoryBudget(device)
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
    exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)
    out_paths = {"pkl": get_det_out_path(exp_path), "vis": get_vis_out_path(exp_path)}
    out_paths["pkl"].mkdir(parents=True, exist_ok=True)
    out_paths["vis"].mkdir(parents=True, exist_ok=True)
    classes = ObjectClasses(cfg.classes_file, bg_classes=cfg.bg_classes, skip_bg=cfg.skip_bg)
    class_names = classes.get_classes_arr()
    save_hydra_config(cfg, exp_path)
    paths = [Path(path) for path in dataset.color_paths]
    print(
        f"[detections] {len(paths)} frames; {WINDOW_FRAMES}-frame source windows; "
        f"full-resolution 1280x720; max {MAX_DETECTIONS} detections/frame; "
        f"YOLO frame batch 1; SAM/CLIP mini-batch cap {MODEL_BATCH}"
    )
    for begin in trange(0, len(paths), WINDOW_FRAMES):
        stop = min(begin + WINDOW_FRAMES, len(paths))
        _process_window(
            paths[begin:stop], list(range(begin, stop)), cfg,
            class_names, device, budget, out_paths,
        )
    if cfg.save_video:
        save_video_detections(exp_path)


if __name__ == "__main__":
    main()
