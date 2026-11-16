"""YOLOE-26-L detections with masked-object or region-proposal features.

This module is deliberately separate from ``streamlined_detections``. Masked
mode uses YOLOE segmentation outputs; region-proposal mode uses standard
YOLOE boxes and rectangular geometry masks. Both cache formats are compatible
with the existing VLM and RGB-D mapping stages.
"""

from __future__ import annotations

import gc
import gzip
import importlib.util
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import hydra
import numpy as np
import supervision as sv
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import trange

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
from conceptgraph.utils.vis import vis_result_fast


WINDOW_FRAMES = 8
MAX_DETECTIONS = 300
ENCODER_BATCH = 4
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.7
CROP_PADDING = 20
ENCODER_DTYPE = torch.float16
MIN_YOLOE_CHECKPOINT_BYTES = 50 * 1024**2
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)


def _resolve_checkpoint(value: str | Path, label: str, *, minimum_bytes: int = 1) -> Path:
    checkpoint = Path(value).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label} checkpoint is missing: {checkpoint}")
    if checkpoint.stat().st_size < minimum_bytes:
        raise FileNotFoundError(
            f"{label} checkpoint is incomplete or undersized: {checkpoint} "
            f"({checkpoint.stat().st_size} bytes; expected at least {minimum_bytes})"
        )
    return checkpoint.resolve()


def _require_cuda_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(
            "The modernized detector requires CUDA; set device=cuda or device=cuda:0. "
            "There is no CPU fallback."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {resolved} was requested, but PyTorch reports CUDA unavailable")
    return resolved


def resolve_yoloe_checkpoint(cfg: DictConfig) -> Path:
    configured = str(cfg.model.yoloe_checkpoint)
    if configured == "auto":
        mode = str(cfg.features.input_mode)
        if mode not in {"masked", "region_proposal"}:
            raise ValueError(
                "features.input_mode must be 'masked' or 'region_proposal'; "
                f"got {mode!r}"
            )
        configured = "yoloe-26l-seg.pt" if mode == "masked" else "yoloe-26l.pt"
    checkpoint = Path(configured).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = get_weights_dir(cfg.repo_root) / checkpoint
    return _resolve_checkpoint(checkpoint, "YOLOE-26-L", minimum_bytes=MIN_YOLOE_CHECKPOINT_BYTES)


def _check_prompt_encoder(cfg: DictConfig) -> Path:
    configured = os.getenv("CONCEPTGRAPH_YOLOE_TEXT_ENCODER") or str(cfg.model.prompt_encoder)
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path(cfg.repo_root) / path
    path = _resolve_checkpoint(path, "YOLOE-26 text encoder", minimum_bytes=100 * 1024**2)
    if path.name != "mobileclip2_b.ts":
        raise ValueError(
            "YOLOE-26 text prompting requires the Ultralytics MobileCLIP2 asset "
            f"named mobileclip2_b.ts; got {path}"
        )
    return path


def _require_local_prompt_tokenizer() -> None:
    """Keep YOLOE prompt setup offline and fail clearly when CLIP is absent."""
    if importlib.util.find_spec("clip") is None:
        raise RuntimeError(
            "YOLOE text prompting requires the project's `clip` dependency. "
            "Install the declared dependencies with `uv sync` before running offline."
        )
    # Ultralytics otherwise attempts an implicit pip install from GitHub when
    # its prompt tokenizer import fails. This pipeline only uses local assets.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def _yoloe_prompt_names(class_names: list[str]) -> list[str]:
    """Make YOLOE-safe text prompts while retaining class-index alignment."""
    prompts = [re.sub(r"\s+", "-", name.strip()) for name in class_names]
    if any(not prompt or " " in prompt for prompt in prompts):
        raise ValueError("YOLOE class prompts must be non-empty and contain no spaces")
    if len(set(prompts)) != len(prompts):
        raise ValueError(
            "YOLOE class prompt normalization produced duplicate names; "
            "use an unambiguous class list."
        )
    return prompts


def _model_snapshot(model_id: str, checkpoint: str | Path | None, hf_home: Path) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if path.is_file():
            path = path.parent
        if not path.is_dir():
            raise FileNotFoundError(
                f"Local model snapshot for {model_id} is missing: {path}. "
                "Prepare it with scripts/download_models.py before running offline."
            )
        return path.resolve()
    folder = hf_home / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    def modified_at(item: Path) -> float:
        try:
            return item.stat().st_mtime
        except OSError:
            return 0.0

    candidates = sorted(folder.glob("*"), key=modified_at, reverse=True)
    for candidate in candidates:
        if (candidate / "config.json").is_file() and any(candidate.glob("*.safetensors")):
            return candidate.resolve()
    raise FileNotFoundError(
        f"Local checkpoint for {model_id} is missing under {folder}. "
        "Prepare it with scripts/download_models.py before running offline."
    )


def _resolve_clip_snapshot(checkpoint: str | Path | None, hf_home: Path) -> Path:
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if path.is_file():
            _resolve_checkpoint(path, "CLIP ViT-H/14", minimum_bytes=1 * 1024**3)
            return path.parent.resolve()
        if path.is_dir() and (path / "open_clip_model.safetensors").is_file():
            return path.resolve()
        raise FileNotFoundError(f"Local CLIP ViT-H/14 checkpoint is missing: {path}")
    root = hf_home / "hub" / "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K" / "snapshots"
    candidates = sorted(root.glob("*/open_clip_model.safetensors"), reverse=True)
    for path in candidates:
        try:
            _resolve_checkpoint(path, "CLIP ViT-H/14", minimum_bytes=1 * 1024**3)
            return path.parent.resolve()
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"Local CLIP ViT-H/14 checkpoint is missing under {root}; prepare it with "
        "scripts/download_models.py --download --only clip."
    )


def _validate_encoder_assets(key: str, snapshot: Path) -> None:
    if key == "clip":
        _resolve_checkpoint(
            snapshot / "open_clip_model.safetensors", "CLIP ViT-H/14",
            minimum_bytes=1 * 1024**3,
        )
        return
    weight_files = [
        path for pattern in ("*.safetensors", "*.bin") for path in snapshot.glob(pattern)
        if path.is_file()
    ]
    if sum(path.stat().st_size for path in weight_files) < 500 * 1024**2:
        raise FileNotFoundError(
            f"No complete model weights were found in {snapshot}; expected at least 500 MiB of local weights."
        )
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"Model config is missing from local snapshot {snapshot}")
    if key == "siglip2":
        if not (snapshot / "preprocessor_config.json").is_file():
            raise FileNotFoundError(f"SigLIP2 processor config is missing from {snapshot}")
        if not any((snapshot / name).is_file() for name in ("tokenizer.json", "spiece.model", "tokenizer.model")):
            raise FileNotFoundError(f"SigLIP2 tokenizer assets are missing from {snapshot}")
    elif key == "fg_clip2":
        processor_files = ("preprocessor_config.json", "processor_config.json")
        if not any((snapshot / name).is_file() for name in processor_files):
            raise FileNotFoundError(f"FG-CLIP2 image processor config is missing from {snapshot}")
        if not any((snapshot / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")):
            raise FileNotFoundError(f"FG-CLIP2 tokenizer assets are missing from {snapshot}")
    else:
        raise ValueError(f"Unsupported modernized feature encoder: {key}")


def _processor_mean(key: str, snapshot: Path) -> tuple[int, int, int]:
    if key == "clip":
        mean = CLIP_IMAGE_MEAN
    elif key == "siglip2":
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
        mean = processor.image_processor.image_mean
        del processor
    elif key == "fg_clip2":
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(
            str(snapshot), trust_remote_code=True, local_files_only=True
        )
        mean = processor.image_mean
        del processor
    else:
        raise ValueError(f"Unsupported modernized feature encoder: {key}")
    return tuple(int(round(float(channel) * 255)) for channel in mean)


class _FeatureEncoder:
    """Small local-only adapter for the two selected shared-space encoders."""

    def __init__(self, key: str, snapshot: Path, device: torch.device):
        self.key = key
        self.device = device
        self.snapshot = snapshot
        self.model = self.processor = self.tokenizer = self.preprocess = None
        if key == "siglip2":
            from transformers import AutoModel, AutoProcessor

            self.model = AutoModel.from_pretrained(
                str(snapshot), torch_dtype=ENCODER_DTYPE, local_files_only=True
            ).to(device).eval()
            self.processor = AutoProcessor.from_pretrained(
                str(snapshot), local_files_only=True
            )
            self.image_mean = tuple(float(x) for x in self.processor.image_processor.image_mean)
            self.revision = getattr(self.model.config, "_commit_hash", None) or snapshot.name
        elif key == "clip":
            import open_clip

            checkpoint = _resolve_checkpoint(snapshot / "open_clip_model.safetensors", "CLIP ViT-H/14", minimum_bytes=1 * 1024**3)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14", pretrained=str(checkpoint)
            )
            self.model = self.model.to(device=device, dtype=ENCODER_DTYPE).eval()
            self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
            normalization = next(
                transform for transform in self.preprocess.transforms
                if hasattr(transform, "mean") and hasattr(transform, "std")
            )
            self.image_mean = tuple(float(x) for x in normalization.mean)
            self.revision = snapshot.name
        elif key == "fg_clip2":
            from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                str(snapshot), trust_remote_code=True, torch_dtype=ENCODER_DTYPE,
                local_files_only=True,
            ).to(device).eval()
            self.processor = AutoImageProcessor.from_pretrained(
                str(snapshot), trust_remote_code=True, local_files_only=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(snapshot), trust_remote_code=True, local_files_only=True
            )
            self.image_mean = tuple(float(x) for x in self.processor.image_mean)
            self.revision = getattr(self.model.config, "_commit_hash", None) or snapshot.name
        else:
            raise ValueError(f"Unsupported modernized feature encoder: {key}")
        if len(self.image_mean) != 3:
            raise ValueError(f"{key} processor must expose an RGB channel mean")

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(
                self.device,
                dtype=ENCODER_DTYPE if value.is_floating_point() else value.dtype,
            ) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty((0, 0), dtype=torch.float32)
        if self.key == "clip":
            batch = torch.stack([self.preprocess(image) for image in images]).to(
                device=self.device, dtype=ENCODER_DTYPE
            )
            values = self.model.encode_image(batch)
        elif self.key == "siglip2":
            inputs = self._move(self.processor(images=images, return_tensors="pt"))
            values = self.model.get_image_features(**inputs)
        else:
            patch_counts = [(im.width // 16) * (im.height // 16) for im in images]
            maximum = max(patch_counts)
            max_patches = (
                1024 if maximum > 784 else 784 if maximum > 576 else
                576 if maximum > 256 else 256 if maximum > 128 else 128
            )
            inputs = self._move(self.processor(
                images=images, max_num_patches=max_patches, return_tensors="pt"
            ))
            values = self.model.get_image_features(**inputs)
        return _normalize(values)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        if self.key == "clip":
            values = self.model.encode_text(
                self.tokenizer([f"a photo of a {text}" for text in texts]).to(self.device)
            )
        elif self.key == "siglip2":
            inputs = self.processor(
                text=[f"a photo of a {text}" for text in texts],
                padding="max_length", truncation=True, max_length=64, return_tensors="pt",
            )
            values = self.model.get_text_features(**self._move(inputs))
        else:
            inputs = self.tokenizer(
                [f"a photo of a {text}" for text in texts],
                padding="max_length", max_length=64, truncation=True, return_tensors="pt",
            )
            inputs = self._move(inputs)
            length = inputs["input_ids"].shape[1]
            inputs["position_ids"] = torch.arange(length, device=self.device).unsqueeze(0).expand(
                inputs["input_ids"].shape[0], -1
            )
            values = self.model.get_text_features(**inputs, walk_type="short")
        return _normalize(values)

    def close(self) -> None:
        self.model = self.processor = self.tokenizer = self.preprocess = None
        gc.collect()
        torch.cuda.empty_cache()


def _normalize(value: Any) -> torch.Tensor:
    if hasattr(value, "pooler_output"):
        value = value.pooler_output
    elif isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Encoder returned {type(value).__name__}; expected a tensor")
    if value.ndim == 3:
        value = value[:, 0]
    if value.ndim != 2:
        raise ValueError(f"Expected [batch, dimension] embedding output, got {tuple(value.shape)}")
    normalized = F.normalize(value.float(), dim=-1)
    if not torch.isfinite(normalized).all():
        raise ValueError("Encoder returned non-finite features")
    return normalized


def _crop(image_rgb: np.ndarray, box: np.ndarray, mask: np.ndarray | None,
          *, fill_rgb: tuple[int, int, int]) -> Image.Image:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = box
    left = max(0, int(np.floor(x1)) - CROP_PADDING)
    top = max(0, int(np.floor(y1)) - CROP_PADDING)
    right = min(width, int(np.ceil(x2)) + CROP_PADDING)
    bottom = min(height, int(np.ceil(y2)) + CROP_PADDING)
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop box {box.tolist()} for source {width}x{height}")
    crop = image_rgb[top:bottom, left:right].copy()
    if mask is not None:
        crop_mask = np.asarray(mask[top:bottom, left:right], dtype=bool)
        crop[~crop_mask] = np.asarray(fill_rgb, dtype=np.uint8)
    return Image.fromarray(crop, mode="RGB")


def _validate_record(record: dict, height: int, width: int, class_count: int) -> None:
    boxes = np.asarray(record["xyxy"])
    scores = np.asarray(record["confidence"]).reshape(-1)
    classes = np.asarray(record["class_id"]).reshape(-1)
    masks = np.asarray(record["mask"])
    n = len(boxes)
    if boxes.shape != (n, 4) or len(scores) != n or len(classes) != n or masks.shape != (n, height, width):
        raise ValueError("YOLOE output fields have inconsistent detection counts or mask dimensions")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("YOLOE returned non-finite boxes or confidence scores")
    if n and (np.any(classes < 0) or np.any(classes >= class_count)):
        raise ValueError("YOLOE returned class IDs outside the configured prompt list")
    if np.any(boxes[:, 0::2] < 0) or np.any(boxes[:, 0::2] > width) or np.any(boxes[:, 1::2] < 0) or np.any(boxes[:, 1::2] > height):
        raise ValueError("YOLOE returned boxes outside the source image")
    if n and (np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1])):
        raise ValueError("YOLOE returned a box with non-positive width or height")
    if n and (np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("YOLOE returned confidence scores outside [0, 1]")
    if "image_crops" in record and len(record["image_crops"]) != n:
        raise ValueError("Padded image crop count does not match YOLOE detections")
    for image_key, text_key in (("image_feats", "text_feats"), ("image_feats_box", "text_feats_box")):
        if image_key not in record or text_key not in record:
            continue
        image_features = np.asarray(record[image_key])
        text_features = np.asarray(record[text_key])
        if image_features.ndim != 2 or image_features.shape[0] != n:
            raise ValueError(f"{image_key} must have one feature row per detection")
        if text_features.ndim != 2 or text_features.shape[0] != class_count:
            raise ValueError(f"{text_key} must have one feature row per class prompt")
        if image_features.shape[1] != text_features.shape[1]:
            raise ValueError(f"{image_key} and {text_key} belong to incompatible feature dimensions")
        for name, features in ((image_key, image_features), (text_key, text_features)):
            if not np.isfinite(features).all():
                raise ValueError(f"{name} contains non-finite values")
            if features.shape[0]:
                norms = np.linalg.norm(features, axis=1)
                if not np.allclose(norms, 1.0, rtol=2e-3, atol=2e-3):
                    raise ValueError(f"{name} rows are not L2-normalized")


def _rectangular_masks(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Build box-shaped masks for geometry when the detector has no mask head."""
    masks = np.zeros((len(boxes), height, width), dtype=bool)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        left = max(0, min(width, int(np.floor(x1))))
        top = max(0, min(height, int(np.floor(y1))))
        right = max(0, min(width, int(np.ceil(x2))))
        bottom = max(0, min(height, int(np.ceil(y2))))
        if right > left and bottom > top:
            masks[index, top:bottom, left:right] = True
    return masks


def _encode_variant(key: str, model_id: str, checkpoint: str | Path | None,
                    images: list[Image.Image], class_names: list[str],
                    class_ids: np.ndarray, hf_home: Path, device: torch.device,
                    budget: NvmlMemoryBudget, *, view: str) -> tuple[np.ndarray, np.ndarray, dict]:
    snapshot = _resolve_clip_snapshot(checkpoint, hf_home) if key == "clip" else _model_snapshot(model_id, checkpoint, hf_home)
    encoder = _FeatureEncoder(key, snapshot, device)
    try:
        with torch.inference_mode():
            text = encoder.encode_text(class_names).cpu()
            budget.check(f"{key} text features complete")
            image_chunks = []
            for begin in range(0, len(images), ENCODER_BATCH):
                image_chunks.append(
                    encoder.encode_images(images[begin : begin + ENCODER_BATCH]).cpu()
                )
                budget.check(f"{key} image features {begin}:{begin + ENCODER_BATCH}")
            image = (
                torch.cat(image_chunks, dim=0)
                if image_chunks else torch.empty((0, text.shape[1]), dtype=torch.float32)
            )
            paired_text = text
        metadata = {
            "encoder": model_id,
            "revision": encoder.revision,
            "checkpoint": str(snapshot),
            "preprocessing": (
                "processor-native RGB; mask-isolated object crops with exterior set to encoder mean"
                if view == "mask" else
                "processor-native RGB; padded proposal crops with 20-pixel context"
            ),
            "text_prompt_template": "a photo of a {class}",
            "normalization": "L2 per row",
        }
        return image.numpy(), paired_text.numpy(), metadata
    finally:
        encoder.close()


def run_window(frame_rgbs: list[np.ndarray], cfg: DictConfig, *, device: torch.device,
               budget: NvmlMemoryBudget) -> list[dict]:
    """Run one bounded window, loading each large model once per feature phase."""
    device = _require_cuda_device(device)
    if not frame_rgbs:
        return []
    for frame_rgb in frame_rgbs:
        if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("Each frame must be a non-empty HxWx3 RGB array")
        if frame_rgb.dtype != np.uint8 or not frame_rgb.size:
            raise ValueError("Each frame must be a non-empty uint8 RGB array")
    height, width = frame_rgbs[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frame_rgbs):
        raise ValueError("All frames in a YOLOE window must have the same native dimensions")
    configured_names = cfg.classes.get("names")
    if configured_names is None:
        classes = ObjectClasses(
            classes_file_path=cfg.classes_file,
            bg_classes=cfg.bg_classes,
            skip_bg=cfg.skip_bg,
        )
        configured_names = classes.get_classes_arr()
    class_names = [str(name) for name in configured_names]
    if not class_names:
        raise ValueError("Configured YOLOE class prompt list is empty")
    prompt_names = _yoloe_prompt_names(class_names)
    yoloe_checkpoint = resolve_yoloe_checkpoint(cfg)
    prompt_encoder = _check_prompt_encoder(cfg)
    hf_home = Path(os.environ.get("HF_HOME", cfg.features.hf_home)).expanduser()
    input_mode = str(cfg.features.input_mode)
    if input_mode not in {"masked", "region_proposal"}:
        raise ValueError(
            "features.input_mode must be 'masked' or 'region_proposal'; "
            f"got {input_mode!r}"
        )
    mask_key = str(cfg.features.mask_encoder)
    box_key = str(cfg.features.box_encoder)
    siglip_model_id = str(cfg.features.mask_model_id)
    fgclip_model_id = str(cfg.features.box_model_id)
    siglip_snapshot = (
        cfg.features.get("mask_checkpoint")
        or cfg.features.get("siglip2_checkpoint")
        or None
    )
    fgclip_snapshot = (
        cfg.features.get("box_checkpoint")
        or cfg.features.get("fg_clip2_checkpoint")
        or None
    )
    if input_mode == "masked":
        active_key, active_model_id, active_snapshot = mask_key, siglip_model_id, siglip_snapshot
        view = "mask"
        feature_key, text_key = "image_feats", "text_feats"
    else:
        active_key, active_model_id, active_snapshot = box_key, fgclip_model_id, fgclip_snapshot
        view = "proposal"
        feature_key, text_key = "image_feats_box", "text_feats_box"
    active_path = (
        _resolve_clip_snapshot(active_snapshot, hf_home)
        if active_key == "clip" else
        _model_snapshot(active_model_id, active_snapshot, hf_home)
    )
    _validate_encoder_assets(active_key, active_path)
    mask_fill = _processor_mean(mask_key, active_path) if input_mode == "masked" else None
    prompt_cwd = Path.cwd() / prompt_encoder.name
    if not prompt_cwd.is_file() or prompt_encoder != prompt_cwd.resolve():
        raise FileNotFoundError(
            f"YOLOE text encoder must be available in the pipeline working directory: {prompt_cwd}. "
            "Download the matching YOLOE-26 checkpoint and run "
            "scripts/download_models.py --download --only yoloe26l yoloe26l_detect first."
        )
    budget.check("before YOLOE load")
    _require_local_prompt_tokenizer()
    from ultralytics import YOLOE, __version__ as ultralytics_version
    version = tuple(int(part) for part in ultralytics_version.split(".")[:3])
    if version < (8, 4, 0):
        raise RuntimeError(
            "YOLOE-26 requires Ultralytics 8.4.0 or newer; "
            f"the installed version is {ultralytics_version}."
        )

    model = YOLOE(str(yoloe_checkpoint))
    model.model.to(device).eval()
    # Ultralytics resolves this model-specific asset relative to the working
    # directory. Require it before set_classes so no first-run network fetch occurs.
    frame_detections = []
    try:
        # YOLOE rejects spaces in prompt names. Hyphenated aliases preserve
        # each class's position; cached labels below remain the original names.
        model.set_classes(prompt_names)
        budget.check("after YOLOE prompt setup")
        for frame_index, frame_rgb in enumerate(frame_rgbs):
            budget.check(f"before YOLOE frame {frame_index}")
            with torch.inference_mode():
                prediction = model.predict(
                    source=Image.fromarray(frame_rgb, mode="RGB"), device=str(device),
                    imgsz=int(cfg.model.image_size), conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS,
                    retina_masks=input_mode == "masked", verbose=False,
                )[0]
            if prediction.boxes is None:
                raise RuntimeError("YOLOE segmentation prediction did not return boxes")
            boxes = prediction.boxes.xyxy.detach().float().cpu().numpy().reshape(-1, 4)
            scores = prediction.boxes.conf.detach().float().cpu().numpy().reshape(-1)
            class_ids = prediction.boxes.cls.detach().long().cpu().numpy().reshape(-1)
            if prediction.masks is None and input_mode == "masked" and len(boxes):
                raise RuntimeError(
                    "The masked input mode requires native YOLOE instance masks; "
                    f"checkpoint {yoloe_checkpoint.name} returned boxes only."
                )
            if prediction.masks is None:
                masks = _rectangular_masks(boxes, height, width)
            else:
                masks = prediction.masks.data.detach().cpu().numpy()
                if masks.shape[-2:] != (height, width):
                    raise ValueError(
                        f"YOLOE native masks have shape {masks.shape[-2:]}; expected {(height, width)}"
                    )
                masks = masks > 0.5
            _validate_record(
                {"xyxy": boxes, "confidence": scores, "class_id": class_ids, "mask": masks},
                height, width, len(class_names),
            )
            frame_detections.append({
                "xyxy": boxes, "confidence": scores, "class_id": class_ids, "mask": masks,
            })
            del prediction
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    budget.check("after YOLOE release")

    padded_crops_by_frame = [
        [_crop(frame_rgb, det["xyxy"][idx], None, fill_rgb=(0, 0, 0)) for idx in range(len(det["xyxy"]))]
        for frame_rgb, det in zip(frame_rgbs, frame_detections)
    ]
    padded_crops = [crop for crops in padded_crops_by_frame for crop in crops]
    all_class_ids = np.concatenate([det["class_id"] for det in frame_detections]) if any(len(det["class_id"]) for det in frame_detections) else np.empty((0,), dtype=np.int64)
    feature_crops = padded_crops
    if input_mode == "masked":
        feature_crops = [
            _crop(frame_rgb, det["xyxy"][idx], det["mask"][idx], fill_rgb=mask_fill)
            for frame_rgb, det in zip(frame_rgbs, frame_detections)
            for idx in range(len(det["xyxy"]))
        ]
    image_features, text_features, feature_meta = _encode_variant(
        active_key, active_model_id, active_path, feature_crops, class_names,
        all_class_ids, hf_home, device, budget, view=view,
    )
    budget.check(f"after {active_model_id} release")
    records = []
    offset = 0
    for index, detection in enumerate(frame_detections):
        count = len(detection["xyxy"])
        next_offset = offset + count
        text_meta = {
            **feature_meta,
            "preprocessing": "encoder text processor; one normalized vector per configured class",
        }
        record = {
            **detection,
            "xyxy": detection["xyxy"].astype(np.float32),
            "confidence": detection["confidence"].astype(np.float32),
            "class_id": detection["class_id"].astype(np.int64),
            "mask": detection["mask"].astype(bool),
            "classes": class_names,
            "image_crops": padded_crops_by_frame[index],
            feature_key: image_features[offset:next_offset],
            text_key: text_features,
            "map_feature_key": feature_key,
            "feature_metadata": {
                "yoloe_prompt_names": prompt_names,
                "input_mode": input_mode,
                feature_key: feature_meta,
                text_key: text_meta,
                "crop_padding_pixels": CROP_PADDING,
                "mask_source": "YOLOE native instance mask" if input_mode == "masked" else "rectangle from YOLOE box",
                **({"mask_exterior_fill": f"{mask_key} processor RGB mean, rounded to uint8"} if input_mode == "masked" else {}),
            },
        }
        _validate_record(record, height, width, len(class_names))
        if len(record[feature_key]) != count:
            raise ValueError("Encoder image feature counts do not align with YOLOE detections")
        records.append(record)
        offset = next_offset
    return records


def run_frame(frame_rgb: np.ndarray, cfg: DictConfig, *, device: torch.device,
              budget: NvmlMemoryBudget) -> dict:
    """Run one configured RGB frame through the same bounded-window path."""
    return run_window([frame_rgb], cfg, device=device, budget=budget)[0]


def save_gobs(path: str | Path, record: dict) -> None:
    """Atomically save one gzip-compressed pickle detection record."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        with gzip.open(temporary, "wb") as stream:
            pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_gobs(path: str | Path) -> dict:
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def _read_rgb(path: Path, *, expected_height: int, expected_width: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read RGB frame: {path}")
    if image.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            f"Expected native {expected_width}x{expected_height} RGB frame at {path}; "
            f"got {image.shape[1]}x{image.shape[0]}"
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="modernized_detections")
def main(cfg: DictConfig) -> None:
    device = _require_cuda_device(cfg.device)
    budget = NvmlMemoryBudget(device)
    dataset = get_dataset(
        dataconfig=cfg.dataset_config, start=cfg.start, end=cfg.end,
        stride=cfg.stride, basedir=cfg.dataset_root, sequence=cfg.scene_id,
        desired_height=cfg.desired_height, desired_width=cfg.desired_width,
        device="cpu", dtype=torch.float,
    )
    class_config = ObjectClasses(
        classes_file_path=cfg.classes_file, bg_classes=cfg.bg_classes,
        skip_bg=cfg.skip_bg,
    )
    OmegaConf.update(cfg, "classes.names", class_config.get_classes_arr(), force_add=True)
    exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)
    paths = {"pkl": get_det_out_path(exp_path), "vis": get_vis_out_path(exp_path)}
    save_hydra_config(cfg, exp_path)
    frame_paths = [Path(path) for path in dataset.color_paths]
    print(
        f"[modernized] YOLOE {resolve_yoloe_checkpoint(cfg)}; "
        f"input mode {cfg.features.input_mode}; "
        f"feature encoder {cfg.features.mask_model_id if cfg.features.input_mode == 'masked' else cfg.features.box_model_id}; "
        f"{len(frame_paths)} frames; {MAX_DETECTIONS} max detections"
    )
    for begin in trange(0, len(frame_paths), WINDOW_FRAMES):
        stop = min(begin + WINDOW_FRAMES, len(frame_paths))
        window_paths = frame_paths[begin:stop]
        window_rgb = [
            _read_rgb(
                path,
                expected_height=int(dataset.orig_height),
                expected_width=int(dataset.orig_width),
            )
            for path in window_paths
        ]
        records = run_window(window_rgb, cfg, device=device, budget=budget)
        for frame_index in range(begin, stop):
            path = frame_paths[frame_index]
            rgb = window_rgb[frame_index - begin]
            record = records[frame_index - begin]
            visualization = vis_result_fast(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                sv.Detections(
                    xyxy=record["xyxy"], confidence=record["confidence"],
                    class_id=record["class_id"], mask=record["mask"],
                ),
                record["classes"],
            )[0]
            vis_path = (paths["vis"] / path.name).with_suffix(".jpg")
            if not cv2.imwrite(str(vis_path), visualization):
                raise OSError(f"Could not write detection visualization: {vis_path}")
            save_gobs(paths["pkl"] / f"{path.stem}.pkl.gz", record)
            print(f"[modernized] {path.name}: {len(record['xyxy'])} instances")


if __name__ == "__main__":
    main()
