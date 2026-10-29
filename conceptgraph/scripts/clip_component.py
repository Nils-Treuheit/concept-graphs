"""Ground-truth benchmark for the ConceptGraphs object image/text component.

The benchmark uses COCO 2017 validation instance masks, boxes, and human
category labels as fixed, oracle proposals. It evaluates the current padded-box
crop and a masked-object variant at the embedding insertion point:

1. zero-shot identification of each ground-truth object from the COCO labels;
2. text-query retrieval of the matching objects from the full object bank.

Run from the repository root after releasing local model engines:

    python3 /home/nils/Project_Code/codex_opencode_pipeline/src/executor.py release-local
    .venv/bin/python -m conceptgraph.scripts.clip_component

COCO images are expected at ``/mnt/HDD1/Project_Data/public_datasets/coco/val2017``
and annotations at ``.../annotations/instances_val2017.json``. Override them
with ``--image-root`` and ``--annotations``. The benchmark is GPU-only.

DINOv3 is deliberately not reported as a text-query replacement: it provides
image features without the shared text/image space this component uses. A
separate tracked-object data-association benchmark is needed to assess DINOv3
at that visual-only mapping use.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "conceptgraph").is_dir():
            return candidate
    raise RuntimeError("Could not locate the ConceptGraphs repository root")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_COCO_ROOT = Path("/mnt/HDD1/Project_Data/public_datasets/coco")
DEFAULT_CAPTIONS = DEFAULT_COCO_ROOT / "annotations" / "captions_val2017.json"
DEFAULT_HF_HOME = Path("/mnt/HDD1/unsloth_and_hugging_face_models/huggingface")
MODEL_IDS = {
    "clip": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "siglip2-base": "google/siglip2-base-patch16-224",
    "siglip2-so400m": "google/siglip2-so400m-patch14-384",
    "fg-clip2-base": "qihoo360/fg-clip2-base",
    "fg-clip2-so400m": "qihoo360/fg-clip2-so400m",
    "metaclip2": "facebook/metaclip-2-worldwide-l14",
}
MIN_CLIP_CHECKPOINT_BYTES = 1 * 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_COCO_ROOT / "val2017")
    parser.add_argument(
        "--annotations", type=Path,
        default=DEFAULT_COCO_ROOT / "annotations" / "instances_val2017.json",
    )
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS,
                        help="COCO image-level human captions for caption-to-label retrieval")
    parser.add_argument("--models", nargs="+", choices=(*MODEL_IDS,), default=list(MODEL_IDS))
    parser.add_argument("--hf-home", type=Path, default=Path(os.getenv("HF_HOME", DEFAULT_HF_HOME)))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=128,
                        help="Text encoding batch size, used for the caption bank")
    parser.add_argument(
        "--max-objects", type=int, default=-1,
        help="Optional deterministic limit for a short run; default evaluates every non-crowd object.",
    )
    parser.add_argument("--padding", type=int, default=20,
                        help="Pixels of context around each ground-truth box, matching the pipeline.")
    parser.add_argument(
        "--timing-repeats", type=int, default=1,
        help="Full object-bank encoding passes per condition; metrics use the final pass.",
    )
    parser.add_argument("--output", type=Path,
                        help="JSON output (default: .agentic_workflow/workspace/clip-component-benchmark/...).")
    parser.add_argument("--check-data-only", action="store_true",
                        help="Check COCO inputs and local model checkpoints without using the GPU")
    return parser.parse_args()


def _load_observations(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str], dict]:
    image_root = args.image_root.expanduser().resolve()
    annotation_path = args.annotations.expanduser().resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"COCO image directory not found: {image_root}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO instance annotations not found: {annotation_path}")

    coco = json.loads(annotation_path.read_text())
    categories = sorted(coco["categories"], key=lambda row: int(row["id"]))
    category_ids = [int(row["id"]) for row in categories]
    category_names = [str(row["name"]).strip() for row in categories]
    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    images = {int(row["id"]): row for row in coco["images"]}
    annotations = sorted(coco["annotations"], key=lambda row: int(row["id"]))
    annotations = [row for row in annotations if not row.get("iscrowd", 0)]
    if args.max_objects > 0:
        annotations = annotations[:args.max_objects]

    observations = []
    for annotation in annotations:
        category_id = int(annotation["category_id"])
        if category_id not in category_index:
            continue
        image_record = images[int(annotation["image_id"])]
        image_id = int(image_record["id"])
        path = image_root / image_record["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"COCO image is missing: {path}")
        x, y, width, height = (float(value) for value in annotation["bbox"])
        left = max(0, int(np.floor(x)) - args.padding)
        top = max(0, int(np.floor(y)) - args.padding)
        right = min(int(image_record["width"]), int(np.ceil(x + width)) + args.padding)
        bottom = min(int(image_record["height"]), int(np.ceil(y + height)) + args.padding)
        if right <= left or bottom <= top:
            continue
        observations.append({
            "image_path": str(path),
            "crop_box": (left, top, right, bottom),
            "region_box": (
                max(0.0, x), max(0.0, y),
                min(float(image_record["width"]), x + width),
                min(float(image_record["height"]), y + height),
            ),
            "segmentation": annotation["segmentation"],
            "category_index": category_index[category_id],
            "category_id": category_id,
            "annotation_id": int(annotation["id"]),
            "image_id": image_id,
        })
    if not observations:
        raise ValueError("No valid non-crowd COCO ground-truth instances were loaded")
    counts = np.bincount(
        [row["category_index"] for row in observations], minlength=len(category_names)
    )
    info = {
        "dataset": "COCO 2017 validation instances",
        "image_root": str(image_root),
        "annotations": str(annotation_path),
        "proposal_source": "human-annotated COCO instance masks and bounding boxes",
        "label_source": "human-annotated COCO category labels",
        "images_used": len({row["image_id"] for row in observations}),
        "ground_truth_objects": len(observations),
        "categories": len(category_names),
        "objects_per_category": {
            category_names[index]: int(count)
            for index, count in enumerate(counts) if count
        },
    }
    return observations, category_names, info


CAPTION_ALIASES = {
    "person": ("people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children"),
    "bicycle": ("bike", "bikes",),
    "motorcycle": ("motorbike", "motorbikes"),
    "airplane": ("plane", "planes", "jet", "jets"),
    "traffic light": ("stoplight", "stoplights"),
    "handbag": ("purse", "purses", "bag", "bags"),
    "suitcase": ("luggage",),
    "sports ball": ("sports balls",),
    "baseball glove": ("baseball gloves",),
    "tennis racket": ("tennis racquet", "tennis racquets"),
    "wine glass": ("wine glasses",),
    "couch": ("sofa", "sofas"),
    "potted plant": ("houseplant", "houseplants"),
    "dining table": ("dining tables",),
    "tv": ("television", "televisions", "tv", "tvs"),
    "laptop": ("computer", "computers", "notebook computer", "notebook computers"),
    "remote": ("remote control", "remote controls"),
    "cell phone": ("cellphone", "cellphones", "mobile phone", "mobile phones", "phone", "phones"),
    "refrigerator": ("fridge", "fridges"),
    "donut": ("doughnut", "doughnuts"),
    "hair drier": ("hair dryer", "hair dryers"),
    "teddy bear": ("teddy bears",),
    "cup": ("mug", "mugs"),
}


def _caption_category_matches(caption: str, category: str) -> bool:
    terms = (category, *CAPTION_ALIASES.get(category, ()))
    for term in terms:
        if term == "person":
            pattern = r"\bperson\b|\bpersons\b"
        elif term in {"knife", "knives"}:
            pattern = r"\bknife\b|\bknives\b"
        elif term in {"scissors"}:
            pattern = r"\bscissors\b"
        elif term in {"bus"}:
            pattern = r"\bbus\b|\bbuses\b"
        else:
            escaped = re.escape(term)
            plural = "" if term.endswith("s") else "s?"
            pattern = rf"\b{escaped}{plural}\b"
        if re.search(pattern, caption, flags=re.IGNORECASE):
            return True
    return False


def _load_caption_queries(caption_path: Path, observations: list[dict],
                          categories: list[str]) -> tuple[list[dict], dict]:
    if not caption_path.is_file():
        raise FileNotFoundError(f"COCO caption annotations not found: {caption_path}")
    caption_data = json.loads(caption_path.read_text(encoding="utf-8"))
    observed_categories: dict[int, set[int]] = {}
    object_indices: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(observations):
        image_id = int(row["image_id"])
        category_index = int(row["category_index"])
        observed_categories.setdefault(image_id, set()).add(category_index)
        object_indices.setdefault((image_id, category_index), []).append(index)

    queries = []
    grounded_pairs = 0
    for annotation in caption_data["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in observed_categories:
            continue
        caption = str(annotation["caption"]).strip()
        matched_categories = [
            index for index, category in enumerate(categories)
            if index in observed_categories[image_id]
            and _caption_category_matches(caption, category)
        ]
        if matched_categories:
            grounded_pairs += len(matched_categories)
        queries.append({
            "caption_id": int(annotation["id"]),
            "image_id": image_id,
            "caption": caption,
            "category_indices": matched_categories,
            "positive_indices": {
                index: object_indices[(image_id, index)] for index in matched_categories
            },
        })
    info = {
        "source": str(caption_path.expanduser().resolve()),
        "caption_count_for_used_images": len(queries),
        "captions_with_at_least_one_gt_category_mention": sum(bool(row["category_indices"]) for row in queries),
        "caption_category_query_pairs": grounded_pairs,
        "matching": "case-insensitive word-boundary keyword matching against COCO category names plus listed common aliases; only labels present in the same image are considered positives",
        "grounding_limit": "COCO captions describe whole images, not individually annotated instances. Positive regions are assigned by image ID and mentioned category, so this is a weakly grounded caption-to-label retrieval measure.",
    }
    return queries, info


def _encode_text_batched(encoder: Encoder, texts: list[str], batch_size: int,
                         device: torch.device, prompt_template: bool,
                         fg_walk_type: str = "short") -> tuple[torch.Tensor, float]:
    if not texts:
        return torch.empty((0, 0)), 0.0
    chunks = []
    started = time.perf_counter()
    for begin in range(0, len(texts), batch_size):
        with torch.inference_mode():
            chunks.append(encoder.encode_text(
                texts[begin:begin + batch_size], prompt_template=prompt_template,
                fg_walk_type=fg_walk_type,
            ).cpu())
    torch.cuda.synchronize(device)
    return torch.cat(chunks), time.perf_counter() - started


def _score_caption_queries(image_features: torch.Tensor, caption_features: torch.Tensor,
                           queries: list[dict], categories: list[str],
                           device: torch.device) -> dict:
    if not queries:
        return {
            "caption_category_query_pairs_scored": 0,
            "caption_to_label_retrieval_mrr_macro_category": float("nan"),
            **{f"caption_to_label_retrieval_recall_at_{cutoff}_macro_category": float("nan")
               for cutoff in (1, 5, 10, 50)},
        }
    image_features = F.normalize(image_features.float(), dim=-1).to(device)
    caption_features = F.normalize(caption_features.float(), dim=-1).to(device)
    hits = {1: {}, 5: {}, 10: {}, 50: {}}
    reciprocal_ranks: dict[str, list[float]] = {}
    scored_pairs = 0
    top_k = min(50, len(image_features))
    query_batch_size = 128
    for begin in range(0, len(queries), query_batch_size):
        query_end = min(begin + query_batch_size, len(queries))
        with torch.inference_mode():
            similarities = caption_features[begin:query_end] @ image_features.T
            top_indices = torch.topk(similarities, k=top_k, dim=1).indices.cpu().numpy()
        for local_index, query in enumerate(queries[begin:query_end]):
            ranked = top_indices[local_index]
            for category_index in query["category_indices"]:
                name = categories[category_index]
                positive = set(query["positive_indices"][category_index])
                positions = [rank for rank, object_index in enumerate(ranked, start=1)
                             if int(object_index) in positive]
                first_rank = positions[0] if positions else None
                scored_pairs += 1
                reciprocal_ranks.setdefault(name, []).append(1.0 / first_rank if first_rank else 0.0)
                for cutoff in hits:
                    hits[cutoff].setdefault(name, []).append(
                        float(any(rank <= cutoff for rank in positions))
                    )

    if not scored_pairs:
        return {
            "caption_category_query_pairs_scored": 0,
            "caption_to_label_retrieval_mrr_macro_category": float("nan"),
            **{f"caption_to_label_retrieval_recall_at_{cutoff}_macro_category": float("nan") for cutoff in hits},
        }
    return {
        "caption_category_query_pairs_scored": scored_pairs,
        "caption_to_label_retrieval_mrr_macro_category": float(np.mean([
            np.mean(values) for values in reciprocal_ranks.values()
        ])),
        **{
            f"caption_to_label_retrieval_recall_at_{cutoff}_macro_category": float(np.mean([
                np.mean(values) for values in by_category.values()
            ]))
            for cutoff, by_category in hits.items()
        },
    }


def _resolve_clip_checkpoint(hf_home: Path) -> Path:
    model_repo = MODEL_IDS["clip"]
    snapshot_root = hf_home / "hub" / f"models--{model_repo.replace('/', '--')}" / "snapshots"
    for path in sorted(snapshot_root.glob("*/open_clip_model.safetensors"), reverse=True):
        try:
            resolved = path.resolve(strict=True)
            if resolved.is_file() and resolved.stat().st_size >= MIN_CLIP_CHECKPOINT_BYTES:
                return path.absolute()
        except OSError:
            continue
    raise FileNotFoundError(f"Complete local CLIP ViT-H/14 checkpoint not found under {snapshot_root}")


def _check_local_model_cache(name: str, hf_home: Path) -> Path:
    if name == "clip":
        return _resolve_clip_checkpoint(hf_home)
    model_repo = MODEL_IDS[name]
    snapshot_root = hf_home / "hub" / f"models--{model_repo.replace('/', '--')}" / "snapshots"
    required = ("model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json")
    for snapshot in sorted(snapshot_root.glob("*"), reverse=True):
        if all((snapshot / filename).is_file() for filename in required):
            return snapshot
    missing = ", ".join(required)
    raise FileNotFoundError(
        f"Complete local checkpoint for {name} ({model_repo}) not found under {snapshot_root}; "
        f"expected {missing}"
    )


def _move_inputs(inputs: dict, device: torch.device, dtype: torch.dtype) -> dict:
    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        else:
            moved[key] = value
    return moved


def _normalize_features(value: Any) -> torch.Tensor:
    if hasattr(value, "pooler_output"):
        value = value.pooler_output
    elif isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Encoder returned {type(value).__name__}, expected a tensor")
    if value.ndim == 3:
        value = value[:, 0]
    if value.ndim != 2:
        raise ValueError(f"Expected [batch, dimension] features, got {tuple(value.shape)}")
    return F.normalize(value.float(), dim=-1)


class Encoder:
    def __init__(self, name: str, hf_home: Path, device: torch.device):
        self.name = name
        self.device = device
        self.dtype = torch.float16
        self.model = self.processor = self.tokenizer = self.preprocess = None
        self.image_mean = None
        started = time.perf_counter()
        if name == "clip":
            import open_clip
            checkpoint = _resolve_clip_checkpoint(hf_home)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14", pretrained=str(checkpoint)
            )
            self.model = self.model.to(device=device, dtype=self.dtype).eval()
            self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
            normalization = next(
                (transform for transform in self.preprocess.transforms
                 if hasattr(transform, "mean") and hasattr(transform, "std")),
                None,
            )
            if normalization is not None:
                self.image_mean = tuple(float(channel) for channel in normalization.mean)
            self.model_id = MODEL_IDS[name]
            self.revision = checkpoint.parent.name
        elif name.startswith("siglip2-"):
            from transformers import AutoModel, AutoProcessor
            self.model_id = MODEL_IDS[name]
            self.model = AutoModel.from_pretrained(
                self.model_id, torch_dtype=self.dtype, local_files_only=True
            ).to(device).eval()
            self.revision = getattr(self.model.config, "_commit_hash", None)
            self.processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
            self.image_mean = tuple(self.processor.image_processor.image_mean)
        elif name.startswith("fg-clip2-"):
            from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
            self.model_id = MODEL_IDS[name]
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype=self.dtype,
                local_files_only=True,
            ).to(device).eval()
            self.revision = getattr(self.model.config, "_commit_hash", None)
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_id, trust_remote_code=True, local_files_only=True
            )
            self.image_mean = tuple(self.processor.image_mean)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True, local_files_only=True
            )
        elif name == "metaclip2":
            from transformers import AutoModel, AutoProcessor
            self.model_id = MODEL_IDS[name]
            self.model = AutoModel.from_pretrained(
                self.model_id, torch_dtype=self.dtype, local_files_only=True
            ).to(device).eval()
            self.revision = getattr(self.model.config, "_commit_hash", None)
            self.processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True)
            self.image_mean = tuple(self.processor.image_processor.image_mean)
        else:
            raise ValueError(f"Unknown model: {name}")
        if self.image_mean is None or len(self.image_mean) != 3:
            raise ValueError(f"Model {name} has no valid three-channel image normalization mean")
        self.load_seconds = time.perf_counter() - started

    @property
    def masked_background_rgb(self) -> tuple[int, int, int]:
        """Nearest 8-bit RGB representation of this model's processor mean."""
        return tuple(int(round(channel * 255)) for channel in self.image_mean)

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        if self.name == "clip":
            batch = torch.stack([self.preprocess(image) for image in images]).to(
                device=self.device, dtype=self.dtype
            )
            return _normalize_features(self.model.encode_image(batch))
        if self.name.startswith("siglip2-"):
            inputs = _move_inputs(self.processor(images=images, return_tensors="pt"), self.device, self.dtype)
            return _normalize_features(self.model.get_image_features(**inputs))
        if self.name.startswith("fg-clip2-"):
            max_patches = max(_fgclip2_patch_budget(image) for image in images)
            inputs = self.processor(images=images, max_num_patches=max_patches, return_tensors="pt")
            inputs = _move_inputs(inputs, self.device, self.dtype)
            return _normalize_features(self.model.get_image_features(**inputs))
        if self.name == "metaclip2":
            inputs = _move_inputs(self.processor(images=images, return_tensors="pt"), self.device, self.dtype)
            return _normalize_features(self.model.get_image_features(**inputs))
        raise ValueError(f"Unknown model: {self.name}")

    def encode_text(self, texts: list[str], prompt_template: bool = True,
                    fg_walk_type: str = "short") -> torch.Tensor:
        prompts = [f"a photo of a {name}" for name in texts] if prompt_template else texts
        if self.name == "clip":
            return _normalize_features(self.model.encode_text(self.tokenizer(prompts).to(self.device)))
        if self.name.startswith("siglip2-"):
            inputs = self.processor(
                text=prompts, padding="max_length", truncation=True,
                max_length=64, return_tensors="pt",
            )
            return _normalize_features(self.model.get_text_features(**_move_inputs(inputs, self.device, self.dtype)))
        if self.name.startswith("fg-clip2-"):
            inputs = self.tokenizer(
                prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
            )
            inputs = _move_inputs(inputs, self.device, self.dtype)
            # The cached FG-CLIP2 remote model registers its ``position_ids`` buffer
            # twice, which can leave invalid indices in that buffer on this runtime.
            # Supply the standard 0..sequence_length-1 positions explicitly.
            sequence_length = inputs["input_ids"].shape[1]
            inputs["position_ids"] = torch.arange(
                sequence_length, device=self.device, dtype=torch.long
            ).unsqueeze(0).expand(inputs["input_ids"].shape[0], -1)
            return _normalize_features(
                self.model.get_text_features(**inputs, walk_type=fg_walk_type)
            )
        if self.name == "metaclip2":
            inputs = self.processor(
                text=prompts, padding="max_length", truncation=True,
                max_length=77, return_tensors="pt",
            )
            return _normalize_features(self.model.get_text_features(**_move_inputs(inputs, self.device, self.dtype)))
        raise ValueError(f"Unknown model: {self.name}")

    def encode_regions(self, images: list[Image.Image],
                       boxes: list[list[tuple[float, float, float, float]]]) -> torch.Tensor:
        """Encode full images with FG-CLIP2's native RoIAlign region API."""
        if not self.name.startswith("fg-clip2-"):
            raise ValueError("Native RoI region features are only available for FG-CLIP2")
        if len(images) != len(boxes) or any(not regions for regions in boxes):
            raise ValueError("Each image must have at least one ground-truth region box")
        max_patches = max(_fgclip2_patch_budget(image) for image in images)
        inputs = self.processor(images=images, max_num_patches=max_patches, return_tensors="pt")
        image_sizes = [(image.height, image.width) for image in images]
        region_infos = [[list(box) for box in regions] for regions in boxes]
        inputs = _move_inputs(inputs, self.device, self.dtype)
        # The cached remote helper builds FP32 RoIs unconditionally, while its
        # feature map follows the model FP16 dtype. CUDA torchvision requires
        # matching dtypes, so reproduce its RoIAlign projection with matched boxes.
        from torchvision.ops import roi_align
        dense_features = self.model.get_image_dense_feature(**inputs)
        region_features = []
        for image_index, image_boxes in enumerate(region_infos):
            feature_h, feature_w = (int(value) for value in inputs["spatial_shapes"][image_index].tolist())
            height, width = image_sizes[image_index]
            feature_map = dense_features[image_index, :feature_h * feature_w]
            feature_map = feature_map.view(feature_h, feature_w, -1).permute(2, 0, 1).unsqueeze(0)
            rois = []
            for x1, y1, x2, y2 in image_boxes:
                rois.append([
                    0,
                    (x1 / width) * feature_w,
                    (y1 / height) * feature_h,
                    (x2 / width) * feature_w,
                    (y2 / height) * feature_h,
                ])
            roi_tensor = torch.tensor(rois, device=feature_map.device, dtype=feature_map.dtype)
            pooled = roi_align(
                feature_map, roi_tensor, output_size=(1, 1), spatial_scale=1.0,
                sampling_ratio=-1, aligned=True,
            )
            region_features.append(pooled.squeeze(-1).squeeze(-1))
        return _normalize_features(torch.cat(region_features, dim=0))

    def close(self) -> None:
        self.model = self.processor = self.tokenizer = self.preprocess = None
        gc.collect()
        torch.cuda.empty_cache()


def _fgclip2_patch_budget(image: Image.Image) -> int:
    """Follow the official FG-CLIP2 inference patch-count buckets."""
    patch_count = (image.width // 16) * (image.height // 16)
    if patch_count > 784:
        return 1024
    if patch_count > 576:
        return 784
    if patch_count > 256:
        return 576
    if patch_count > 128:
        return 256
    return 128


def _materialize_images(rows: list[dict], image_cache: OrderedDict,
                        crop_input: str,
                        background_rgb: tuple[int, int, int]) -> list[Image.Image]:
    crops = []
    for row in rows:
        path = row["image_path"]
        image = image_cache.get(path)
        if image is None:
            with Image.open(path) as source:
                image = source.convert("RGB")
            image_cache[path] = image
        image_cache.move_to_end(path)
        while len(image_cache) > 32:
            image_cache.popitem(last=False)
        left, top, right, bottom = row["crop_box"]
        crop = image.crop((left, top, right, bottom))
        if crop_input == "ground-truth-masked-object":
            object_mask = Image.new("L", crop.size, 0)
            draw = ImageDraw.Draw(object_mask)
            segmentation = row["segmentation"]
            if not isinstance(segmentation, list):
                raise ValueError("COCO non-crowd instance segmentation must contain polygon lists")
            for polygon in segmentation:
                if len(polygon) < 6:
                    continue
                points = [
                    (float(polygon[index]) - left, float(polygon[index + 1]) - top)
                    for index in range(0, len(polygon) - 1, 2)
                ]
                draw.polygon(points, fill=255)
            if object_mask.getbbox() is None:
                raise ValueError(f"Ground-truth mask is empty for COCO annotation {row['annotation_id']}")
            neutral = Image.new("RGB", crop.size, background_rgb)
            neutral.paste(crop, mask=object_mask)
            crop = neutral
        crops.append(crop)
    return crops


def _measure_embeddings(encoder: Encoder, observations: list[dict], batch_size: int,
                        timing_repeats: int, crop_input: str
                        ) -> tuple[torch.Tensor, float, list[float]]:
    timings = []
    features = None
    image_cache: OrderedDict[str, Image.Image] = OrderedDict()
    for _ in range(timing_repeats):
        chunks = []
        total_encode_seconds = 0.0
        for begin in range(0, len(observations), batch_size):
            images = _materialize_images(
                observations[begin:begin + batch_size], image_cache, crop_input,
                encoder.masked_background_rgb,
            )
            started = time.perf_counter()
            with torch.inference_mode():
                chunks.append(encoder.encode_images(images).cpu())
            torch.cuda.synchronize(encoder.device)
            total_encode_seconds += time.perf_counter() - started
        timings.append(total_encode_seconds)
        features = torch.cat(chunks)
    image_cache.clear()
    return features, float(np.median(timings)), timings


def _measure_fg_native_regions(encoder: Encoder, observations: list[dict], batch_size: int,
                               timing_repeats: int) -> tuple[torch.Tensor, float, list[float]]:
    """Run FG-CLIP2 dense full-image features and its native RoIAlign head."""
    if not encoder.name.startswith("fg-clip2-"):
        raise ValueError("Native FG-CLIP2 region measurement requested for another model")
    by_image: OrderedDict[str, list[int]] = OrderedDict()
    for index, row in enumerate(observations):
        by_image.setdefault(row["image_path"], []).append(index)
    image_groups = list(by_image.items())
    timings = []
    features = None
    for _ in range(timing_repeats):
        output = [None] * len(observations)
        total_encode_seconds = 0.0
        for begin in range(0, len(image_groups), batch_size):
            group = image_groups[begin:begin + batch_size]
            images = []
            region_boxes = []
            for path, indices in group:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                images.append(image)
                region_boxes.append([observations[index]["region_box"] for index in indices])
            started = time.perf_counter()
            with torch.inference_mode():
                encoded = encoder.encode_regions(images, region_boxes).cpu()
            torch.cuda.synchronize(encoder.device)
            total_encode_seconds += time.perf_counter() - started
            cursor = 0
            for (_, indices) in group:
                for index in indices:
                    output[index] = encoded[cursor]
                    cursor += 1
            for image in images:
                image.close()
        timings.append(total_encode_seconds)
        features = torch.stack(output)
    return features, float(np.median(timings)), timings


def _average_precision(scores: np.ndarray, relevant: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = relevant[order].astype(np.float64)
    cumulative = np.cumsum(ranked)
    precision_at_relevant = cumulative[ranked.astype(bool)] / (np.flatnonzero(ranked) + 1)
    return float(precision_at_relevant.sum() / relevant.sum()) if relevant.any() else float("nan")


def _score(image_features: torch.Tensor, text_features: torch.Tensor,
           observations: list[dict], category_names: list[str], device: torch.device) -> dict:
    targets = np.asarray([row["category_index"] for row in observations], dtype=np.int64)
    image_features = F.normalize(image_features.float(), dim=-1).to(device)
    text_features = F.normalize(text_features.float(), dim=-1).to(device)
    chunks = []
    with torch.inference_mode():
        for begin in range(0, len(observations), 2048):
            chunks.append((image_features[begin:begin + 2048] @ text_features.T).cpu().numpy())
    similarities = np.concatenate(chunks, axis=0)
    predictions = similarities.argmax(axis=1)
    topk_count = min(5, similarities.shape[1])
    topk = np.argpartition(-similarities, kth=topk_count - 1, axis=1)[:, :topk_count]
    top1 = predictions == targets
    top5 = np.any(topk == targets[:, None], axis=1)
    present = np.unique(targets)
    per_category = {}
    aps = []
    retrieval_r1, retrieval_r5, retrieval_r10, retrieval_r50 = [], [], [], []
    for category_index in present:
        relevant = targets == category_index
        scores = similarities[:, category_index]
        ap = _average_precision(scores, relevant)
        aps.append(ap)
        order = np.argsort(-scores, kind="stable")
        hits = relevant[order]
        per_category[category_names[int(category_index)]] = {
            "ground_truth_objects": int(relevant.sum()),
            "zero_shot_top1_accuracy": float(top1[relevant].mean()),
            "query_average_precision": ap,
            "query_recall_at_1": float(hits[:1].any()),
            "query_recall_at_5": float(hits[:5].any()),
            "query_recall_at_10": float(hits[:10].any()),
            "query_recall_at_50": float(hits[:50].any()),
        }
        retrieval_r1.append(float(hits[:1].any()))
        retrieval_r5.append(float(hits[:5].any()))
        retrieval_r10.append(float(hits[:10].any()))
        retrieval_r50.append(float(hits[:50].any()))
    return {
        "zero_shot_top1_micro": float(top1.mean()),
        "zero_shot_top1_macro_category": float(np.mean([top1[targets == c].mean() for c in present])),
        "zero_shot_top5_micro": float(top5.mean()),
        "object_query_retrieval_map_macro_category": float(np.mean(aps)),
        "object_query_retrieval_recall_at_1_macro_query": float(np.mean(retrieval_r1)),
        "object_query_retrieval_recall_at_5_macro_query": float(np.mean(retrieval_r5)),
        "object_query_retrieval_recall_at_10_macro_query": float(np.mean(retrieval_r10)),
        "object_query_retrieval_recall_at_50_macro_query": float(np.mean(retrieval_r50)),
        "per_category": per_category,
    }


def _run_model(name: str, args: argparse.Namespace, observations: list[dict],
               categories: list[str], caption_queries: list[dict],
               device: torch.device) -> dict:
    print(f"[clip-component] loading {name}: {MODEL_IDS[name]}", flush=True)
    encoder = Encoder(name, args.hf_home.expanduser().resolve(), device)
    model_resident_cuda_gib = torch.cuda.memory_allocated(device) / (1024**3)
    torch.cuda.reset_peak_memory_stats(device)
    text_started = time.perf_counter()
    with torch.inference_mode():
        text_features = encoder.encode_text(categories).cpu()
    torch.cuda.synchronize(device)
    text_seconds = time.perf_counter() - text_started
    scored_caption_queries = [row for row in caption_queries if row["category_indices"]]
    caption_features, caption_text_seconds = _encode_text_batched(
        encoder,
        [row["caption"] for row in scored_caption_queries],
        args.text_batch_size,
        device,
        prompt_template=False,
    )
    conditions = {}
    for crop_input in ("ground-truth-box", "ground-truth-masked-object"):
        print(f"[clip-component] {name}: encoding {crop_input}", flush=True)
        warmup_cache: OrderedDict[str, Image.Image] = OrderedDict()
        warmup_images = _materialize_images(
            observations[:args.batch_size], warmup_cache, crop_input,
            encoder.masked_background_rgb,
        )
        with torch.inference_mode():
            encoder.encode_images(warmup_images)
        torch.cuda.synchronize(device)
        warmup_cache.clear()
        features, encode_seconds, pass_times = _measure_embeddings(
            encoder, observations, args.batch_size, args.timing_repeats, crop_input
        )
        metrics = _score(features, text_features, observations, categories, device)
        metrics.update(_score_caption_queries(
            features, caption_features, scored_caption_queries, categories, device
        ))
        conditions[crop_input] = {
            "image_encode_seconds_median": encode_seconds,
            "image_encode_seconds_per_pass": pass_times,
            "object_label_retrieval_inference_seconds": encode_seconds + text_seconds,
            "caption_to_label_inference_seconds": encode_seconds + caption_text_seconds,
            "ground_truth_objects_per_second": len(observations) / encode_seconds,
            "milliseconds_per_object": encode_seconds * 1000 / len(observations),
            **metrics,
        }
        del features
    if name.startswith("fg-clip2-"):
        print(f"[clip-component] {name}: encoding native full-image RoIAlign regions", flush=True)
        region_text_started = time.perf_counter()
        region_text_features = encoder.encode_text(categories, fg_walk_type="box").cpu()
        torch.cuda.synchronize(device)
        region_text_seconds = time.perf_counter() - region_text_started
        region_caption_features, region_caption_text_seconds = _encode_text_batched(
            encoder,
            [row["caption"] for row in scored_caption_queries],
            args.text_batch_size,
            device,
            prompt_template=False,
            fg_walk_type="box",
        )
        region_features, region_encode_seconds, region_pass_times = _measure_fg_native_regions(
            encoder, observations, args.batch_size, args.timing_repeats
        )
        region_metrics = _score(region_features, region_text_features, observations, categories, device)
        region_metrics.update(_score_caption_queries(
            region_features, region_caption_features, scored_caption_queries, categories, device
        ))
        conditions["ground-truth-native-roi"] = {
            "image_encode_seconds_median": region_encode_seconds,
            "image_encode_seconds_per_pass": region_pass_times,
            "object_label_retrieval_inference_seconds": (
                region_encode_seconds + region_text_seconds
            ),
            "caption_to_label_inference_seconds": (
                region_encode_seconds + region_caption_text_seconds
            ),
            "ground_truth_objects_per_second": len(observations) / region_encode_seconds,
            "milliseconds_per_object": region_encode_seconds * 1000 / len(observations),
            "text_space": "FG-CLIP2 region-caption alignment: RoIAlign region features against walk_type=box text features",
            "category_text_encode_seconds": region_text_seconds,
            **region_metrics,
        }
        del region_features, region_text_features, region_caption_features
    result = {
        "name": name,
        "model_id": encoder.model_id,
        "model_revision": encoder.revision,
        "processor_image_mean_rgb_0_1": list(encoder.image_mean),
        "mask_background_rgb_8bit": list(encoder.masked_background_rgb),
        "model_load_seconds": encoder.load_seconds,
        "model_resident_cuda_allocated_gib": model_resident_cuda_gib,
        "text_encode_seconds": text_seconds,
        "caption_text_encode_seconds": caption_text_seconds,
        "native_region_category_text_encode_seconds": (
            region_text_seconds if name.startswith("fg-clip2-") else None
        ),
        "native_region_caption_text_encode_seconds": (
            region_caption_text_seconds if name.startswith("fg-clip2-") else None
        ),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "vram_measurement_scope": "per-process PyTorch CUDA allocated/reserved bytes; excludes VRAM owned by concurrent processes",
        "feature_dimension": int(text_features.shape[1]),
        "embedding_storage_mib_float32": (
            (len(observations) + len(categories)) * int(text_features.shape[1]) * 4 / (1024**2)
        ),
        "input_conditions": conditions,
    }
    encoder.close()
    del text_features
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.text_batch_size < 1 or args.padding < 0 or args.timing_repeats < 1:
        raise ValueError("batch sizes and timing repeats must be positive; --padding cannot be negative")
    if args.max_objects == 0 or args.max_objects < -1:
        raise ValueError("--max-objects must be positive or -1 for all objects")
    os.environ["HF_HOME"] = str(args.hf_home.expanduser().resolve())
    observations, categories, dataset_info = _load_observations(args)
    if args.check_data_only:
        if not args.captions.expanduser().is_file():
            raise FileNotFoundError(f"COCO caption annotations not found: {args.captions}")
        for name in args.models:
            checkpoint = _check_local_model_cache(name, args.hf_home.expanduser().resolve())
            print(f"[clip-component] local checkpoint ready: {name} ({checkpoint})")
        print(f"[clip-component] COCO inputs ready: {len(observations)} GT objects, {len(categories)} categories")
        print(f"[clip-component] COCO captions ready: {args.captions.expanduser().resolve()}")
        return 0
    caption_queries, caption_info = _load_caption_queries(
        args.captions.expanduser().resolve(), observations, categories
    )
    from conceptgraph.utils.model_utils import require_cuda_device
    device = require_cuda_device("cuda:0")
    print(f"[clip-component] {len(observations)} ground-truth objects, "
          f"{len(categories)} text queries on {device}", flush=True)

    results = []
    for name in args.models:
        result = _run_model(name, args, observations, categories, caption_queries, device)
        results.append(result)
        production_crop = result["input_conditions"]["ground-truth-box"]
        masked_crop = result["input_conditions"]["ground-truth-masked-object"]
        print(
            f"[clip-component] {name}: box query mAP="
            f"{production_crop['object_query_retrieval_map_macro_category']:.4f}, "
            f"masked-object query mAP="
            f"{masked_crop['object_query_retrieval_map_macro_category']:.4f}",
            flush=True,
        )
    baseline = next((row for row in results if row["name"] == "clip"), None)
    if baseline:
        for row in results:
            for crop_input, result in row["input_conditions"].items():
                if crop_input not in baseline["input_conditions"]:
                    continue
                base = baseline["input_conditions"][crop_input]
                result["macro_top1_delta_vs_clip"] = (
                    result["zero_shot_top1_macro_category"] - base["zero_shot_top1_macro_category"]
                )
                result["query_map_delta_vs_clip"] = (
                    result["object_query_retrieval_map_macro_category"]
                    - base["object_query_retrieval_map_macro_category"]
                )
                result["throughput_ratio_vs_clip"] = (
                    result["ground_truth_objects_per_second"] / base["ground_truth_objects_per_second"]
                )
                for task_metric in ("object_label_retrieval_inference_seconds", "caption_to_label_inference_seconds"):
                    result[f"{task_metric}_delta_vs_clip"] = result[task_metric] - base[task_metric]
                    result[f"{task_metric}_ratio_vs_clip"] = result[task_metric] / base[task_metric]
            for key in ("model_load_seconds", "model_resident_cuda_allocated_gib", "peak_cuda_allocated_gib", "peak_cuda_reserved_gib"):
                row[f"{key}_delta_vs_clip"] = row[key] - baseline[key]
                row[f"{key}_ratio_vs_clip"] = row[key] / baseline[key] if baseline[key] else float("nan")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        REPO_ROOT / ".agentic_workflow/workspace/clip-component-benchmark"
        / f"coco-val-{timestamp}.json"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "ConceptGraphs GT object embedding and text-query retrieval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": {
            "primary": "macro-average AP for category-text queries retrieving matching ground-truth objects from the complete object bank",
            "secondary": "macro-average zero-shot category top-1 accuracy, top-5 accuracy, query recall@K, and encoder throughput",
            "query_scope": "each query is a shared category name such as 'a photo of a chair'; this measures category retrieval, not arbitrary referring expressions",
            "inputs": {
                "ground-truth-box": "padded human-annotated COCO boxes; production-faithful crop, since current ConceptGraphs CLIP embeddings use padded boxes and do not apply mask pixels",
                "ground-truth-masked-object": "same proposals and labels, but pixels outside each human-annotated instance polygon use the model-specific processor mean RGB; these pixels normalize to approximately zero and isolate object appearance from context",
                "ground-truth-native-roi": "FG-CLIP2-only supplemental input: the full source image and oracle GT box are passed through FG-CLIP2's native RoIAlign feature API; box text embeddings use walk_type=box. This is a model-specific fine-grained path and is not the apples-to-apples crop replacement path used by the main two conditions.",
            },
            "scope": "fixed oracle proposals, masks, and labels isolate the image/text embedding component; category-name queries proxy free-form user descriptions; this does not measure proposal quality, temporal object aggregation, or full 3D mapping replay",
            "domain_limit": "COCO categories and images differ from the project's Replica and user-recorded scenes",
            "caption_to_label": "Each human COCO image caption is used as a text query. Keyword-matched COCO category labels present in that same image define relevant GT object crops. Regions are ranked over the full object bank using each model's image/text embeddings. This is weakly grounded because captions describe complete images rather than individual instances.",
            "dino": "not included in text-query scores because DINOv3 has no joint image/text embedding space; test it separately on ground-truth tracked-object association if that visual-only use is under consideration",
        },
        "dataset": dataset_info,
        "captions": caption_info,
        "settings": {
            "models": args.models,
            "batch_size": args.batch_size,
            "text_batch_size": args.text_batch_size,
            "padding_pixels": args.padding,
            "timing_repeats": args.timing_repeats,
            "device": str(device),
            "text_prompt_template": "a photo of a {category}",
        },
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"[clip-component] wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[clip-component] ERROR: {exc}", file=sys.stderr)
        raise
