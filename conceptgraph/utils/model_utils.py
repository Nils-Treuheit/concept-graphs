import os
from conceptgraph.utils.general_utils import measure_time
# from line_profiler import profile
# from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import torch
import cv2
from PIL import Image
from scipy.spatial.distance import cosine


YOLO_WORLD_INPUT_SIZE = 640
SAM_L_INPUT_SIZE = 1024


def resize_and_letterbox(image: np.ndarray, size: int, color: int = 114):
    """Resize an image proportionally into a square model input.

    Returns the padded image and the exact scale/padding needed to map model
    coordinates back to the source image. The source aspect ratio is preserved.
    """
    if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be a non-empty HxW or HxWxC array")
    height, width = image.shape[:2]
    gain = min(size / width, size / height)
    resized_width = min(size, max(1, int(round(width * gain))))
    resized_height = min(size, max(1, int(round(height * gain))))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    if image.ndim == 2:
        canvas = np.full((size, size), color, dtype=image.dtype)
    else:
        canvas = np.full((size, size, image.shape[2]), color, dtype=image.dtype)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    transform = {
        "source_shape": (height, width),
        "resized_shape": (resized_height, resized_width),
        "gain_x": resized_width / width,
        "gain_y": resized_height / height,
        "left": left,
        "top": top,
    }
    return canvas, transform


def boxes_from_letterbox(boxes: np.ndarray, transform: dict) -> np.ndarray:
    """Map XYXY boxes from a letterboxed model canvas to source coordinates."""
    boxes = np.asarray(boxes).copy()
    if boxes.size == 0:
        return boxes.reshape((-1, 4))
    boxes = boxes.reshape((-1, 4))
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform["left"]) / transform["gain_x"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform["top"]) / transform["gain_y"]
    height, width = transform["source_shape"]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes


def boxes_to_letterbox(boxes: np.ndarray, transform: dict) -> np.ndarray:
    """Map source-image XYXY boxes onto a model's letterboxed input canvas."""
    boxes = np.asarray(boxes).copy()
    if boxes.size == 0:
        return boxes.reshape((-1, 4))
    boxes = boxes.reshape((-1, 4))
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * transform["gain_x"] + transform["left"]
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * transform["gain_y"] + transform["top"]
    return boxes


def masks_from_letterbox(masks: np.ndarray, transform: dict) -> np.ndarray:
    """Remove square padding from masks and restore source-image resolution."""
    masks = np.asarray(masks)
    height, width = transform["source_shape"]
    resized_height, resized_width = transform["resized_shape"]
    left, top = transform["left"], transform["top"]
    content = masks[..., top : top + resized_height, left : left + resized_width]
    if content.shape[-2:] == (height, width):
        return content
    leading_shape = content.shape[:-2]
    flat = content.reshape((-1, resized_height, resized_width))
    interpolation = cv2.INTER_NEAREST
    restored = [cv2.resize(mask, (width, height), interpolation=interpolation) for mask in flat]
    return np.stack(restored).reshape((*leading_shape, height, width))

def require_cuda_device(device) -> torch.device:
    """Validate and resolve a CUDA device for neural inference.

    Returns the resolved ``torch.device`` only when the requested device is a
    CUDA device and ``torch.cuda.is_available()`` is true. This is the single
    CUDA validation seam shared by the detection and mapping entry points; it
    must be called before any YOLO, SAM or CLIP model is constructed.

    Raises
    ------
    ValueError
        If the requested device is not a CUDA device (e.g. ``"cpu"``). A CPU
        fallback is never returned.
    RuntimeError
        If a CUDA device is requested but CUDA is not available.

    This helper performs no model loading, allocates no inference tensors, and
    does not fall back to CPU.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(
            "Neural inference in the detection/mapping path requires a CUDA "
            f"device, but was given '{device}'. Set the pipeline device to a "
            "CUDA device (e.g. 'cuda:0'); there is no CPU fallback."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"A CUDA device ({dev}) was requested, but PyTorch reports that "
            "CUDA is not available. Check that a CUDA-enabled PyTorch build is "
            "installed and a GPU is reachable."
        )
    return dev

# def get_sam_predictor(cfg) -> SamPredictor:
#     if cfg.sam_variant == "sam":
#         sam = sam_model_registry[cfg.sam_encoder_version](checkpoint=cfg.sam_checkpoint_path)
#         sam.to(cfg.device)
#         sam_predictor = SamPredictor(sam)
#         return sam_predictor
    
#     if cfg.sam_variant == "mobilesam":
#         from MobileSAM.setup_mobile_sam import setup_model
#         # MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/mobile_sam.pt")
#         # checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
#         checkpoint = torch.load(cfg.mobile_sam_path)
#         mobile_sam = setup_model()
#         mobile_sam.load_state_dict(checkpoint, strict=True)
#         mobile_sam.to(device=cfg.device)
        
#         sam_predictor = SamPredictor(mobile_sam)
#         return sam_predictor

#     elif cfg.sam_variant == "lighthqsam":
#         from LightHQSAM.setup_light_hqsam import setup_model
#         HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/sam_hq_vit_tiny.pth")
#         checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
#         light_hqsam = setup_model()
#         light_hqsam.load_state_dict(checkpoint, strict=True)
#         light_hqsam.to(device=cfg.device)
        
#         sam_predictor = SamPredictor(light_hqsam)
#         return sam_predictor
        
#     elif cfg.sam_variant == "fastsam":
#         raise NotImplementedError
#     else:
#         raise NotImplementedError

# Prompting SAM with detected boxes in a batch
def get_sam_segmentation_from_xyxy_batched(sam_predictor, image: np.ndarray, xyxy_tensor: torch.Tensor) -> torch.Tensor:
    
    sam_predictor.set_image(image)
    
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(xyxy_tensor, image.shape[:2])
    
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    
    return masks.squeeze()

# Prompting SAM with detected boxes in a batch
def get_sam_segmentation_from_xyxy(sam_predictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    
    sam_predictor.set_image(image)
    
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)
    
def compute_clip_features(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    backup_image = image.copy()
    
    image = Image.fromarray(image)
    
    # padding = args.clip_padding  # Adjust the padding amount as needed
    padding = 20  # Adjust the padding amount as needed
    
    image_crops = []
    image_feats = []
    text_feats = []

    
    for idx in range(len(detections.xyxy)):
        # Get the crop of the mask with padding
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

        # Check and adjust padding to avoid going beyond the image borders
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # Apply the adjusted padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # Get the preprocessed image for clip from the crop 
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to(device)

        crop_feat = clip_model.encode_image(preprocessed_image)
        crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
        
        class_id = detections.class_id[idx]
        tokenized_text = clip_tokenizer([classes[class_id]]).to(device)
        text_feat = clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
        crop_feat = crop_feat.cpu().numpy()
        text_feat = text_feat.cpu().numpy()

        image_crops.append(cropped_image)
        image_feats.append(crop_feat)
        text_feats.append(text_feat)
        
    # turn the list of feats into np matrices
    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)

    return image_crops, image_feats, text_feats

# @profile
def compute_clip_features_batched(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    if len(detections.xyxy) == 0:
        return [], np.empty((0, 0), dtype=np.float32), []
    
    image = Image.fromarray(image)
    padding = 20  # Adjust the padding amount as needed
    
    image_crops = []
    preprocessed_images = []
    text_tokens = []
    
    # Prepare data for batch processing
    for idx in range(len(detections.xyxy)):
        x_min, y_min, x_max, y_max = detections.xyxy[idx]
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)

        class_id = detections.class_id[idx]
        text_tokens.append(classes[class_id])
        image_crops.append(cropped_image)
    
    # Convert lists to batches
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
    text_tokens_batch = clip_tokenizer(text_tokens).to(device)
    
    # Batch inference
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # text_features = clip_model.encode_text(text_tokens_batch)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy
    image_feats = image_features.cpu().numpy()
    # text_feats = text_features.cpu().numpy()
    # image_feats = []
    text_feats = []
    
    return image_crops, image_feats, text_feats


def compute_ft_vector_closeness_statistics(unbatched, batched):
    # Initialize lists to store statistics
    mad = []  # Mean Absolute Difference
    max_diff = []  # Maximum Absolute Difference
    mrd = []  # Mean Relative Difference
    cosine_sim = []  # Cosine Similarity

    for i in range(len(unbatched)):
        diff = np.abs(unbatched[i] - batched[i])
        mad.append(np.mean(diff))
        max_diff.append(np.max(diff))
        mrd.append(np.mean(diff / (np.abs(batched[i]) + 1e-8)))  # Adding a small value to avoid division by zero
        cosine_sim.append(1 - cosine(unbatched[i].flatten(), batched[i].flatten()))  # 1 - cosine distance to get similarity

    # Convert lists to numpy arrays for easy statistics
    mad = np.array(mad)
    max_diff = np.array(max_diff)
    mrd = np.array(mrd)
    cosine_sim = np.array(cosine_sim)

    # Print statistics
    print(f"Mean Absolute Difference: {np.mean(mad)}")
    print(f"Maximum Absolute Difference: {np.max(max_diff)}")
    print(f"Mean Relative Difference: {np.mean(mrd)}")
    print(f"Mean Cosine Similarity: {np.mean(cosine_sim)}")
    print(f"Min Cosine Similarity: {np.min(cosine_sim)}")
