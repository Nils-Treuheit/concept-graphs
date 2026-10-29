'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import os
import copy
import json
import subprocess
import sys
import uuid
from pathlib import Path
import pickle
import gzip

# Third-party imports
import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from tqdm import trange
from open3d.io import read_pinhole_camera_parameters
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv
from collections import Counter

# Local application/library specific imports
from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun, 
    orr_log_annotated_image, 
    orr_log_camera, 
    orr_log_depth_image, 
    orr_log_edges, 
    orr_log_objs_pcd_and_bbox, 
    orr_log_rgb_image, 
    orr_log_world_view_coordinates,
    orr_log_vlm_image
)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.vlm import consolidate_captions, get_obj_rel_from_image_gpt4v
from conceptgraph.utils.llm_client import get_llm_client, get_vlm_client
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses, 
    find_existing_image_path, 
    get_det_out_path, 
    get_exp_out_path, 
    get_vlm_annotated_image_path, 
    get_weights_dir,
    handle_rerun_saving, 
    load_saved_detections, 
    load_saved_hydra_json_config, 
    make_vlm_edges_and_captions, 
    measure_time, 
    save_detection_results,
    save_edge_json, 
    save_hydra_config,
    save_obj_json, 
    save_objects_for_frame, 
    save_pointcloud, 
    should_exit_early, 
    vis_render_image
)
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import (
    OnlineObjectRenderer, 
    save_video_from_frames, 
    vis_result_fast_on_depth, 
    vis_result_for_vlm, 
    vis_result_fast, 
    save_video_detections
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    merge_objects, 
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_cfg,
    process_edges,
    process_pcd,
    processing_needed,
    resize_gobs
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches
)
from conceptgraph.utils.model_utils import (
    SAM_L_INPUT_SIZE,
    YOLO_WORLD_INPUT_SIZE,
    boxes_from_letterbox,
    boxes_to_letterbox,
    compute_clip_features_batched,
    masks_from_letterbox,
    require_cuda_device,
    resize_and_letterbox,
)
from conceptgraph.utils.gpu_memory_budget import NvmlMemoryBudget
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections


# Disable torch gradient computation
torch.set_grad_enabled(False)


def _task_overrides_without(overrides, blocked_keys):
    """Keep the user's public Hydra overrides while replacing stage controls."""
    forwarded = []
    for override in overrides:
        if "=" not in override:
            continue
        key = override.split("=", 1)[0].lstrip("+~")
        if key not in blocked_keys:
            forwarded.append(override)
    return forwarded


def _run_forced_detection_pipeline(cfg: DictConfig) -> None:
    """Run detector, VLM, then cached mapping as separate GPU-owner processes."""
    overrides = list(HydraConfig.get().overrides.task)
    common = [
        f"dataset_root={cfg.dataset_root}",
        f"dataset_config={cfg.dataset_config}",
        f"scene_id={cfg.scene_id}",
        f"start={cfg.start}",
        f"end={cfg.end}",
        f"stride={cfg.stride}",
        "device=cuda",
        f"classes_file={cfg.classes_file}",
        f"bg_classes={json.dumps(list(cfg.bg_classes))}",
        f"skip_bg={str(bool(cfg.skip_bg)).lower()}",
    ]
    detector_args = [
        sys.executable, "-m", "conceptgraph.scripts.streamlined_detections",
        "--config-name", "streamlined_detections",
        *common,
        f"repo_root={cfg.repo_root}",
        f"exp_suffix={cfg.detections_exp_suffix}",
        "desired_height=720",
        "desired_width=1280",
        "save_video=false",
    ]
    print("[pipeline] Stage 1/3: full-resolution CUDA detections")
    subprocess.run(detector_args, check=True)

    vlm_args = [
        sys.executable, "-m", "conceptgraph.scripts.vlm_annotations",
        "--config-name", "vlm_annotations",
        *common,
        f"detections_exp_suffix={cfg.detections_exp_suffix}",
        "desired_height=720",
        "desired_width=1280",
        f"make_edges={str(bool(cfg.make_edges)).lower()}",
        "overwrite=true",
    ]
    print("[pipeline] Stage 2/3: local VLM annotations and verified model release")
    subprocess.run(vlm_args, check=True)

    mapping_args = [
        sys.executable, "-m", "conceptgraph.slam.rerun_realtime_mapping",
        "--config-name", "rerun_realtime_mapping",
        *_task_overrides_without(
            overrides,
            {"force_detection", "save_detections", "make_edges"},
        ),
        "force_detection=false",
        "save_detections=false",
        "make_edges=false",
        # The staged offline map runs without requiring the optional Rerun CLI.
        "use_rerun=false",
    ]
    print("[pipeline] Stage 3/3: cached RGB-D mapping with CUDA Faiss")
    subprocess.run(mapping_args, check=True)

# A logger for this file
@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="rerun_realtime_mapping")
# @profile
def main(cfg : DictConfig):
    # Preserve the public Hydra command. The forced path exits before creating
    # any mapping state, and the final child is explicitly cache-only.
    if bool(cfg.force_detection):
        _run_forced_detection_pipeline(cfg)
        return

    # Validate/resolve the CUDA device before any neural model is constructed
    # and regardless of the detection cache state (independent of
    # `run_detections` below).
    cuda_device = require_cuda_device(cfg.device)
    # The cache-only map stage still owns CUDA through Open3D and GPU-Faiss.
    # Require the same initial headroom and enforce aggregate plus per-process
    # limits before and throughout the mapping loop.
    memory_budget = NvmlMemoryBudget(cuda_device)

    tracker = MappingTracker()
    
    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)
    orr.init("realtime_mapping")
    orr.spawn()

    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(project="concept-graphs", 
            #    entity="concept-graphs",
                config=cfg_to_dict(cfg),
               )
    cfg = process_cfg(cfg)

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    orr_log_world_view_coordinates(dataset.name)
    # cam_K = dataset.get_cam_K()

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    # For visualization
    if cfg.vis_render:
        view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
        obj_renderer = OnlineObjectRenderer(
            view_param = view_param,
            base_objects = None, 
            gray_map = False,
        )
        frames = []
    # output folder for this mapping experiment
    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)

    # output folder of the detections experiment to use
    det_exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix, make_dir=False)

    # we need to make sure to use the same classes as the ones used in the detections
    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg['classes_file'], 
        bg_classes=detections_exp_cfg['bg_classes'], 
        skip_bg=detections_exp_cfg['skip_bg']
    )

    # if we need to do detections
    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)
    
    prev_adjusted_pose = None

    if run_detections:
        print("\n".join(["Running detections..."] * 10))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        ## Initialize the detection models
        # Move only the internal torch modules to CUDA; reassigning the
        # Ultralytics wrapper itself (e.g. `detection_model = ...to(...)`) would
        # lose the wrapper methods (set_classes, predict, ...).
        weights_dir = get_weights_dir(cfg.repo_root)
        detection_model = measure_time(YOLO)(str(weights_dir / "yolov8l-world.pt"))
        detection_model.model.to(cuda_device)
        sam_predictor = SAM(str(weights_dir / "sam_l.pt")) # SAM('mobile_sam.pt') # UltraLytics SAM
        sam_predictor.model.to(cuda_device)
        # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(cuda_device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # Set the classes for the detection model
        detection_model.set_classes(obj_classes.get_classes_arr())
    else:
        print("\n".join(["NOT Running detections..."] * 10))

    # Local models: text calls -> Qwen3.8 (Ollama), vision calls -> Qwen3-VL (vLLM)
    llm_client = get_llm_client()
    vlm_client = get_vlm_client()

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0
    for frame_idx in trange(len(dataset)):
        memory_budget.check(f"mapping before frame {frame_idx}")
        tracker.curr_frame_idx = frame_idx
        counter+=1
        orr.set_time("frame", sequence=frame_idx)

        # Check if we should exit early only if the flag hasn't been set yet
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True

        # If exit early flag is set and we're not at the last frame, skip this iteration
        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        # Read info about current frame from dataset
        # color image
        color_path = Path(dataset.color_paths[frame_idx])
        image_original_pil = Image.open(color_path)
        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        # Keep the dataset-resolution image for RGB-D mapping. Detection and
        # VLM inference below use the original full-resolution source image.
        mapping_rgb = color_np.astype(np.uint8) # (H, W, 3)
        image_rgb = mapping_rgb
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Load image detections for the current frame
        raw_gobs = None
        gobs = None # stands for grounded observations
        detections_path = det_exp_pkl_path / (color_path.stem + ".pkl.gz")
        
        vis_save_path_for_vlm = get_vlm_annotated_image_path(det_exp_vis_path, color_path)
        vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(det_exp_vis_path, color_path, w_edges=True)
        
        if run_detections:
            results = None
            # opencv can't read Path objects...
            image = cv2.imread(str(color_path)) # This will in BGR color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Each model receives an aspect-preserving square input at its
            # configured inference size. Predictions are restored to the
            # original camera-image coordinate frame before models are joined.
            yolo_input, yolo_transform = resize_and_letterbox(
                image, YOLO_WORLD_INPUT_SIZE
            )
            results = detection_model.predict(
                yolo_input, conf=0.1, verbose=False, device=str(cuda_device),
                imgsz=YOLO_WORLD_INPUT_SIZE, rect=False,
            )
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
            yolo_boxes_canvas = results[0].boxes.xyxy.cpu().numpy()
            xyxy_np = boxes_from_letterbox(yolo_boxes_canvas, yolo_transform)

            # if there are detections,
            # Get Masks Using SAM or MobileSAM
            if len(xyxy_np) != 0:
                sam_input, sam_transform = resize_and_letterbox(
                    image, SAM_L_INPUT_SIZE
                )
                sam_boxes = boxes_to_letterbox(xyxy_np, sam_transform)
                sam_boxes_tensor = torch.as_tensor(
                    sam_boxes, dtype=torch.float32, device=cuda_device
                )
                sam_out = sam_predictor.predict(
                    sam_input, bboxes=sam_boxes_tensor, verbose=False,
                    device=str(cuda_device), imgsz=SAM_L_INPUT_SIZE, rect=False,
                )
                masks_tensor = sam_out[0].masks.data
                masks_canvas = masks_tensor.cpu().numpy()
                masks_np = masks_from_letterbox(masks_canvas, sam_transform)
            else:
                masks_np = np.empty((0, *image.shape[:2]), dtype=bool)

            # Create a detections object that we will save later
            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )
            
            # Make the edges
            labels, edges, edge_image, captions = make_vlm_edges_and_captions(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, cfg.make_edges, vlm_client)

            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), cuda_device)

            # increment total object detections
            tracker.increment_total_detections(len(curr_det.xyxy))

            # Save results
            # Convert the detections to a dict. The elements are in np.array
            results = {
                # add new uuid for each detection 
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "detection_class_labels": detection_class_labels,
                "labels": labels,
                "edges": edges,
                "captions": captions,
            }

            raw_gobs = results

            # save the detections if needed
            if cfg.save_detections:

                vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                # Visualize and save the annotated image
                annotated_image, labels = vis_result_fast(image, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)

                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                if depth_image_rgb.shape[:2] != image.shape[:2]:
                    depth_image_rgb = cv2.resize(
                        depth_image_rgb,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                annotated_depth_image, labels = vis_result_fast_on_depth(depth_image_rgb, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
                save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)
        else:
            # Support current and old saving formats
            # - "<stem>"            : a folder (old) or a file (new)
            # - "<stem>.pkl.gz"     : the format of the detections stage
            # - "<frame number>"    : the old numeric naming
            candidates = [
                det_exp_pkl_path / color_path.stem,
                det_exp_pkl_path / f"{color_path.stem}.pkl.gz",
            ]
            if color_path.stem.isdigit():
                candidates.append(det_exp_pkl_path / f"{int(color_path.stem):06}")

            raw_gobs = None
            for candidate in candidates:
                # load_saved_detections() appends ".pkl.gz" itself, so the
                # plain file has to be accepted as well.
                if os.path.exists(candidate) or os.path.exists(f"{candidate}.pkl.gz"):
                    raw_gobs = load_saved_detections(candidate)
                    break

            if raw_gobs is None:
                # if no detections, throw an error
                raise FileNotFoundError(
                    f"No detections found for frame {frame_idx} at paths \n"
                    + " or \n".join(str(c) for c in candidates)
                )

        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose
        
        prev_adjusted_pose = orr_log_camera(intrinsics, adjusted_pose, prev_adjusted_pose, cfg.image_width, cfg.image_height, frame_idx)
        
        orr_log_rgb_image(color_path)
        orr_log_annotated_image(color_path, det_exp_vis_path)
        orr_log_depth_image(depth_tensor)
        orr_log_vlm_image(vis_save_path_for_vlm)
        orr_log_vlm_image(vis_save_path_for_vlm_edges, label="w_edges")

        # resize the observation if needed
        # YOLO/SAM and saved detections can be full resolution, while the
        # dataset RGB-D tensors may be downsampled (for example 640x360).
        # Resize masks and boxes to the mapping frame before pairing them with
        # depth or RGB values for point-cloud projection.
        resized_gobs = resize_gobs(raw_gobs, mapping_rgb)
        # filter the observations
        filtered_gobs = filter_gobs(resized_gobs, mapping_rgb,
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

        gobs = filtered_gobs

        if len(gobs['mask']) == 0: # no detections in this frame
            continue

        # this helps make sure things like pillows on couches are separate objects
        gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array,
            masks=gobs['mask'],
            cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
            image_rgb=mapping_rgb,
            trans_pose=adjusted_pose,
            min_points_threshold=cfg.min_points_threshold,
            spatial_sim_type=cfg.spatial_sim_type,
            obj_pcd_max_points=cfg.obj_pcd_max_points,
            device=cfg.device,
        )
        memory_budget.check(f"mapping after RGB-D projection for frame {frame_idx}")

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    pcd=obj["pcd"],
                )

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )

        if len(detection_list) == 0: # no detections, skip
            continue

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(objects) == 0:
            objects.extend(detection_list)
            tracker.increment_total_objects(len(detection_list))
            owandb.log({
                    "total_objects_so_far": tracker.get_total_objects(),
                    "objects_this_frame": len(detection_list),
                })
            continue 

        ### compute similarities and then merge
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg['spatial_sim_type'], 
            detection_list=detection_list, 
            objects=objects,
            downsample_voxel_size=cfg['downsample_voxel_size']
        )
        memory_budget.check(f"mapping after spatial similarity for frame {frame_idx}")

        visual_sim = compute_visual_similarities(detection_list, objects)
        memory_budget.check(f"mapping after visual similarity for frame {frame_idx}")

        agg_sim = aggregate_similarities(
            match_method=cfg['match_method'], 
            phys_bias=cfg['phys_bias'], 
            spatial_sim=spatial_sim, 
            visual_sim=visual_sim
        )

        # Perform matching of detections to existing objects
        match_indices = match_detections_to_objects(
            agg_sim=agg_sim, 
            detection_threshold=cfg['sim_threshold']  # Use the sim_threshold from the configuration
        )

        # Now merge the detected objects into the existing objects based on the match indices
        objects = merge_obj_matches(
            detection_list=detection_list, 
            objects=objects, 
            match_indices=match_indices,
            downsample_voxel_size=cfg['downsample_voxel_size'], 
            dbscan_remove_noise=cfg['dbscan_remove_noise'], 
            dbscan_eps=cfg['dbscan_eps'], 
            dbscan_min_points=cfg['dbscan_min_points'], 
            spatial_sim_type=cfg['spatial_sim_type'], 
            device=cfg['device']
            # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
        )
        memory_budget.check(f"mapping after object association for frame {frame_idx}")
        # fix the class names for objects
        # they should be the most popular name, not the first name
        for idx, obj in enumerate(objects):
            temp_class_name = obj["class_name"]
            curr_obj_class_id_counter = Counter(obj['class_id'])
            most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
            most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
            if temp_class_name != most_common_class_name:
                obj["class_name"] = most_common_class_name

        map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges, frame_idx)
        is_final_frame = frame_idx == len(dataset) - 1
        if is_final_frame:
            print("Final frame detected. Performing final post-processing...")

        # Clean up outlier edges
        edges_to_delete = []
        for curr_map_edge in map_edges.edges_by_index.values():
            curr_obj1_idx = curr_map_edge.obj1_idx
            curr_obj2_idx = curr_map_edge.obj2_idx
            obj1_class_name = objects[curr_obj1_idx]['class_name'] 
            obj2_class_name = objects[curr_obj2_idx]['class_name']
            curr_first_detected = curr_map_edge.first_detected
            curr_num_det = curr_map_edge.num_detections
            if (frame_idx - curr_first_detected > 5) and curr_num_det < 2:
                edges_to_delete.append((curr_obj1_idx, curr_obj2_idx))
        for edge in edges_to_delete:
            map_edges.delete_edge(edge[0], edge[1])
        ### Perform post-processing periodically if told so

        # Denoising
        if processing_needed(
            cfg["denoise_interval"],
            cfg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device'], 
                objects=objects
            )
            memory_budget.check(f"mapping after denoise for frame {frame_idx}")

        # Filtering
        if processing_needed(
            cfg["filter_interval"],
            cfg["run_filter_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = filter_objects(
                obj_min_points=cfg['obj_min_points'], 
                obj_min_detections=cfg['obj_min_detections'], 
                objects=objects,
                map_edges=map_edges
            )

        # Merging
        if processing_needed(
            cfg["merge_interval"],
            cfg["run_merge_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            # The edges of the objects are always tracked (the relations of
            # the VLM stage are part of the saved detections), so the merge
            # has to update the edge indices as well.
            objects, map_edges = measure_time(merge_objects)(
                merge_overlap_thresh=cfg["merge_overlap_thresh"],
                merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                objects=objects,
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                do_edges=True,
                map_edges=map_edges
            )
            memory_budget.check(f"mapping after merge for frame {frame_idx}")
        orr_log_objs_pcd_and_bbox(objects, obj_classes)
        orr_log_edges(objects, map_edges, obj_classes)

        if cfg.save_objects_all_frames:
            save_objects_for_frame(
                obj_all_frames_out_path,
                frame_idx,
                objects,
                cfg.obj_min_detections,
                adjusted_pose,
                color_path
            )
        
        if cfg.vis_render:
            # render a frame, if needed (not really used anymore since rerun)
            vis_render_image(
                objects,
                obj_classes,
                obj_renderer,
                image_original_pil,
                adjusted_pose,
                frames,
                frame_idx,
                color_path,
                cfg.obj_min_detections,
                cfg.class_agnostic,
                cfg.debug_render,
                is_final_frame,
                cfg.exp_out_path,
                cfg.exp_suffix,
            )

        if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
            # save the pointcloud
            save_pointcloud(
                exp_suffix=cfg.exp_suffix,
                exp_out_path=exp_out_path,
                cfg=cfg,
                objects=objects,
                obj_classes=obj_classes,
                latest_pcd_filepath=cfg.latest_pcd_filepath,
                create_symlink=True
            )

        owandb.log({
            "frame_idx": frame_idx,
            "counter": counter,
            "exit_early_flag": exit_early_flag,
            "is_final_frame": is_final_frame,
        })

        tracker.increment_total_objects(len(objects))
        tracker.increment_total_detections(len(detection_list))
        owandb.log({
                "total_objects": tracker.get_total_objects(),
                "objects_this_frame": len(objects),
                "total_detections": tracker.get_total_detections(),
                "detections_this_frame": len(detection_list),
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
                })
    # LOOP OVER -----------------------------------------------------
    memory_budget.check("mapping after final frame")
    
    # Consolidate captions 
    # NOTE: this is a *text* call -> it needs the local text model (Ollama).
    # Only one model fits into the GPU at a time, so this step can be skipped
    # here (consolidate_captions=False) and run afterwards with
    # scripts/consolidate_captions_local.py.
    if cfg.consolidate_captions:
        for object in objects:
            obj_captions = object['captions'][:20]
            consolidated_caption = consolidate_captions(llm_client, obj_captions)
            object['consolidated_caption'] = consolidated_caption
    else:
        print("Skipping the caption consolidation (cfg.consolidate_captions=False).")
        for object in objects:
            object.setdefault('consolidated_caption', '')

    handle_rerun_saving(cfg.use_rerun, cfg.save_rerun, cfg.exp_suffix, exp_out_path)

    # Save the pointcloud
    if cfg.save_pcd:
        save_pointcloud(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            cfg=cfg,
            objects=objects,
            obj_classes=obj_classes,
            latest_pcd_filepath=cfg.latest_pcd_filepath,
            create_symlink=True,
            edges=map_edges
        )

    if cfg.save_json:
        save_obj_json(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            objects=objects
        )
        
        save_edge_json(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            objects=objects,
            edges=map_edges
        )

    # Save metadata if all frames are saved
    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / f"meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': obj_classes.get_classes_arr(),
                'class_colors': obj_classes.get_class_color_dict_by_index(),
            }, f)

    if run_detections:
        if cfg.save_video:
            save_video_detections(det_exp_path)

    owandb.finish()

if __name__ == "__main__":
    main()
