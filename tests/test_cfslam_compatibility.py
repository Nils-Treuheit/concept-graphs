"""Compatibility contracts for the original CFSLAM entrypoints; no GPU work."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conceptgraph.slam import utils
from conceptgraph.slam.slam_classes import DetectionList


def _inputs():
    cfg = SimpleNamespace(
        device="cpu", min_points_threshold=1, mask_area_threshold=0,
        max_bbox_area_ratio=1.0, mask_conf_threshold=0.0, skip_bg=False,
        downsample_voxel_size=0.025, dbscan_remove_noise=False,
        dbscan_eps=0.1, dbscan_min_points=1, spatial_sim_type="iou",
    )
    return dict(
        cfg=cfg, image=np.zeros((3, 3, 3), dtype=np.uint8),
        depth_array=np.ones((3, 3)), cam_K=np.eye(3), idx=7,
        trans_pose=np.eye(4), class_names=["chair", "wall", "table"],
        BG_CLASSES=["wall"], color_path="frame.png",
        gobs=dict(
            classes=["chair", "wall", "table"],
            xyxy=np.tile([0, 0, 2, 2], (3, 1)),
            mask=np.ones((3, 3, 3), dtype=bool),
            class_id=np.array([0, 1, 2]), confidence=np.array([0.9, 0.8, 0.7]),
            image_feats=np.eye(3, dtype=np.float32),
            text_feats=np.eye(3, dtype=np.float32),
        ),
    )


class _Cloud:
    points = np.zeros((5, 3))

    def transform(self, pose):
        return self


def test_original_cfslam_converter_preserves_order_and_metadata(monkeypatch):
    converter = getattr(utils, "gobs_to_detection_list", None)
    assert callable(converter), "CFSLAM imports gobs_to_detection_list from slam.utils"
    monkeypatch.setattr(utils, "create_object_pcd", lambda *a, **k: _Cloud())
    monkeypatch.setattr(utils, "process_pcd", lambda pcd, *a, **k: pcd)
    monkeypatch.setattr(utils, "get_bounding_box", lambda *a, **k: SimpleNamespace(volume=lambda: 1.0))
    foreground, background = converter(**_inputs())
    assert isinstance(foreground, DetectionList)
    assert isinstance(background, DetectionList)
    assert [d["class_name"] for d in foreground] == [["chair"], ["table"]]
    assert [d["class_name"] for d in background] == [["wall"]]
    for detection, mask_idx in zip([foreground[0], background[0], foreground[1]], range(3)):
        assert detection["image_idx"] == [7]
        assert detection["mask_idx"] == [mask_idx]
        assert detection["class_id"] == [mask_idx]
        assert detection["color_path"] == ["frame.png"]
        assert detection["num_detections"] == 1
        assert {"pcd", "bbox", "clip_ft"} <= detection.keys()


def test_original_cfslam_converter_reports_missing_required_metadata():
    converter = getattr(utils, "gobs_to_detection_list", None)
    assert callable(converter), "CFSLAM imports gobs_to_detection_list from slam.utils"
    inputs = _inputs()
    del inputs["gobs"]["class_id"]
    with pytest.raises((KeyError, ValueError), match="class_id"):
        converter(**inputs)


def test_original_cfslam_converter_skip_bg_preserves_foreground(monkeypatch):
    inputs = _inputs()
    inputs["cfg"].skip_bg = True
    monkeypatch.setattr(utils, "create_object_pcd", lambda *a, **k: _Cloud())
    monkeypatch.setattr(utils, "process_pcd", lambda pcd, *a, **k: pcd)
    monkeypatch.setattr(utils, "get_bounding_box", lambda *a, **k: SimpleNamespace(volume=lambda: 1.0))

    foreground, background = utils.gobs_to_detection_list(**inputs)

    assert isinstance(foreground, DetectionList)
    assert isinstance(background, DetectionList)
    assert [d["class_name"] for d in foreground] == [["chair"], ["table"]]
    assert background == [], "skip_bg must prevent background detections reaching the caller"


def test_scenegraph_overlap_call_passes_scene_map_and_voxel_size():
    # Execute the real call expression without importing model-heavy scenegraph code.
    source = Path(__file__).resolve().parents[1] / "conceptgraph/scenegraph/build_scenegraph_cfslam.py"
    tree = ast.parse(source.read_text())
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_scenegraph")
    calls = [n for n in ast.walk(build) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "compute_overlap_matrix_general"]
    assert len(calls) == 1
    scene_map = object()
    args = SimpleNamespace(downsample_voxel_size=0.025)

    def overlap(objects_a, objects_b=None, downsample_voxel_size=None):
        assert objects_a is scene_map, "First overlap argument must be scene_map"
        assert objects_b is None, "Scenegraph computes self-overlap"
        assert downsample_voxel_size == args.downsample_voxel_size

    eval(compile(ast.Expression(calls[0]), str(source), "eval"),
         {"compute_overlap_matrix_general": overlap, "scene_map": scene_map, "args": args})
