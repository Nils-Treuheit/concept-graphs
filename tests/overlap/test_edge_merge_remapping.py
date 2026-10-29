"""Exercise real GPU-Faiss consolidation and graph-edge remapping together."""

import faiss
import numpy as np
import open3d as o3d
import torch

from conceptgraph.slam import utils
from conceptgraph.slam.slam_classes import MapEdge, MapEdgeMapping, MapObjectList


def _require_gpu_faiss():
    assert torch.cuda.is_available(), "edge consolidation requires CUDA"
    assert faiss.get_num_gpus() > 0, "edge consolidation requires GPU Faiss"


def _object(object_id, offset):
    corners = np.asarray(
        [
            [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1],
            [0.1, 0.1, 0.0], [0.1, 0.0, 0.1], [0.0, 0.1, 0.1], [0.1, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    corners[:, 0] += offset
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(corners)
    return {
        "id": object_id,
        "pcd": cloud,
        "bbox": cloud.get_axis_aligned_bounding_box(),
        "clip_ft": torch.tensor([1.0, 0.0], device="cuda:0"),
        "num_detections": 1,
        "n_points": len(corners),
    }


def _add_edge(edges, source, target, relation, count):
    edge = MapEdge(source, target, relation, num_detections=count)
    edges.edges_by_index[(source, target)] = edge
    edges.edges_by_uuid[
        (edges.objects[source]["id"], edges.objects[target]["id"])
    ] = edge


def test_gpu_consolidation_remaps_edges_and_drops_merge_created_self_loops():
    _require_gpu_faiss()
    utils.clear_gpu_faiss_index_cache()
    objects = MapObjectList(
        [
            _object("duplicate-a", 0.0),
            _object("duplicate-b", 0.0),
            _object("duplicate-c", 0.0),
            _object("other", 2.0),
        ],
        device="cuda:0",
    )
    edges = MapEdgeMapping(objects)
    _add_edge(edges, 0, 3, "on top of", 2)
    _add_edge(edges, 1, 3, "on top of", 3)
    _add_edge(edges, 2, 3, "on top of", 5)
    _add_edge(edges, 3, 0, "left of", 4)
    _add_edge(edges, 3, 1, "left of", 6)
    _add_edge(edges, 3, 2, "left of", 7)
    _add_edge(edges, 0, 1, "near", 7)
    _add_edge(edges, 1, 2, "near", 8)
    _add_edge(edges, 0, 2, "near", 9)

    merged, updated_edges = utils.merge_objects(
        merge_overlap_thresh=0.5,
        merge_visual_sim_thresh=0.5,
        merge_text_sim_thresh=0.5,
        objects=objects,
        downsample_voxel_size=0.025,
        dbscan_remove_noise=False,
        dbscan_eps=0.0,
        dbscan_min_points=0,
        spatial_sim_type="axis_aligned",
        device="cuda:0",
        do_edges=True,
        map_edges=edges,
    )

    assert len(merged) == 2
    assert updated_edges.objects is merged
    assert set(updated_edges.edges_by_index) == {(0, 1), (1, 0)}
    assert {edge.rel_type for edge in updated_edges.edges_by_index.values()} == {
        "on top of",
        "left of",
    }
    assert updated_edges.edges_by_index[(0, 1)].num_detections == 10
    assert updated_edges.edges_by_index[(1, 0)].num_detections == 17
    assert set(updated_edges.edges_by_uuid) == {
        (merged[0]["id"], merged[1]["id"]),
        (merged[1]["id"], merged[0]["id"]),
    }
    for (source, target), edge in updated_edges.edges_by_index.items():
        assert (edge.obj1_idx, edge.obj2_idx) == (source, target)

    utils.clear_gpu_faiss_index_cache()
    torch.cuda.synchronize()
