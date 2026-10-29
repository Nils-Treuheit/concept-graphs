"""Stage 5 of the local ConceptGraphs pipeline: build the scene graph.

The scene graph is built from the map of the mapping stage:

* the **nodes** are the 3D objects; their tag is derived from the detected
  class and the consolidated caption with the **local text model**
  (Qwen3.8 served by Ollama),
* the **edges** are the spatial relations that the VLM found between the
  objects ("on top of" / "under"), aggregated over all frames.

The result is written as ``scene_graph_<exp_suffix>.json`` next to the map.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
from pathlib import Path

from conceptgraph.utils.llm_client import get_llm_client

SYSTEM_PROMPT_OBJECT_TAG = """
You are an agent that names the objects of an indoor scene for a 3D scene graph.

For every object you are given the class that was predicted by an object
detector and a caption that describes the object in the scene.  Your task is to
write a short, specific name (at most four words) for the object, e.g.
"wooden side table", "white ceramic mug" or "potted plant".

Answer with a JSON object of the form
{"object_tag": "<short name>"}
and nothing else.
"""


def clean_tag(tag: str, fallback: str) -> str:
    tag = (tag or "").strip().strip('"').strip()
    tag = re.sub(r"\s+", " ", tag)
    if not tag:
        return fallback
    return tag[:60]


def build_object_tag(client, class_name: str, caption: str) -> str:
    """Ask the local text model for a short, specific name of the object."""
    fallback = (class_name or "object").strip() or "object"
    if not caption:
        return fallback

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_OBJECT_TAG},
        {
            "role": "user",
            "content": (
                f"Detected class: {class_name}\n"
                f"Caption: {caption}\n\n"
                "What is the short name of this object?"
            ),
        },
    ]

    try:
        response = client.chat.completions.create(
            model=getattr(client, "model", None),
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = (response.choices[0].message.content or "").strip()
        tag = json.loads(content).get("object_tag", "")
        return clean_tag(tag, fallback)
    except Exception as exc:  # pragma: no cover - depends on the local server
        print(f"Could not name the object ({exc}), using the detected class.")
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp_out_path",
        required=True,
        help="Folder of the mapping experiment (contains pcd_<exp_suffix>.pkl.gz)",
    )
    parser.add_argument("--exp_suffix", required=True, help="Suffix of the mapping experiment")
    parser.add_argument(
        "--min_edge_detections",
        type=int,
        default=1,
        help="Relations that were seen in fewer frames are dropped",
    )
    args = parser.parse_args()

    exp_out_path = Path(args.exp_out_path)
    with gzip.open(exp_out_path / f"pcd_{args.exp_suffix}.pkl.gz", "rb") as f:
        results = pickle.load(f)

    objects = results["objects"]
    edges = (results.get("edges") or {}).get("edges", [])

    client = get_llm_client()
    print(f"Using the local text model '{client.model}' at {client.base_url}")

    nodes = []
    for obj_idx, curr_obj in enumerate(objects):
        class_name = curr_obj.get("class_name", "")
        caption = curr_obj.get("consolidated_caption", "") or ""
        tag = build_object_tag(client, class_name, caption)
        nodes.append(
            {
                "id": curr_obj.get("curr_obj_num", obj_idx + 1),
                "object_tag": tag,
                "detected_class": class_name,
                "object_caption": caption,
                "num_detections": len(curr_obj.get("captions") or []),
            }
        )
        print(f"[{obj_idx + 1}/{len(objects)}] {class_name} -> {tag}")

    scene_edges = []
    for edge in edges:
        if edge.get("num_detections", 0) < args.min_edge_detections:
            continue
        obj1_idx, obj2_idx = edge["obj1_index"], edge["obj2_index"]
        if obj1_idx >= len(nodes) or obj2_idx >= len(nodes):
            continue
        rel_type = edge.get("rel_type", "next to")
        scene_edges.append(
            {
                "object_1_id": nodes[obj1_idx]["id"],
                "object_1_tag": nodes[obj1_idx]["object_tag"],
                "object_2_id": nodes[obj2_idx]["id"],
                "object_2_tag": nodes[obj2_idx]["object_tag"],
                "relationship": rel_type,
                "num_detections": edge.get("num_detections", 0),
            }
        )

    scene_graph = {"nodes": nodes, "edges": scene_edges}
    out_path = exp_out_path / f"scene_graph_{args.exp_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(scene_graph, f, indent=2)
    print(f"Saved the scene graph with {len(nodes)} nodes and {len(scene_edges)} edges to {out_path}")


if __name__ == "__main__":
    main()
