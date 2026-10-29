"""Regression check for the RealSense mapper's process_edges call."""

import ast
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
MAPPER_RELATIVE_PATH = Path("conceptgraph/slam/r3d_stream_rerun_realtime_mapping.py")
MAPPER_CANDIDATES = (
    TEST_DIR.parent / "implementation-role" / "repo" / MAPPER_RELATIVE_PATH,
    TEST_DIR.parent / MAPPER_RELATIVE_PATH,
)
MAPPER_PATH = next((path for path in MAPPER_CANDIDATES if path.is_file()), MAPPER_CANDIDATES[0])


def test_realsense_mapping_passes_current_frame_idx_to_process_edges():
    module = ast.parse(MAPPER_PATH.read_text(encoding="utf-8"), filename=str(MAPPER_PATH))
    mapping_function = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_run_mapping"
    )
    calls = [
        node
        for node in ast.walk(mapping_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "process_edges"
    ]

    assert len(calls) == 1, "expected exactly one process_edges call in _run_mapping"
    assert calls[0].args, "process_edges must receive positional arguments"
    final_argument = calls[0].args[-1]
    assert isinstance(final_argument, ast.Name) and final_argument.id == "frame_idx", (
        "_run_mapping must pass the current frame_idx as process_edges' final positional argument"
    )
