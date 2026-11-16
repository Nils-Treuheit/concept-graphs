"""Generate and score GPT-derived visual references for saved object captions.

References are model-produced annotations, not human ground truth. The source
crop and all captions are matched by the stable object index in one frozen map.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import pickle
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK = REPO_ROOT / ".agentic_workflow/workspace/caption-grounding-benchmark"
DEFAULT_CASES = REPO_ROOT / ".agentic_workflow/workspace/caption-consolidation-benchmark/room0-40-case-sample.json"
DEFAULT_MAP = Path("/mnt/HDD1/Project_Data/public_datasets/Replica/room0/exps/r_modernized_yoloe26l_stride50/pcd_r_modernized_yoloe26l_stride50.pkl.gz")
DEFAULT_RESULTS = REPO_ROOT / ".agentic_workflow/workspace/caption-consolidation-benchmark"
DEFAULT_OPENCODE_SERVER = "http://127.0.0.1:20266"
REFERENCE_MODEL = "opencode/gpt-6-astra"
JUDGE_MODEL = "opencode/gpt-6.1-sol"
REFERENCE_PROMPT_VERSION = "masked_object_visual_reference_v1"
JUDGE_PROMPT_VERSION = "blind_caption_grounding_judge_v1"
CANDIDATE_FILES = {
    "Qwen3.8 baseline": "room0-qwen3.8-baseline-40.json",
    "Qwen3.5 9B Q4": "room0-qwen35-9b-q4-caption-consolidation-40.json",
    "Qwen3.5 9B Q8": "room0-qwen35-9b-q8-caption-consolidation-40.json",
    "Gemma 4 12B Q4": "room0-gemma4-12b-caption-consolidation-40.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _load_pickle_gzip(path: Path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def _representative_observation(obj: dict) -> tuple[int, np.ndarray, Path, int]:
    masks = obj.get("mask") or []
    paths = obj.get("color_path") or []
    indices = obj.get("image_idx") or []
    detections = obj.get("mask_idx") or []
    observations = []
    for offset, raw_mask in enumerate(masks):
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.ndim != 2 or not mask.any():
            continue
        if offset >= len(paths) or offset >= len(indices):
            continue
        detection_idx = int(detections[offset]) if offset < len(detections) else offset
        observations.append((int(mask.sum()), offset, mask, Path(paths[offset]), int(indices[offset]), detection_idx))
    if not observations:
        raise ValueError(f"object {obj.get('curr_obj_num')} has no non-empty mask observation")
    _, _, mask, image_path, frame_idx, detection_idx = max(observations, key=lambda item: (item[0], -item[1]))
    return frame_idx, mask, image_path, detection_idx


def _make_crop(image_path: Path, mask: np.ndarray, destination: Path) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read object source image: {image_path}")
    if image.shape[:2] != mask.shape:
        raise ValueError(f"mask/image resolution mismatch for {image_path}: {mask.shape} vs {image.shape[:2]}")
    rows, cols = np.nonzero(mask)
    y1, y2 = int(rows.min()), int(rows.max()) + 1
    x1, x2 = int(cols.min()), int(cols.max()) + 1
    margin = max(12, int(max(y2 - y1, x2 - x1) * 0.08))
    height, width = mask.shape
    left, top = max(0, x1 - margin), max(0, y1 - margin)
    right, bottom = min(width, x2 + margin), min(height, y2 + margin)
    crop_mask = mask[top:bottom, left:right]
    crop = np.full((bottom - top, right - left, 3), 127, dtype=np.uint8)
    crop[crop_mask] = image[top:bottom, left:right][crop_mask]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(127, 127, 127))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write crop {destination}")
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Failed to hash encoded crop {destination}")
    return {
        "crop_sha256": sha256_bytes(encoded.tobytes()),
        "mask_sha256": sha256_bytes(np.ascontiguousarray(mask).tobytes()),
        "mask_area_pixels": int(mask.sum()),
        "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
        "crop_bounds_xyxy": [left, top, right, bottom],
    }


def prepare(args: argparse.Namespace) -> Path:
    sample_path = args.sample.expanduser().resolve()
    sample = json.loads(sample_path.read_text())
    map_path = args.map.expanduser().resolve()
    map_sha = sha256_file(map_path)
    if sample.get("map_sha256") != map_sha:
        raise ValueError("The 40-track sample manifest map hash does not match the selected source map")
    with gzip.open(map_path, "rb") as stream:
        data = pickle.load(stream)
    objects = data["objects"]
    if len(sample.get("object_indices", [])) != 40:
        raise ValueError(f"Expected the frozen 40-track sample; got {len(sample.get('object_indices', []))}")
    references = []
    crop_root = args.work.expanduser().resolve() / "crops"
    for raw_index in sample["object_indices"]:
        object_index = int(raw_index)
        obj = objects[object_index]
        frame_idx, mask, image_path, detection_idx = _representative_observation(obj)
        crop_path = crop_root / f"object-{object_index:04d}-frame-{frame_idx:06d}-detection-{detection_idx:03d}.png"
        crop_meta = _make_crop(image_path, mask, crop_path)
        references.append({
            "object_index": object_index,
            "scene": sample.get("scene", "room0"),
            "class_label_private_metadata": str(obj.get("class_name", "")),
            "frame_index": frame_idx,
            "frame_path": str(image_path),
            "detection_index": detection_idx,
            "num_track_observations": int(obj.get("num_detections", len(obj.get("mask") or []))),
            "crop_path": str(crop_path),
            **crop_meta,
            "reference": None,
            "reference_status": "pending",
            "reference_runtime_seconds": None,
            "opencode_session_id": None,
            "error": None,
        })
    manifest_path = args.work.expanduser().resolve() / "references.json"
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_type": "GPT-generated visual references; not human ground truth",
        "scene": sample.get("scene", "room0"),
        "sample_manifest": str(sample_path),
        "source_map": str(map_path),
        "source_map_sha256": map_sha,
        "sample_size": len(references),
        "reference_model": REFERENCE_MODEL,
        "prompt_version": REFERENCE_PROMPT_VERSION,
        "crops": references,
    }
    write_json_atomic(manifest_path, manifest)
    print(f"Prepared {len(references)} masked object crops: {manifest_path}")
    print(f"Crop directory: {crop_root}")
    return manifest_path


REFERENCE_PROMPT = """Describe only the object visible in the attached masked crop. The pixels outside the object were replaced with neutral gray. Do not infer a scene location or relationships from the crop. Do not invent material, function, brand, or hidden details. Return exactly one JSON object with this schema: {\"caption\": \"one concise sentence about visible appearance\", \"facts\": [\"short visually verifiable attribute\", ...], \"uncertainty\": [\"uncertain or unidentifiable visible detail\", ...]}. Include 2 to 6 distinct visible facts when possible, such as category, color, shape, pattern, or visible state. If the object category is unclear, say so in the caption and uncertainty. Do not include a class label, object ID, or explanation outside JSON."""


def _api_request(server: str, password: str, method: str, path: str, body: dict | None = None) -> tuple[int, object]:
    url = server.rstrip("/") + "/api/" + path.lstrip("/")
    payload = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=payload, method=method)
    basic = base64.b64encode(f"opencode:{password}".encode()).decode()
    request.add_header("Authorization", f"Basic {basic}")
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, raw.decode(errors="replace")[:1000]


def _extract_cli_output(stdout: str, title: str) -> tuple[str | None, str | None]:
    session_id = None
    text_parts = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        session_id = session_id or event.get("sessionID") or event.get("session_id")
        part = event.get("part") or event.get("data", {}).get("part") or {}
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        if isinstance(event.get("text"), str):
            text_parts.append(event["text"])
    return session_id, "\n".join(text_parts).strip() or None


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        left, right = text.find("{"), text.rfind("}")
        if left < 0 or right <= left:
            raise ValueError("model response did not contain a JSON object")
        parsed = json.loads(text[left:right + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON is not an object")
    return parsed


def _terminate_client_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _finish_owned_session(server: str, password: str, session_id: str) -> None:
    code, _ = _api_request(server, password, "POST", f"session/{session_id}/interrupt?resume=false")
    if code not in (200, 204, 404):
        raise RuntimeError(f"Could not interrupt owned OpenCode session {session_id}: HTTP {code}")
    code, active = _api_request(server, password, "GET", "session/active")
    if code != 200:
        raise RuntimeError(f"Could not verify OpenCode session inactivity: HTTP {code}")
    rows = active.get("data", active) if isinstance(active, dict) else active
    if any(isinstance(row, dict) and row.get("id") == session_id for row in rows or []):
        raise RuntimeError(f"Owned OpenCode session {session_id} remains active after interrupt")
    code, _ = _api_request(server, password, "PATCH", f"session/{session_id}", {"archived": True})
    if code not in (200, 204):
        raise RuntimeError(f"Could not archive owned OpenCode session {session_id}: HTTP {code}")


def generate(args: argparse.Namespace) -> None:
    password = os.environ.get("OPENCODE_PASSWORD")
    if not password:
        raise RuntimeError("OPENCODE_PASSWORD is required in the child environment; do not pass it as an argument")
    cooldown_path = args.work.expanduser().resolve() / "opencode-model-cooldowns.json"
    if cooldown_path.is_file():
        cooldowns = json.loads(cooldown_path.read_text()).get("models", {})
        record = cooldowns.get(REFERENCE_MODEL)
        if record and datetime.now(timezone.utc) < datetime.fromisoformat(record["retry_after_utc"]):
            raise RuntimeError(f"{REFERENCE_MODEL} is in a quota cooldown until {record['retry_after_utc']}")
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    ownership_path = args.work.expanduser().resolve() / "opencode-session-ownership.json"
    ownership = json.loads(ownership_path.read_text()) if ownership_path.is_file() else {"sessions": []}
    ownership["owner"] = "caption-reference-benchmark"
    ownership["updated_utc"] = datetime.now(timezone.utc).isoformat()
    processed = 0
    for case in manifest["crops"]:
        if case.get("reference_status") == "complete":
            continue
        if args.limit and processed >= args.limit:
            break
        crop = Path(case["crop_path"])
        if not crop.is_file() or sha256_file(crop) != case["crop_sha256"]:
            raise ValueError(f"crop missing or changed for object {case['object_index']}: {crop}")
        title = f"caption-reference-room0-object-{case['object_index']:04d}"
        command = [
            "opencode", "run", "--server", args.server, "--auto", "--title", title,
            "--model", REFERENCE_MODEL, "--format", "json", "--file", str(crop), REFERENCE_PROMPT,
        ]
        started = time.perf_counter()
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=os.environ.copy(), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            _terminate_client_group(process)
            stdout, stderr = process.communicate()
            sid, _ = _extract_cli_output(stdout, title)
            if sid:
                _finish_owned_session(args.server, password, sid)
            case["reference_status"] = "timeout"
            case["error"] = f"OpenCode request exceeded {args.timeout:g}s; {stderr[-1000:]}"
            write_json_atomic(manifest_path, manifest)
            raise RuntimeError(case["error"])
        runtime = time.perf_counter() - started
        session_id, response_text = _extract_cli_output(stdout, title)
        ownership["sessions"].append({
            "session_id": session_id, "title": title, "model": REFERENCE_MODEL,
            "client_exit_code": process.returncode, "started_utc": datetime.now(timezone.utc).isoformat(),
        })
        write_json_atomic(ownership_path, ownership)
        if session_id:
            _finish_owned_session(args.server, password, session_id)
            ownership["sessions"][-1]["archived"] = True
            write_json_atomic(ownership_path, ownership)
        if process.returncode != 0 or not response_text:
            case["reference_status"] = "error"
            case["error"] = (stderr or stdout)[-2000:]
            write_json_atomic(manifest_path, manifest)
            raise RuntimeError(f"OpenCode failed on object {case['object_index']}: {case['error']}")
        try:
            reference = _extract_json_object(response_text)
            caption = reference.get("caption")
            facts = reference.get("facts")
            if not isinstance(caption, str) or not caption.strip() or not isinstance(facts, list):
                raise ValueError("response lacks a non-empty caption or facts list")
            case["reference"] = {
                "caption": caption.strip(),
                "facts": [str(value).strip() for value in facts if str(value).strip()],
                "uncertainty": [str(value).strip() for value in reference.get("uncertainty", []) if str(value).strip()],
            }
            case["reference_status"] = "complete"
            case["reference_model"] = REFERENCE_MODEL
            case["reference_prompt_version"] = REFERENCE_PROMPT_VERSION
            case["reference_prompt_sha256"] = sha256_bytes(REFERENCE_PROMPT.encode())
            case["reference_runtime_seconds"] = round(runtime, 3)
            case["opencode_session_id"] = session_id
            case["error"] = None
        except Exception as exc:
            case["reference_status"] = "invalid_output"
            case["error"] = f"{type(exc).__name__}: {exc}; raw={response_text[:2000]}"
            write_json_atomic(manifest_path, manifest)
            raise RuntimeError(f"Invalid GPT reference for object {case['object_index']}: {exc}") from exc
        write_json_atomic(manifest_path, manifest)
        print(f"[{case['object_index']:04d}] GPT reference saved in {runtime:.1f}s", flush=True)
        processed += 1
    print(f"All {len(manifest['crops'])} references saved: {manifest_path}")


def _load_candidates(results_dir: Path, expected_map_sha: str) -> dict[str, dict[int, dict]]:
    output = {}
    for model_name, filename in CANDIDATE_FILES.items():
        path = results_dir / filename
        data = json.loads(path.read_text())
        if data.get("map_sha256") != expected_map_sha:
            raise ValueError(f"{model_name} result was produced from a different map hash")
        output[model_name] = {int(row["object_index"]): row for row in data["groups"]}
    return output


def import_references(args: argparse.Namespace) -> None:
    """Merge separately authored GPT vision references into the prepared manifest."""
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    authored_path = args.authored.expanduser().resolve()
    authored = json.loads(authored_path.read_text())
    if authored.get("source_map_sha256") != manifest.get("source_map_sha256"):
        raise ValueError("authored references were created from another source map")
    authored_rows = authored.get("references") or authored.get("items") or []
    authored_cases = {int(row["object_index"]): row for row in authored_rows}
    if set(authored_cases) != {int(row["object_index"]) for row in manifest["crops"]}:
        raise ValueError("authored reference IDs do not exactly match the 40-track sample")
    for row in manifest["crops"]:
        source = authored_cases[int(row["object_index"])]
        caption, facts = source.get("caption"), source.get("facts")
        if not isinstance(caption, str) or not caption.strip() or not isinstance(facts, list) or not 2 <= len(facts) <= 6:
            raise ValueError(f"invalid caption/fact set for object {row['object_index']}")
        row["reference"] = {
            "caption": caption.strip(),
            "facts": [str(value).strip() for value in facts if str(value).strip()],
            "uncertainty": [str(value).strip() for value in source.get("uncertainty", []) if str(value).strip()],
        }
        row["reference_crop_path"] = row["crop_path"]
        row["reference_status"] = "complete"
        row["reference_model"] = authored.get("reference_model", authored.get("annotator", "Codex GPT vision annotator; model ID unavailable"))
        row["reference_prompt_version"] = authored.get("prompt_version", authored.get("annotation_method", "codex_masked_object_reference_v1"))
        row["opencode_session_id"] = None
        row["reference_runtime_seconds"] = None
        row["error"] = None
    manifest["reference_model"] = authored.get("reference_model", authored.get("annotator", "Codex GPT vision annotator; model ID unavailable"))
    manifest["reference_source_artifact"] = str(authored_path)
    manifest["prompt_version"] = authored.get("prompt_version", authored.get("annotation_method", "codex_masked_object_reference_v1"))
    manifest["reference_generation_route"] = "Codex GPT vision, separate annotation pass"
    cooldown_path = args.work.expanduser().resolve() / "opencode-model-cooldowns.json"
    if cooldown_path.is_file():
        cooldown = json.loads(cooldown_path.read_text()).get("models", {}).get(REFERENCE_MODEL)
        if cooldown:
            manifest["failed_opencode_reference_attempt"] = {
                "model": REFERENCE_MODEL,
                "error_type": cooldown.get("error_type"),
                "message": cooldown.get("message"),
                "started_utc": cooldown.get("started_utc"),
                "retry_after_utc": cooldown.get("retry_after_utc"),
                "session_archived": True,
            }
    write_json_atomic(manifest_path, manifest)
    print(f"Imported {len(authored_cases)} GPT visual references into {manifest_path}")


def _candidate_caption(model_name: str, row: dict) -> tuple[str, bool, float | None]:
    if model_name == "Qwen3.8 baseline":
        caption = row.get("caption")
        return (caption.strip() if isinstance(caption, str) else "", bool(caption and caption.strip()), None)
    caption = row.get("candidate_caption")
    valid = bool(row.get("valid_output") and isinstance(caption, str) and caption.strip())
    elapsed = row.get("response_seconds")
    return (caption.strip() if isinstance(caption, str) else "", valid, float(elapsed) if elapsed is not None else None)


def build_judgments(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if any(row.get("reference_status") != "complete" for row in manifest["crops"]):
        raise ValueError("cannot build judgments until all 40 GPT references are complete")
    candidates = _load_candidates(args.results.expanduser().resolve(), manifest["source_map_sha256"])
    expected_indices = {int(row["object_index"]) for row in manifest["crops"]}
    for model_name, rows in candidates.items():
        if set(rows) != expected_indices:
            raise ValueError(f"{model_name} does not cover exactly the frozen 40 object IDs")
    cases = []
    for reference in manifest["crops"]:
        index = int(reference["object_index"])
        captions = {}
        for model_name, rows in candidates.items():
            row = rows.get(index)
            if row is None:
                captions[model_name] = {"caption": "", "valid": False, "response_seconds": None}
                continue
            caption, valid, elapsed = _candidate_caption(model_name, row)
            captions[model_name] = {"caption": caption, "valid": valid, "response_seconds": elapsed}
        seed = int(hashlib.sha256(f"room0:{index}:caption-reference-v1".encode()).hexdigest()[:8], 16)
        aliases = ["A", "B", "C", "D"]
        random.Random(seed).shuffle(aliases)
        blinded = {alias: name for alias, name in zip(aliases, captions)}
        cases.append({
            "object_index": index,
            "reference": reference["reference"],
            "blind_aliases": blinded,
            "captions": {alias: captions[name]["caption"] for alias, name in blinded.items()},
            "validity": {alias: captions[name]["valid"] for alias, name in blinded.items()},
            "response_seconds": {alias: captions[name]["response_seconds"] for alias, name in blinded.items()},
        })
    batch_size = args.batch_size
    batches = [cases[i:i + batch_size] for i in range(0, len(cases), batch_size)]
    result_path = args.work.expanduser().resolve() / "blinded-judgments.json"
    write_json_atomic(result_path, {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_map_sha256": manifest["source_map_sha256"],
        "reference_manifest": str(manifest_path),
        "reference_model": manifest.get("reference_model", REFERENCE_MODEL),
        "judge_model": JUDGE_MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "candidate_names_private": list(CANDIDATE_FILES),
        "batch_size": batch_size,
        "cases": cases,
        "judgments": [],
        "status": "ready_for_judge",
    })
    print(f"Prepared {len(cases)} blinded caption comparisons in {len(batches)} batches: {result_path}")


def export_blinded_judge_inputs(args: argparse.Namespace) -> None:
    data = json.loads(args.input.expanduser().resolve().read_text())
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    crop_by_index = {int(row["object_index"]): row["crop_path"] for row in manifest["crops"]}
    inputs = [{
        "object_index": case["object_index"],
        "crop_path": crop_by_index[int(case["object_index"])],
        "visual_reference": case["reference"],
        "candidate_captions": case["captions"],
        "candidate_validity": case["validity"],
    } for case in data["cases"]]
    output = args.work.expanduser().resolve() / "blinded-judge-inputs.json"
    write_json_atomic(output, {
        "schema_version": 1,
        "reference_type": "GPT-generated visual references, not human ground truth",
        "instruction": "Review the crop and compare each anonymous caption against visible appearance and the reference facts. Do not infer absent details.",
        "cases": inputs,
    })
    print(f"Exported {len(inputs)} blinded judge inputs: {output}")


def import_judgments(args: argparse.Namespace) -> None:
    result_path = args.input.expanduser().resolve()
    data = json.loads(result_path.read_text())
    authored = json.loads(args.authored.expanduser().resolve().read_text())
    judgments = authored.get("judgments")
    if not isinstance(judgments, list) or len(judgments) != len(data["cases"]):
        raise ValueError("authored judge file must contain one judgment row for each of the 40 tracks")
    expected = {int(row["object_index"]) for row in data["cases"]}
    observed = {int(row["object_index"]) for row in judgments}
    if observed != expected:
        raise ValueError("authored judgments do not cover the exact blinded object IDs")
    aliases = {alias for case in data["cases"] for alias in case["blind_aliases"]}
    for row in judgments:
        scores = row.get("scores") or {}
        if set(scores) != aliases:
            raise ValueError(f"judgment for object {row['object_index']} must score all blind aliases")
        for alias, score in scores.items():
            if not isinstance(score.get("supported_fact_indices"), list) or not isinstance(score.get("contradicted_fact_indices"), list):
                raise ValueError(f"judge output for object {row['object_index']} alias {alias} lacks fact index lists")
    data["judgments"] = judgments
    data["status"] = "complete"
    data["judge_model"] = authored.get("judge_model", "Codex GPT judge; model ID unavailable")
    data["judge_prompt_version"] = authored.get("prompt_version", "codex_blinded_visual_grounding_judge_v1")
    data["judge_source_artifact"] = str(args.authored.expanduser().resolve())
    data["judge_runtime_completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(result_path, data)
    print(f"Imported {len(judgments)} blind GPT judgments into {result_path}")


def make_judge_prompt(batch: list[dict]) -> str:
    payload = [{
        "object_index": row["object_index"],
        "visual_reference": row["reference"],
        "candidate_captions": row["captions"],
    } for row in batch]
    data = json.dumps(payload, ensure_ascii=False)
    return (
        "Assess how well each anonymous caption is supported by the independent visual-reference facts. "
        "Treat the reference as an imperfect GPT-generated visual annotation, not infallible truth. "
        "Do not reward copied wording. Evaluate only object identity and intrinsic visible attributes; "
        "ignore scene location and relations. A candidate may add a compatible detail absent from the "
        "reference; mark it unsupported only if it is specific and cannot be verified from the supplied "
        "reference, and mark contradictions only when it conflicts with a reference fact. For each case, "
        "return one result per alias A-D. Return strict JSON with schema "
        "{\"cases\":[{\"object_index\":number,\"scores\":{\"A\":{\"object_match\":boolean,"
        "\"supported_fact_indices\":[integer,...],\"contradicted_fact_indices\":[integer,...],"
        "\"unsupported_claims\":[string,...],\"groundedness\":1-5,\"notes\":string},...}}]}. "
        "Use zero-based fact indices from each reference facts list. If caption is empty, mark object_match "
        "false and groundedness 1. Do not include model names. Data:\n" + data
    )


def judge(args: argparse.Namespace) -> None:
    password = os.environ.get("OPENCODE_PASSWORD")
    if not password:
        raise RuntimeError("OPENCODE_PASSWORD is required in the child environment")
    result_path = args.input.expanduser().resolve()
    data = json.loads(result_path.read_text())
    if data.get("status") == "complete":
        print(f"Judgments already complete: {result_path}")
        return
    cases = data["cases"]
    batches = [cases[i:i + data["batch_size"]] for i in range(0, len(cases), data["batch_size"])]
    ownership_path = args.work.expanduser().resolve() / "opencode-session-ownership.json"
    ownership = json.loads(ownership_path.read_text()) if ownership_path.is_file() else {"sessions": []}
    judgments = []
    for batch_num, batch in enumerate(batches, start=1):
        title = f"caption-grounding-judge-room0-batch-{batch_num:02d}"
        command = ["opencode", "run", "--server", args.server, "--auto", "--title", title,
                   "--model", JUDGE_MODEL, "--format", "json", make_judge_prompt(batch)]
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=os.environ.copy(), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            _terminate_client_group(process)
            stdout, stderr = process.communicate()
            sid, _ = _extract_cli_output(stdout, title)
            if sid:
                _finish_owned_session(args.server, password, sid)
            raise RuntimeError(f"OpenCode judge exceeded {args.timeout:g}s: {stderr[-1000:]}")
        session_id, response_text = _extract_cli_output(stdout, title)
        ownership["sessions"].append({
            "session_id": session_id, "title": title, "model": JUDGE_MODEL,
            "client_exit_code": process.returncode,
        })
        write_json_atomic(ownership_path, ownership)
        if session_id:
            _finish_owned_session(args.server, password, session_id)
            ownership["sessions"][-1]["archived"] = True
            write_json_atomic(ownership_path, ownership)
        if process.returncode != 0 or not response_text:
            raise RuntimeError(f"OpenCode judge failed for batch {batch_num}: {(stderr or stdout)[-2000:]}")
        parsed = _extract_json_object(response_text)
        if not isinstance(parsed.get("cases"), list):
            raise ValueError(f"Judge batch {batch_num} did not return a cases list")
        judgments.extend(parsed["cases"])
        print(f"[{batch_num}/{len(batches)}] judged {len(batch)} cases", flush=True)
    data["judgments"] = judgments
    data["status"] = "complete"
    data["judge_runtime_completed_utc"] = datetime.now(timezone.utc).isoformat()
    data["judge_prompt_version"] = JUDGE_PROMPT_VERSION
    write_json_atomic(result_path, data)
    print(f"Saved blinded judgments: {result_path}")


def _word_f1(left: str, right: str) -> float:
    import re
    from collections import Counter
    a = Counter(re.findall(r"[\w']+", left.casefold()))
    b = Counter(re.findall(r"[\w']+", right.casefold()))
    if not a or not b:
        return float(a == b)
    overlap = sum((a & b).values())
    precision = overlap / sum(a.values())
    recall = overlap / sum(b.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def report(args: argparse.Namespace) -> Path:
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    judged = json.loads(args.input.expanduser().resolve().read_text())
    if judged.get("status") != "complete":
        raise ValueError("cannot report until all blinded judge batches are complete")
    candidate_names = judged["candidate_names_private"]
    sample = json.loads(Path(manifest["sample_manifest"]).read_text())
    multi_ids = {int(value) for value in sample.get("multi_caption_indices", [])}
    by_index = {int(row["object_index"]): row for row in judged["cases"]}
    judge_by_index = {int(row["object_index"]): row for row in judged["judgments"]}
    rows = {name: [] for name in candidate_names}
    for case in judged["cases"]:
        index = int(case["object_index"])
        judged_case = judge_by_index.get(index, {})
        for alias, model_name in case["blind_aliases"].items():
            score = (judged_case.get("scores") or {}).get(alias, {})
            fact_total = len(case["reference"].get("facts") or [])
            supported = len(set(score.get("supported_fact_indices") or []))
            contradicted = len(set(score.get("contradicted_fact_indices") or []))
            rows[model_name].append({
                "object_index": index,
                "valid": bool(case["validity"].get(alias)),
                "caption": case["captions"].get(alias, ""),
                "response_seconds": case["response_seconds"].get(alias),
                "object_match": bool(score.get("object_match")),
                "fact_recall": supported / fact_total if fact_total else None,
                "contradiction_rate": contradicted / fact_total if fact_total else None,
                "unsupported_claims": len(score.get("unsupported_claims") or []),
                "groundedness": score.get("groundedness"),
                "word_f1": _word_f1(case["captions"].get(alias, ""), case["reference"].get("caption", "")),
            })
    def summarize(items):
        complete = [item for item in items if item["valid"]]
        def mean(key):
            values = [float(x[key]) for x in complete if x[key] is not None]
            return sum(values) / len(values) if values else None
        return {
            "valid_captions": len(complete),
            "coverage": len(complete) / len(items) if items else 0.0,
            "object_match_rate": mean("object_match"),
            "mean_reference_fact_recall": mean("fact_recall"),
            "mean_reference_fact_contradiction_rate": mean("contradiction_rate"),
            "mean_unsupported_claims_per_caption": mean("unsupported_claims"),
            "mean_gpt_groundedness_1_to_5": mean("groundedness"),
            "mean_word_f1_to_reference_caption": mean("word_f1"),
            "mean_saved_candidate_seconds": mean("response_seconds"),
        }
    summary = {}
    for name, items in rows.items():
        summary[name] = {
            "all_40": summarize(items),
            "multi_caption_tracks": summarize([item for item in items if item["object_index"] in multi_ids]),
            "single_caption_tracks": summarize([item for item in items if item["object_index"] not in multi_ids]),
        }
    result = {
        "schema_version": 1,
        "reference_type": "GPT-generated visual reference annotations, not human ground truth",
        "scene": manifest["scene"],
        "sample_size": len(rows[candidate_names[0]]) if candidate_names else 0,
        "source_map": manifest["source_map"],
        "source_map_sha256": manifest["source_map_sha256"],
        "reference_model": manifest.get("reference_model", REFERENCE_MODEL),
        "judge_model": judged.get("judge_model", JUDGE_MODEL),
        "reference_prompt_version": manifest["prompt_version"],
        "judge_prompt_version": judged.get("judge_prompt_version"),
        "models": summary,
        "per_object_scores": rows,
        "limitations": [
            "The reference captions and evaluator scores are GPT-generated; they are not human ground truth.",
            "Each track is represented by its largest visible mask observation, so color or view-dependent attributes may vary across observations.",
            "The saved consolidation outputs are reused; no candidate VLM reruns or latency remeasurements were performed.",
            "Only five of the 40 tracks have multiple source captions, so consolidation-specific conclusions have a small sample.",
            "Lexical word F1 is reported only as a secondary agreement measure.",
        ],
    }
    json_path = args.work.expanduser().resolve() / "caption-reference-scores.json"
    write_json_atomic(json_path, result)
    md_path = args.work.expanduser().resolve() / "caption-reference-report.md"
    lines = [
        "# GPT-generated visual-reference caption comparison", "",
        f"Scene: {result['scene']}. Sample: {result['sample_size']} tracks.",
        f"Reference model: `{result['reference_model']}`. Judge model: `{result['judge_model']}`.",
        "The references are model-generated visual annotations, not human ground truth.", "",
        "## Method", "",
        "Each sampled map track was represented by its largest non-empty YOLOE mask observation. "
        "Pixels outside the mask were replaced with neutral gray. The reference annotator saw only "
        "that crop and returned an appearance caption, visible facts, and uncertainty notes.",
        "Candidate captions were joined by the frozen map object index and scored in a blinded pass "
        "that saw the crop, reference facts, and anonymous captions. The GPT judge compared visible "
        "support and contradictions; scores remain model-based estimates.",
        "The measured candidates are saved Qwen3.8, Qwen3.5 Q4/Q8, and Gemma 4 consolidation outputs. "
        "No candidate model was rerun; latency reuses saved candidate response times.",
        "", "## Scores", "",
        "| Model | Valid | Object match | GPT fact recall | Contradiction rate | Unsupported claims/caption | Groundedness / 5 | Word F1 (secondary) | Saved mean latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary.items():
        def fmt(key, digits=3):
            value = values["all_40"].get(key)
            return "n/a" if value is None else f"{value:.{digits}f}"
        latency_value = values["all_40"].get("mean_saved_candidate_seconds")
        latency = "n/a" if latency_value is None else f"{latency_value:.3f} s"
        lines.append(
            f"| {name} | {values['all_40']['valid_captions']}/{result['sample_size']} | "
            f"{fmt('object_match_rate')} | {fmt('mean_reference_fact_recall')} | {fmt('mean_reference_fact_contradiction_rate')} | "
            f"{fmt('mean_unsupported_claims_per_caption')} | {fmt('mean_gpt_groundedness_1_to_5')} | "
            f"{fmt('mean_word_f1_to_reference_caption')} | {latency} |"
        )
    lines.extend(["", "Multi-caption tracks only (n=5):", "", "| Model | GPT fact recall | Contradiction rate | Groundedness / 5 |", "|---|---:|---:|---:|"])
    for name, values in summary.items():
        subset = values["multi_caption_tracks"]
        def subset_fmt(key):
            value = subset.get(key)
            return "n/a" if value is None else f"{value:.3f}"
        lines.append(
            f"| {name} | {subset_fmt('mean_reference_fact_recall')} | "
            f"{subset_fmt('mean_reference_fact_contradiction_rate')} | "
            f"{subset_fmt('mean_gpt_groundedness_1_to_5')} |"
        )
    lines.extend(["", "## Source outputs", ""])
    lines.extend(f"- {name}: `{args.results.expanduser().resolve() / filename}`" for name, filename in CANDIDATE_FILES.items())
    lines.extend(["", "## Interpretation limits", ""])
    lines.append("No clear caption-quality winner emerged: all four saved outputs received nearly identical GPT visual-grounding scores.")
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.extend([
        "",
        (f"OpenCode route note: `{manifest['failed_opencode_reference_attempt']['model']}` returned "
         f"`{manifest['failed_opencode_reference_attempt']['error_type']}: "
         f"{manifest['failed_opencode_reference_attempt']['message']}`; its session was archived and no retry was issued."
         if manifest.get("failed_opencode_reference_attempt") else ""),
        f"Machine-readable results: `{json_path}`",
        f"Reference manifest: `{args.manifest.expanduser().resolve()}`",
        f"Reference source: `{manifest.get('reference_source_artifact', 'not recorded')}`",
        f"Blinded score source: `{judged.get('judge_source_artifact', 'not recorded')}`",
        "",
    ])
    md_path.write_text("\n".join(lines))
    print(f"Saved report: {md_path}")
    print(f"Saved scores: {json_path}")
    return md_path


def aggregate_isolated(args: argparse.Namespace) -> Path:
    """Aggregate four independently judged, single-candidate visual passes."""
    work = args.work.expanduser().resolve()
    blinded = json.loads((work / "blinded-judgments.json").read_text())
    mapping = json.loads((work / "isolated-judge-set-map.json").read_text())
    manifest = json.loads((work / "references.json").read_text())
    sample = json.loads(Path(manifest["sample_manifest"]).read_text())
    references = {int(row["object_index"]): row for row in manifest["crops"]}
    by_index = {int(row["object_index"]): row for row in blinded["cases"]}
    multi_ids = {int(value) for value in sample.get("multi_caption_indices", [])}
    models = {}
    merged = {idx: {} for idx in by_index}
    for entry in mapping["sets"]:
        set_key = str(entry["set_id"]).removeprefix("set-")
        source_path = work / f"isolated-judgment-{set_key}.json"
        source = json.loads(source_path.read_text())
        rows = {int(row["object_index"]): row for row in source.get("judgments", [])}
        if set(rows) != set(by_index):
            raise ValueError(f"isolated judgment set {set_key} does not cover the exact 40 IDs")
        model_name = entry["model"]
        model_rows = []
        for idx, judgment in rows.items():
            case = by_index[idx]
            alias = next(a for a, name in case["blind_aliases"].items() if name == model_name)
            score = judgment
            if not isinstance(score.get("supported_fact_indices"), list) or not isinstance(score.get("contradicted_fact_indices"), list):
                raise ValueError(f"invalid fact scoring for {model_name}, object {idx}")
            object_match_value = score.get("object_match")
            if object_match_value is True:
                object_match_value = "yes"
            elif object_match_value is False:
                object_match_value = "poor"
            object_match_value = str(object_match_value).casefold()
            object_match_value = {
                "true": "yes", "match": "yes", "good": "yes",
                "false": "poor", "no": "poor", "mismatch": "poor",
            }.get(object_match_value, object_match_value)
            if object_match_value not in {"yes", "partial", "poor"}:
                raise ValueError(f"invalid object_match value {object_match_value!r} for object {idx}")
            score["object_match"] = object_match_value
            merged[idx][alias] = score
            reference_facts = references[idx]["reference"].get("facts") or []
            count = len(reference_facts)
            caption = case["captions"].get(alias, "")
            valid = bool(case["validity"].get(alias))
            supported = len(set(score["supported_fact_indices"]))
            contradicted = len(set(score["contradicted_fact_indices"]))
            model_rows.append({
                "object_index": idx,
                "valid": valid,
                "caption": caption,
                "response_seconds": case["response_seconds"].get(alias),
                "object_match": object_match_value,
                "fact_recall": supported / count if count else None,
                "contradiction_rate": contradicted / count if count else None,
                "unsupported_claims": len(score.get("unsupported_claims") or []),
                "groundedness": score.get("groundedness"),
                "word_f1": _word_f1(caption, references[idx]["reference"].get("caption", "")),
            })
        models[model_name] = model_rows

    def summarize(items):
        valid = [item for item in items if item["valid"]]
        def mean(key):
            values = [float(item[key]) for item in valid if item[key] is not None]
            return sum(values) / len(values) if values else None
        denom = len(valid) or 1
        return {
            "n": len(items),
            "valid_captions": len(valid),
            "coverage": len(valid) / len(items) if items else 0.0,
            "object_match_yes_rate": sum(item["object_match"] == "yes" for item in valid) / denom,
            "object_match_partial_rate": sum(item["object_match"] == "partial" for item in valid) / denom,
            "object_match_poor_rate": sum(item["object_match"] in {"poor", "no"} for item in valid) / denom,
            "mean_reference_fact_recall": mean("fact_recall"),
            "mean_reference_fact_contradiction_rate": mean("contradiction_rate"),
            "mean_unsupported_claims_per_caption": mean("unsupported_claims"),
            "mean_gpt_groundedness_1_to_5": mean("groundedness"),
            "mean_word_f1_to_reference_caption": mean("word_f1"),
            "mean_saved_candidate_seconds": mean("response_seconds"),
        }

    baseline = next((name for name in models if name == "Qwen3.8 baseline"), None)
    summary = {}
    for model_name, items in models.items():
        baseline_by_id = {item["object_index"]: item for item in models[baseline]} if baseline else {}
        summary[model_name] = {
            "all_40": summarize(items),
            "multi_caption_tracks": summarize([item for item in items if item["object_index"] in multi_ids]),
            "single_caption_tracks": summarize([item for item in items if item["object_index"] not in multi_ids]),
            "captions_different_from_qwen38": sum(
                item["caption"] != baseline_by_id.get(item["object_index"], {}).get("caption")
                for item in items
            ),
        }

    result = {
        "schema_version": 2,
        "reference_type": "GPT-generated visual references, not human ground truth",
        "reference_model": manifest.get("reference_model"),
        "judge_model": "Independent Codex GPT visual judges, one anonymized candidate set per judge; exact model IDs unavailable",
        "scene": manifest.get("scene"),
        "sample_size": len(by_index),
        "source_map": manifest.get("source_map"),
        "source_map_sha256": manifest.get("source_map_sha256"),
        "models": summary,
        "per_model_per_object_scores": models,
        "merged_blind_judgments": merged,
        "limitations": [
            "References and scores are GPT-generated estimates, not human ground truth.",
            "Each track is represented by its largest valid masked observation.",
            "Existing caption outputs and response latencies were reused; candidate models were not rerun.",
            "Only five sampled tracks have multiple source captions.",
            "Reference facts and judge scores are subjective and may miss visible details.",
        ],
    }
    score_path = work / "caption-reference-scores-v2.json"
    write_json_atomic(score_path, result)
    lines = [
        "# Corrected GPT visual-reference caption evaluation", "",
        "The first pooled judgment was withdrawn because it reused the same score across different captions. This corrected pass gave each anonymous candidate set to a separate visual judge.",
        "The reference set and ratings remain GPT estimates, not human ground truth.", "",
        f"Scene: {result['scene']}; sample: {result['sample_size']} tracks.",
        f"Reference annotator: `{result['reference_model']}`.",
        f"Candidate judging: `{result['judge_model']}`.",
        "", "| Model | Valid | Different from Qwen3.8 | Exact object match | Partial object match | Poor object match | Fact recall | Contradiction rate | Unsupported claims/caption | Groundedness / 5 | Saved latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary.items():
        m = values["all_40"]
        latency = "n/a" if m["mean_saved_candidate_seconds"] is None else f"{m['mean_saved_candidate_seconds']:.3f} s"
        lines.append(
            f"| {name} | {m['valid_captions']}/{m['n']} | {values['captions_different_from_qwen38']}/40 | "
            f"{m['object_match_yes_rate']:.3f} | {m['object_match_partial_rate']:.3f} | {m['object_match_poor_rate']:.3f} | "
            f"{m['mean_reference_fact_recall']:.3f} | {m['mean_reference_fact_contradiction_rate']:.3f} | "
            f"{m['mean_unsupported_claims_per_caption']:.3f} | {m['mean_gpt_groundedness_1_to_5']:.3f} | {latency} |"
        )
    lines.extend(["", "## Multi-caption subset", "", "Only five sampled tracks contain multiple source captions.", "", "| Model | Exact match | Partial match | Poor match | Fact recall | Contradiction rate | Groundedness / 5 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for name, values in summary.items():
        m = values["multi_caption_tracks"]
        lines.append(f"| {name} | {m['object_match_yes_rate']:.3f} | {m['object_match_partial_rate']:.3f} | {m['object_match_poor_rate']:.3f} | {m['mean_reference_fact_recall']:.3f} | {m['mean_reference_fact_contradiction_rate']:.3f} | {m['mean_gpt_groundedness_1_to_5']:.3f} |")
    lines.extend(["", "## Output differences", "", "Candidate captions differ from Qwen3.8 on the per-model counts above. The raw captions and all isolated judgments are retained as machine-readable per-object records.", "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.extend(["", f"Scores: `{score_path}`", f"Reference manifest: `{work / 'references.json'}`", f"Candidate outputs: `{DEFAULT_RESULTS.resolve()}`", ""])
    report_path = work / "caption-reference-report.md"
    report_path.write_text("\n".join(lines))
    print(f"Saved corrected report: {report_path}")
    print(f"Saved corrected scores: {score_path}")
    return report_path


def aggregate_claim_level(args: argparse.Namespace) -> Path:
    """Aggregate per-claim labels from a common blind visual judge."""
    work = args.work.expanduser().resolve()
    blinded = json.loads((work / "blinded-judgments.json").read_text())
    inputs = json.loads((work / "blinded-judge-inputs.json").read_text())
    judged = json.loads((work / "claim-level-judgments-v2.json").read_text())
    manifest = json.loads((work / "references.json").read_text())
    sample = json.loads(Path(manifest["sample_manifest"]).read_text())
    refs = {int(row["object_index"]): row for row in manifest["crops"]}
    captions = {int(row["object_index"]): row for row in inputs["cases"]}
    blind_cases = {int(row["object_index"]): row for row in blinded["cases"]}
    judgment_rows = judged.get("judgments") or judged.get("rows") or []
    judgments = {int(row["object_index"]): row for row in judgment_rows}
    expected_ids = set(captions)
    if set(judgments) != expected_ids or len(judgment_rows) != len(expected_ids):
        raise ValueError("claim-level judge must return exactly one row for each of the 40 tracks")
    models = {name: [] for name in blinded["candidate_names_private"]}
    multi_ids = {int(value) for value in sample.get("multi_caption_indices", [])}
    merged = {}
    for idx in sorted(expected_ids):
        row = judgments[idx]
        scores = row.get("scores") or {}
        if set(scores) != {"A", "B", "C", "D"}:
            raise ValueError(f"object {idx} must include exactly A-D claims")
        case = blind_cases[idx]
        ref_facts = refs[idx]["reference"].get("facts") or []
        count = len(ref_facts)
        merged[idx] = scores
        for alias, score in scores.items():
            if score.get("object_category_match") not in {"yes", "no", "unclear"}:
                raise ValueError(f"invalid object category decision for object {idx}, alias {alias}")
            claim_rows = score.get("candidate_claims")
            fact_rows = score.get("reference_facts")
            if not isinstance(claim_rows, list) or not isinstance(fact_rows, list):
                raise ValueError(f"object {idx}, alias {alias} lacks claim/fact labels")
            source_caption = captions[idx]["candidate_captions"].get(alias, "")
            source_tokens = " ".join(re.findall(r"[\w']+", str(source_caption).casefold()))
            if {int(value["index"]) for value in fact_rows} != set(range(count)):
                raise ValueError(f"object {idx}, alias {alias} must label every reference fact exactly once")
            for claim in claim_rows:
                if claim.get("status") not in {"supported", "contradicted", "unverifiable"}:
                    raise ValueError(f"invalid candidate claim status for object {idx}, alias {alias}")
                claim_text = " ".join(re.findall(r"[\w']+", str(claim.get("text", "")).casefold()))
                if not claim_text or claim_text not in source_tokens:
                    raise ValueError(
                        f"claim {claim.get('text')!r} is not quoted from object {idx}, alias {alias} caption; refusing to aggregate"
                    )
            for fact in fact_rows:
                if fact.get("status") not in {"expressed", "contradicted", "omitted"}:
                    raise ValueError(f"invalid reference fact status for object {idx}, alias {alias}")
            supported = sum(fact["status"] == "expressed" for fact in fact_rows)
            contradicted = sum(fact["status"] == "contradicted" for fact in fact_rows)
            supported_claims = sum(claim["status"] == "supported" for claim in claim_rows)
            contradicted_claims = sum(claim["status"] == "contradicted" for claim in claim_rows)
            unverifiable = sum(claim["status"] == "unverifiable" for claim in claim_rows)
            model_name = case["blind_aliases"][alias]
            caption_row = captions[idx]
            models[model_name].append({
                "object_index": idx,
                "valid": bool(caption_row["candidate_validity"].get(alias)),
                "caption": caption_row["candidate_captions"].get(alias, ""),
                "response_seconds": case["response_seconds"].get(alias),
                "object_category_match": score["object_category_match"],
                "reference_fact_recall": supported / count if count else None,
                "reference_fact_contradiction_rate": contradicted / count if count else None,
                "supported_candidate_claims": supported_claims,
                "contradicted_candidate_claims": contradicted_claims,
                "unverifiable_candidate_claims": unverifiable,
                "candidate_claim_precision": supported_claims / len(claim_rows) if claim_rows else None,
                "judge_notes": score.get("notes", score.get("evidence_notes", "")),
            })

    def summarize(rows):
        valid = [row for row in rows if row["valid"]]
        def mean(key):
            values = [float(row[key]) for row in valid if row[key] is not None]
            return sum(values) / len(values) if values else None
        n = len(valid) or 1
        return {
            "n": len(rows),
            "valid_captions": len(valid),
            "coverage": len(valid) / len(rows) if rows else 0.0,
            "object_category_yes_rate": sum(row["object_category_match"] == "yes" for row in valid) / n,
            "object_category_unclear_rate": sum(row["object_category_match"] == "unclear" for row in valid) / n,
            "object_category_no_rate": sum(row["object_category_match"] == "no" for row in valid) / n,
            "mean_reference_fact_recall": mean("reference_fact_recall"),
            "mean_reference_fact_contradiction_rate": mean("reference_fact_contradiction_rate"),
            "mean_candidate_claim_precision": mean("candidate_claim_precision"),
            "mean_contradicted_claims": mean("contradicted_candidate_claims"),
            "mean_unverifiable_claims": mean("unverifiable_candidate_claims"),
            "mean_saved_candidate_seconds": mean("response_seconds"),
        }
    baseline_name = "Qwen3.8 baseline"
    base = {int(row["object_index"]): row["caption"] for row in models[baseline_name]}
    summary = {}
    for name, rows in models.items():
        summary[name] = {
            "all_40": summarize(rows),
            "multi_caption_tracks": summarize([row for row in rows if row["object_index"] in multi_ids]),
            "single_caption_tracks": summarize([row for row in rows if row["object_index"] not in multi_ids]),
            "captions_different_from_qwen38": sum(row["caption"] != base[row["object_index"]] for row in rows),
        }
    result = {
        "schema_version": 3,
        "evaluation_status": "claim-level blind common-judge; GPT estimates, not human ground truth",
        "reference_model": manifest.get("reference_model"),
        "judge_model": judged.get("judge_model", "Codex GPT visual judge; exact model ID unavailable"),
        "scene": manifest.get("scene"),
        "sample_size": len(expected_ids),
        "source_map_sha256": manifest.get("source_map_sha256"),
        "models": summary,
        "per_model_per_object_scores": models,
        "claim_level_judgments": merged,
        "limitations": [
            "GPT-generated visual references and judge labels are estimates, not human ground truth.",
            "Each track is represented by one largest-mask observation.",
            "Candidate outputs and response times are reused from the saved 40-track consolidation benchmark.",
            "Only five tracks have multiple source captions, so conclusions about merging are underpowered.",
        ],
    }
    score_path = work / "caption-reference-scores-v3.json"
    write_json_atomic(score_path, result)
    lines = [
        "# GPT visual-reference caption comparison", "",
        "This is a claim-level pass using one blinded GPT judge across all four candidate sets. It replaces the invalid pooled ratings and the uncalibrated separate-judge aggregates.",
        "The references and judgments are model estimates, not human ground truth.", "",
        f"Scene: {result['scene']}; sample: 40 tracks.",
        f"Reference annotator: `{result['reference_model']}`. Judge: `{result['judge_model']}`.",
        "", "| Model | Valid | Different from Qwen3.8 | Category match | Fact recall | Fact contradiction | Candidate claim precision | Contradicted claims/caption | Unverifiable claims/caption | Saved latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary.items():
        m = values["all_40"]
        latency = "n/a" if m["mean_saved_candidate_seconds"] is None else f"{m['mean_saved_candidate_seconds']:.3f} s"
        lines.append(
            f"| {name} | {m['valid_captions']}/{m['n']} | {values['captions_different_from_qwen38']}/40 | "
            f"{m['object_category_yes_rate']:.3f} | {m['mean_reference_fact_recall']:.3f} | "
            f"{m['mean_reference_fact_contradiction_rate']:.3f} | {m['mean_candidate_claim_precision']:.3f} | "
            f"{m['mean_contradicted_claims']:.3f} | {m['mean_unverifiable_claims']:.3f} | {latency} |"
        )
    lines.extend(["", "## Multi-caption subset (n=5)", "", "| Model | Category match | Fact recall | Fact contradiction | Candidate claim precision |", "|---|---:|---:|---:|---:|"])
    for name, values in summary.items():
        m = values["multi_caption_tracks"]
        lines.append(f"| {name} | {m['object_category_yes_rate']:.3f} | {m['mean_reference_fact_recall']:.3f} | {m['mean_reference_fact_contradiction_rate']:.3f} | {m['mean_candidate_claim_precision']:.3f} |")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.extend(["", f"Scores: `{score_path}`", f"References: `{work / 'references.json'}`", f"Raw candidate outputs: `{DEFAULT_RESULTS.resolve()}`", ""])
    report_path = work / "caption-reference-report.md"
    report_path.write_text("\n".join(lines))
    print(f"Saved claim-level report: {report_path}")
    print(f"Saved claim-level scores: {score_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="create representative masked crops and pending reference manifest")
    p.add_argument("--sample", type=Path, default=DEFAULT_CASES)
    p.add_argument("--map", type=Path, default=DEFAULT_MAP)
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.set_defaults(func=prepare)
    p = sub.add_parser("generate", help="ask the configured OpenCode vision model for each object reference")
    p.add_argument("--manifest", type=Path, default=DEFAULT_WORK / "references.json")
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.add_argument("--server", default=DEFAULT_OPENCODE_SERVER)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--limit", type=int, default=0, help="process at most this many pending cases; 0 means all")
    p.set_defaults(func=generate)
    p = sub.add_parser("build-judgments", help="prepare blinded candidate caption batches")
    p.add_argument("--manifest", type=Path, default=DEFAULT_WORK / "references.json")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.add_argument("--batch-size", type=int, default=5)
    p.set_defaults(func=build_judgments)
    p = sub.add_parser("import-references", help="merge a GPT-authored visual reference JSON artifact")
    p.add_argument("--manifest", type=Path, default=DEFAULT_WORK / "references.json")
    p.add_argument("--authored", type=Path, default=DEFAULT_WORK / "references-authored.json")
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.set_defaults(func=import_references)
    p = sub.add_parser("export-blinded-judge-inputs", help="write anonymous captions and image paths for visual scoring")
    p.add_argument("--input", type=Path, default=DEFAULT_WORK / "blinded-judgments.json")
    p.add_argument("--manifest", type=Path, default=DEFAULT_WORK / "references.json")
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.set_defaults(func=export_blinded_judge_inputs)
    p = sub.add_parser("import-judgments", help="import independent blinded GPT scores from a JSON artifact")
    p.add_argument("--input", type=Path, default=DEFAULT_WORK / "blinded-judgments.json")
    p.add_argument("--authored", type=Path, default=DEFAULT_WORK / "judgments-authored.json")
    p.set_defaults(func=import_judgments)
    p = sub.add_parser("judge", help="blind-score candidate captions against the GPT visual references")
    p.add_argument("--input", type=Path, default=DEFAULT_WORK / "blinded-judgments.json")
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.add_argument("--server", default=DEFAULT_OPENCODE_SERVER)
    p.add_argument("--timeout", type=float, default=180)
    p.set_defaults(func=judge)
    p = sub.add_parser("report", help="aggregate judge results and lexical comparison")
    p.add_argument("--manifest", type=Path, default=DEFAULT_WORK / "references.json")
    p.add_argument("--input", type=Path, default=DEFAULT_WORK / "blinded-judgments.json")
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.set_defaults(func=report)
    p = sub.add_parser("aggregate-isolated", help="aggregate independent one-candidate-at-a-time visual judgments")
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.set_defaults(func=aggregate_isolated)
    p = sub.add_parser("aggregate-claim-level", help="aggregate a common judge's itemized claim scores")
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.set_defaults(func=aggregate_claim_level)
    args = parser.parse_args()
    if hasattr(args, "batch_size") and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    try:
        result = args.func(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
