"""Fail-closed lifecycle controls for the local model-control service."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_MODEL_CONTROL_URL = "http://127.0.0.1:20268"


def _request(path: str, *, method: str = "GET", payload: dict | None = None, timeout: float = 15.0):
    base = os.getenv("OPENCODE_MODEL_CONTROL_URL", DEFAULT_MODEL_CONTROL_URL).rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("model-control URL must be a loopback HTTP URL")
    headers = {"Accept": "application/json"}
    data = None
    if method == "POST":
        headers.update({"Content-Type": "application/json", "X-Model-Unload": "1"})
        data = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            code, body = response.status, response.read()
    except urllib.error.HTTPError as exc:
        code, body = exc.code, exc.read()
    result = json.loads(body.decode("utf-8")) if body else {}
    if not isinstance(result, dict):
        raise RuntimeError(f"model-control {path} returned a non-object response")
    return code, result


def _released(state: dict) -> bool:
    """Require the app's explicit engine inventory to show all engines released."""
    if state.get("ollama_loaded") != []:
        return False
    if state.get("vllm_listening") is not False or state.get("unsloth_listening") is not False:
        return False
    ollama = state.get("ollama")
    if isinstance(ollama, dict):
        servers = ollama.get("servers")
        if not isinstance(servers, list) or not servers:
            return False
        for server in servers:
            if not isinstance(server, dict):
                return False
            if server.get("models") != [] or server.get("gpu_runner_processes") != []:
                return False
    vllm = state.get("vllm")
    if isinstance(vllm, dict) and vllm.get("processes") != []:
        return False
    unsloth = state.get("unsloth")
    if isinstance(unsloth, dict) and unsloth.get("pids") != []:
        return False
    return True


def get_model_control_status() -> dict:
    """Return the loopback control app's current validated status object."""
    code, state = _request("/status")
    if code != 200:
        raise RuntimeError(f"model-control /status returned HTTP {code}: {state}")
    return state


def require_idle_model_control() -> None:
    """Fail unless managed inference engines and unmanaged GPU jobs are idle."""
    state = get_model_control_status()
    if not _released(state):
        raise RuntimeError("model-control reports an active managed inference engine")
    ollama = state.get("ollama")
    unmatched = ollama.get("unmatched_gpu_processes") if isinstance(ollama, dict) else None
    if unmatched != []:
        raise RuntimeError(
            "model-control reports unmanaged GPU processes; refusing to start inference: "
            f"{unmatched}"
        )
    djev = state.get("djev")
    if isinstance(djev, dict) and (
        djev.get("ready") or djev.get("process_groups_alive")
    ):
        raise RuntimeError("model-control reports an active D-JEV inference engine")
    jobs = state.get("jobs")
    terminal_states = {
        "done", "complete", "completed", "success", "succeeded", "failed", "error",
    }
    if not isinstance(jobs, dict):
        raise RuntimeError("model-control status omitted its job inventory")
    for job_name, job in jobs.items():
        if not isinstance(job, dict) or str(job.get("status", "")).lower() not in terminal_states:
            raise RuntimeError(
                f"model-control job {job_name!r} is active or has unknown status: {job}"
            )


def verify_ollama_gpu_model(model: str) -> list[int]:
    """Require the requested model and at least one GPU runner in app status."""
    code, state = _request("/status")
    if code != 200:
        raise RuntimeError(f"model-control /status returned HTTP {code}: {state}")
    loaded = state.get("ollama_loaded")
    if not isinstance(loaded, list):
        raise RuntimeError("model-control status omitted the Ollama loaded-model inventory")

    def names(rows):
        result = []
        for row in rows:
            if isinstance(row, str):
                result.append(row)
            elif isinstance(row, dict):
                result.extend(
                    str(row[key]) for key in ("name", "model", "id", "model_name")
                    if row.get(key) is not None
                )
        return result

    expected = model.casefold().removesuffix(":latest")
    loaded_names = names(loaded)
    if not any(name.casefold().removesuffix(":latest") == expected for name in loaded_names):
        raise RuntimeError(
            f"Ollama does not report the requested VLM loaded ({model}); "
            f"loaded models: {loaded_names}"
        )

    ollama = state.get("ollama")
    servers = ollama.get("servers") if isinstance(ollama, dict) else None
    if not isinstance(servers, list):
        raise RuntimeError("model-control status omitted Ollama GPU-runner inventory")
    runner_pids = set()
    for server in servers:
        if not isinstance(server, dict):
            continue
        server_models = names(server.get("models", []))
        if server_models and not any(
            name.casefold().removesuffix(":latest") == expected for name in server_models
        ):
            continue
        runners = server.get("gpu_runner_processes")
        if not isinstance(runners, list):
            raise RuntimeError("model-control status omitted Ollama GPU-runner process details")
        for runner in runners:
            if isinstance(runner, dict):
                pid = runner.get("pid", runner.get("process_id"))
            else:
                pid = runner
            try:
                runner_pids.add(int(pid))
            except (TypeError, ValueError):
                continue
    if not runner_pids:
        raise RuntimeError(
            f"The requested Ollama model ({model}) has no observable GPU runner process"
        )
    return sorted(runner_pids)


def load_ollama_gpu_model(
    model: str,
    *,
    base_url: str = "http://127.0.0.1:20261",
    keep_alive: int = -1,
    timeout: float = 180.0,
    poll_interval: float = 1.0,
) -> list[int]:
    """Load one Ollama model through model-control and confirm its GPU runner."""
    code, response = _request(
        "/load",
        method="POST",
        payload={"base_url": base_url, "model": model, "keep_alive": keep_alive},
        timeout=min(timeout, 30.0),
    )
    if code not in {200, 202}:
        raise RuntimeError(f"model-control /load returned HTTP {code}: {response}")
    if response.get("status") in {"error", "failed"}:
        raise RuntimeError(f"model-control load job failed: {response}")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return verify_ollama_gpu_model(model)
        except RuntimeError as exc:
            code, state = _request("/status", timeout=min(timeout, 15.0))
            if code != 200:
                raise RuntimeError(f"model-control /status returned HTTP {code}: {state}")
            load_job = (state.get("jobs") or {}).get("load", {})
            if isinstance(load_job, dict) and load_job.get("status") in {"error", "failed"}:
                raise RuntimeError(f"model-control load job failed: {load_job}") from exc
            time.sleep(poll_interval)
    raise TimeoutError(
        f"model-control did not confirm the GPU runner for {model} within {timeout:.0f}s"
    )


def _job_status(state: dict) -> str | None:
    jobs = state.get("jobs")
    if not isinstance(jobs, dict):
        return None
    for name, job in jobs.items():
        if "unload" not in str(name).lower() and "release" not in str(name).lower():
            continue
        if isinstance(job, dict) and isinstance(job.get("status"), str):
            return job["status"].lower()
    return None


def unload_all_models(*, timeout: float = 120.0, poll_interval: float = 1.0) -> None:
    """Unload managed local engines and wait until model-control confirms release."""
    code, response = _request("/unload", method="POST", timeout=min(timeout, 30.0))
    if code not in {200, 202}:
        raise RuntimeError(f"model-control /unload returned HTTP {code}: {response}")

    deadline = time.monotonic() + timeout
    response_job = response.get("job")
    requires_job_completion = False
    if isinstance(response_job, dict):
        response_status = response_job.get("status")
        if response_status in {"error", "failed"}:
            raise RuntimeError(f"model-control unload job failed: {response_job}")
        requires_job_completion = response_status not in {
            "done", "complete", "completed", "success", "succeeded",
        }
    requires_job_completion = requires_job_completion or any(
        key in response for key in ("job_id", "jobId", "operation_id", "operationId")
    )
    while time.monotonic() < deadline:
        status_code, state = _request("/status", timeout=min(timeout, 15.0))
        job_status = _job_status(state)
        if job_status in {"error", "failed"}:
            raise RuntimeError(f"model-control unload job failed: {job_status}")
        job_finished = not requires_job_completion and job_status is None or job_status in {
            "done", "complete", "completed", "success", "succeeded",
        }
        if status_code == 200 and job_finished and _released(state):
            return
        time.sleep(poll_interval)
    raise TimeoutError("model-control did not confirm that all managed models were released")
