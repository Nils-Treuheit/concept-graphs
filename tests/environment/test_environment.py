"""Slice A contract tests: reproducible checkout environment.

Pure manifest checks of ``PROJECT_ROOT/pyproject.toml`` and
``PROJECT_ROOT/uv.lock``. ``$PROJECT_ROOT`` selects the staged checkout; when
unset (for example after promotion), the nearest ancestor containing both
manifests is used. Expected values are inlined from the frozen Slice A
fixture. No network, shell, GPU, or model access; the clean ``uv sync
--locked`` run and the CUDA verifier are root-owned acceptance separate from
these tests.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import pytest

# Frozen Slice A fixture values (inlined on purpose).

CLIP_NAME = "clip"
CLIP_URL = "https://github.com/ultralytics/CLIP.git"
CLIP_REV = "7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40"

#: Locked package name -> native source directory that must survive.
NATIVE_SOURCES = {
    "faiss-gpu-cu12": "third_party/faiss",
    "pytorch3d": "third_party/pytorch3d",
}

#: Locked versions, CUDA local suffix included, that must be preserved.
TORCH_VERSIONS = {
    "torch": "2.14.1+cu130",
    "torchvision": "0.29.1+cu130",
    "torchaudio": "2.11.0+cu130",
}

#: Named ``[tool.uv] index`` in pyproject and the registry it locks to.
TORCH_INDEX = "pytorch-cu130"
TORCH_REGISTRY = "https://download.pytorch.org/whl/cu130"

OPENAI_NAME = "openai"


def _normalize(name: object) -> str:
    """PEP 503 normalization for a distribution or source key."""
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def _dep_name(requirement: str) -> str:
    """Normalized distribution name at the head of a PEP 508 requirement."""
    match = re.match(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", requirement)
    if match is None:
        pytest.fail(f"cannot parse a distribution name from {requirement!r}")
    return _normalize(match.group(0))


def _parse_git_ref(value: str) -> tuple[str, str | None]:
    """Split ``git+<url>@<ref>`` into ``(url, ref)``; ``ref`` may be None."""
    v = value.strip().removeprefix("git+")
    url, sep, ref = v.rpartition("@")
    return (v, None) if not sep else (url, ref)


def _norm_path(path: object) -> str:
    """POSIX-normalized path without leading ``./`` or trailing ``/``."""
    return re.sub(r"^\./", "", str(path).replace("\\", "/")).rstrip("/")


def _load_toml(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        pytest.fail(f"{path} is not parseable TOML, which the contract requires: {exc}")
    assert isinstance(data, dict), f"{path} must contain a top-level TOML table"
    return data


def _uv_table(pyproject: dict) -> dict | None:
    tool = pyproject.get("tool")
    uv_cfg = tool.get("uv") if isinstance(tool, dict) else None
    return uv_cfg if isinstance(uv_cfg, dict) else None


def _lock_packages(lock: dict) -> dict[str, dict]:
    """Normalized name -> ``[[package]]`` table for every locked package."""
    entries = lock.get("package")
    assert isinstance(entries, list), "uv.lock must declare [[package]] entries"
    packages: dict[str, dict] = {}
    for entry in entries:
        assert isinstance(entry, dict) and isinstance(entry.get("name"), str), (
            f"malformed [[package]] entry in uv.lock: {entry!r}"
        )
        packages.setdefault(_normalize(entry["name"]), entry)
    return packages


def _declared_index_names(index: object) -> set[str]:
    """Names of indexes declared under ``[tool.uv] index`` (list or mapping)."""
    if isinstance(index, list):
        return {
            entry["name"]
            for entry in index
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        }
    if isinstance(index, dict):
        return {key for key in index if isinstance(key, str)}
    return set()


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Checkout under review: ``$PROJECT_ROOT`` or the nearest ancestor
    containing both ``pyproject.toml`` and ``uv.lock``."""
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        for filename in ("pyproject.toml", "uv.lock"):
            assert (root / filename).is_file(), f"{root} is missing {filename}"
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "uv.lock").is_file():
            return parent
    pytest.fail(
        "no checkout found: set PROJECT_ROOT to a directory containing "
        "pyproject.toml and uv.lock"
    )


@pytest.fixture(scope="session")
def pyproject(project_root: Path) -> dict:
    return _load_toml(project_root / "pyproject.toml")


@pytest.fixture(scope="session")
def lock(project_root: Path) -> dict:
    return _load_toml(project_root / "uv.lock")


def test_clip_dependency_pins_git_source(pyproject: dict, lock: dict) -> None:
    """clip is an active dependency pinned to the fixture Git URL and commit,
    and uv.lock locks it from that same Git source."""
    dependencies = pyproject.get("project", {}).get("dependencies") or []
    clip_dep = next(
        (d for d in dependencies if isinstance(d, str) and _dep_name(d) == CLIP_NAME),
        None,
    )
    assert clip_dep is not None, (
        "[project] dependencies must add the required detector dependency 'clip'"
    )

    pinned: tuple[str | None, str | None] | None = None
    uv_cfg = _uv_table(pyproject)
    sources = uv_cfg.get("sources") if uv_cfg else None
    if isinstance(sources, dict):
        key = next((k for k in sources if _normalize(k) == CLIP_NAME), None)
        if key is not None:
            spec = sources[key]
            if isinstance(spec, dict):
                pinned = (spec.get("git"), spec.get("rev"))
            elif isinstance(spec, str):
                pinned = _parse_git_ref(spec)
            else:
                pytest.fail(
                    f"[tool.uv.sources] {key!r} must declare a git url and rev: {spec!r}"
                )
    if pinned is None and "@" in clip_dep:
        pinned = _parse_git_ref(clip_dep.partition("@")[2])
    assert pinned == (CLIP_URL, CLIP_REV), (
        f"clip must be pinned to {CLIP_URL} at {CLIP_REV}; "
        f"got {pinned!r} for dependency {clip_dep!r}"
    )

    package = _lock_packages(lock).get(CLIP_NAME)
    assert package is not None, "uv.lock must lock a 'clip' package"
    source = package.get("source")
    locked = str(source.get("git")) if isinstance(source, dict) else ""
    assert locked, f"locked clip must come from a Git source, got {source!r}"
    assert CLIP_URL in locked and CLIP_REV in locked, (
        f"locked clip source {locked!r} must pin {CLIP_URL} at {CLIP_REV}"
    )


def test_explicit_checkout_mode(pyproject: dict) -> None:
    """``[tool.uv] package`` is explicitly false (checkout execution, legacy
    setup.py distribution identity)."""
    uv_cfg = _uv_table(pyproject)
    assert uv_cfg is not None, "pyproject.toml must define a [tool.uv] table"
    assert "package" in uv_cfg, (
        "[tool.uv] package must be set explicitly, not left to the default"
    )
    assert uv_cfg["package"] is False, (
        f"[tool.uv] package must be the boolean false, got {uv_cfg['package']!r}"
    )


def test_locked_torch_native_sources_and_index_preserved(
    pyproject: dict, lock: dict
) -> None:
    """Locked torch keeps the frozen versions and cu130 registry, native
    packages keep their local-directory lock sources, and the named torch
    index survives in pyproject."""
    packages = _lock_packages(lock)

    for name, version in TORCH_VERSIONS.items():
        package = packages.get(_normalize(name))
        assert package is not None, f"uv.lock must lock {name!r}"
        assert package.get("version") == version, (
            f"{name} locked at {package.get('version')!r}; the contract "
            f"preserves {version!r} (no unnecessary upgrade)"
        )
        source = package.get("source")
        registry = str(source.get("registry", "")).rstrip("/") if isinstance(source, dict) else ""
        assert registry == TORCH_REGISTRY, (
            f"{name} must lock from {TORCH_REGISTRY!r}, got {source!r}"
        )

    for name, directory in NATIVE_SOURCES.items():
        package = packages.get(_normalize(name))
        assert package is not None, f"uv.lock must lock the native source {name!r}"
        source = package.get("source")
        assert isinstance(source, dict) and source.get("directory") is not None, (
            f"{name} must keep its local-directory lock source, got {source!r}"
        )
        assert _norm_path(source["directory"]) == _norm_path(directory), (
            f"{name} lock source {source['directory']!r} != expected {directory!r}"
        )

    uv_cfg = _uv_table(pyproject)
    assert uv_cfg is not None, "pyproject.toml must define a [tool.uv] table"
    declared = _declared_index_names(uv_cfg.get("index"))
    assert TORCH_INDEX in declared, (
        f"pyproject.toml must preserve the {TORCH_INDEX!r} index; "
        f"declared: {sorted(declared)}"
    )


def test_no_openai_in_active_project_graph(pyproject: dict, lock: dict) -> None:
    """openai is absent from every active declaration and the locked graph."""
    requirements: list = []
    requirements.extend(pyproject.get("project", {}).get("dependencies") or [])
    groups = pyproject.get("dependency-groups")
    if isinstance(groups, dict):
        for group in groups.values():
            if isinstance(group, list):
                requirements.extend(group)
    uv_cfg = _uv_table(pyproject)
    sources: dict = {}
    if uv_cfg is not None:
        requirements.extend(uv_cfg.get("dependencies") or [])
        uv_sources = uv_cfg.get("sources")
        if isinstance(uv_sources, dict):
            sources.update(uv_sources)

    offenders = [
        req for req in requirements
        if isinstance(req, str) and _dep_name(req) == OPENAI_NAME
    ]
    offenders.extend(k for k in sources if _normalize(k) == OPENAI_NAME)
    assert not offenders, (
        f"openai must not be in the active project graph; found: {offenders}"
    )
    assert OPENAI_NAME not in _lock_packages(lock), (
        "uv.lock must not contain an openai package"
    )


def test_python_minimum_remains_312(pyproject: dict) -> None:
    """requires-python keeps 3.12 as the current supported minimum."""
    requires = pyproject.get("project", {}).get("requires-python")
    assert isinstance(requires, str) and requires.strip(), (
        "[project] requires-python must be declared"
    )
    match = re.search(r"(>=|~=)\s*(\d+(?:\.\d+)+)", requires.replace(" ", ""))
    assert match is not None, (
        f"requires-python {requires!r} must declare an explicit lower bound"
    )
    assert (match.group(1), match.group(2)) in ((">=", "3.12"), ("~=", "3.12")), (
        f"requires-python {requires!r} must keep 3.12 as the minimum "
        f"(>=3.12 or ~=3.12)"
    )
