#!/usr/bin/env python
"""Verify that every package in the project's uv environment works as intended.

Run with the project venv:

    .venv/bin/python scripts/verify_env.py

The script exercises the CPU *and* the GPU (RTX 5090 / sm_120) code paths of
every installed package and prints a PASS/FAIL report. It exits with status 1
if any check fails.
"""

from __future__ import annotations

import importlib
import platform
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

RESULTS: list[tuple[str, str, bool, str]] = []


@dataclass
class Section:
    name: str
    checks: list[tuple[str, bool, str]] = field(default_factory=list)


def check(name: str, fn) -> None:
    """Run a check function, recording pass/failure without aborting."""
    try:
        detail = fn()
        RESULTS.append((name, "", True, detail or ""))
    except Exception as exc:  # noqa: BLE001 - we want to report every failure
        RESULTS.append((name, "", False, f"{type(exc).__name__}: {exc}"))
        if "-v" in sys.argv:
            traceback.print_exc()


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def report() -> int:
    print("\n" + "=" * 78)
    failed = [r for r in RESULTS if not r[2]]
    current = None
    for name, _, ok, detail in RESULTS:
        group = name.split("::")[0]
        if group != current:
            current = group
            print(f"\n[{group}]")
        label = name.split("::", 1)[1] if "::" in name else name
        status = "PASS" if ok else "FAIL"
        line = f"  {status}  {label}"
        if detail:
            line += f"  ({detail})"
        print(line)
    print("\n" + "=" * 78)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("\nFailures:")
        for name, _, _, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    print("All checks passed.")
    return 0


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def _site_packages_integrity():
    """Every file listed in an installed package's RECORD must exist.

    uv removes files by path, so two wheels that install to the same path
    (e.g. the nvidia-*-cu12 and nvidia-*-cu13 builds) can delete each
    other's files when one of them is pruned.
    """
    import csv
    import site

    sp = Path(site.getsitepackages()[0])
    broken = {}
    for rec in sp.glob("*.dist-info/RECORD"):
        missing = sum(
            1 for row in csv.reader(rec.open()) if row and not (sp / row[0]).exists()
        )
        if missing:
            broken[rec.parent.name.removesuffix(".dist-info")] = missing
    assert not broken, f"missing files in installed packages: {broken}"
    return "all installed packages intact"


section("environment")
check("env::python", lambda: f"{platform.python_version()} ({sys.executable})")
check("env::platform", lambda: platform.platform())
check("env::site-packages integrity", _site_packages_integrity)


# --------------------------------------------------------------------------
# torch
# --------------------------------------------------------------------------
def _torch_basic():
    import torch

    x = torch.randn(256, 256)
    y = (x @ x).sum().item()
    assert isinstance(y, float)
    return f"torch {torch.__version__}, cuda build {torch.version.cuda}"


def _torch_cuda():
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    assert cap == (12, 0), f"unexpected compute capability {cap}"
    x = torch.randn(1024, 1024, device="cuda")
    out = (x @ x).sum().item()
    torch.cuda.synchronize()
    return f"{dev} sm_{cap[0]}{cap[1]}, matmul ok"


def _torch_autograd():
    import torch

    x = torch.randn(512, 512, device="cuda", requires_grad=True)
    y = (torch.sin(x) @ x).sum()
    y.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    torch.cuda.synchronize()
    return "backward pass on GPU ok"


def _torch_conv():
    import torch

    c = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    x = torch.randn(8, 3, 64, 64, device="cuda")
    y = c(x)
    assert y.shape == (8, 16, 64, 64)
    torch.cuda.synchronize()
    return f"conv2d output {tuple(y.shape)}"


def _torch_amp():
    import torch

    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = torch.randn(512, 512, device="cuda")
        y = (x @ x).float().sum()
    torch.cuda.synchronize()
    return f"bf16 autocast ok (dtype {y.dtype})"


def _torch_half():
    import torch

    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = (a @ a).sum().float().item()
    assert torch.isfinite(torch.tensor(b))
    torch.cuda.synchronize()
    return "fp16 matmul ok"


def _torch_dtypes():
    import torch

    t = torch.tensor([1.5, 2.5], device="cuda", dtype=torch.bfloat16)
    assert t.dtype == torch.bfloat16
    return "bfloat16 tensor ok"


def _torch_mem():
    import torch

    free, total = torch.cuda.mem_get_info(0)
    return f"GPU memory free {free / 2**30:.1f} GiB / total {total / 2**30:.1f} GiB"


def _torch_stream():
    import torch

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x = torch.randn(1024, 1024, device="cuda")
        (x @ x).sum()
    torch.cuda.synchronize()
    return "non-default stream ok"


def _torch_cudnn():
    import torch

    torch.backends.cudnn.benchmark = True
    c = torch.nn.Conv2d(8, 8, 3, padding=1).cuda()
    x = torch.randn(4, 8, 32, 32, device="cuda")
    c(x)
    torch.cuda.synchronize()
    return f"cudnn {torch.backends.cudnn.version()}"


def _torch_multi_gpu():
    import torch

    n = torch.cuda.device_count()
    return f"{n} CUDA device(s) visible"


def _torch_numpy_interop():
    import numpy as np
    import torch

    a = np.random.rand(64, 64).astype("float32")
    t = torch.from_numpy(a).cuda()
    assert np.allclose(t.cpu().numpy(), a)
    return "numpy<->torch cuda interop ok"


check("torch::import + cpu ops", _torch_basic)
check("torch::cuda available + matmul", _torch_cuda)
check("torch::autograd on GPU", _torch_autograd)
check("torch::conv2d on GPU", _torch_conv)
check("torch::cudnn conv", _torch_cudnn)
check("torch::autocast bf16 on GPU", _torch_amp)
check("torch::fp16 matmul on GPU", _torch_half)
check("torch::bfloat16 dtype", _torch_dtypes)
check("torch::non-default stream", _torch_stream)
check("torch::device count", _torch_multi_gpu)
check("torch::GPU memory info", _torch_mem)
check("torch::numpy interop", _torch_numpy_interop)


# --------------------------------------------------------------------------
# torchvision
# --------------------------------------------------------------------------
def _torchvision():
    import torchvision

    return f"torchvision {torchvision.__version__}"


def _torchvision_model():
    import torch
    from torchvision.models import resnet18

    m = resnet18(weights=None).cuda().eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 224, 224, device="cuda"))
    assert y.shape == (2, 1000)
    torch.cuda.synchronize()
    return "resnet18 forward on GPU ok"


def _torchvision_ops():
    import torch
    from torchvision.ops import nms, box_iou

    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]],
        device="cuda",
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
    keep = nms(boxes, scores, 0.5)
    assert keep.numel() == 2
    iou = box_iou(boxes, boxes)
    torch.cuda.synchronize()
    return f"nms/box_iou on GPU ok (keep={keep.tolist()})"


def _torchvision_transforms():
    from PIL import Image
    from torchvision import transforms

    img = Image.new("RGB", (32, 32), (120, 30, 200))
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    x = t(img)
    assert x.shape == (3, 32, 32)
    return "PIL->tensor transforms ok"


check("torchvision::import", _torchvision)
check("torchvision::resnet18 on GPU", _torchvision_model)
check("torchvision::nms/box_iou on GPU", _torchvision_ops)
check("torchvision::transforms with PIL", _torchvision_transforms)


# --------------------------------------------------------------------------
# torchaudio
# --------------------------------------------------------------------------
def _torchaudio():
    import torchaudio

    return f"torchaudio {torchaudio.__version__}"


def _torchaudio_signal():
    import torch
    import torchaudio

    wav = torch.sin(torch.arange(16000, dtype=torch.float32) * 0.05).unsqueeze(0)
    spec = torchaudio.transforms.Spectrogram(n_fft=400)(wav)
    assert spec.shape[-1] > 0
    return f"spectrogram {tuple(spec.shape)}"


def _torchaudio_gpu():
    import torch
    import torchaudio

    wav = torch.randn(1, 16000, device="cuda")
    resampled = torchaudio.functional.resample(wav, 16000, 8000)
    assert resampled.shape[-1] == 8000
    torch.cuda.synchronize()
    return f"resample on GPU ok {tuple(resampled.shape)}"


def _torchaudio_backends():
    import torch
    import torchaudio

    # torchaudio >= 2.9 routes file I/O through torchcodec
    path = "/tmp/opencode/_verify_torchaudio.wav"
    wav = torch.sin(torch.arange(8000, dtype=torch.float32) * 0.05).unsqueeze(0)
    torchaudio.save(path, wav, 16000)
    loaded, sr = torchaudio.load(path)
    assert sr == 16000, f"unexpected sample rate {sr}"
    assert loaded.shape[-1] == 8000, f"unexpected length {loaded.shape}"
    return f"io roundtrip ok ({loaded.shape[0]} ch @ {sr} Hz)"


def _torchcodec():
    import torch
    from torchcodec.decoders import AudioDecoder

    path = "/tmp/opencode/_verify_torchaudio.wav"
    wav = torch.sin(torch.arange(8000, dtype=torch.float32) * 0.05).unsqueeze(0)
    import torchaudio

    torchaudio.save(path, wav, 16000)
    samples = AudioDecoder(path, sample_rate=16000).get_all_samples()
    assert samples.data.shape[-1] == 8000, f"unexpected length {samples.data.shape}"
    return f"torchcodec decoder ok ({samples.data.shape})"


check("torchaudio::import", _torchaudio)
check("torchaudio::spectrogram (cpu)", _torchaudio_signal)
check("torchaudio::resample on GPU", _torchaudio_gpu)
check("torchaudio::file io (torchcodec)", _torchaudio_backends)
check("torchcodec::audio decoder", _torchcodec)


# --------------------------------------------------------------------------
# triton
# --------------------------------------------------------------------------
def _triton():
    import triton

    return f"triton {triton.__version__}"


def _triton_kernel():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 4096
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    torch.cuda.synchronize()
    assert torch.allclose(out, x + y, atol=1e-5)
    return "triton kernel compiled and ran on GPU"


check("triton::import", _triton)
check("triton::kernel on GPU", _triton_kernel)


# --------------------------------------------------------------------------
# pytorch3d
# --------------------------------------------------------------------------
def _p3d_import():
    import pytorch3d
    from pytorch3d import _C

    return f"pytorch3d {pytorch3d.__version__}, ext {_C.__file__.split('/')[-1]}"


def _p3d_knn_cpu():
    import torch
    from pytorch3d.ops import knn_points

    p = torch.rand(1, 500, 3)
    q = torch.rand(1, 50, 3)
    r = knn_points(q, p, K=4)
    assert r.idx.shape == (1, 50, 4)
    return "knn_points CPU ok"


def _p3d_knn_gpu():
    import torch
    from pytorch3d.ops import knn_points

    p = torch.rand(1, 20000, 3, device="cuda")
    q = torch.rand(1, 256, 3, device="cuda")
    r = knn_points(q, p, K=16)
    assert r.idx.shape == (1, 256, 16)
    torch.cuda.synchronize()
    return "knn_points GPU ok (sm_120 kernels)"


def _p3d_sample_gpu():
    import torch
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import ico_sphere

    m = ico_sphere(level=3)
    mesh = Meshes(verts=[m.verts_packed()], faces=[m.faces_packed()]).to("cuda")
    s = sample_points_from_meshes(mesh, 8192)
    assert s.shape == (1, 8192, 3)
    torch.cuda.synchronize()
    return "sample_points_from_meshes GPU ok"


def _p3d_chamfer_gpu():
    import torch
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import ico_sphere

    m = ico_sphere(level=2)
    mesh = Meshes(verts=[m.verts_packed()], faces=[m.faces_packed()]).to("cuda")
    x = sample_points_from_meshes(mesh, 1024)
    loss, _ = chamfer_distance(x, x)
    assert loss.item() < 1e-6, f"chamfer distance of a point cloud to itself is {loss.item()}"
    torch.cuda.synchronize()
    return f"chamfer_distance GPU ok (loss {loss.item():.2e})"


def _p3d_rasterizer_gpu():
    import torch
    from pytorch3d.renderer import (
        FoVPerspectiveCameras,
        MeshRasterizer,
        RasterizationSettings,
    )
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import ico_sphere

    m = ico_sphere(level=2)
    mesh = Meshes(verts=[m.verts_packed()], faces=[m.faces_packed()]).to("cuda")
    cameras = FoVPerspectiveCameras(fov=60.0, device="cuda")
    settings = RasterizationSettings(image_size=64, blur_radius=0.0, faces_per_pixel=1)
    frag = MeshRasterizer(cameras=cameras, raster_settings=settings)(mesh)
    assert frag.pix_to_face.shape[0] == 1
    torch.cuda.synchronize()
    return f"MeshRasterizer GPU ok (pix_to_face {tuple(frag.pix_to_face.shape)})"


def _p3d_ball_query_gpu():
    import torch
    from pytorch3d.ops import ball_query, knn_gather

    p = torch.rand(1, 5000, 3, device="cuda")
    q = torch.rand(1, 128, 3, device="cuda")
    out = ball_query(q, p, radius=0.2, K=8)
    assert out.idx.shape == (1, 128, 8)
    gathered = knn_gather(p, out.idx)
    assert gathered.shape == (1, 128, 8, 3)
    torch.cuda.synchronize()
    return "ball_query / knn_gather GPU ok"


check("pytorch3d::import + extension", _p3d_import)
check("pytorch3d::knn_points (cpu)", _p3d_knn_cpu)
check("pytorch3d::knn_points (GPU)", _p3d_knn_gpu)
check("pytorch3d::sample_points_from_meshes (GPU)", _p3d_sample_gpu)
check("pytorch3d::chamfer_distance (GPU)", _p3d_chamfer_gpu)
check("pytorch3d::MeshRasterizer (GPU)", _p3d_rasterizer_gpu)
check("pytorch3d::ball_query/knn_gather (GPU)", _p3d_ball_query_gpu)


# --------------------------------------------------------------------------
# faiss
# --------------------------------------------------------------------------
def _faiss_import():
    import faiss

    return f"faiss {faiss.__version__}, {faiss.get_num_gpus()} GPU(s)"


def _faiss_cpu():
    import faiss
    import numpy as np

    d = 32
    xb = np.random.rand(2000, d).astype("float32")
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    D, I = index.search(xb[:5], 3)
    assert (I[:, 0] == np.arange(5)).all(), "self-search mismatch"
    return "IndexFlatL2 (CPU) ok"


def _faiss_gpu_flat():
    import faiss
    import numpy as np

    d = 64
    xb = np.random.rand(5000, d).astype("float32")
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, d)
    index.add(xb)
    D, I = index.search(xb[:5], 3)
    assert (I[:, 0] == np.arange(5)).all(), "self-search mismatch"
    return "GpuIndexFlatL2 search on GPU ok"


def _faiss_gpu_ip():
    import faiss
    import numpy as np

    d = 128
    xb = np.random.rand(3000, d).astype("float32")
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, d)
    index.add(xb)
    D, I = index.search(xb[:3], 5)
    assert I.shape == (3, 5)
    return "GpuIndexFlatIP search on GPU ok"


def _faiss_gpu_ivf():
    import faiss
    import numpy as np

    d = 64
    nt, nb = 4000, 2000
    xt = np.random.rand(nt, d).astype("float32")
    xb = np.random.rand(nb, d).astype("float32")
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexIVFFlat(res, d, 32)
    index.train(xt)
    index.add(xb)
    index.nprobe = 4
    D, I = index.search(xb[:5], 3)
    assert I.shape == (5, 3)
    return "GpuIndexIVFFlat train+search on GPU ok"


def _faiss_gpu_torch_bridge():
    import faiss
    import numpy as np
    import torch

    d = 32
    xb = np.random.rand(1000, d).astype("float32")
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, d)
    index.add(xb)
    q = np.random.rand(4, d).astype("float32")
    D, I = index.search(q, 2)
    # same query through torch, to check the numpy/torch boundary
    qt = torch.from_numpy(q)
    assert qt.shape == (4, d)
    return "faiss GPU <-> torch interop ok"


check("faiss::import + gpu count", _faiss_import)
check("faiss::IndexFlatL2 (CPU)", _faiss_cpu)
check("faiss::GpuIndexFlatL2 (GPU)", _faiss_gpu_flat)
check("faiss::GpuIndexFlatIP (GPU)", _faiss_gpu_ip)
check("faiss::GpuIndexIVFFlat (GPU)", _faiss_gpu_ivf)
check("faiss::GPU <-> torch interop", _faiss_gpu_torch_bridge)


# --------------------------------------------------------------------------
# numpy and the rest of the scientific stack
# --------------------------------------------------------------------------
def _numpy():
    import numpy as np

    a = np.linspace(0, 1, 1000)
    assert np.isclose(np.trapezoid(a, a), 0.5)
    return f"numpy {np.__version__}"


def _pillow():
    from PIL import Image

    img = Image.new("RGB", (16, 16), (10, 20, 30))
    assert img.convert("L").size == (16, 16)
    return "pillow image ops ok"


def _sympy():
    import sympy

    x = sympy.symbols("x")
    assert sympy.solve(x**2 - 4, x) == [-2, 2]
    return f"sympy {sympy.__version__}"


def _networkx():
    import networkx as nx

    g = nx.path_graph(5)
    assert nx.shortest_path_length(g, 0, 4) == 4
    return f"networkx {nx.__version__}"


def _jinja2():
    from jinja2 import Template

    assert Template("{{ a + 1 }}").render(a=41) == "42"
    return "jinja2 templating ok"


def _fsspec():
    import fsspec

    with fsspec.open("memory://sample.txt", "w") as f:
        f.write("hello")
    with fsspec.open("memory://sample.txt", "r") as f:
        assert f.read() == "hello"
    return "fsspec memory filesystem ok"


def _iopath():
    from iopath.common.file_io import PathManager

    pm = PathManager()
    assert pm is not None
    return "iopath PathManager ok"


def _tqdm():
    from tqdm import tqdm

    total = sum(1 for _ in tqdm(range(10), disable=True))
    assert total == 10
    return "tqdm ok"


def _filelock():
    from filelock import FileLock

    with FileLock("/tmp/opencode/.verify_filelock.lock"):
        pass
    return "filelock ok"


def _portalocker():
    import portalocker

    with portalocker.Lock("/tmp/opencode/.verify_portalocker.lock", mode="w"):
        pass
    return "portalocker ok"


def _packaging():
    from packaging.version import Version

    assert Version("2.14.1") < Version("2.15.0")
    return "packaging version comparison ok"


def _markupsafe():
    from markupsafe import escape

    assert escape("<b>") == "&lt;b&gt;"
    return "markupsafe ok"


def _mpmath():
    import mpmath

    assert abs(mpmath.pi - 3.14159265) < 1e-6
    return "mpmath ok"


def _typing_extensions():
    from typing_extensions import TypedDict

    class _T(TypedDict):
        a: int

    return "typing_extensions ok"


def _setuptools():
    import setuptools

    return f"setuptools {setuptools.__version__}"


def _pip():
    import pip

    return f"pip {pip.__version__}"


def _wheel():
    import wheel

    return f"wheel {wheel.__version__}"


def _cuda_toolkit():
    import importlib.metadata as md

    try:
        version = md.version("cuda-toolkit")
    except md.PackageNotFoundError:
        version = "n/a"
    return f"cuda-toolkit {version}"


check("numpy::import + ops", _numpy)
check("pillow::import + ops", _pillow)
check("sympy::import + solve", _sympy)
check("networkx::import + ops", _networkx)
check("jinja2::import + render", _jinja2)
check("fsspec::memory fs", _fsspec)
check("iopath::PathManager", _iopath)
check("tqdm::import", _tqdm)
check("filelock::import + lock", _filelock)
check("portalocker::import + lock", _portalocker)
check("packaging::Version", _packaging)
check("markupsafe::escape", _markupsafe)
check("mpmath::import", _mpmath)
check("typing_extensions::TypedDict", _typing_extensions)
check("setuptools::import", _setuptools)
check("pip::import", _pip)
check("wheel::import", _wheel)
check("cuda-toolkit::dist-info", _cuda_toolkit)


# --------------------------------------------------------------------------
# CUDA runtime bindings
# --------------------------------------------------------------------------
def _cuda_bindings():
    try:
        from cuda.bindings import driver
    except ImportError:  # older layout
        from cuda import cuda as driver  # type: ignore

    assert driver.cuInit(0)[0] == driver.CUresult.CUDA_SUCCESS
    n = driver.cuDeviceGetCount()[1]
    return f"cuda-bindings: {n} device(s) via driver API"


def _cuda_pathfinder():
    import cuda.pathfinder as pathfinder

    return f"cuda-pathfinder {getattr(pathfinder, '__version__', 'ok')}"


def _nvidia_runtime_pkgs():
    from pathlib import Path

    import nvidia

    root = Path(list(nvidia.__path__)[0])
    libdir = root / "cu13" / "lib"
    required = {
        "libcudart.so.13": "nvidia-cuda-runtime",
        "libcublas.so.13": "nvidia-cublas",
        "libcublasLt.so.13": "nvidia-cublas",
        "libnvJitLink.so.13": "nvidia-nvjitlink",
        "libcurand.so.10": "nvidia-curand",
        "libcusparse.so.12": "nvidia-cusparse",
        "libcusolver.so.12": "nvidia-cusolver",
        "libcufft.so.12": "nvidia-cufft",
        "libnvrtc.so.13": "nvidia-cuda-nvrtc",
        "libcupti.so.13": "nvidia-cuda-cupti",
    }
    missing = [f for f in required if not (libdir / f).exists()]
    assert not missing, f"missing CUDA 13 runtime libs in {libdir}: {missing}"
    return f"{len(required)} CUDA 13 runtime libs present in {libdir}"


def _nvidia_optional_pkgs():
    from pathlib import Path

    import nvidia

    root = Path(list(nvidia.__path__)[0])
    found = []
    for libdir in sorted(root.glob("*/lib")):
        found += [p.name for p in libdir.glob("*.so*") if "cudnn" in p.name or "nccl" in p.name]
    assert any("libcudnn.so" in f for f in found), f"no cudnn library found under {root}"
    assert any("libnccl.so" in f for f in found), f"no nccl library found under {root}"
    return f"cudnn/nccl libs present: {', '.join(sorted(set(found)))}"


def _torch_arch_list():
    import torch

    archs = torch.cuda.get_arch_list()
    assert "sm_120" in archs, f"sm_120 missing from torch arch list: {archs}"
    return f"torch arch list: {', '.join(archs)}"


def _sm120_cubins(lib, label):
    import subprocess

    out = subprocess.run(
        ["cuobjdump", "--list-elf", str(lib)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    arches = sorted({ln.split(".sm_")[1].split(".")[0] for ln in out.splitlines() if ".sm_" in ln})
    assert any(a == "120" for a in arches), f"{label} has no sm_120 cubins: {arches}"
    n = sum(1 for ln in out.splitlines() if ".sm_120." in ln)
    return f"{label}: {n} sm_120 cubins"


def _p3d_sm120():
    import pytorch3d
    from pytorch3d import _C

    return _sm120_cubins(_C.__file__, "pytorch3d _C")


def _faiss_sm120():
    import faiss

    lib = Path(faiss.__file__).parent / "libfaiss.so"
    return _sm120_cubins(lib, "faiss libfaiss.so")


def _nccl():
    import torch

    return f"nccl available: {torch.distributed.is_nccl_available()}"


check("cuda-bindings::driver API", _cuda_bindings)
check("cuda-pathfinder::import", _cuda_pathfinder)
check("nvidia-*-cu13::runtime libs", _nvidia_runtime_pkgs)
check("nvidia-*-cu13::optional libs", _nvidia_optional_pkgs)
check("torch::nccl available", _nccl)
check("torch::sm_120 in arch list", _torch_arch_list)
check("pytorch3d::sm_120 cubins", _p3d_sm120)
check("faiss::sm_120 cubins", _faiss_sm120)


if __name__ == "__main__":
    sys.exit(report())
