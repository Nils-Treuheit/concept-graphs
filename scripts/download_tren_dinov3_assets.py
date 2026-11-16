#!/usr/bin/env python3
"""Prepare the DINOv3 and tokenizer assets required by T-REN.

Purpose
-------
T-REN expects two checkpoint files in the experiment save directory:

* ``dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`` (ViT-L/16 backbone)
* ``dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth``
  (DINOtxt vision head plus text encoder)
* ``bpe_simple_vocab_16e6.txt.gz`` (DINOv3 text tokenizer vocabulary)

Two preparation modes are supported.

Hugging Face mode (default)
    The backbone is converted from a local Hugging Face snapshot of
    ``facebook/dinov3-vitl16-pretrain-lvd1689m``.  The conversion is verified
    against a tensor fingerprint of the official Meta state dict, so the
    produced ``.pth`` is functionally identical even though the container
    bytes differ.  The DINOtxt head is downloaded from a public Hugging Face
    mirror and verified with the official full SHA-256.

Meta mode
    Both original files are downloaded from the official Meta host using the
    time-limited URLs from the DINOv3 access email.  This is the original
    behaviour of this script.

The script never automates license acceptance, login, cookies, or any web
form.  It only reads local Hugging Face caches, downloads public Hugging Face
files, or downloads from the official Meta host with URLs supplied by the
user.

Usage
-----
Hugging Face mode, using the snapshot already present in ``$HF_HOME``::

    python3 scripts/download_tren_dinov3_assets.py \\
        --output-dir <path>

Convert a specific snapshot directory or ``model.safetensors`` file::

    python3 scripts/download_tren_dinov3_assets.py \\
        --hf-backbone-snapshot <path> \\
        --output-dir <path>

Download the original ``.pth`` files from the verified Hugging Face mirror
instead of converting locally::

    python3 scripts/download_tren_dinov3_assets.py \\
        --backbone-mode download \\
        --output-dir <path>

Meta mode, using the official email URLs::

    python3 scripts/download_tren_dinov3_assets.py \\
        --source meta \\
        --url-file <path-to-url-file> \\
        --output-dir <path>

The ``--output-dir`` argument is required unless ``$HF_HOME`` is set, in which
case the default output directory is ``$HF_HOME/t_ren-dinov3-assets``.

Keep the Meta URL-list file private.  Do not commit it to Git or share it.
The URLs are time-limited and tied to your Meta DINOv3 access email.
"""

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import shutil
import ssl
import sys
import textwrap
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKBONE_FILENAME = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
HEAD_FILENAME = "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
BPE_URL = "https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz"
BPE_SIZE = 1_356_917
BPE_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"

# Full SHA-256 of the official Meta assets.  The 8-hex suffixes in the
# filenames are the prefixes of these hashes.
BACKBONE_SHA256 = "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
HEAD_SHA256 = "a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0"

# Canonical fingerprint of the official ViT-L/16 LVD1689M state dict.
#
# A converted .pth cannot match the official file's SHA-256 because the zip
# container and pickle metadata differ.  Instead, this fingerprint covers the
# semantic payload: for every key in sorted order, the key name, tensor shape,
# tensor dtype, and raw tensor bytes are hashed.  It is computed from the
# official ``dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`` and therefore
# proves that a conversion carries exactly the official weights and buffers.
BACKBONE_FINGERPRINT = (
    "580d3cf3cc64bac71d45f0e3ab68ba1eb8b260777dd7cb500b11373e455e1365"
)

# Hugging Face locations.
HF_BACKBONE_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
HF_BACKBONE_FILE = "model.safetensors"

# Public mirrors that host the official Meta .pth assets.  Every downloaded
# file is checked against the full official SHA-256 above, so a mirror cannot
# substitute different weights without the check failing.
HF_ASSET_REPO = "PIA-SPACE-LAB/dinov3_vitl16_dinotxt_vision_head_and_text_encoder"
HF_ASSET_HEAD_FILE = HEAD_FILENAME
HF_ASSET_BACKBONE_FILE = BACKBONE_FILENAME

# Alternative mirror, useful with ``--hf-asset-repo`` and ``--hf-head-file``:
#   backseollgi/dinov3_vitl16_dinotxt_vision_head_and_text_encoder
#   dinov3_vitl16_dinotxt_vision_head_and_text_encoder.pth

# Expected ViT-L/16 architecture.  Checked before conversion so that a
# ViT-B, ViT-S, ViT-H+, or SAT493M snapshot is rejected with a clear message.
EXPECTED_EMBED_DIM = 1024
EXPECTED_DEPTH = 24
EXPECTED_HEADS = 16
EXPECTED_PATCH_SIZE = 16
EXPECTED_REGISTER_TOKENS = 4

OFFICIAL_HOST = "dl.fbaipublicfiles.com"
PART_SUFFIX = ".part"

# Partial and final file permission (owner read/write only)
_PART_FILE_MODE = 0o600

# Minimum expected size for a valid .pth asset (heuristic safety gate)
_MIN_PTH_SIZE = 1_000_000  # 1 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(msg, exit_code=1):
    sys.stderr.write(f"Error: {msg}\n")
    sys.exit(exit_code)


def _find_git_root():
    """Return the Git worktree root, or None if not inside a repo.

    Starts from the script's own location so the check is stable
    regardless of the caller's working directory.
    """
    script_dir = pathlib.Path(__file__).resolve()
    cwd = script_dir
    while True:
        if (cwd / ".git").exists():
            return cwd
        parent = cwd.parent
        if parent == cwd:
            return None
        cwd = parent


def _is_inside_git_worktree(path, git_root):
    if git_root is None:
        return False
    try:
        pathlib.Path(path).resolve().relative_to(git_root.resolve())
        return True
    except ValueError:
        return False


def _sha256_file(filepath):
    """Return the full SHA-256 hex digest of *filepath*."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _human_size(nbytes):
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _torch_load_state_dict(filepath):
    """Load a torch state dict with weights_only=True.

    Raises on an invalid or unsafe file.  The caller converts failures into
    clear user-facing messages.
    """
    import torch  # Lazy import so Meta-only downloads need no torch.

    try:
        obj = torch.load(filepath, map_location="cpu", weights_only=True)
    except TypeError:
        # Very old torch versions do not accept weights_only.
        obj = torch.load(filepath, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError("checkpoint does not contain a state dict")
    return obj


# ---------------------------------------------------------------------------
# Hugging Face snapshot discovery and conversion
# ---------------------------------------------------------------------------

def _require_hf_dependencies():
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import safetensors  # noqa: F401
    except ImportError:
        missing.append("safetensors")
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        missing.append("huggingface_hub")
    if missing:
        _error(
            "Hugging Face mode requires the following Python packages: "
            + ", ".join(missing)
            + ". Install them with: pip install torch safetensors huggingface_hub"
        )


def _default_hub_cache():
    cache = os.environ.get("HF_HUB_CACHE")
    if cache:
        return pathlib.Path(cache).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return pathlib.Path(hf_home).expanduser() / "hub"
    return pathlib.Path.home() / ".cache" / "huggingface" / "hub"


def _local_snapshot_candidates(repo_id):
    """Return local snapshot directories for *repo_id*, newest first."""
    cache_root = _default_hub_cache()
    repo_dir = cache_root / ("models--" + repo_id.replace("/", "--"))
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return []
    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def _resolve_local_backbone(explicit_path, repo_id):
    """Locate a local ``model.safetensors`` for the backbone.

    Returns ``(model_path, config_path_or_none)``.  ``explicit_path`` may be a
    snapshot directory or a direct path to ``model.safetensors``.
    """
    if explicit_path:
        p = pathlib.Path(explicit_path).expanduser()
        if p.is_dir():
            model_path = p / HF_BACKBONE_FILE
            config_path = p / "config.json"
        else:
            model_path = p
            config_path = p.parent / "config.json"
        if not model_path.is_file():
            _error(
                f"--hf-backbone-snapshot does not contain {HF_BACKBONE_FILE}: "
                f"{model_path}"
            )
        return model_path, config_path if config_path.is_file() else None

    # Prefer huggingface_hub, which knows about revision and cache settings.
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            snapshot_dir = pathlib.Path(
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[HF_BACKBONE_FILE],
                    local_files_only=True,
                )
            )
            model_path = snapshot_dir / HF_BACKBONE_FILE
            if model_path.is_file():
                config_path = snapshot_dir / "config.json"
                return model_path, config_path if config_path.is_file() else None
        except LocalEntryNotFoundError:
            pass
        except Exception:  # noqa: BLE001, S110 - fall back to manual discovery
            pass
    except ImportError:
        pass

    for snapshot_dir in _local_snapshot_candidates(repo_id):
        model_path = snapshot_dir / HF_BACKBONE_FILE
        if model_path.is_file():
            config_path = snapshot_dir / "config.json"
            return model_path, config_path if config_path.is_file() else None

    return None


def _check_snapshot_config(config_path):
    """Reject snapshots that are not ViT-L/16 with a clear message."""
    if config_path is None:
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        return

    checks = {
        "hidden_size": EXPECTED_EMBED_DIM,
        "num_hidden_layers": EXPECTED_DEPTH,
        "num_attention_heads": EXPECTED_HEADS,
        "patch_size": EXPECTED_PATCH_SIZE,
        "num_register_tokens": EXPECTED_REGISTER_TOKENS,
    }
    for key, expected in checks.items():
        if key in config and config[key] != expected:
            _error(
                f"The Hugging Face snapshot is not DINOv3 ViT-L/16: "
                f"{key}={config[key]!r}, expected {expected}. "
                "Use the facebook/dinov3-vitl16-pretrain-lvd1689m snapshot."
            )
    if config.get("model_type") not in (None, "dinov3_vit"):
        _error(
            f"Unexpected model_type in the snapshot: "
            f"{config.get('model_type')!r}. "
            "Use the facebook/dinov3-vitl16-pretrain-lvd1689m snapshot."
        )


def _expected_hf_keys(depth=EXPECTED_DEPTH):
    keys = {
        "embeddings.cls_token",
        "embeddings.mask_token",
        "embeddings.register_tokens",
        "embeddings.patch_embeddings.weight",
        "embeddings.patch_embeddings.bias",
        "norm.weight",
        "norm.bias",
    }
    for i in range(depth):
        p = f"layer.{i}."
        keys.update(
            {
                p + "attention.q_proj.weight",
                p + "attention.k_proj.weight",
                p + "attention.v_proj.weight",
                p + "attention.o_proj.weight",
                p + "attention.q_proj.bias",
                p + "attention.v_proj.bias",
                p + "attention.o_proj.bias",
                p + "layer_scale1.lambda1",
                p + "layer_scale2.lambda1",
                p + "mlp.up_proj.weight",
                p + "mlp.up_proj.bias",
                p + "mlp.down_proj.weight",
                p + "mlp.down_proj.bias",
                p + "norm1.weight",
                p + "norm1.bias",
                p + "norm2.weight",
                p + "norm2.bias",
            }
        )
    return keys


def _validate_hf_state_dict(hf):
    import torch

    expected = _expected_hf_keys()
    actual = set(hf)
    missing = expected - actual
    extra = actual - expected
    if missing:
        _error(
            f"The Hugging Face checkpoint is missing {len(missing)} expected "
            f"tensor(s), for example: {sorted(missing)[:3]}. "
            "Use the facebook/dinov3-vitl16-pretrain-lvd1689m snapshot."
        )
    if extra:
        _error(
            f"The Hugging Face checkpoint has {len(extra)} unexpected "
            f"tensor(s), for example: {sorted(extra)[:3]}. "
            "This converter targets the DINOv3 ViT-L/16 LVD1689M checkpoint."
        )

    embed = hf["embeddings.patch_embeddings.weight"]
    if tuple(embed.shape) != (EXPECTED_EMBED_DIM, 3, EXPECTED_PATCH_SIZE, EXPECTED_PATCH_SIZE):
        _error(
            "The Hugging Face checkpoint does not look like ViT-L/16: "
            f"patch embedding shape is {tuple(embed.shape)}, expected "
            f"({EXPECTED_EMBED_DIM}, 3, {EXPECTED_PATCH_SIZE}, {EXPECTED_PATCH_SIZE})."
        )
    for name, tensor in hf.items():
        if tensor.dtype != torch.float32:
            _error(
                f"The Hugging Face checkpoint contains non-float32 tensor "
                f"{name} ({tensor.dtype}). This converter expects float32."
            )


def _roped_periods_bf16():
    """Reconstruct Meta's ``rope_embed.periods`` buffer.

    Mirrors ``RopePositionEmbedding._init_weights`` for the ViT-L/16 config
    (base=100, embed_dim=1024, num_heads=16).  The official checkpoint stores
    this buffer as bfloat16 even though the hub code builds it in fp32.
    """
    import torch

    d_head = EXPECTED_EMBED_DIM // EXPECTED_HEADS
    ar = torch.arange(d_head // 4, dtype=torch.bfloat16)
    return 100.0 ** (2 * ar / (d_head // 2))


def _hf_to_meta_state_dict(hf):
    """Convert Hugging Face DINOv3 keys to Meta's DINOv3 state dict."""
    import collections

    import torch

    sd = collections.OrderedDict()
    sd["cls_token"] = hf["embeddings.cls_token"]
    sd["storage_tokens"] = hf["embeddings.register_tokens"]
    sd["mask_token"] = hf["embeddings.mask_token"].reshape(1, -1)
    sd["patch_embed.proj.weight"] = hf["embeddings.patch_embeddings.weight"]
    sd["patch_embed.proj.bias"] = hf["embeddings.patch_embeddings.bias"]
    sd["rope_embed.periods"] = _roped_periods_bf16()

    for i in range(EXPECTED_DEPTH):
        p = f"layer.{i}."
        b = f"blocks.{i}."

        sd[b + "norm1.weight"] = hf[p + "norm1.weight"]
        sd[b + "norm1.bias"] = hf[p + "norm1.bias"]

        # Meta fuses q, k, and v into one linear layer.  The HF conversion
        # omits k_proj.bias because key bias is masked (see bias_mask below).
        q_w = hf[p + "attention.q_proj.weight"]
        k_w = hf[p + "attention.k_proj.weight"]
        v_w = hf[p + "attention.v_proj.weight"]
        sd[b + "attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = hf[p + "attention.q_proj.bias"]
        v_b = hf[p + "attention.v_proj.bias"]
        sd[b + "attn.qkv.bias"] = torch.cat(
            [q_b, torch.zeros_like(q_b), v_b], dim=0
        )
        # The official checkpoint masks all qkv biases, so the buffer is zero.
        sd[b + "attn.qkv.bias_mask"] = torch.zeros(
            3 * EXPECTED_EMBED_DIM, dtype=q_b.dtype
        )

        sd[b + "attn.proj.weight"] = hf[p + "attention.o_proj.weight"]
        sd[b + "attn.proj.bias"] = hf[p + "attention.o_proj.bias"]
        sd[b + "ls1.gamma"] = hf[p + "layer_scale1.lambda1"]

        sd[b + "norm2.weight"] = hf[p + "norm2.weight"]
        sd[b + "norm2.bias"] = hf[p + "norm2.bias"]

        sd[b + "mlp.fc1.weight"] = hf[p + "mlp.up_proj.weight"]
        sd[b + "mlp.fc1.bias"] = hf[p + "mlp.up_proj.bias"]
        sd[b + "mlp.fc2.weight"] = hf[p + "mlp.down_proj.weight"]
        sd[b + "mlp.fc2.bias"] = hf[p + "mlp.down_proj.bias"]
        sd[b + "ls2.gamma"] = hf[p + "layer_scale2.lambda1"]

    sd["norm.weight"] = hf["norm.weight"]
    sd["norm.bias"] = hf["norm.bias"]
    return sd


def _tensor_fingerprint(state_dict):
    """Hash the semantic payload of a state dict.

    Format: SHA-256 over, for each key in sorted order, the UTF-8 key name,
    the shape tuple, the dtype string, and the raw little-endian tensor bytes,
    each terminated by a NUL byte.
    """
    import torch

    h = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(b"\0")
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(b"\0")
        h.update(tensor.view(torch.uint8).numpy().tobytes())
        h.update(b"\0")
    return h.hexdigest()


def _load_hf_safetensors(model_path):
    from safetensors import safe_open

    with safe_open(str(model_path), framework="pt", device="cpu") as f:
        names = list(f.keys())
        return {key: f.get_tensor(key) for key in names}


def _convert_backbone_from_snapshot(model_path, config_path):
    """Load, validate, convert, and fingerprint-check an HF snapshot."""
    _check_snapshot_config(config_path)
    print(f"Reading Hugging Face weights: {model_path}")
    hf = _load_hf_safetensors(model_path)
    _validate_hf_state_dict(hf)
    print(f"Converting {len(hf)} Hugging Face tensors to Meta DINOv3 keys ...")
    state_dict = _hf_to_meta_state_dict(hf)

    fingerprint = _tensor_fingerprint(state_dict)
    if fingerprint != BACKBONE_FINGERPRINT:
        _error(
            "The converted state dict does not match the official "
            "DINOv3 ViT-L/16 LVD1689M weights. The local snapshot may be a "
            "different checkpoint or a fine-tuned model. Use "
            "--backbone-mode download to fetch the official .pth instead."
        )
    print(
        f"Converted state dict matches the official weights "
        f"(fingerprint {fingerprint[:12]}...)."
    )
    return state_dict


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _verify_backbone_file(path):
    """Return 'official', 'converted', or None for *path*."""
    try:
        digest = _sha256_file(path)
    except OSError:
        return None
    if digest == BACKBONE_SHA256:
        return "official"
    try:
        state_dict = _torch_load_state_dict(path)
    except Exception:  # noqa: BLE001 - unreadable files are simply not verified
        return None
    try:
        if _tensor_fingerprint(state_dict) == BACKBONE_FINGERPRINT:
            return "converted"
    except Exception:  # noqa: BLE001 - malformed tensors are simply not verified
        return None
    return None


def _verify_head_file(path):
    try:
        return _sha256_file(path) == HEAD_SHA256
    except OSError:
        return False


def _verify_sha256(path, expected):
    try:
        return _sha256_file(path) == expected
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Atomic promotion
# ---------------------------------------------------------------------------

def _promote_part(part, dest, verify, label):
    """Atomically move *part* to *dest* without overwriting an existing file.

    ``verify`` is a callable that returns a truthy value when a file is valid.
    Returns ``True`` when this call installed the file.
    """
    part = pathlib.Path(part)
    dest = pathlib.Path(dest)
    try:
        os.link(str(part), str(dest))
    except FileExistsError:
        # Another process created the destination.  Accept it only if valid.
        if verify(str(dest)):
            try:
                os.unlink(str(part))
            except OSError:
                pass
            print(f"Already present (race): {dest.name}")
            return False
        try:
            os.unlink(str(part))
        except OSError:
            pass
        _error(
            f"Destination {dest.name} was replaced by a mismatched file. "
            "Remove it and re-run."
        )
    except OSError:
        try:
            os.unlink(str(part))
        except OSError:
            pass
        _error(
            f"Safe promotion for {label} is unavailable. "
            "Check filesystem hard-link support and permissions."
        )
    try:
        os.unlink(str(part))
    except OSError:
        print(
            f"WARNING: {label} is saved and verified, but its temporary "
            f"partial link remains at {part}; remove it manually.",
            file=sys.stderr,
        )
    return True


# ---------------------------------------------------------------------------
# Installing generated and downloaded files
# ---------------------------------------------------------------------------

def _install_backbone_state_dict(state_dict, dest):
    """Verify, save, re-verify, and atomically promote a converted backbone."""
    import torch

    dest = pathlib.Path(dest)
    if dest.is_symlink():
        _error(
            f"Destination {dest.name} is a symlink. "
            "Assets must not be installed through symlinks."
        )
    if dest.exists():
        kind = _verify_backbone_file(str(dest))
        if kind == "official":
            print(f"Already present (official bytes): {dest.name}")
            return
        if kind == "converted":
            print(f"Already present (converted, verified): {dest.name}")
            return
        _error(
            f"Existing file {dest.name} is not the expected DINOv3 ViT-L/16 "
            "backbone. Remove it and re-run."
        )

    part = pathlib.Path(str(dest) + PART_SUFFIX)
    if part.is_symlink() or part.exists():
        _error(
            f"Partial file for {dest.name} already exists. "
            "Remove it manually before re-running."
        )

    fd = None
    try:
        fd = os.open(
            str(part),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            _PART_FILE_MODE,
        )
        with os.fdopen(fd, "wb") as f:
            fd = None
            torch.save(state_dict, f)
    except Exception as exc:  # noqa: BLE001 - report any save failure uniformly
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(str(part))
        except OSError:
            pass
        _error(f"Could not write {dest.name}: {exc}")

    # Re-read the saved file so a broken serialization cannot slip through.
    if _verify_backbone_file(str(part)) != "converted":
        try:
            os.unlink(str(part))
        except OSError:
            pass
        _error(
            f"Saved {dest.name} failed verification after writing. "
            "The output filesystem may have corrupted the file."
        )

    size = part.stat().st_size
    installed = _promote_part(part, dest, _verify_backbone_file, dest.name)
    if installed:
        print(
            f"Saved {dest.name} "
            f"({_human_size(size)}, converted and fingerprint-verified)"
        )


def _install_hf_file(source, dest, expected_sha256, label):
    """Verify an HF-cached file and install it at *dest* without overwriting."""
    source = pathlib.Path(os.path.realpath(str(source)))
    dest = pathlib.Path(dest)

    if dest.is_symlink():
        _error(
            f"Destination {dest.name} is a symlink. "
            "Assets must not be installed through symlinks."
        )
    if dest.exists():
        if _verify_sha256(str(dest), expected_sha256):
            print(f"Already present (SHA-256 verified): {dest.name}")
            return
        _error(
            f"Existing file {dest.name} failed the official SHA-256 check. "
            "Remove it and re-run."
        )

    if not source.is_file():
        _error(f"Downloaded {label} is missing: {source}")

    actual = _sha256_file(str(source))
    if actual != expected_sha256:
        _error(
            f"Downloaded {label} failed the official SHA-256 check "
            f"(expected {expected_sha256[:12]}..., got {actual[:12]}...). "
            "The mirror may be stale or modified. Do not use this file."
        )

    # Prefer a hard link into the HF cache (no 2 GB copy).  Fall back to an
    # atomic copy when the output directory is on another filesystem or when
    # the cache file is accessible to group/other users.
    linkable = False
    try:
        linkable = not (os.stat(str(source)).st_mode & 0o077)
    except OSError:
        linkable = False
    if linkable:
        try:
            os.link(str(source), str(dest))
            print(
                f"Saved {dest.name} "
                f"({_human_size(source.stat().st_size)}, SHA-256 verified)"
            )
            return
        except FileExistsError:
            if _verify_sha256(str(dest), expected_sha256):
                print(f"Already present (race): {dest.name}")
                return
            _error(
                f"Destination {dest.name} was replaced by a mismatched file. "
                "Remove it and re-run."
            )
        except OSError:
            pass

    part = pathlib.Path(str(dest) + PART_SUFFIX)
    if part.is_symlink() or part.exists():
        _error(
            f"Partial file for {dest.name} already exists. "
            "Remove it manually before re-running."
        )
    fd = None
    try:
        fd = os.open(
            str(part),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            _PART_FILE_MODE,
        )
        with os.fdopen(fd, "wb") as out:
            fd = None
            with open(str(source), "rb") as inp:
                shutil.copyfileobj(inp, out, 1024 * 1024)
    except Exception as exc:  # noqa: BLE001 - report any copy failure uniformly
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(str(part))
        except OSError:
            pass
        _error(f"Could not copy {label} to {dest}: {exc}")

    if not _verify_sha256(str(part), expected_sha256):
        try:
            os.unlink(str(part))
        except OSError:
            pass
        _error(f"Copied {label} failed the official SHA-256 check.")

    size = part.stat().st_size
    installed = _promote_part(
        part,
        dest,
        lambda p: _verify_sha256(p, expected_sha256),
        dest.name,
    )
    if installed:
        print(f"Saved {dest.name} ({_human_size(size)}, SHA-256 verified)")


def _hf_hub_download(repo_id, filename, revision, cache_dir, label):
    """Download a file from Hugging Face, returning its local cache path."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except ImportError as exc:
        _error(
            "Hugging Face mode requires huggingface_hub. "
            f"Install it with: pip install huggingface_hub ({exc})"
        )

    try:
        return pathlib.Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        )
    except GatedRepoError:
        _error(
            f"Repository {repo_id} is gated. Request access on Hugging Face, "
            "then run: huggingface-cli login"
        )
    except RepositoryNotFoundError:
        _error(
            f"Repository {repo_id} was not found. Check the repository name, "
            "your network, and any HF_ENDPOINT setting."
        )
    except EntryNotFoundError:
        _error(
            f"File {filename!r} was not found in {repo_id}. "
            "Check --hf-asset-repo, --hf-head-file, or --hf-backbone-file."
        )
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            _error(
                f"Hugging Face rejected the download of {label} (HTTP {status}). "
                "The mirror repositories are public, so unset a stale HF_TOKEN "
                "or run: huggingface-cli login"
            )
        _error(f"Hugging Face download of {label} failed: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface unexpected hub errors
        _error(f"Hugging Face download of {label} failed: {exc}")


# ---------------------------------------------------------------------------
# Hugging Face mode orchestration
# ---------------------------------------------------------------------------

def _existing_backbone_is_ready(dest):
    """Print and return True when *dest* is a verified backbone."""
    if dest.is_symlink():
        _error(
            f"Destination {dest.name} is a symlink. "
            "Assets must not be installed through symlinks."
        )
    if not dest.exists():
        return False
    kind = _verify_backbone_file(str(dest))
    if kind == "official":
        print(f"Already present (official bytes): {dest.name}")
        return True
    if kind == "converted":
        print(f"Already present (converted, verified): {dest.name}")
        return True
    _error(
        f"Existing file {dest.name} is not the expected DINOv3 ViT-L/16 "
        "backbone. Remove it and re-run."
    )


def _existing_sha_file_is_ready(dest, expected_sha256):
    """Print and return True when *dest* matches *expected_sha256*."""
    if dest.is_symlink():
        _error(
            f"Destination {dest.name} is a symlink. "
            "Assets must not be installed through symlinks."
        )
    if not dest.exists():
        return False
    if _verify_sha256(str(dest), expected_sha256):
        print(f"Already present (SHA-256 verified): {dest.name}")
        return True
    _error(
        f"Existing file {dest.name} failed the official SHA-256 check. "
        "Remove it and re-run."
    )


def _prepare_from_huggingface(output_dir, args):
    _require_hf_dependencies()

    dest_backbone = output_dir / BACKBONE_FILENAME
    dest_head = output_dir / HEAD_FILENAME

    if not _existing_backbone_is_ready(dest_backbone):
        mode = args.backbone_mode
        snapshot = None
        if mode in ("auto", "convert"):
            resolved = _resolve_local_backbone(
                args.hf_backbone_snapshot, args.hf_backbone_repo
            )
            if resolved is not None:
                snapshot = resolved
            elif mode == "convert":
                _error(
                    "No local Hugging Face snapshot of "
                    f"{args.hf_backbone_repo} was found. Download it first with: "
                    f"huggingface-cli download {args.hf_backbone_repo} "
                    f"{HF_BACKBONE_FILE} --local-dir <dir>, "
                    "then pass --hf-backbone-snapshot <dir>, or use "
                    "--backbone-mode download."
                )

        if snapshot is not None:
            model_path, config_path = snapshot
            state_dict = _convert_backbone_from_snapshot(model_path, config_path)
            _install_backbone_state_dict(state_dict, dest_backbone)
        else:
            print(
                "No local Hugging Face snapshot found; downloading the "
                f"official backbone from {args.hf_asset_repo} ..."
            )
            src = _hf_hub_download(
                repo_id=args.hf_asset_repo,
                filename=args.hf_backbone_file,
                revision=args.hf_revision,
                cache_dir=args.hf_cache_dir,
                label="backbone",
            )
            _install_hf_file(src, dest_backbone, BACKBONE_SHA256, "backbone")

    if not _existing_sha_file_is_ready(dest_head, HEAD_SHA256):
        print(f"Fetching the DINOtxt head from {args.hf_asset_repo} ...")
        src = _hf_hub_download(
            repo_id=args.hf_asset_repo,
            filename=args.hf_head_file,
            revision=args.hf_revision,
            cache_dir=args.hf_cache_dir,
            label="DINOtxt head",
        )
        _install_hf_file(src, dest_head, HEAD_SHA256, "DINOtxt head")


# ---------------------------------------------------------------------------
# Meta download helpers
# ---------------------------------------------------------------------------

def _stream_download(url, outfile):
    """Stream a file from *url* to *outfile* (an already-open file object).

    Returns the number of bytes written, or raises on HTTP errors.
    URLs and query strings are never printed.
    """
    parsed = urlparse(url)
    path = parsed.path
    if parsed.query:
        path = path + "?" + parsed.query

    ctx = ssl.create_default_context()

    conn = None
    try:
        conn = http.client.HTTPSConnection(parsed.hostname, context=ctx)
        conn.request("GET", path)
        resp = conn.getresponse()
    except Exception as exc:  # normalize transport errors
        if conn is not None:
            conn.close()
        raise RuntimeError(
            "Network error - could not reach the server. "
            "Check connectivity and try again."
        ) from exc

    try:
        status = resp.status
        # Reject redirects without following them
        if 300 <= status <= 399:
            raise RuntimeError(
                f"HTTP {status} - redirect encountered. "
                "Download links must not be followed."
            )

        raw_length = resp.getheader("Content-Length")
        if raw_length is not None:
            try:
                length = int(raw_length)
                if length < 0:
                    raise ValueError("negative")
            except (ValueError, TypeError):
                raise RuntimeError(
                    "Content-Length header was malformed. "
                    "The server returned an invalid value."
                )
        else:
            length = None

        if status == 403:
            raise RuntimeError(
                "HTTP 403 Forbidden - missing Meta DINOv3 access. "
                "Accept the license at the DINOv3 GitHub page and use the "
                "URL from the official email."
            )
        if status in (401, 410):
            if status == 410:
                raise RuntimeError(
                    "HTTP 410 Gone - the download link has expired. "
                    "Request a new URL from the official DINOv3 access email."
                )
            raise RuntimeError(
                f"HTTP {status} - authentication required. "
                "Accept the license and use the URL from the official email."
            )
        if 400 <= status < 500 and status not in (401, 403, 410):
            raise RuntimeError(
                f"HTTP {status} - request rejected. "
                "The link may be expired or the access revoked."
            )
        if status >= 500:
            raise RuntimeError(
                f"HTTP {status} - server error. "
                "Retry after a short wait; contact Meta support if persistent."
            )

        # Successful response (2xx) - write content
        if length is not None:
            # Positive Content-Length: require exact byte count
            total = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                outfile.write(chunk)
                total += len(chunk)
            if total != length:
                raise RuntimeError(
                    f"Download interrupted - expected {length} bytes "
                    f"but received {total} bytes."
                )
            return total
        else:
            # Absent Content-Length: accept only after complete
            # stream and minimum-size check
            total = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                outfile.write(chunk)
                total += len(chunk)
            if total < _MIN_PTH_SIZE:
                raise RuntimeError(
                    f"Download interrupted - file too small ({total} bytes). "
                    f"Expected at least {_MIN_PTH_SIZE:,} bytes."
                )
            return total
    finally:
        resp.close()
        conn.close()


def _prepare_bpe(output_dir):
    """Download and verify the small official DINOv3 tokenizer vocabulary."""
    dest = output_dir / BPE_FILENAME
    part = output_dir / (BPE_FILENAME + PART_SUFFIX)

    if dest.is_symlink():
        _error(f"Destination {BPE_FILENAME} is a symlink. Refusing to follow it.")
    if dest.is_file():
        if _verify_sha256(str(dest), BPE_SHA256):
            print(f"Already present (SHA-256 verified): {BPE_FILENAME}")
            return
        _error(f"Existing {BPE_FILENAME} failed SHA-256 verification. Remove it and re-run.")
    if part.is_symlink() or part.exists():
        _error(f"Partial file for {BPE_FILENAME} already exists. Remove it and re-run.")

    fd = None
    try:
        fd = os.open(str(part), os.O_WRONLY | os.O_CREAT | os.O_EXCL, _PART_FILE_MODE)
        with os.fdopen(fd, "wb") as outfile:
            fd = None
            size = _stream_download(BPE_URL, outfile)
    except Exception as exc:  # noqa: BLE001 - remove partial bytes and report a bounded error
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            part.unlink()
        except OSError:
            pass
        _error(f"Could not download {BPE_FILENAME} from the official DINOv3 host: {exc}")

    if size != BPE_SIZE or not _verify_sha256(str(part), BPE_SHA256):
        try:
            part.unlink()
        except OSError:
            pass
        _error(f"Downloaded {BPE_FILENAME} failed its pinned size or SHA-256 check.")

    if _promote_part(part, dest, lambda path: _verify_sha256(path, BPE_SHA256), "DINOv3 tokenizer"):
        print(f"Saved {dest.name} ({_human_size(size)}, SHA-256 verified)")


def _read_url_file(url_file_path, git_root):
    url_file = pathlib.Path(url_file_path)
    if url_file.is_symlink():
        _error("URL-list file must not be a symlink")
    if not url_file.is_file():
        _error("URL-list file not found")
    if _is_inside_git_worktree(url_file, git_root):
        _error(
            "URL-list file must not be inside the Git working tree. "
            "Place it outside the repository and re-run."
        )

    stat = url_file.stat()
    mode = stat.st_mode & 0o777
    if mode & 0o077:
        _error(
            f"URL-list file has overly permissive permissions ({oct(mode)}). "
            "Run: chmod 600 <url-file>"
        )

    urls = {}
    try:
        with open(url_file, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    _error(
                        f"Line {lineno}: malformed entry "
                        "(missing key=URL separator)"
                    )
                key, _, value = line.partition("=")
                key = key.strip().lower()
                value = value.strip()
                if key not in ("backbone", "dinotxt"):
                    _error(f"Line {lineno}: unknown key")
                if key in urls:
                    _error(f"Line {lineno}: duplicate key")
                urls[key] = value
    except OSError:
        _error("Cannot read the URL-list file.")

    if "backbone" not in urls or "dinotxt" not in urls:
        _error("URL-list file must contain exactly two lines: backbone= and dinotxt=")

    parsed_urls = {}
    for key, url in urls.items():
        try:
            parsed = urlparse(url)
            port = parsed.port
        except ValueError:
            _error(f"{key}: malformed URL or port; use the official email URL.")
        if parsed.scheme != "https":
            _error(f"{key}: URL scheme must be https.")
        if parsed.hostname is None or parsed.hostname != OFFICIAL_HOST:
            _error(
                f"{key}: host must be the official Meta download host. "
                "Only the official Meta host is accepted."
            )
        if port is not None and port != 443:
            _error(
                f"{key}: non-standard port. "
                "Only the standard HTTPS port 443 is allowed."
            )
        if parsed.username or parsed.password:
            _error(
                f"{key}: URL must not contain embedded user credentials. "
                "Use the bare URL from the official access email."
            )
        parsed_urls[key] = url
    return parsed_urls


def _prepare_from_meta(output_dir, args, git_root):
    parsed_urls = _read_url_file(args.url_file, git_root)

    assets = [
        {
            "name": BACKBONE_FILENAME,
            "role": "backbone",
            "sha256": BACKBONE_SHA256,
            "verify": _verify_backbone_file,
        },
        {
            "name": HEAD_FILENAME,
            "role": "dinotxt",
            "sha256": HEAD_SHA256,
            "verify": _verify_head_file,
        },
    ]

    errors = []
    for asset in assets:
        role = asset["role"]
        filename = asset["name"]
        url = parsed_urls[role]
        dest = output_dir / filename
        part = output_dir / (filename + PART_SUFFIX)
        part_path_str = str(part)

        if dest.is_symlink():
            _error(
                f"Destination {filename} is a symlink. "
                "Downloads must not replace files through symlinks."
            )
        if dest.is_file():
            kind = asset["verify"](str(dest))
            if kind is True or kind in ("official", "converted"):
                if kind == "converted":
                    print(f"Already present (converted, verified): {filename}")
                elif kind == "official":
                    print(f"Already present (official bytes): {filename}")
                else:
                    print(f"Already present: {filename}")
                continue
            _error(
                f"Existing file {filename} failed verification. "
                "Remove the mismatched file and re-run."
            )

        if part.is_symlink() or part.exists():
            _error(
                f"Partial file for {filename} already exists. "
                "Remove it manually before re-running."
            )

        print(f"Downloading {filename} ...")

        f = None
        part_created = False
        try:
            fd = os.open(
                part_path_str,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                _PART_FILE_MODE,
            )
            part_created = True
            try:
                f = os.fdopen(fd, "wb")
            except Exception:
                os.close(fd)
                raise
            nbytes = _stream_download(url, f)
            f.close()
            f = None
        except FileExistsError:
            print(
                f"FAIL: {filename} - partial file appeared unexpectedly",
                file=sys.stderr,
            )
            errors.append((filename, "download failed"))
            continue
        except Exception as exc:  # noqa: BLE001 - report any download failure
            if f is not None:
                try:
                    f.close()
                except Exception:  # noqa: BLE001, S110 - best-effort cleanup
                    pass
            if part_created:
                try:
                    os.unlink(part_path_str)
                except OSError:
                    pass
            print(
                f"FAIL: {filename} - {exc}",
                file=sys.stderr,
            )
            errors.append((filename, "download or file creation failed"))
            continue

        # --- Verification ---
        try:
            if not asset["verify"](part_path_str):
                raise ValueError("verification failed")
        except Exception:  # noqa: BLE001 - verification means any failure rejects the file
            try:
                os.unlink(part_path_str)
            except OSError:
                pass
            print(
                f"FAIL: {filename} - verification failed",
                file=sys.stderr,
            )
            errors.append((filename, "verification failed"))
            continue

        installed = _promote_part(part_path_str, dest, asset["verify"], filename)
        if installed:
            print(
                f"Saved {filename} "
                f"({_human_size(nbytes)}, SHA-256 verified)"
            )

    if errors:
        _error(
            f"{len(errors)} download(s) failed. "
            "Partial files have been cleaned up. "
            "Resolve the above errors and re-run."
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the DINOv3 weights and tokenizer required by T-REN. In Hugging Face "
            "mode (default), the ViT-L/16 backbone is converted from a local "
            "facebook/dinov3-vitl16-pretrain-lvd1689m snapshot and the DINOtxt "
            "head is downloaded from a public Hugging Face mirror. Both are "
            "verified: the converted backbone by a tensor fingerprint of the "
            "official state dict, and the head by the official full SHA-256. "
            "In Meta mode, both original .pth files are downloaded from the "
            "official Meta host using the URLs from the DINOv3 access email."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            Hugging Face mode using the snapshot already in $HF_HOME::

                python3 scripts/download_tren_dinov3_assets.py \\
                    --output-dir ~/.cache/t_ren-dinov3-assets

            Convert an explicit snapshot::

                python3 scripts/download_tren_dinov3_assets.py \\
                    --hf-backbone-snapshot /path/to/dinov3-vitl16-snapshot \\
                    --output-dir /path/to/assets

            Download the original .pth files from the verified HF mirror::

                python3 scripts/download_tren_dinov3_assets.py \\
                    --backbone-mode download \\
                    --output-dir /path/to/assets

            Meta mode with the official email URLs::

                python3 scripts/download_tren_dinov3_assets.py \\
                    --source meta \\
                    --url-file url-list.txt \\
                    --output-dir /path/to/assets

            The Hugging Face asset mirror is public. Its files are accepted
            only when their full SHA-256 matches the official Meta hashes.
            Keep the Meta URL-list file private. Do not commit it to Git.
        """),
    )
    parser.add_argument(
        "--source",
        choices=("auto", "hf", "meta"),
        default="auto",
        help=(
            "Where to obtain the assets. 'hf' uses local Hugging Face caches "
            "and public Hugging Face mirrors. 'meta' uses the official Meta "
            "email URLs from --url-file. 'auto' (default) chooses 'meta' when "
            "--url-file is provided, otherwise 'hf'."
        ),
    )
    parser.add_argument(
        "--url-file",
        default=None,
        help=(
            "Meta mode only. Path to a private URL-list file with exactly two "
            "lines: backbone=<URL> and dinotxt=<URL>. Create this file from "
            "the download URLs received in the Meta DINOv3 access email. Keep "
            "this file private; do not commit it to Git."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory in which to save the prepared files. Required unless "
            "$HF_HOME is set, in which case the default is "
            "$HF_HOME/t_ren-dinov3-assets."
        ),
    )
    parser.add_argument(
        "--backbone-mode",
        choices=("auto", "convert", "download"),
        default="auto",
        help=(
            "Hugging Face mode only. 'convert' converts a local "
            "facebook/dinov3-vitl16-pretrain-lvd1689m snapshot to the "
            "required Meta .pth. 'download' downloads the original .pth from "
            "the verified Hugging Face mirror. 'auto' (default) converts when "
            "a local snapshot is found and downloads otherwise."
        ),
    )
    parser.add_argument(
        "--hf-backbone-repo",
        default=HF_BACKBONE_REPO,
        help=(
            "Hugging Face repository used to locate the local ViT-L/16 "
            f"snapshot. Default: {HF_BACKBONE_REPO}."
        ),
    )
    parser.add_argument(
        "--hf-backbone-snapshot",
        default=None,
        help=(
            "Path to a local snapshot directory or model.safetensors file. "
            "Overrides automatic discovery in the Hugging Face cache."
        ),
    )
    parser.add_argument(
        "--hf-asset-repo",
        default=HF_ASSET_REPO,
        help=(
            "Hugging Face repository that mirrors the official Meta .pth "
            f"files. Default: {HF_ASSET_REPO}. Files are accepted only when "
            "their full SHA-256 matches the official hashes."
        ),
    )
    parser.add_argument(
        "--hf-head-file",
        default=HF_ASSET_HEAD_FILE,
        help=(
            "Filename of the DINOtxt head inside --hf-asset-repo. "
            f"Default: {HF_ASSET_HEAD_FILE}."
        ),
    )
    parser.add_argument(
        "--hf-backbone-file",
        default=HF_ASSET_BACKBONE_FILE,
        help=(
            "Filename of the original backbone .pth inside --hf-asset-repo, "
            f"used by --backbone-mode download. Default: {HF_ASSET_BACKBONE_FILE}."
        ),
    )
    parser.add_argument(
        "--hf-revision",
        default=None,
        help="Optional Hugging Face revision (branch, tag, or commit) for both repos.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help=(
            "Optional Hugging Face cache directory for downloads. Defaults to "
            "the standard HF_HOME/HF_HUB_CACHE location."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()

    git_root = _find_git_root()
    if git_root is None:
        _error(
            "The script must be located inside a Git working tree. "
            "Ensure the production script resides under the repository."
        )

    source = args.source
    if source == "auto":
        source = "meta" if args.url_file else "hf"
    if source == "meta" and not args.url_file:
        _error("Meta mode requires --url-file.")
    if source == "hf" and args.url_file:
        _error(
            "--url-file is only used with Meta mode. Use --source meta, or "
            "omit --url-file for Hugging Face mode."
        )

    # --- Resolve output directory ---
    output_dir = args.output_dir
    if output_dir is None:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            output_dir = os.path.join(hf_home, "t_ren-dinov3-assets")
        else:
            _error(
                "--output-dir is required when $HF_HOME is not set. "
                "Example: --output-dir ~/.cache/t_ren-dinov3-assets"
            )

    output_dir = pathlib.Path(output_dir).expanduser()
    if output_dir.is_symlink():
        _error(
            "Output directory must not be a symlink. "
            "Provide a real directory path and re-run."
        )
    if _is_inside_git_worktree(output_dir, git_root):
        _error(
            "Output directory must not be inside the Git working tree "
            "(weight files must remain outside Git). "
            "Use a location outside the repository, or set $HF_HOME."
        )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        _error("Cannot create the output directory. Check its path and permissions.")

    print(f"Output directory: {output_dir}")

    if source == "meta":
        _prepare_from_meta(output_dir, args, git_root)
    else:
        _prepare_from_huggingface(output_dir, args)

    _prepare_bpe(output_dir)
    print("All assets prepared and verified.")


if __name__ == "__main__":
    main()
