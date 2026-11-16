#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"

ensure_checkout() {
    local name="$1"
    local path="$2"
    local remote="$3"
    local tag="$4"
    local expected_commit="$5"
    local patch_file="$6"

    if [[ -e "$path" ]]; then
        if ! git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            printf 'Refusing to replace existing non-Git dependency directory: %s\n' "$path" >&2
            printf 'Move or archive it yourself, then rerun this script. No files were changed.\n' >&2
            return 1
        fi
    else
        git clone --depth 1 --branch "$tag" "$remote" "$path"
    fi

    local actual_commit
    actual_commit="$(git -C "$path" rev-parse HEAD)"
    if [[ "$actual_commit" != "$expected_commit" ]]; then
        printf '%s checkout mismatch: expected %s, found %s\n' "$name" "$expected_commit" "$actual_commit" >&2
        return 1
    fi

    if git -C "$path" apply --unidiff-zero --reverse --check "$patch_file" >/dev/null 2>&1; then
        : # The local patch is already applied.
    elif git -C "$path" apply --unidiff-zero --check "$patch_file" >/dev/null 2>&1; then
        git -C "$path" apply --unidiff-zero "$patch_file"
    else
        printf 'Could not safely apply the pinned %s patch to %s. No files were overwritten.\n' "$name" "$path" >&2
        return 1
    fi
}

ensure_checkout \
    "FAISS" \
    "$THIRD_PARTY/faiss" \
    "https://github.com/facebookresearch/faiss.git" \
    "v1.15.1" \
    "75c755c5a1fc854756aaf3bd1131213aeb21d8d1" \
    "$THIRD_PARTY/patches/faiss-cuda13.patch"

ensure_checkout \
    "PyTorch3D" \
    "$THIRD_PARTY/pytorch3d" \
    "https://github.com/facebookresearch/pytorch3d.git" \
    "v0.7.9" \
    "33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba" \
    "$THIRD_PARTY/patches/pytorch3d-cuda13.patch"

FAISS_CONFIG="$THIRD_PARTY/faiss/pyproject.toml"
LOCAL_FAISS_CONFIG="$THIRD_PARTY/patches/faiss-pyproject.toml"
if [[ -e "$FAISS_CONFIG" ]]; then
    if ! cmp -s "$FAISS_CONFIG" "$LOCAL_FAISS_CONFIG"; then
        printf 'Refusing to overwrite existing FAISS build config: %s\n' "$FAISS_CONFIG" >&2
        printf 'Compare it with %s and resolve the difference manually.\n' "$LOCAL_FAISS_CONFIG" >&2
        exit 1
    fi
else
    cp -- "$LOCAL_FAISS_CONFIG" "$FAISS_CONFIG"
fi

printf 'Pinned external sources and local patches are ready. Run `uv sync` from %s.\n' "$ROOT"
