#!/usr/bin/env python3
"""Download exactly two official Meta DINOv3 assets required by T-REN.

Purpose
-------
Download the DINOv3 ViT-L/16 pre-train backbone and the DINOtxt vision-head
plus text-encoder .pth files after the user has accepted the Meta license and
received the official email download URLs.

This script downloads from the official Meta host only. It does not automate
license acceptance, login, cookies, or any form. It does not fall back to
Hugging Face, community mirrors, or other sources.

Usage
-----
Create a private URL-list file containing exactly two lines::

    backbone=<URL>
    dinotxt=<URL>

Run::

    python3 output/scripts/download_tren_dinov3_assets.py \\
        --url-file <path-to-url-file> \\
        --output-dir <path>

The --output-dir argument is required unless $HF_HOME is set, in which case the
default output directory is ``$HF_HOME/t_ren-dinov3-assets``.

Keep the URL-list file private. Do not commit it to Git or share it. The URLs
are time-limited and tied to your Meta DINOv3 access email.
"""

import argparse
import hashlib
import http.client
import os
import pathlib
import ssl
import sys
import textwrap
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_ASSETS = [
    {
        "name": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "role": "backbone",
        "prefix": "8aa4cbdd",
    },
    {
        "name": "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
        "role": "dinotxt",
        "prefix": "a442d8f5",
    },
]

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


def _compute_sha256_prefix(filepath):
    """Compute the first 8 hex digits of the SHA-256 of *filepath*."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:8]


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

    try:
        conn = http.client.HTTPSConnection(parsed.hostname, context=ctx)
        conn.request("GET", path)
        resp = conn.getresponse()
    except Exception:
        conn.close()
        raise RuntimeError(
            "Network error - could not reach the server. "
            "Check connectivity and try again."
        )

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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Download exactly two official Meta DINOv3 assets required by T-REN: "
            "the ViT-L/16 pre-train backbone and the DINOtxt vision-head plus "
            "text-encoder. These files are gated behind Meta's license acceptance. "
            "This script does not automate acceptance, login, or cookies. "
            "It downloads only from the official Meta host "
            "(dl.fbaipublicfiles.com) and verifies SHA-256 prefixes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example
            -------
            1. Accept the Meta DINOv3 license at https://github.com/facebookresearch/dinov3
            2. Wait for the official email with download URLs.
            3. Create a private file ``url-list.txt`` containing::

                   backbone=https://dl.fbaipublicfiles.com/…
                   dinotxt=https://dl.fbaipublicfiles.com/…

            4. Run::

                   python3 output/scripts/download_tren_dinov3_assets.py \\
                       --url-file url-list.txt \\
                       --output-dir ~/.cache/t_ren-dinov3-assets

            Keep the URL-list file private. Do not commit it to Git.
        """),
    )
    parser.add_argument(
        "--url-file",
        required=True,
        help=(
            "Path to a private URL-list file with exactly two lines: "
            "backbone=<URL> and dinotxt=<URL>. Create this file from the "
            "download URLs received in the Meta DINOv3 access email. Keep "
            "this file private; do not commit it to Git."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory in which to save the downloaded files. Required unless "
            "$HF_HOME is set, in which case the default is "
            "$HF_HOME/t_ren-dinov3-assets."
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

    # --- Validate URL-list file ---
    url_file = pathlib.Path(args.url_file)
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

    # --- Parse URL file ---
    urls = {}
    try:
        with open(url_file, "r") as f:
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

    # --- Validate URLs ---
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
        if port is not None:
            if port != 443:
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

    # --- Ensure output directory exists ---
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        _error("Cannot create the output directory. Check its path and permissions.")

    # --- Download each asset ---
    errors = []

    for asset in EXPECTED_ASSETS:
        role = asset["role"]
        filename = asset["name"]
        expected_prefix = asset["prefix"]
        url = parsed_urls[role]

        dest = output_dir / filename
        part = output_dir / (filename + PART_SUFFIX)
        part_path_str = str(part)

        # --- Check destination existence ---
        if dest.is_symlink():
            _error(
                f"Destination {filename} is a symlink. "
                "Downloads must not replace files through symlinks."
            )
        if dest.is_file():
            try:
                prefix = _compute_sha256_prefix(str(dest))
            except OSError:
                _error(f"Existing file {filename} cannot be verified; inspect it and retry.")
            if prefix == expected_prefix:
                print(f"Already present: {filename}")
                continue
            else:
                _error(
                    f"Existing file {filename} failed SHA-256 prefix check. "
                    "Remove the mismatched file and re-run."
                )

        # --- Refuse pre-existing partial ---
        if part.is_symlink() or part.exists():
            _error(
                f"Partial file for {filename} already exists. "
                "Remove it manually before re-running."
            )

        print(f"Downloading {filename} ...")

        # --- Exclusively create the partial ---
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
        except Exception:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
            if part_created:
                try:
                    os.unlink(part_path_str)
                except OSError:
                    pass
            print(
                f"FAIL: {filename} - download or file creation failed",
                file=sys.stderr,
            )
            errors.append((filename, "download or file creation failed"))
            continue

        # --- Hash check ---
        try:
            prefix = _compute_sha256_prefix(part_path_str)
        except OSError:
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

        if prefix != expected_prefix:
            try:
                os.unlink(part_path_str)
            except OSError:
                pass
            _error(
                f"SHA-256 prefix mismatch for {filename}: "
                f"expected {expected_prefix}, got {prefix}. "
                "The downloaded file is corrupted or not the correct asset."
            )

        # --- Atomic promotion: hard link (no-replace) then unlink part ---
        try:
            os.link(part_path_str, str(dest))
        except FileExistsError:
            # Racing destination appeared - inspect without modifying
            race_dest = output_dir / filename
            if race_dest.is_symlink():
                os.unlink(part_path_str)
                _error(
                    f"Destination {filename} is a symlink. "
                    "Downloads must not replace files through symlinks."
                )
            if race_dest.is_file():
                try:
                    race_prefix = _compute_sha256_prefix(str(race_dest))
                except OSError:
                    os.unlink(part_path_str)
                    _error(
                        f"Destination {filename} could not be verified. "
                        "Cannot proceed safely."
                    )
                if race_prefix == expected_prefix:
                    os.unlink(part_path_str)
                    print(f"Already present (race): {filename}")
                    continue
                else:
                    os.unlink(part_path_str)
                    _error(
                        f"Destination {filename} was replaced by a "
                        "mismatched file. Remove it and re-run."
                    )
            os.unlink(part_path_str)
            _error(
                f"Unexpected destination state for {filename}. "
                "Cannot promote the downloaded file safely."
            )
        except OSError:
            try:
                os.unlink(part_path_str)
            except OSError:
                pass
            print(
                f"FAIL: {filename} - safe promotion unavailable; "
                "check filesystem hard-link support and permissions",
                file=sys.stderr,
            )
            errors.append((filename, "safe promotion unavailable"))
            continue

        try:
            os.unlink(part_path_str)
        except OSError:
            print(
                f"WARNING: {filename} is saved and verified, but its temporary "
                f"partial link remains at {part_path_str}; remove it manually.",
                file=sys.stderr,
            )
        print(f"Saved {filename} ({nbytes:,} bytes, SHA-256 prefix {prefix})")

    if errors:
        _error(
            f"{len(errors)} download(s) failed. "
            "Partial files have been cleaned up. "
            "Resolve the above errors and re-run."
        )

    print("All assets downloaded and verified.")


if __name__ == "__main__":
    main()
