"""Fetch the two IWADs the Doom envs need into data/wads/.

    uv run python scripts/fetch_wads.py

DoomTakeCover's take_cover scenario ships its own WAD inside the vizdoom
pip package, so this script is only for DoomCampaign, which plays full
maps and needs a real IWAD on disk: id Software's shareware doom1.wad
(episode one, free to redistribute since 1995) and Freedoom's
freedoom1.wad (a from-scratch, libre replacement for id's registered
IWAD, for anyone who would rather not use id's assets at all). Each
file's SHA256 is pinned below and checked after download. A file
already on disk that verifies against its pin is left alone rather than
re-fetched; a downloaded file that does not match its pin is a hard
failure, not a wad quietly sitting in data/wads/ with the wrong bytes
behind a name that says it is fine.

Freedoom publishes phase 1 and phase 2 bundled in one release zip, so
freedoom1.wad's entry names the zip member to pull out; doom1.wad's
mirror serves the raw file directly.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

WADS_DIR = Path("data/wads")

# id Software's own mirrors have gone dark and come back more than once
# over the decades; this points at a plain, direct file (not a DOS-era
# self-extracting archive) so the download side of this script stays a
# single GET. The pinned hash is what protects against a mirror serving
# something else under the same name, not the choice of mirror itself.
_DOOM1_URL = ("https://raw.githubusercontent.com/Doom-Utils/"
             "shareware-collection/master/Doom%201.9/doom1.wad")
_FREEDOOM_ZIP_URL = ("https://github.com/freedoom/freedoom/releases/"
                     "download/v0.13.0/freedoom-0.13.0.zip")


@dataclass(frozen=True)
class WadSpec:
    """One IWAD to fetch.

    filename is where it lands under the destination directory. url is
    what to GET: a raw wad, or a zip when member is set, in which case
    member names the entry inside the zip to extract. sha256 is the
    pin the final wad bytes must match, checked whether the file was
    just downloaded or was already sitting on disk.
    """
    filename: str
    url: str
    sha256: str
    member: str | None = None


# Verified by downloading each mirror and hashing the bytes by hand
# during research for this script; if a
# mirror ever moves, re-pin by fetching the new file and running this
# script with --print-hash on it, after checking the new bytes are the
# wad they claim to be some other way (a second independent mirror, a
# hash published by the project itself).
WADS = (
    WadSpec(
        filename="doom1.wad",
        url=_DOOM1_URL,
        sha256="1d7d43be501e67d927e415e0b8f3e29c3bf33075e859721816f652a526cac771",
    ),
    WadSpec(
        filename="freedoom1.wad",
        url=_FREEDOOM_ZIP_URL,
        sha256="7323bcc168c5a45ff10749b339960e98314740a734c30d4b9f3337001f9e703d",
        member="freedoom-0.13.0/freedoom1.wad",
    ),
)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verifies(path: Path, sha256: str) -> bool:
    """True when path exists and its content hashes to sha256."""
    return path.exists() and sha256_of(path.read_bytes()) == sha256


def fetch_bytes(url: str) -> bytes:
    """Plain GET. The only function here that touches the network, so
    tests exercise everything else by passing a fake in its place."""
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def fetch_wad(spec: WadSpec, dest_dir: Path,
             fetch_fn: Callable[[str], bytes] = fetch_bytes) -> Path:
    """Get spec into dest_dir and return where it landed.

    A file already at the destination that verifies is left in place
    and fetch_fn is never called: refetching a wad that is already
    correct just burns bandwidth against someone else's mirror. A
    downloaded file that does not match spec.sha256 raises instead of
    being written, so a bad download never leaves a wad on disk under
    the right name with the wrong bytes.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / spec.filename
    if verifies(dest, spec.sha256):
        print(f"{spec.filename}: already present and verified at {dest}")
        return dest

    print(f"{spec.filename}: fetching from {spec.url}")
    raw = fetch_fn(spec.url)
    data = raw if spec.member is None else _extract_member(raw, spec.member)

    digest = sha256_of(data)
    if digest != spec.sha256:
        raise SystemExit(
            f"{spec.filename}: SHA256 mismatch after download, expected "
            f"{spec.sha256} but got {digest}; refusing to write it to "
            f"{dest}"
        )

    dest.write_bytes(data)
    print(f"{spec.filename}: verified and saved to {dest}")
    return dest


def _extract_member(zip_bytes: bytes, member: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        return archive.read(member)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch doom1.wad and freedoom1.wad into data/wads/")
    parser.add_argument("--dest", type=Path, default=WADS_DIR,
                        help="destination directory (default data/wads)")
    parser.add_argument(
        "--print-hash", type=Path, default=None, metavar="PATH",
        help="print the SHA256 of an existing file and exit, for "
             "re-pinning a wad by hand instead of fetching anything")
    args = parser.parse_args()

    if args.print_hash is not None:
        print(sha256_of(args.print_hash.read_bytes()))
        return

    for spec in WADS:
        fetch_wad(spec, args.dest)


if __name__ == "__main__":
    main()
