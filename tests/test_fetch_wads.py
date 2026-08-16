"""fetch_wads.py is a standalone script, not a package module, so it is
loaded here by file path. Every test below passes a fake fetch_fn in
place of the real network call: nothing here ever makes a request."""

import hashlib
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_wads.py"
_spec = importlib.util.spec_from_file_location("fetch_wads", _SCRIPT_PATH)
fetch_wads = importlib.util.module_from_spec(_spec)
# dataclass's own field resolution looks the module up by name in
# sys.modules, so it has to be registered there before exec_module runs.
sys.modules["fetch_wads"] = fetch_wads
_spec.loader.exec_module(fetch_wads)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_pinned_wads_have_well_formed_hashes_and_names():
    names = {spec.filename for spec in fetch_wads.WADS}
    assert names == {"doom1.wad", "freedoom1.wad"}
    for spec in fetch_wads.WADS:
        assert len(spec.sha256) == 64
        int(spec.sha256, 16)  # raises if it is not plain hex
        assert spec.url.startswith("https://")


def test_verifies_true_for_matching_content(tmp_path):
    data = b"some wad bytes"
    path = tmp_path / "doom1.wad"
    path.write_bytes(data)
    assert fetch_wads.verifies(path, _sha256(data))


def test_verifies_false_for_missing_or_mismatched_file(tmp_path):
    path = tmp_path / "doom1.wad"
    assert not fetch_wads.verifies(path, _sha256(b"anything"))
    path.write_bytes(b"wrong bytes")
    assert not fetch_wads.verifies(path, _sha256(b"right bytes"))


def test_fetch_wad_refuses_to_redownload_a_verified_file(tmp_path):
    data = b"already correct"
    spec = fetch_wads.WadSpec(filename="doom1.wad", url="unused",
                              sha256=_sha256(data))
    dest_dir = tmp_path
    (dest_dir / spec.filename).write_bytes(data)

    def fetch_fn(url):
        raise AssertionError("fetch_fn must not be called for a file "
                             "that already verifies")

    result = fetch_wads.fetch_wad(spec, dest_dir, fetch_fn=fetch_fn)
    assert result == dest_dir / spec.filename
    assert (dest_dir / spec.filename).read_bytes() == data


def test_fetch_wad_downloads_and_writes_when_missing(tmp_path):
    data = b"the real wad bytes"
    spec = fetch_wads.WadSpec(filename="doom1.wad", url="http://example/x",
                              sha256=_sha256(data))

    calls = []

    def fetch_fn(url):
        calls.append(url)
        return data

    result = fetch_wads.fetch_wad(spec, tmp_path, fetch_fn=fetch_fn)
    assert calls == [spec.url]
    assert result == tmp_path / "doom1.wad"
    assert result.read_bytes() == data


def test_fetch_wad_overwrites_a_file_that_does_not_verify(tmp_path):
    good = b"the correct bytes"
    spec = fetch_wads.WadSpec(filename="doom1.wad", url="http://example/x",
                              sha256=_sha256(good))
    dest = tmp_path / spec.filename
    dest.write_bytes(b"stale or corrupt content")

    result = fetch_wads.fetch_wad(spec, tmp_path, fetch_fn=lambda url: good)
    assert result.read_bytes() == good


def test_fetch_wad_hash_mismatch_fails_loudly_and_writes_nothing(tmp_path):
    spec = fetch_wads.WadSpec(filename="doom1.wad", url="http://example/x",
                              sha256=_sha256(b"expected bytes"))

    with pytest.raises(SystemExit):
        fetch_wads.fetch_wad(spec, tmp_path, fetch_fn=lambda url: b"wrong bytes")
    assert not (tmp_path / spec.filename).exists()


def test_fetch_wad_extracts_named_member_from_a_zip(tmp_path):
    wad_bytes = b"freedoom1 payload"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("freedoom-0.13.0/freedoom1.wad", wad_bytes)
        archive.writestr("freedoom-0.13.0/freedoom2.wad", b"phase 2, ignored")
    zip_bytes = buf.getvalue()

    spec = fetch_wads.WadSpec(
        filename="freedoom1.wad", url="http://example/freedoom.zip",
        sha256=_sha256(wad_bytes), member="freedoom-0.13.0/freedoom1.wad",
    )
    result = fetch_wads.fetch_wad(spec, tmp_path, fetch_fn=lambda url: zip_bytes)
    assert result.read_bytes() == wad_bytes


def test_fetch_wad_creates_the_destination_directory(tmp_path):
    data = b"bytes"
    spec = fetch_wads.WadSpec(filename="doom1.wad", url="http://example/x",
                              sha256=_sha256(data))
    dest_dir = tmp_path / "wads" / "nested"
    assert not dest_dir.exists()

    fetch_wads.fetch_wad(spec, dest_dir, fetch_fn=lambda url: data)
    assert (dest_dir / "doom1.wad").read_bytes() == data


def test_print_hash_mode_reports_a_file_hash_without_touching_wads(
    tmp_path, capsys, monkeypatch
):
    data = b"contents to hash"
    target = tmp_path / "some.wad"
    target.write_bytes(data)

    monkeypatch.setattr(
        "sys.argv", ["fetch_wads.py", "--print-hash", str(target)]
    )
    fetch_wads.main()
    out = capsys.readouterr().out.strip()
    assert out == _sha256(data)
