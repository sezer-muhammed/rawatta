# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/download_weights.sh against a local mock of the Hugging Face API.

The mock answers the two requests the script makes (paths-info for the LFS SHA-256 and size,
and the resolve download with HTTP Range support) and checks the bearer token. The script is
pointed at it with HF_ENDPOINT. No network, no weights.

    python3 -m pytest tools/tests -q
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "download_weights.sh"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

pytestmark = pytest.mark.skipif(shutil.which("curl") is None or shutil.which("sh") is None,
                                reason="needs sh and curl")

TOKEN = "hf_test_secret_token_0123456789"
FILE = "vggt_omega_1b_512.pt"


class Mock:
    def __init__(self, payload: bytes, delay: float = 0.0, meta_status: int = 200):
        self.payload = payload
        self.meta_status = meta_status  # forced status of the paths-info request (403: no access)
        self.delay = delay  # seconds between 16 KiB chunks of a download
        self.requests: list[tuple[str, str, str | None]] = []  # (method, path, Range header)
        mock = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _authorised(self) -> bool:
                if self.headers.get("Authorization") == f"Bearer {TOKEN}":
                    return True
                body = b'{"error":"Access to model facebook/VGGT-Omega is restricted."}'
                self.send_response(401)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return False

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                mock.requests.append(("POST", self.path, None))
                if not self._authorised():
                    return
                if mock.meta_status != 200:
                    body = b'{"error":"forced"}'
                    self.send_response(mock.meta_status)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                assert "/api/models/facebook/VGGT-Omega/paths-info/" in self.path
                body = json.dumps([{"type": "file", "path": FILE, "size": len(mock.payload),
                                    "lfs": {"oid": hashlib.sha256(mock.payload).hexdigest(),
                                            "size": len(mock.payload)}}]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                rng = self.headers.get("Range")
                mock.requests.append(("GET", self.path, rng))
                if not self._authorised():
                    return
                assert self.path.startswith("/facebook/VGGT-Omega/resolve/") and self.path.endswith(FILE)
                data, start = mock.payload, 0
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                try:
                    for i in range(start, len(data), 1 << 14):
                        self.wfile.write(data[i:i + (1 << 14)])
                        if mock.delay:
                            self.wfile.flush()
                            time.sleep(mock.delay)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"

    def gets(self):
        return [r for r in self.requests if r[0] == "GET"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    e = {k: v for k, v in os.environ.items() if not k.startswith(("HF_", "RAWATTA_", "XDG_"))}
    e.update(HOME=str(home), TMPDIR=str(tmp), PATH=f"{Path(sys.executable).parent}:{e.get('PATH', '/usr/bin:/bin')}")
    tok = tmp_path / "token"
    tok.write_text(TOKEN + "\n")
    return e, tok, tmp


def run(args, e, mock=None):
    e = dict(e)
    if mock:
        e["HF_ENDPOINT"] = mock.endpoint
    r = subprocess.run(["sh", str(SCRIPT), *args], capture_output=True, text=True, env=e, timeout=120)
    assert TOKEN not in r.stdout and TOKEN not in r.stderr  # never printed
    return r


@pytest.fixture
def mock():
    m = Mock(os.urandom(300_000))
    yield m
    m.close()


def test_help_prints_only_usage(env):
    e, _, _ = env
    r = run(["--help"], e)
    assert r.returncode == 0 and r.stdout.startswith("usage: download_weights.sh [--token-file PATH] [--dir DIR] "
                                                     "[--fp8] [--keep-pt] [--no-convert]")
    assert "set -eu" not in r.stdout and "REPO=" not in r.stdout and r.stderr == ""


def test_bad_arguments(env):
    e, _, _ = env
    assert run(["--bogus"], e).returncode == 2
    assert run(["--dir"], e).returncode == 2


def test_fresh_download_and_rerun(env, mock, tmp_path):
    e, tok, tmp = env
    out = tmp_path / "w"
    r = run(["--token-file", str(tok), "--dir", str(out), "--no-convert"], e, mock)
    assert r.returncode == 0, r.stderr
    assert (out / FILE).read_bytes() == mock.payload and "sha256 ok" in r.stdout
    assert not list(tmp.iterdir())  # the header file is gone
    assert all("/resolve/1041e80fc0e911235d3426b0a3d9a81075111a53/" in p for m, p, _ in mock.requests if m == "GET")
    n = len(mock.gets())
    r = run(["--token-file", str(tok), "--dir", str(out), "--no-convert"], e, mock)
    assert r.returncode == 0 and "already complete" in r.stdout and len(mock.gets()) == n


def test_resume_partial(env, mock, tmp_path):
    e, tok, _ = env
    out = tmp_path / "w"
    out.mkdir()
    (out / FILE).write_bytes(mock.payload[:100_000])
    r = run(["--token-file", str(tok), "--dir", str(out), "--no-convert"], e, mock)
    assert r.returncode == 0, r.stderr
    assert mock.gets()[-1][2] == "bytes=100000-" and (out / FILE).read_bytes() == mock.payload


@pytest.mark.parametrize("corrupt", ["full", "partial", "oversize"])
def test_corrupt_file_is_replaced(env, mock, tmp_path, corrupt):
    e, tok, _ = env
    out = tmp_path / "w"
    out.mkdir()
    bad = bytes(len(mock.payload))
    (out / FILE).write_bytes({"full": bad, "partial": bad[:1000], "oversize": bad + b"x"}[corrupt])
    r = run(["--token-file", str(tok), "--dir", str(out), "--no-convert"], e, mock)
    assert r.returncode == 0, r.stderr
    assert (out / FILE).read_bytes() == mock.payload
    assert ("from scratch" if corrupt == "partial" else "fails the checksum") in r.stderr


def test_token_sources(env, mock, tmp_path):
    e, _, _ = env
    out = tmp_path / "w"
    r = run(["--dir", str(out), "--no-convert"], e, mock)
    assert r.returncode == 1 and "no Hugging Face token" in r.stderr
    r = run(["--token-file", str(tmp_path / "nope"), "--dir", str(out), "--no-convert"], e, mock)
    assert r.returncode == 1 and "missing or empty" in r.stderr
    r = run(["--dir", str(out), "--no-convert"], {**e, "HF_TOKEN": TOKEN}, mock)
    assert r.returncode == 0, r.stderr
    (out / FILE).unlink()
    hf = Path(e["HOME"]) / ".cache" / "huggingface"
    hf.mkdir(parents=True)
    (hf / "token").write_text(TOKEN)
    r = run(["--dir", str(out), "--no-convert"], e, mock)
    assert r.returncode == 0, r.stderr


def test_rawatta_cache_is_the_default_folder(env, mock, tmp_path):
    e, tok, _ = env
    r = run(["--token-file", str(tok), "--no-convert"], {**e, "RAWATTA_CACHE": str(tmp_path / "c")}, mock)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "c" / FILE).read_bytes() == mock.payload


def test_bad_token(env, mock, tmp_path):
    """Hugging Face answers 401 to an invalid or expired token."""
    e, _, _ = env
    tok = tmp_path / "bad"
    tok.write_text("hf_wrong")
    r = run(["--token-file", str(tok), "--dir", str(tmp_path / "w"), "--no-convert"], e, mock)
    assert r.returncode == 1 and "rejected the token (401)" in r.stderr and not mock.gets()


def test_no_access_403(env, tmp_path):
    e, tok, _ = env
    m = Mock(b"x" * 1000, meta_status=403)
    try:
        r = run(["--token-file", str(tok), "--dir", str(tmp_path / "w"), "--no-convert"], e, m)
        assert r.returncode == 1 and "no access to facebook/VGGT-Omega (403)" in r.stderr, r.stderr
        assert "request access" in r.stderr and not m.gets()
    finally:
        m.close()


def test_other_status_is_printed(env, tmp_path):
    e, tok, _ = env
    m = Mock(b"x" * 1000, meta_status=500)
    try:
        r = run(["--token-file", str(tok), "--dir", str(tmp_path / "w"), "--no-convert"], e, m)
        assert r.returncode == 1 and "HTTP 500" in r.stderr, r.stderr
    finally:
        m.close()


def test_network_error_000(env, tmp_path):
    e, tok, _ = env
    m = Mock(b"x")
    endpoint = m.endpoint
    m.close()  # nothing listens on the port any more: curl gets no reply (status 000)
    r = run(["--token-file", str(tok), "--dir", str(tmp_path / "w"), "--no-convert"], {**e, "HF_ENDPOINT": endpoint})
    assert r.returncode == 1 and "network error" in r.stderr, r.stderr


def test_ctrl_c_stops_at_once(env, tmp_path):
    e, tok, tmp = env
    m = Mock(os.urandom(4 << 20), delay=0.05)  # about 13 s to serve
    try:
        out = tmp_path / "w"
        p = subprocess.Popen(["sh", str(SCRIPT), "--token-file", str(tok), "--dir", str(out), "--no-convert"],
                             env={**e, "HF_ENDPOINT": m.endpoint}, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True)
        t0 = time.time()
        while not ((out / FILE).is_file() and (out / FILE).stat().st_size > 0) and time.time() - t0 < 30:
            time.sleep(0.05)
        os.killpg(p.pid, signal.SIGINT)  # what Ctrl-C does: the whole foreground process group
        t1 = time.time()
        p.communicate(timeout=10)
        assert p.returncode == 130 and time.time() - t1 < 5
        assert not list(tmp.iterdir())  # header file removed
        assert len(m.gets()) == 1  # no retry loop after the interrupt
    finally:
        m.close()


def test_download_and_convert_via_symlink(env, tmp_path):
    import test_convert_weights as tcw

    import convert_weights as cw

    e, tok, _ = env
    ck, _ = tcw._handmade_state(tmp_path)
    m = Mock(ck.read_bytes())
    try:
        link_dir = tmp_path / "bin"
        link_dir.mkdir()
        (link_dir / "download-weights").symlink_to(SCRIPT)
        out = tmp_path / "w"
        w = out / "vggt_omega_1b_512.safetensors"
        e2 = {**e, "HF_ENDPOINT": m.endpoint}
        r = subprocess.run([str(link_dir / "download-weights"), "--token-file", str(tok), "--dir", str(out)],
                           capture_output=True, text=True, env=e2, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
        assert cw.check_layout(w)["format"] == "rawatta-vggt-1" and not (out / FILE).exists()
        assert "pass --weights" in r.stdout
        w.unlink()
        r = subprocess.run([str(link_dir / "download-weights"), "--token-file", str(tok), "--dir", str(out),
                            "--keep-pt"], capture_output=True, text=True, env={**e2, "RAWATTA_WEIGHTS": str(w)},
                           timeout=120)
        assert r.returncode == 0 and "RAWATTA_WEIGHTS points there" in r.stdout and (out / FILE).exists()
        default = Path(e["HOME"]) / ".cache" / "rawatta"
        r = subprocess.run([str(link_dir / "download-weights"), "--token-file", str(tok)],
                           capture_output=True, text=True, env=e2, timeout=120)
        assert r.returncode == 0 and "rawatta's default path" in r.stdout
        assert (default / "vggt_omega_1b_512.safetensors").is_file()
    finally:
        m.close()
