"""Native desktop launcher for BetterCroLidar."""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn

from .server import app


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_ready(url: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.15)
    return False


def main() -> int:
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}/"

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    if not _wait_ready(f"http://127.0.0.1:{port}/api/info"):
        print("Failed to start BetterCroLidar backend.", file=sys.stderr)
        return 1

    print(f"BetterCroLidar running at {url}")

    # Prefer pywebview if installed; otherwise launch chromeless desktop app window
    try:
        import webview  # type: ignore

        webview.create_window("BetterCroLidar — Croatia LiDAR Studio", url, width=1480, height=920)
        webview.start()
        server.should_exit = True
        return 0
    except ImportError:
        pass

    browser = (
        shutil.which("brave-browser")
        or shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if browser:
        profile_dir = tempfile.mkdtemp(prefix="better_cro_lidar_")
        proc = subprocess.Popen(
            [
                browser,
                f"--app={url}",
                f"--user-data-dir={profile_dir}",
                "--class=BetterCroLidar",
                "--window-size=1480,920",
                "--no-first-run",
            ]
        )
        proc.wait()
        shutil.rmtree(profile_dir, ignore_errors=True)
        server.should_exit = True
        return 0

    # Fallback: open default browser and keep server running
    subprocess.Popen(["xdg-open", url])
    try:
        while t.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.should_exit = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
