"""Submit TF15 inference to Kaggle T4, monitor it, and download its outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env.kaggle.local")
RUNNER = HERE / "kaggle_tf15_runner.py"
BUILD = HERE / ".runtime/kaggle-tf15"
OUTPUT = HERE / "outputs/kaggle"
SLUG = "istl-tf15-full-universe"
KAGGLE = str(Path(sys.executable).with_name("kaggle"))


def configured_username() -> str:
    return os.environ.get("KAGGLE_USERNAME", "").strip()


def execute(command: list[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def authenticated() -> bool:
    probe = subprocess.run([KAGGLE, "kernels", "list", "--mine", "--page-size", "1"], text=True, capture_output=True)
    return probe.returncode == 0


def prepare_kernel(username: str) -> str:
    kernel_id = f"{username}/{SLUG}"
    BUILD.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNNER, BUILD / RUNNER.name)
    metadata = {
        "id": kernel_id,
        "title": "ISTL TF15 Full Universe",
        "code_file": RUNNER.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (BUILD / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return kernel_id


def status(kernel_id: str) -> tuple[str, str]:
    result = execute([KAGGLE, "kernels", "status", kernel_id], capture=True)
    message = (result.stdout + "\n" + result.stderr).strip()
    lowered = message.lower()
    # New Kaggle CLI formats states as KernelWorkerStatus.RUNNING/ERROR.
    # Check terminal keywords in the full response before any generic parsing.
    if "error" in lowered or "failed" in lowered:
        state = "error"
    elif "complete" in lowered:
        state = "complete"
    elif "running" in lowered:
        state = "running"
    elif "queued" in lowered:
        state = "queued"
    elif "cancel" in lowered:
        state = "cancelled"
    else:
        match = re.search(r'has status[ :]?[ "\']*([a-z.]+)', lowered)
        state = match.group(1) if match else lowered
    return state, message


def submit_and_wait(username: str, poll_seconds: int = 30, timeout_minutes: int = 720) -> Path:
    if not authenticated():
        raise RuntimeError(
            "Kaggle CLI belum login. Jalankan `uv run kaggle auth login` satu kali, "
            "selesaikan otorisasi browser, lalu ulangi cell ini."
        )
    kernel_id = prepare_kernel(username)
    execute([KAGGLE, "kernels", "push", "-p", str(BUILD), "--accelerator", "NvidiaTeslaT4"])
    deadline = time.monotonic() + timeout_minutes * 60
    while time.monotonic() < deadline:
        state, message = status(kernel_id)
        print(message, flush=True)
        if any(word in state for word in ("complete", "completed")):
            break
        if any(word in state for word in ("error", "failed", "cancel")):
            raise RuntimeError(f"Kaggle job stopped with status: {message}")
        time.sleep(poll_seconds)
    else:
        raise TimeoutError(f"Kaggle job did not finish within {timeout_minutes} minutes")

    destination = OUTPUT / time.strftime("%Y%m%d-%H%M%S")
    destination.mkdir(parents=True, exist_ok=True)
    execute([KAGGLE, "kernels", "output", kernel_id, "-p", str(destination), "-o"])
    print("Downloaded Kaggle results:", destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default=configured_username())
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if not args.username:
        parser.error("Provide --username YOUR_KAGGLE_USERNAME or set KAGGLE_USERNAME")
    submit_and_wait(args.username, args.poll_seconds)


if __name__ == "__main__":
    main()
