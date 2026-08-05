"""Kaggle-side entrypoint for full-universe TF15 inference."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPOSITORY = "https://github.com/zzeiidann/SIER.git"
REPO = Path("/tmp/SIER")
OUTPUT = Path("/kaggle/working/tf15_results")


def run(command: list[str], retries: int = 1, reset_path: Path | None = None, **kwargs) -> None:
    for attempt in range(1, retries + 1):
        if attempt > 1 and reset_path and reset_path.exists():
            shutil.rmtree(reset_path)
        print(f"+ [{attempt}/{retries}]", " ".join(command), flush=True)
        try:
            subprocess.run(command, check=True, **kwargs)
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise
            delay = min(30, 5 * attempt)
            print(f"Command failed; retrying in {delay}s...", flush=True)
            time.sleep(delay)


def main() -> None:
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "einops==0.8.1", "huggingface-hub==0.33.1", "safetensors==0.6.2",
        "pandas>=2.2,<3", "pyarrow>=16", "tqdm>=4.66",
    ])
    if REPO.exists():
        shutil.rmtree(REPO)
    clone_environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    run(
        ["git", "clone", "--depth", "1", REPOSITORY, str(REPO)],
        retries=5, reset_path=REPO, env=clone_environment,
    )
    run(["git", "-C", str(REPO), "lfs", "install", "--local"])
    run([
        "git", "-C", str(REPO), "lfs", "pull", "--include",
        "Kronos IDX FineTune 15 Minutes/results/2026-07-30/refit-run-e4/production_model/**",
    ], retries=5)

    daily = REPO / "Daily Screener"
    sys.path.insert(0, str(daily))
    from project_tf15_next_session import run_projection

    ranking, _, metadata = run_projection(
        output_dir=OUTPUT,
        max_tickers=None,
        lookback=240,
        paths=5,
        batch_size=16,
        target_date=None,
    )
    print(metadata)
    print(ranking.head(30).to_string(index=False))
    print("Kaggle output:", OUTPUT)


if __name__ == "__main__":
    main()
