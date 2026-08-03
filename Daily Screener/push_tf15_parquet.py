"""Commit and push only the canonical TF15 parquet after a local update."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
PARQUET = Path("Kronos IDX FineTune 15 Minutes/data/idx_kronos_all_15m.parquet")


def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["git", "-C", str(ROOT), *arguments]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=check, text=True)


def push_parquet() -> bool:
    if not (ROOT / PARQUET).exists():
        raise FileNotFoundError(ROOT / PARQUET)
    git("add", "--", str(PARQUET))
    unchanged = git("diff", "--cached", "--quiet", "--", str(PARQUET), check=False)
    if unchanged.returncode == 0:
        print("Parquet tidak berubah; tidak ada commit/push baru.")
        return False
    if unchanged.returncode != 1:
        raise RuntimeError(f"git diff failed with exit code {unchanged.returncode}")

    stamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M WIB")
    # The pathspec keeps unrelated staged or dirty files out of this commit.
    git("commit", "-m", f"Update TF15 market data through {stamp}", "--", str(PARQUET))
    git("push", "origin", "main")
    print("Canonical TF15 parquet committed and pushed to origin/main.")
    return True


if __name__ == "__main__":
    push_parquet()
