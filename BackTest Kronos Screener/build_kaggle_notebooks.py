from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
REPOSITORY_URL = "https://github.com/zzeiidann/ISTL.git"

MODELS = [
    {
        "slug": "validated_no_refit_e15",
        "label": "Validated no-refit epoch 15",
        "path": "Kronos IDX FineTune/results/2026-07-30/validated-no-refit-e15/kronos_base_idx_all/best_model",
    },
    {
        "slug": "refit_run_validated_e4",
        "label": "Refit run — validated epoch 4 checkpoint",
        "path": "Kronos IDX FineTune/results/2026-07-30/refit-run-e4/kronos_base_idx_all/best_model",
    },
    {
        "slug": "production_refit_e4",
        "label": "Production refit epoch 4",
        "path": "Kronos IDX FineTune/results/2026-07-30/refit-run-e4/production_model",
    },
]


def lines(text: str) -> list[str]:
    return dedent(text).strip().splitlines(True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def build_notebook(model: dict, horizon: int) -> dict:
    run_name = f"{model['slug']}_{horizon}d"
    checkpoint_file = f"{model['path']}/model.safetensors"
    horizon_description = "Day 1 only" if horizon == 1 else "Day 1 through Day 5"
    cells = [
        markdown(
            f"""
            # Kronos daily +5% screener backtest — {model['label']} — {horizon}D

            Frozen Kronos checkpoint: **{model['label']}**
            Evaluation horizons: **{horizon_description}**

            For every rolling origin in the last 42 eligible sessions, the model
            uses only the previous 120 sessions as context. Each horizon filters
            positive predicted close gain, keeps at most 100 candidates, then
            Optuna tunes one global rank-weight vector to select 30 stocks.

            A hit means actual target-day high is at least 5% above the actual
            previous trading-session close. Kronos weights are never updated.
            """
        ),
        markdown("## 1. Clone ISTL and pull only this model checkpoint from Git LFS"),
        code(
            f"""
            from pathlib import Path
            import os, subprocess

            REPO = Path("/kaggle/working/ISTL")
            if not REPO.exists():
                env = os.environ.copy()
                env["GIT_LFS_SKIP_SMUDGE"] = "1"
                subprocess.run(
                    ["git", "clone", "{REPOSITORY_URL}", str(REPO)],
                    check=True,
                    env=env,
                )
            else:
                subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)

            subprocess.run(
                ["git", "-C", str(REPO), "lfs", "pull", "--include={checkpoint_file}"],
                check=True,
            )
            checkpoint = REPO / "{model['path']}"
            assert (checkpoint / "model.safetensors").exists(), checkpoint
            print("Repository:", REPO)
            print("Checkpoint:", checkpoint)
            """
        ),
        markdown("## 2. Install P100/T4-compatible runtime"),
        code(
            """
            # CUDA 11.8 supports Kaggle Tesla P100 (sm_60) and T4 (sm_75).
            %pip install -q --upgrade torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
            %pip install -q einops==0.8.1 huggingface_hub==0.33.1 safetensors==0.6.2 pyarrow optuna==4.4.0 tqdm
            """
        ),
        markdown("## 3. Run rolling inference and Optuna 400-trial optimization"),
        code(
            f"""
            import subprocess, sys

            command = [
                sys.executable,
                str(REPO / "BackTest Kronos Screener" / "run_backtest.py"),
                "--repo", str(REPO),
                "--model-path", str(checkpoint),
                "--run-name", "{run_name}",
                "--horizon-mode", "{horizon}",
                "--trials", "400",
                "--backtest-sessions", "42",
                "--paths", "5",
                "--batch-size", "32",
                "--top-positive", "100",
                "--select", "30",
                "--lookback", "120",
                "--seed", "42",
            ]
            print("Running:", " ".join(command))
            subprocess.run(command, check=True)
            """
        ),
        markdown("## 4. Inspect and package outputs"),
        code(
            f"""
            import json, shutil
            import pandas as pd
            from IPython.display import display

            output_dir = Path("/kaggle/working/backtest_kronos_screener/{run_name}")
            summary = json.loads((output_dir / "backtest_summary.json").read_text())
            weights = json.loads((output_dir / "best_weights.json").read_text())
            metrics = pd.read_csv(output_dir / "win_rate_by_horizon.csv")

            print(json.dumps(summary, indent=2))
            display(metrics)
            display(
                pd.Series(weights, name="weight")
                .sort_values(key=abs, ascending=False)
                .rename_axis("feature")
                .to_frame()
            )

            archive = shutil.make_archive(
                f"/kaggle/working/{run_name}", "zip", output_dir
            )
            print("Download:", archive)
            """
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


for model in MODELS:
    for horizon in (1, 5):
        output = ROOT / f"backtest_{horizon}d_{model['slug']}.ipynb"
        output.write_text(json.dumps(build_notebook(model, horizon), indent=1), encoding="utf-8")
        print(output)
