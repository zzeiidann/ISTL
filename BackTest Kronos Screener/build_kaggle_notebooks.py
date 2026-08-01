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
            Optuna tunes one global rank-weight vector against **all valid
            rolling origins** to select 30 stocks with no chronological split.
            The reported win rate is the full-timeframe optimization objective.

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
        markdown("## 3. Run rolling inference and Optuna 1,500-trial optimization"),
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
                "--trials", "1500",
                "--backtest-sessions", "42",
                "--paths", "5",
                "--batch-size", "32",
                "--top-positive", "100",
                "--select", "30",
                "--min-universe-ratio", "0.80",
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
            from IPython.display import Javascript, display

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

            # Start downloading automatically as soon as the ZIP exists.
            archive_name = Path(archive).name
            download_url = f"/files/kaggle/working/{{archive_name}}"
            display(Javascript(
                "const link=document.createElement('a');"
                f"link.href={{json.dumps(download_url)}};"
                f"link.download={{json.dumps(archive_name)}};"
                "document.body.appendChild(link);"
                "link.click();"
                "link.remove();"
            ))
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


def build_july_31_and_august_week_screen_notebook() -> dict:
    cells = [
        markdown(
            """
            # Screen IDX 31 Juli dan 3–7 Agustus 2026

            Notebook ini menjalankan dua jadwal forecast:

            - origin **30 Juli 2026** → target **31 Juli 2026**;
            - origin **31 Juli 2026** → target **3–7 Agustus 2026**.

            Dua checkpoint yang digunakan:

            1. Validated no-refit epoch 15.
            2. Production refit epoch 4.

            Setiap checkpoint memakai `best_weights.json` miliknya sendiri dari
            `BackTest Results`. Kandidat awal adalah maksimum 100 saham dengan
            prediksi close positif. Skor akhir adalah jumlah
            `percentile_rank(feature) × weight`, lalu dipilih top 30 per model.

            Bobot berasal dari backtest 1D full-timeframe. Bobot yang sama
            diterapkan pada setiap horizon target 3–7 Agustus sesuai kebutuhan
            screening ini.

            Bobot positif menyukai nilai feature yang tinggi; bobot negatif
            menekan nilai feature yang tinggi dan relatif menyukai nilai rendah.
            """
        ),
        markdown("## 1. Clone repository dan tarik dua checkpoint"),
        code(
            f"""
            from pathlib import Path
            import os, subprocess

            REPO = Path("/kaggle/working/ISTL")
            if not REPO.exists():
                env = os.environ.copy()
                env["GIT_LFS_SKIP_SMUDGE"] = "1"
                subprocess.run(["git", "clone", "{REPOSITORY_URL}", str(REPO)], check=True, env=env)
            else:
                subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)

            MODEL_CONFIGS = [
                {{
                    "slug": "validated_no_refit_e15",
                    "label": "Validated no-refit epoch 15",
                    "checkpoint": REPO / "Kronos IDX FineTune/results/2026-07-30/validated-no-refit-e15/kronos_base_idx_all/best_model",
                    "weights": REPO / "BackTest Kronos Screener/BackTest Results/validated_no_refit_e15_1d/summary/best_weights.json",
                }},
                {{
                    "slug": "production_refit_e4",
                    "label": "Production refit epoch 4",
                    "checkpoint": REPO / "Kronos IDX FineTune/results/2026-07-30/refit-run-e4/production_model",
                    "weights": REPO / "BackTest Kronos Screener/BackTest Results/production_refit_e4_1d/summary/best_weights.json",
                }},
            ]
            includes = ",".join(str(config["checkpoint"].relative_to(REPO) / "model.safetensors") for config in MODEL_CONFIGS)
            subprocess.run(["git", "-C", str(REPO), "lfs", "pull", f"--include={{includes}}"], check=True)
            for config in MODEL_CONFIGS:
                assert (config["checkpoint"] / "model.safetensors").exists(), config["checkpoint"]
                assert config["weights"].exists(), config["weights"]
            print("Repository:", REPO)
            """
        ),
        markdown("## 2. Install runtime Kaggle P100/T4"),
        code(
            """
            %pip install -q --upgrade torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
            %pip install -q einops==0.8.1 huggingface_hub==0.33.1 safetensors==0.6.2 pyarrow optuna==4.4.0 tqdm yfinance
            """
        ),
        markdown("## 3. Siapkan data, runtime, dan arti feature"),
        code(
            """
            import gc, importlib.util, json, sys
            import numpy as np
            import pandas as pd
            import torch
            from IPython.display import display

            runner_path = REPO / "BackTest Kronos Screener/run_backtest.py"
            spec = importlib.util.spec_from_file_location("kronos_backtest", runner_path)
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)

            # Refresh actual OHLCV 31 Juli sebelum membentuk stock features.
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "Kronos IDX FineTune/update_daily_parquet.py"),
                    "--date", "2026-07-31",
                ],
                check=True,
            )

            device = runner.configure_runtime(42)
            prices = runner.load_prices(REPO)
            featured = runner.add_stock_features(prices)
            available_dates = set(featured["date"])
            JULY_30 = pd.Timestamp("2026-07-30")
            JULY_31 = pd.Timestamp("2026-07-31")
            AUGUST_TARGETS = list(pd.date_range("2026-08-03", "2026-08-07", freq="D"))
            assert JULY_30 in available_dates, "Data 30 Juli 2026 belum tersedia."

            assert JULY_31 in available_dates, "Unduhan yfinance 31 Juli tidak masuk ke parquet."

            SCREEN_JOBS = [
                {
                    "name": "2026-07-31",
                    "origin": JULY_30,
                    "target_dates": [JULY_31],
                    "horizons": [1],
                },
                {
                    "name": "2026-08-03_to_07",
                    "origin": JULY_31,
                    "target_dates": AUGUST_TARGETS,
                    "horizons": [1, 2, 3, 4, 5],
                },
            ]

            kronos_dir = REPO / "Kronos IDX FineTune/Kronos"
            if not (kronos_dir / "model/kronos.py").exists():
                kronos_dir = Path("/kaggle/working/Kronos")
                if not kronos_dir.exists():
                    subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos.git", str(kronos_dir)], check=True)
                subprocess.run(["git", "-C", str(kronos_dir), "checkout", "67b630e67f6a18c9e9be918d9b4337c960db1e9a"], check=True)
            sys.path.insert(0, str(kronos_dir))
            from model import Kronos, KronosPredictor, KronosTokenizer

            FEATURE_MEANINGS = {
                "pred_high_gain_mean": "Rata-rata estimasi kenaikan high dari 5 jalur forecast",
                "pred_high_gain_median": "Median estimasi kenaikan high",
                "pred_hit5_probability": "Proporsi jalur forecast yang memprediksi high +5%",
                "pred_close_gain_mean": "Rata-rata estimasi kenaikan close",
                "pred_close_up_probability": "Proporsi jalur yang memprediksi close naik",
                "pred_range_mean": "Estimasi lebar high-low harian",
                "pred_high_gain_dispersion": "Ketidakpastian estimasi kenaikan high",
                "pred_low_gain_mean": "Rata-rata estimasi posisi low terhadap previous close",
                "hit5_rate_20": "Frekuensi historis menyentuh +5% dalam 20 sesi",
                "hit5_rate_60": "Frekuensi historis menyentuh +5% dalam 60 sesi",
                "mom_5": "Momentum close 5 sesi",
                "mom_20": "Momentum close 20 sesi",
                "volatility_20": "Volatilitas return 20 sesi, annualized",
                "volume_ratio_5_60": "Volume rata-rata 5 sesi / median 60 sesi",
                "turnover_accel": "Akselerasi turnover 5 sesi terhadap median 60 sesi",
                "range_20": "Rentang high-low selama 20 sesi",
                "drawdown_60": "Posisi close terhadap high 60 sesi",
            }
            """
        ),
        markdown("## 4. Tampilkan bobot dan arah screening setiap model"),
        code(
            """
            weight_rows = []
            for config in MODEL_CONFIGS:
                weights = json.loads(config["weights"].read_text())
                for feature, weight in weights.items():
                    weight_rows.append({
                        "model": config["label"],
                        "feature": feature,
                        "weight": weight,
                        "arah_screen": "pilih nilai tinggi" if weight > 0 else "tekan nilai tinggi",
                        "yang_diukur": FEATURE_MEANINGS[feature],
                    })
            weight_table = pd.DataFrame(weight_rows)
            display(weight_table.sort_values(["model", "weight"], key=lambda s: s.abs() if s.name == "weight" else s, ascending=[True, False]))
            """
        ),
        markdown("## 5. Forecast dan screen top 30 per model"),
        code(
            """
            OUTPUT_DIR = Path("/kaggle/working/screen_2026-07-31_and_2026-08-03_to_07")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device).eval()
            all_scored = []
            all_selected = []

            for config in MODEL_CONFIGS:
                print("Running", config["label"])
                model = Kronos.from_pretrained(str(config["checkpoint"])).to(device).eval()
                predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=512)
                weights = json.loads(config["weights"].read_text())
                model_candidates = []
                model_selected = []
                for job in SCREEN_JOBS:
                    forecast = runner.forecast_origin(
                        predictor=predictor,
                        featured=featured,
                        origin=job["origin"],
                        future_dates=job["target_dates"],
                        horizons=job["horizons"],
                        paths=5,
                        batch_size=32,
                        lookback=120,
                        seed=42,
                    )
                    candidates = runner.prepare_candidate_panel(forecast, top_positive=100, select=30)
                    candidates["secondary_score"] = runner.score_with_weights(candidates, weights)
                    candidates["score_percentile"] = candidates.groupby("target_date")["secondary_score"].rank(pct=True, method="average")
                    candidates["model"] = config["slug"]
                    for feature in runner.SCORE_FEATURES:
                        candidates[f"contribution_{feature}"] = candidates[f"r_{feature}"] * weights[feature]
                    contribution_columns = [f"contribution_{feature}" for feature in runner.SCORE_FEATURES]
                    candidates["top_support"] = candidates[contribution_columns].idxmax(axis=1).str.removeprefix("contribution_")
                    candidates["top_penalty"] = candidates[contribution_columns].idxmin(axis=1).str.removeprefix("contribution_")

                    selected_parts = []
                    for _, daily in candidates.groupby("target_date", sort=True):
                        chosen = daily.nlargest(30, "secondary_score").copy()
                        chosen["selected_rank"] = np.arange(1, len(chosen) + 1)
                        selected_parts.append(chosen)
                    selected = pd.concat(selected_parts, ignore_index=True)
                    model_candidates.append(candidates)
                    model_selected.append(selected)

                model_candidates = pd.concat(model_candidates, ignore_index=True)
                model_selected = pd.concat(model_selected, ignore_index=True)
                model_candidates.to_csv(OUTPUT_DIR / f"{config['slug']}_all_candidates.csv", index=False)
                model_selected.to_csv(OUTPUT_DIR / f"{config['slug']}_top30_by_date.csv", index=False)
                all_scored.append(model_candidates)
                all_selected.append(model_selected)
                for target_date, daily in model_selected.groupby("target_date", sort=True):
                    print(config["label"], pd.Timestamp(target_date).date())
                    display(daily[["selected_rank", "ticker", "secondary_score", "pred_close_gain_mean", "pred_hit5_probability", "top_support", "top_penalty"]])

                del predictor, model
                gc.collect()
                torch.cuda.empty_cache()
            """
        ),
        markdown("## 6. Consensus dua model dan download output"),
        code(
            """
            scored = pd.concat(all_scored, ignore_index=True)
            selected = pd.concat(all_selected, ignore_index=True)
            consensus = (
                scored.groupby(["target_date", "ticker"], as_index=False)
                .agg(models_available=("model", "nunique"), mean_score_percentile=("score_percentile", "mean"))
            )
            selected_counts = selected.groupby(["target_date", "ticker"])["model"].nunique().rename("models_selected")
            consensus = consensus.merge(selected_counts, on=["target_date", "ticker"], how="left").fillna({"models_selected": 0})
            consensus["models_selected"] = consensus["models_selected"].astype(int)
            consensus = consensus.sort_values(
                ["target_date", "models_selected", "mean_score_percentile"],
                ascending=[True, False, False],
            )
            consensus["consensus_rank"] = consensus.groupby("target_date").cumcount() + 1
            consensus = consensus[consensus["consensus_rank"].le(30)].copy()
            consensus.to_csv(OUTPUT_DIR / "consensus_top30_by_date.csv", index=False)
            weight_table.to_csv(OUTPUT_DIR / "weight_meanings.csv", index=False)
            for target_date, daily in consensus.groupby("target_date", sort=True):
                print("Consensus", pd.Timestamp(target_date).date())
                display(daily[["consensus_rank", "ticker", "models_selected", "models_available", "mean_score_percentile"]])

            import shutil
            from IPython.display import Javascript
            archive = shutil.make_archive("/kaggle/working/screen_2026-07-31_and_2026-08-03_to_07", "zip", OUTPUT_DIR)
            print("Download:", archive)
            display(Javascript(
                "const a=document.createElement('a');"
                "a.href='/files/kaggle/working/screen_2026-07-31_and_2026-08-03_to_07.zip';"
                "a.download='screen_2026-07-31_and_2026-08-03_to_07.zip';"
                "document.body.appendChild(a);a.click();a.remove();"
            ))
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

screen_output = ROOT / "screen_2026_07_31_backtest_weights.ipynb"
screen_output.write_text(json.dumps(build_july_31_and_august_week_screen_notebook(), indent=1), encoding="utf-8")
print(screen_output)
