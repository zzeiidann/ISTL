"""Generate the Kaggle/Colab TF15 versus TF1D daily screener notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "screen_2026_08_03_tf15_vs_tf1d.ipynb"


def lines(text: str) -> list[str]:
    return dedent(text).strip().splitlines(True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}


cells = [
    md("""
    # Daily Screener — Kronos TF15 vs TF1D (3 August 2026)

    Dua ranking Top 30 yang independen:

    1. **TF15** — return bar pertama 09:00–09:15 WIB pada 3 Agustus.
    2. **TF1D** — return close harian 3 Agustus terhadap close 31 Juli.

    Semua context berakhir 31 Juli 2026. Lima sampled paths dipakai untuk
    menghitung expected return, median, probability up, dan downside P10.
    """),
    md("## 1. Runtime setup"),
    code("""
    # Blackwell (sm_120) needs a CUDA 12.8 PyTorch wheel. Run this cell first.
    # If torch was already imported in this session, restart the kernel once
    # after installation and then Run All.
    %pip install -q --upgrade torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    %pip install -q einops==0.8.1 huggingface_hub==0.33.1 safetensors==0.6.2 pyarrow plotly tqdm
    """),
    code("""
    from pathlib import Path
    import gc, json, os, random, shutil, subprocess, sys

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import torch
    from IPython.display import display
    from tqdm.auto import tqdm

    SEED = 42
    N_PATHS = 5
    BATCH_SIZE = 32
    AS_OF = pd.Timestamp("2026-07-31 23:59:59")
    FORECAST_DATE = pd.Timestamp("2026-08-03")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    if DEVICE.type == "cuda":
        capability = torch.cuda.get_device_capability(0)
        expected_arch = f"sm_{capability[0]}{capability[1]}"
        compiled_arches = set(torch.cuda.get_arch_list())
        print({"torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0), "required_arch": expected_arch,
               "compiled_arches": sorted(compiled_arches)})
        if expected_arch not in compiled_arches:
            raise RuntimeError(
                f"PyTorch {torch.__version__} tidak menyediakan {expected_arch}. "
                "Restart Kaggle session setelah cell instalasi, lalu Run All dari awal."
            )
        smoke = torch.ones(8, device=DEVICE)
        torch.cuda.synchronize()
        del smoke
        torch.cuda.empty_cache()
        print("CUDA architecture smoke test passed.")

    WORKSPACE = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/content") if Path("/content").exists() else Path.cwd()
    REPO = Path.cwd().resolve() if Path.cwd().name == "SIER" and (Path.cwd() / ".git").exists() else WORKSPACE / "SIER"
    if not (REPO / ".git").exists():
        clone_env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(["git", "clone", "https://github.com/zzeiidann/SIER.git", str(REPO)], check=True, env=clone_env)
    else:
        subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only", "origin", "main"], check=True)

    LFS_INCLUDE = (
        "Kronos IDX FineTune/data/idx_kronos_all_daily.parquet,"
        "Kronos IDX FineTune 15 Minutes/data/idx_kronos_all_15m.parquet,"
        "Kronos IDX FineTune 15 Minutes/results/2026-07-30/refit-run-e4/production_model/**,"
        "Kronos IDX FineTune/results/2026-07-31/tokenizer-idx-e10-predictor-e4/**"
    )
    subprocess.run(["git", "-C", str(REPO), "lfs", "install", "--local"], check=True)
    subprocess.run(["git", "-C", str(REPO), "lfs", "pull", "--include", LFS_INCLUDE], check=True)
    print("Repository and required LFS files ready:", REPO)
    DATA_15M = REPO / "Kronos IDX FineTune 15 Minutes/data/idx_kronos_all_15m.parquet"
    DATA_1D = REPO / "Kronos IDX FineTune/data/idx_kronos_all_daily.parquet"
    DAILY_RUN = REPO / "Kronos IDX FineTune/results/2026-07-31/tokenizer-idx-e10-predictor-e4"
    DAILY_MODEL = DAILY_RUN / "kronos_base_idx_all/best_model"
    DAILY_TOKENIZER = DAILY_RUN / "tokenizer_idx_best"
    KRONOS_DIR = REPO / "Kronos IDX FineTune/Kronos"
    REPO_MODEL_15M = REPO / "Kronos IDX FineTune 15 Minutes/results/2026-07-30/refit-run-e4/production_model"

    RUNTIME = Path("/kaggle/working/daily_screener") if Path("/kaggle/working").exists() else Path("/content/daily_screener") if Path("/content").exists() else REPO / "Daily Screener/outputs"
    RUNTIME.mkdir(parents=True, exist_ok=True)
    KRONOS_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
    if not (KRONOS_DIR / "model/kronos.py").exists():
        # A normal GitHub clone does not populate the Kronos submodule. Clone the
        # pinned upstream source explicitly so Kaggle/Colab never depends on
        # submodule state in the parent repository.
        KRONOS_DIR = RUNTIME / "Kronos-runtime"
        if not (KRONOS_DIR / "model/kronos.py").exists():
            if KRONOS_DIR.exists():
                shutil.rmtree(KRONOS_DIR)
            subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos.git", str(KRONOS_DIR)], check=True)
            subprocess.run(["git", "-C", str(KRONOS_DIR), "checkout", KRONOS_COMMIT], check=True)
        print("Loaded pinned upstream Kronos source:", KRONOS_DIR)

    MODEL_15M = REPO_MODEL_15M

    for required in (DATA_15M, DATA_1D, DAILY_MODEL / "model.safetensors", DAILY_TOKENIZER / "model.safetensors", MODEL_15M / "model.safetensors", KRONOS_DIR / "model/kronos.py"):
        if not required.exists(): raise FileNotFoundError(required)
    sys.path.insert(0, str(KRONOS_DIR))
    from model import Kronos, KronosTokenizer, KronosPredictor
    print({"device": str(DEVICE), "repo": str(REPO), "model_15m": str(MODEL_15M), "output": str(RUNTIME)})
    """),
    md("## 2. Load and audit the two timeframes"),
    code("""
    FEATURES = ["open", "high", "low", "close", "volume", "amount"]

    def load_prices(path):
        frame = pd.read_parquet(path)
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
        frame["date"] = pd.to_datetime(frame["date"])
        if frame["date"].dt.tz is not None:
            frame["date"] = frame["date"].dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
        frame = frame[frame.date <= AS_OF].sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
        frame["volume"] = frame["volume"].fillna(0).clip(lower=0)
        frame["amount"] = frame.get("amount", frame[["open", "high", "low", "close"]].mean(axis=1) * frame["volume"])
        return frame.dropna(subset=["open", "high", "low", "close"])

    prices_15m, prices_1d = load_prices(DATA_15M), load_prices(DATA_1D)
    common = sorted(set(prices_15m.ticker) & set(prices_1d.ticker))
    print(f"TF15: {len(prices_15m):,} rows / {prices_15m.ticker.nunique()} ticker / last {prices_15m.date.max()}")
    print(f"TF1D: {len(prices_1d):,} rows / {prices_1d.ticker.nunique()} ticker / last {prices_1d.date.max()}")
    print(f"Common universe: {len(common)}")
    """),
    md("## 3. Shared causal multi-step inference"),
    code("""
    def future_schedule(frame, intraday):
        if not intraday:
            return pd.Series(pd.bdate_range(FORECAST_DATE, periods=2))
        observed = frame.assign(clock=frame.date.dt.strftime("%H:%M"))
        clock_frequency = observed.groupby("clock")["date"].count().sort_values(ascending=False)
        clocks = sorted(clock_frequency.head(20).index)
        if len(clocks) < 20:
            raise RuntimeError(f"Observed intraday schedule hanya memiliki {len(clocks)} clock bars")
        values = []
        for day in pd.bdate_range(FORECAST_DATE, periods=2):
            values.extend(day + pd.Timedelta(hours=int(x[:2]), minutes=int(x[3:])) for x in clocks)
        return pd.Series(values)

    def build_contexts(frame, tickers, lookback, schedule):
        contexts, x_times, y_times, names, anchors, skipped = [], [], [], [], {}, []
        for ticker in tickers:
            g = frame[frame.ticker.eq(ticker)].tail(lookback)
            if len(g) < lookback:
                skipped.append((ticker, len(g))); continue
            contexts.append(g[FEATURES].copy())
            x_times.append(pd.Series(g.date.to_numpy()))
            y_times.append(schedule.copy())
            names.append(ticker); anchors[ticker] = float(g.close.iloc[-1])
        return contexts, x_times, y_times, names, anchors, skipped

    def rank_paths(paths, step, label):
        selected = paths[paths.horizon_step.eq(step)]
        ranking = selected.groupby("ticker").agg(
            anchor_close=("anchor_close", "first"), expected_close=("forecast_close", "mean"),
            expected_return=("return", "mean"), median_return=("return", "median"),
            probability_up=("return", lambda x: float((x > 0).mean())),
            downside_p10=("return", lambda x: float(np.quantile(x, 0.10))),
            dispersion=("return", "std"),
        ).reset_index().sort_values(["expected_return", "probability_up"], ascending=False)
        ranking["rank"] = np.arange(1, len(ranking) + 1)
        ranking["scope"] = label
        return ranking

    def screen_horizon(model_path, tokenizer_path, frame, tickers, lookback, intraday, label):
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path)).to(DEVICE).eval()
        model = Kronos.from_pretrained(str(model_path)).to(DEVICE).eval()
        predictor = KronosPredictor(model, tokenizer, device=str(DEVICE), max_context=512)
        schedule = future_schedule(frame, intraday)
        contexts, x_times, y_times, names, anchors, skipped = build_contexts(frame, tickers, lookback, schedule)
        rows = []
        with torch.inference_mode():
            for path_id in range(N_PATHS):
                torch.manual_seed(SEED + 10_000 * (1 if intraday else 2) + path_id)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + 10_000 * (1 if intraday else 2) + path_id)
                for start in tqdm(range(0, len(names), BATCH_SIZE), desc=f"{label} path {path_id+1}"):
                    stop = start + BATCH_SIZE
                    preds = predictor.predict_batch(
                        df_list=contexts[start:stop], x_timestamp_list=x_times[start:stop],
                        y_timestamp_list=y_times[start:stop], pred_len=len(schedule), T=0.8,
                        top_p=0.9, top_k=0, sample_count=1, verbose=False,
                    )
                    for ticker, pred in zip(names[start:stop], preds):
                        for step in ([1, 21] if intraday else [1, 2]):
                            close = float(pred["close"].iloc[step - 1])
                            rows.append({"ticker": ticker, "path_id": path_id, "horizon_step": step,
                                         "forecast_time": schedule.iloc[step - 1], "forecast_close": close,
                                         "anchor_close": anchors[ticker], "return": close / anchors[ticker] - 1})
        paths = pd.DataFrame(rows)
        rankings = {
            "today": rank_paths(paths, 1, f"{label} 2026-08-03"),
            "tomorrow": rank_paths(paths, 21 if intraday else 2, f"{label} 2026-08-04"),
        }
        del predictor, model, tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return rankings, paths, skipped
    """),
    md("## 4. TF15 — first 15-minute bars of 3 and 4 August"),
    code("""
    tokenizer_15m_id = "NeoQuasar/Kronos-Tokenizer-base"
    ranks_15m, paths_15m, skipped_15m = screen_horizon(
        MODEL_15M, tokenizer_15m_id, prices_15m, common, 240, True, "TF15"
    )
    top30_15m, top30_15m_tomorrow = ranks_15m["today"].head(30).copy(), ranks_15m["tomorrow"].head(30).copy()
    print("TF15 TODAY — first bar 3 August"); display(top30_15m)
    print("TF15 TOMORROW — first bar 4 August (causal step 21)"); display(top30_15m_tomorrow)
    """),
    md("## 5. TF1D — daily bars of 3 and 4 August"),
    code("""
    ranks_1d, paths_1d, skipped_1d = screen_horizon(
        DAILY_MODEL, DAILY_TOKENIZER, prices_1d, common, 120, False, "TF1D"
    )
    top30_1d, top30_1d_tomorrow = ranks_1d["today"].head(30).copy(), ranks_1d["tomorrow"].head(30).copy()
    print("TF1D TODAY — 3 August"); display(top30_1d)
    print("TF1D TOMORROW — 4 August (causal step 2)"); display(top30_1d_tomorrow)
    """),
    md("## 6. Compare TF15 and TF1D for each date"),
    code("""
    def compare_lists(tf15, tf1d):
        out = tf15[["ticker","rank","expected_return","probability_up"]].rename(columns={"rank":"rank_15m","expected_return":"return_15m","probability_up":"p_up_15m"}).merge(
            tf1d[["ticker","rank","expected_return","probability_up"]].rename(columns={"rank":"rank_1d","expected_return":"return_1d","probability_up":"p_up_1d"}), on="ticker", how="outer")
        out["in_both_top30"] = out.rank_15m.notna() & out.rank_1d.notna()
        return out.sort_values(["in_both_top30","rank_15m","rank_1d"], ascending=[False,True,True])

    comparison = compare_lists(top30_15m, top30_1d)
    comparison_tomorrow = compare_lists(top30_15m_tomorrow, top30_1d_tomorrow)
    print("3 August overlap:", int(comparison.in_both_top30.sum())); display(comparison)
    print("4 August overlap:", int(comparison_tomorrow.in_both_top30.sum())); display(comparison_tomorrow)

    chart_data = pd.concat([top30_15m_tomorrow, top30_1d_tomorrow])
    fig = px.bar(chart_data, x="expected_return", y="ticker", color="scope", barmode="group",
                 orientation="h", title="4 August 2026 — TF15 First Bar vs TF1D", template="plotly_white")
    fig.update_layout(height=1000, yaxis={"categoryorder":"total ascending"}, xaxis_tickformat="+.1%")
    fig.show()
    """),
    md("## 7. Save outputs"),
    code("""
    top30_15m.to_csv(RUNTIME / "top30_tf15_first_bar_2026_08_03.csv", index=False)
    top30_1d.to_csv(RUNTIME / "top30_tf1d_2026_08_03.csv", index=False)
    top30_15m_tomorrow.to_csv(RUNTIME / "top30_tf15_first_bar_2026_08_04.csv", index=False)
    top30_1d_tomorrow.to_csv(RUNTIME / "top30_tf1d_2026_08_04.csv", index=False)
    comparison.to_csv(RUNTIME / "comparison_tf15_vs_tf1d_2026_08_03.csv", index=False)
    comparison_tomorrow.to_csv(RUNTIME / "comparison_tf15_vs_tf1d_2026_08_04.csv", index=False)
    paths_15m.to_parquet(RUNTIME / "paths_tf15_first_bar.parquet", index=False)
    paths_1d.to_parquet(RUNTIME / "paths_tf1d_today.parquet", index=False)
    metadata = {
        "forecast_dates": ["2026-08-03", "2026-08-04"], "context_end": str(AS_OF),
        "tf15_model": str(MODEL_15M), "tf15_lookback_bars": 240, "tf15_horizon_bars": 21,
        "tf1d_model": str(DAILY_MODEL), "tf1d_tokenizer": str(DAILY_TOKENIZER),
        "tf1d_lookback_bars": 120, "tf1d_horizon_bars": 2,
        "sample_paths": N_PATHS, "common_universe": len(common),
        "eligible_tf15": len(ranks_15m["today"]), "eligible_tf1d": len(ranks_1d["today"]),
        "top30_overlap": int(comparison.in_both_top30.sum()),
        "top30_overlap_tomorrow": int(comparison_tomorrow.in_both_top30.sum()),
        "warning": "Statistical forecasts, not investment advice.",
    }
    (RUNTIME / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    archive = shutil.make_archive(str(RUNTIME), "zip", RUNTIME)
    print("Outputs:", RUNTIME)
    print("Archive:", archive)
    for path in sorted(RUNTIME.iterdir()): print("-", path.name)
    """),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"gpuType": "A100", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
