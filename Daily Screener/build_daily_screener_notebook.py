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
    %pip install -q einops==0.8.1 huggingface_hub==0.33.1 safetensors==0.6.2 pyarrow plotly tqdm
    """),
    code("""
    from pathlib import Path
    import gc, json, os, random, shutil, sys, zipfile

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

    if Path("/content").exists() and not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as exc:
            print("Drive mount skipped:", exc)

    SEARCH_ROOTS = [Path.cwd(), Path("/kaggle/working"), Path("/kaggle/input"), Path("/content"), Path("/content/drive/MyDrive")]
    SEARCH_ROOTS = [p for p in SEARCH_ROOTS if p.exists()]

    def find_named(name, want_dir=False):
        for root in SEARCH_ROOTS:
            if root.name == name and root.is_dir() == want_dir:
                return root.resolve()
            direct = root / name
            if direct.exists() and direct.is_dir() == want_dir:
                return direct.resolve()
            for hit in root.glob(f"**/{name}"):
                if hit.is_dir() == want_dir:
                    return hit.resolve()
        raise FileNotFoundError(f"{name} tidak ditemukan di Kaggle/Colab/local search roots")

    REPO = find_named("ISTL", want_dir=True)
    DATA_15M = REPO / "Kronos IDX FineTune 15 Minutes/data/idx_kronos_all_15m.parquet"
    DATA_1D = REPO / "Kronos IDX FineTune/data/idx_kronos_all_daily.parquet"
    DAILY_RUN = REPO / "Kronos IDX FineTune/results/2026-07-31/tokenizer-idx-e10-predictor-e4"
    DAILY_MODEL = DAILY_RUN / "kronos_base_idx_all/best_model"
    DAILY_TOKENIZER = DAILY_RUN / "tokenizer_idx_best"
    KRONOS_DIR = REPO / "Kronos IDX FineTune/Kronos"
    ZIP_15M = find_named("kronos_idx_15m_outputs.zip")

    RUNTIME = Path("/kaggle/working/daily_screener") if Path("/kaggle/working").exists() else Path("/content/daily_screener") if Path("/content").exists() else REPO / "Daily Screener/outputs"
    RUNTIME.mkdir(parents=True, exist_ok=True)
    MODEL_15M = RUNTIME / "model_15m"
    if not (MODEL_15M / "model.safetensors").exists():
        with zipfile.ZipFile(ZIP_15M) as archive:
            for member in ("production_model/model.safetensors", "production_model/config.json", "production_model/README.md"):
                archive.extract(member, RUNTIME)
        extracted = RUNTIME / "production_model"
        if MODEL_15M.exists(): shutil.rmtree(MODEL_15M)
        extracted.rename(MODEL_15M)

    for required in (DATA_15M, DATA_1D, DAILY_MODEL / "model.safetensors", DAILY_TOKENIZER / "model.safetensors", MODEL_15M / "model.safetensors"):
        if not required.exists(): raise FileNotFoundError(required)
    sys.path.insert(0, str(KRONOS_DIR))
    from model import Kronos, KronosTokenizer, KronosPredictor
    print({"device": str(DEVICE), "repo": str(REPO), "zip_15m": str(ZIP_15M), "output": str(RUNTIME)})
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
    md("## 3. Shared one-step inference"),
    code("""
    def build_contexts(frame, tickers, lookback, intraday):
        contexts, x_times, y_times, names, anchors, skipped = [], [], [], [], {}, []
        target = pd.Timestamp("2026-08-03 09:00:00") if intraday else FORECAST_DATE
        for ticker in tickers:
            g = frame[frame.ticker.eq(ticker)].tail(lookback)
            if len(g) < lookback:
                skipped.append((ticker, len(g))); continue
            contexts.append(g[FEATURES].copy())
            x_times.append(pd.Series(g.date.to_numpy()))
            y_times.append(pd.Series([target]))
            names.append(ticker); anchors[ticker] = float(g.close.iloc[-1])
        return contexts, x_times, y_times, names, anchors, skipped

    def screen_one(model_path, tokenizer_path, frame, tickers, lookback, intraday, label):
        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path)).to(DEVICE).eval()
        model = Kronos.from_pretrained(str(model_path)).to(DEVICE).eval()
        predictor = KronosPredictor(model, tokenizer, device=str(DEVICE), max_context=512)
        contexts, x_times, y_times, names, anchors, skipped = build_contexts(frame, tickers, lookback, intraday)
        rows = []
        with torch.inference_mode():
            for path_id in range(N_PATHS):
                torch.manual_seed(SEED + 10_000 * (1 if intraday else 2) + path_id)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + 10_000 * (1 if intraday else 2) + path_id)
                for start in tqdm(range(0, len(names), BATCH_SIZE), desc=f"{label} path {path_id+1}"):
                    stop = start + BATCH_SIZE
                    preds = predictor.predict_batch(
                        df_list=contexts[start:stop], x_timestamp_list=x_times[start:stop],
                        y_timestamp_list=y_times[start:stop], pred_len=1, T=0.8,
                        top_p=0.9, top_k=0, sample_count=1, verbose=False,
                    )
                    for ticker, pred in zip(names[start:stop], preds):
                        close = float(pred["close"].iloc[0])
                        rows.append({"ticker": ticker, "path_id": path_id, "forecast_close": close,
                                     "anchor_close": anchors[ticker], "return": close / anchors[ticker] - 1})
        paths = pd.DataFrame(rows)
        ranking = paths.groupby("ticker").agg(
            anchor_close=("anchor_close", "first"), expected_close=("forecast_close", "mean"),
            expected_return=("return", "mean"), median_return=("return", "median"),
            probability_up=("return", lambda x: float((x > 0).mean())),
            downside_p10=("return", lambda x: float(np.quantile(x, 0.10))),
            dispersion=("return", "std"),
        ).reset_index().sort_values(["expected_return", "probability_up"], ascending=False)
        ranking["rank"] = np.arange(1, len(ranking) + 1)
        ranking["timeframe"] = label
        del predictor, model, tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return ranking, paths, skipped
    """),
    md("## 4. TF15 — first 15-minute bar of 3 August"),
    code("""
    tokenizer_15m_id = "NeoQuasar/Kronos-Tokenizer-base"
    rank_15m, paths_15m, skipped_15m = screen_one(
        MODEL_15M, tokenizer_15m_id, prices_15m, common, 240, True, "TF15"
    )
    top30_15m = rank_15m.head(30).copy()
    display(top30_15m.style.format({"anchor_close":"{:,.0f}", "expected_close":"{:,.2f}", "expected_return":"{:+.2%}", "median_return":"{:+.2%}", "probability_up":"{:.0%}", "downside_p10":"{:+.2%}", "dispersion":"{:.2%}"}))
    """),
    md("## 5. TF1D — 3 August daily bar"),
    code("""
    rank_1d, paths_1d, skipped_1d = screen_one(
        DAILY_MODEL, DAILY_TOKENIZER, prices_1d, common, 120, False, "TF1D"
    )
    top30_1d = rank_1d.head(30).copy()
    display(top30_1d.style.format({"anchor_close":"{:,.0f}", "expected_close":"{:,.2f}", "expected_return":"{:+.2%}", "median_return":"{:+.2%}", "probability_up":"{:.0%}", "downside_p10":"{:+.2%}", "dispersion":"{:.2%}"}))
    """),
    md("## 6. Compare the two Top 30 lists"),
    code("""
    comparison = top30_15m[["ticker","rank","expected_return","probability_up","downside_p10"]].rename(columns={
        "rank":"rank_15m", "expected_return":"return_15m", "probability_up":"p_up_15m", "downside_p10":"p10_15m"
    }).merge(top30_1d[["ticker","rank","expected_return","probability_up","downside_p10"]].rename(columns={
        "rank":"rank_1d", "expected_return":"return_1d", "probability_up":"p_up_1d", "downside_p10":"p10_1d"
    }), on="ticker", how="outer")
    comparison["in_both_top30"] = comparison.rank_15m.notna() & comparison.rank_1d.notna()
    comparison = comparison.sort_values(["in_both_top30","rank_15m","rank_1d"], ascending=[False,True,True])
    print("Overlap Top 30:", int(comparison.in_both_top30.sum()))
    display(comparison.style.format({"return_15m":"{:+.2%}","p_up_15m":"{:.0%}","p10_15m":"{:+.2%}","return_1d":"{:+.2%}","p_up_1d":"{:.0%}","p10_1d":"{:+.2%}"}))

    chart_data = pd.concat([top30_15m.assign(scope="TF15 first bar"), top30_1d.assign(scope="TF1D today")])
    fig = px.bar(chart_data, x="expected_return", y="ticker", color="scope", barmode="group",
                 orientation="h", title="3 August 2026 — TF15 First Bar vs TF1D", template="plotly_white")
    fig.update_layout(height=1000, yaxis={"categoryorder":"total ascending"}, xaxis_tickformat="+.1%")
    fig.show()
    """),
    md("## 7. Save outputs"),
    code("""
    top30_15m.to_csv(RUNTIME / "top30_tf15_first_bar_2026_08_03.csv", index=False)
    top30_1d.to_csv(RUNTIME / "top30_tf1d_2026_08_03.csv", index=False)
    comparison.to_csv(RUNTIME / "comparison_tf15_vs_tf1d_2026_08_03.csv", index=False)
    paths_15m.to_parquet(RUNTIME / "paths_tf15_first_bar.parquet", index=False)
    paths_1d.to_parquet(RUNTIME / "paths_tf1d_today.parquet", index=False)
    metadata = {
        "forecast_date": str(FORECAST_DATE.date()), "context_end": str(AS_OF),
        "tf15_model": str(MODEL_15M), "tf15_lookback_bars": 240, "tf15_horizon_bars": 1,
        "tf1d_model": str(DAILY_MODEL), "tf1d_tokenizer": str(DAILY_TOKENIZER),
        "tf1d_lookback_bars": 120, "tf1d_horizon_bars": 1,
        "sample_paths": N_PATHS, "common_universe": len(common),
        "eligible_tf15": len(rank_15m), "eligible_tf1d": len(rank_1d),
        "top30_overlap": int(comparison.in_both_top30.sum()),
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
