"""Run local CPU/GPU inference for the next IDX session with the TF15 model."""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Kronos IDX FineTune 15 Minutes/data/idx_kronos_all_15m.parquet"
MODEL = ROOT / "Kronos IDX FineTune 15 Minutes/results/2026-07-30/refit-run-e4/production_model"
OUTPUT = ROOT / "Daily Screener/outputs"
KRONOS_LOCAL = ROOT / "Kronos IDX FineTune/Kronos"
KRONOS_RUNTIME = ROOT / "Daily Screener/.runtime/Kronos"
KRONOS_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def ensure_kronos_source() -> Path:
    if (KRONOS_LOCAL / "model/kronos.py").exists():
        return KRONOS_LOCAL
    if not (KRONOS_RUNTIME / "model/kronos.py").exists():
        KRONOS_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
        if KRONOS_RUNTIME.exists():
            shutil.rmtree(KRONOS_RUNTIME)
        subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos.git", str(KRONOS_RUNTIME)], check=True)
        subprocess.run(["git", "-C", str(KRONOS_RUNTIME), "checkout", KRONOS_COMMIT], check=True)
    return KRONOS_RUNTIME


def next_weekday(value: pd.Timestamp) -> pd.Timestamp:
    result = value.normalize() + pd.Timedelta(days=1)
    while result.weekday() >= 5:
        result += pd.Timedelta(days=1)
    return result


def session_schedule(prices: pd.DataFrame, target: pd.Timestamp) -> pd.Series:
    reference = prices.assign(day=prices.date.dt.normalize(), weekday=prices.date.dt.weekday)
    matching = reference[reference.weekday.eq(target.weekday())]
    counts = matching.groupby("day").size().sort_index()
    if counts.empty:
        raise RuntimeError(f"No historical weekday={target.weekday()} schedule is available")
    reference_day = counts.index[-1]
    clocks = sorted(matching.loc[matching.day.eq(reference_day), "date"].dt.strftime("%H:%M").unique())
    return pd.Series([target + pd.Timedelta(hours=int(clock[:2]), minutes=int(clock[3:])) for clock in clocks])


def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_parquet(path)
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip().str.removesuffix(".JK")
    prices["date"] = pd.to_datetime(prices["date"])
    if prices.date.dt.tz is not None:
        prices["date"] = prices.date.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
    prices = prices.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    prices["volume"] = prices.volume.fillna(0).clip(lower=0)
    prices["amount"] = prices.get("amount", prices[["open", "high", "low", "close"]].mean(axis=1) * prices.volume)
    return prices.dropna(subset=["open", "high", "low", "close"])


def liquid_candidates(prices: pd.DataFrame, count: int, lookback_sessions: int = 20) -> list[str]:
    last_days = sorted(prices.date.dt.normalize().unique())[-lookback_sessions:]
    recent = prices[prices.date.dt.normalize().isin(last_days)]
    minimum_bars = max(1, len(last_days) * 8)
    stats = recent.groupby("ticker").agg(median_amount=("amount", "median"), bars=("date", "size"))
    return stats[stats.bars.ge(minimum_bars)].nlargest(count, "median_amount").index.tolist()


def run_projection(
    data_path: Path = DATA,
    model_path: Path = MODEL,
    output_dir: Path = OUTPUT,
    candidate_count: int = 30,
    lookback: int = 240,
    paths: int = 3,
    batch_size: int | None = None,
    target_date: str | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or (16 if device.type == "cuda" else 2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    prices = load_prices(data_path)
    context_end = prices.date.max()
    target = pd.Timestamp(target_date) if target_date else next_weekday(context_end)
    if target.normalize() <= context_end.normalize():
        raise ValueError("target_date must be after the final actual context date")
    schedule = session_schedule(prices, target)
    candidates = liquid_candidates(prices, candidate_count)

    contexts, x_times, names, anchors = [], [], [], {}
    for ticker in candidates:
        context = prices[prices.ticker.eq(ticker)].tail(lookback)
        if len(context) < lookback:
            continue
        contexts.append(context[FEATURES].copy())
        x_times.append(pd.Series(context.date.to_numpy()))
        names.append(ticker)
        anchors[ticker] = float(context.close.iloc[-1])
    if not names:
        raise RuntimeError("No candidate has enough context bars")

    source = ensure_kronos_source()
    sys.path.insert(0, str(source))
    from model import Kronos, KronosPredictor, KronosTokenizer

    print({"device": str(device), "context_end": str(context_end), "target": str(target.date()), "candidates": len(names)})
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER).to(device).eval()
    model = Kronos.from_pretrained(str(model_path)).to(device).eval()
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=512)
    rows: list[dict] = []
    y_times = [schedule.copy() for _ in names]
    with torch.inference_mode():
        for path_id in range(paths):
            torch.manual_seed(seed + path_id)
            for start in tqdm(range(0, len(names), batch_size), desc=f"path {path_id + 1}/{paths}"):
                stop = start + batch_size
                predictions = predictor.predict_batch(
                    df_list=contexts[start:stop],
                    x_timestamp_list=x_times[start:stop],
                    y_timestamp_list=y_times[start:stop],
                    pred_len=len(schedule),
                    T=0.8,
                    top_p=0.9,
                    top_k=0,
                    sample_count=1,
                    verbose=False,
                )
                for ticker, prediction in zip(names[start:stop], predictions):
                    for step, (forecast_time, close) in enumerate(zip(schedule, prediction["close"]), 1):
                        close = float(close)
                        rows.append(
                            {"ticker": ticker, "path_id": path_id, "horizon_step": step,
                             "forecast_time": forecast_time, "forecast_close": close,
                             "anchor_close": anchors[ticker], "return": close / anchors[ticker] - 1}
                        )

    forecasts = pd.DataFrame(rows)
    first = forecasts[forecasts.horizon_step.eq(1)]
    ranking = first.groupby("ticker").agg(
        anchor_close=("anchor_close", "first"),
        expected_opening_bar_close=("forecast_close", "mean"),
        expected_return=("return", "mean"),
        median_return=("return", "median"),
        probability_up=("return", lambda values: float((values > 0).mean())),
        downside_p10=("return", lambda values: float(np.quantile(values, 0.10))),
    ).reset_index().sort_values(["expected_return", "probability_up"], ascending=False)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    metadata = {
        "context_end": str(context_end), "target_date": str(target.date()),
        "target_bars": len(schedule), "candidate_count": len(names), "lookback_bars": lookback,
        "sample_paths": paths, "device": str(device), "model": str(model_path), "tokenizer": TOKENIZER,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = target.strftime("%Y_%m_%d")
    ranking.to_csv(output_dir / f"top30_tf15_first_bar_{stamp}.csv", index=False)
    forecasts.to_parquet(output_dir / f"paths_tf15_{stamp}.parquet", index=False)
    (output_dir / f"metadata_tf15_{stamp}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    del predictor, model, tokenizer
    gc.collect()
    return ranking, forecasts, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-date", default=None, help="Optional YYYY-MM-DD override for IDX holidays")
    args = parser.parse_args()
    ranking, _, metadata = run_projection(
        data_path=args.data, model_path=args.model, output_dir=args.output,
        candidate_count=args.candidates, lookback=args.lookback, paths=args.paths,
        batch_size=args.batch_size, target_date=args.target_date,
    )
    print(json.dumps(metadata, indent=2))
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
