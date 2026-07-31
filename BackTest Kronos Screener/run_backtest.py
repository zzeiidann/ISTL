from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from tqdm.auto import tqdm


PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
STOCK_FEATURES = [
    "hit5_rate_20",
    "hit5_rate_60",
    "mom_5",
    "mom_20",
    "volatility_20",
    "volume_ratio_5_60",
    "turnover_accel",
    "range_20",
    "drawdown_60",
]
FORECAST_FEATURES = [
    "pred_high_gain_mean",
    "pred_high_gain_median",
    "pred_hit5_probability",
    "pred_close_gain_mean",
    "pred_close_up_probability",
    "pred_range_mean",
    "pred_high_gain_dispersion",
    "pred_low_gain_mean",
]
SCORE_FEATURES = FORECAST_FEATURES + STOCK_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--horizon-mode", type=int, choices=[1, 5], required=True)
    parser.add_argument("--trials", type=int, default=1500)
    parser.add_argument("--backtest-sessions", type=int, default=42)
    parser.add_argument("--paths", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-positive", type=int, default=100)
    parser.add_argument("--select", type=int, default=30)
    parser.add_argument("--min-universe-ratio", type=float, default=0.80)
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_runtime(seed: int) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU Kaggle tidak aktif. Pilih accelerator P100 atau T4.")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(0)
    expected_arch = f"sm_{capability[0]}{capability[1]}"
    if expected_arch not in set(torch.cuda.get_arch_list()):
        raise RuntimeError(
            f"PyTorch {torch.__version__} tidak membawa {expected_arch}; "
            "restart kernel setelah instalasi CUDA 11.8 dari sel pertama."
        )
    print(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": capability,
            "compiled_arches": torch.cuda.get_arch_list(),
        }
    )
    return device


def load_prices(repo: Path) -> pd.DataFrame:
    path = repo / "Kronos IDX FineTune" / "data" / "idx_kronos_all_daily.parquet"
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = frame.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "ticker", *PRICE_COLUMNS])
    return frame


def add_stock_features(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in tqdm(frame.groupby("ticker", sort=False), desc="Stock features", unit="ticker"):
        group = group.sort_values("date").copy()
        close = group["close"]
        returns = close.pct_change()
        previous_close = close.shift(1)
        high_gain = group["high"] / previous_close - 1
        turnover = close * group["volume"]
        hit5 = high_gain.ge(0.05).astype(float)
        # The origin session is fully closed when the next session is forecast,
        # so its realized excursion is legitimate context.
        group["hit5_rate_20"] = hit5.rolling(20, min_periods=10).mean()
        group["hit5_rate_60"] = hit5.rolling(60, min_periods=20).mean()
        group["mom_5"] = close.pct_change(5)
        group["mom_20"] = close.pct_change(20)
        group["volatility_20"] = returns.rolling(20).std() * math.sqrt(252)
        group["volume_ratio_5_60"] = (
            group["volume"].rolling(5).mean()
            / group["volume"].rolling(60).median().replace(0, np.nan)
        )
        group["turnover_accel"] = (
            turnover.rolling(5).mean()
            / turnover.rolling(60).median().replace(0, np.nan)
        )
        group["range_20"] = group["high"].rolling(20).max() / group["low"].rolling(20).min() - 1
        group["drawdown_60"] = close / close.rolling(60).max() - 1
        parts.append(group)
    result = pd.concat(parts, ignore_index=True)
    return result.replace([np.inf, -np.inf], np.nan)


def backtest_origins(calendar: pd.DatetimeIndex, sessions: int, horizon: int) -> list[pd.Timestamp]:
    if len(calendar) <= sessions + horizon:
        raise RuntimeError("Calendar tidak cukup panjang untuk backtest.")
    return list(calendar[-(sessions + horizon) : -horizon])


def eligible_contexts(
    featured: pd.DataFrame,
    origin: pd.Timestamp,
    future_dates: list[pd.Timestamp],
    lookback: int,
) -> tuple[list[str], list[pd.DataFrame], list[pd.Series], list[pd.Series], dict[str, float]]:
    tickers, contexts, x_times, y_times, anchors = [], [], [], [], {}
    origin_frame = featured[featured["date"].le(origin)]
    for ticker, group in origin_frame.groupby("ticker", sort=False):
        context = group.sort_values("date").tail(lookback)
        if len(context) != lookback or context[PRICE_COLUMNS].isna().any().any():
            continue
        if context["date"].iloc[-1] != origin:
            continue
        tickers.append(ticker)
        contexts.append(context[PRICE_COLUMNS].copy())
        x_times.append(pd.Series(context["date"].to_numpy()))
        y_times.append(pd.Series(future_dates))
        anchors[ticker] = float(context["close"].iloc[-1])
    return tickers, contexts, x_times, y_times, anchors


def forecast_origin(
    predictor,
    featured: pd.DataFrame,
    origin: pd.Timestamp,
    future_dates: list[pd.Timestamp],
    horizons: list[int],
    paths: int,
    batch_size: int,
    lookback: int,
    seed: int,
) -> pd.DataFrame:
    tickers, contexts, x_times, y_times, anchors = eligible_contexts(
        featured, origin, future_dates, lookback
    )
    if not tickers:
        raise RuntimeError(f"Tidak ada context eligible pada {origin.date()}.")

    rows = []
    pred_len = max(horizons)
    for path_id in range(paths):
        path_seed = seed + int(origin.strftime("%Y%m%d")) + 10_000 * path_id
        torch.manual_seed(path_seed)
        torch.cuda.manual_seed_all(path_seed)
        for start in tqdm(
            range(0, len(tickers), batch_size),
            desc=f"{origin.date()} path {path_id + 1}/{paths}",
            leave=False,
        ):
            stop = min(start + batch_size, len(tickers))
            predictions = predictor.predict_batch(
                df_list=contexts[start:stop],
                x_timestamp_list=x_times[start:stop],
                y_timestamp_list=y_times[start:stop],
                pred_len=pred_len,
                T=0.8,
                top_p=0.9,
                top_k=0,
                sample_count=1,
                verbose=False,
            )
            for ticker, prediction in zip(tickers[start:stop], predictions):
                prediction = prediction.reset_index(drop=True)
                for horizon in horizons:
                    current = prediction.iloc[horizon - 1]
                    previous_close = anchors[ticker] if horizon == 1 else float(prediction.iloc[horizon - 2]["close"])
                    if not np.isfinite(previous_close) or previous_close <= 0:
                        continue
                    rows.append(
                        {
                            "origin": origin,
                            "target_date": future_dates[horizon - 1],
                            "horizon": horizon,
                            "ticker": ticker,
                            "path_id": path_id,
                            "pred_high_gain": float(current["high"] / previous_close - 1),
                            "pred_close_gain": float(current["close"] / previous_close - 1),
                            "pred_low_gain": float(current["low"] / previous_close - 1),
                            "pred_range": float((current["high"] - current["low"]) / previous_close),
                        }
                    )

    paths_frame = pd.DataFrame(rows)
    paths_frame["pred_hit5"] = paths_frame["pred_high_gain"].ge(0.05)
    paths_frame["pred_close_up"] = paths_frame["pred_close_gain"].gt(0)
    aggregate = (
        paths_frame.groupby(["origin", "target_date", "horizon", "ticker"], as_index=False)
        .agg(
            pred_high_gain_mean=("pred_high_gain", "mean"),
            pred_high_gain_median=("pred_high_gain", "median"),
            pred_hit5_probability=("pred_hit5", "mean"),
            pred_close_gain_mean=("pred_close_gain", "mean"),
            pred_close_up_probability=("pred_close_up", "mean"),
            pred_range_mean=("pred_range", "mean"),
            pred_high_gain_dispersion=("pred_high_gain", "std"),
            pred_low_gain_mean=("pred_low_gain", "mean"),
        )
    )
    aggregate["pred_high_gain_dispersion"] = aggregate["pred_high_gain_dispersion"].fillna(0)

    origin_features = featured.loc[featured["date"].eq(origin), ["ticker", *STOCK_FEATURES]]
    return aggregate.merge(origin_features, on="ticker", how="left")


def attach_actual_labels(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    calendar_position = {date: index for index, date in enumerate(calendar)}
    actual_rows = []
    for target_date in predictions["target_date"].drop_duplicates().sort_values():
        position = calendar_position[pd.Timestamp(target_date)]
        previous_date = calendar[position - 1]
        current = prices.loc[prices["date"].eq(target_date), ["ticker", "high", "volume"]]
        previous = prices.loc[prices["date"].eq(previous_date), ["ticker", "close"]].rename(
            columns={"close": "previous_actual_close"}
        )
        actual = current.merge(previous, on="ticker", how="outer")
        actual["target_date"] = target_date
        actual["actual_high_gain"] = actual["high"] / actual["previous_actual_close"] - 1
        actual["hit5"] = (
            actual["actual_high_gain"].ge(0.05)
            & actual["volume"].fillna(0).gt(0)
        ).astype(int)
        actual_rows.append(actual[["target_date", "ticker", "actual_high_gain", "hit5"]])
    return predictions.merge(pd.concat(actual_rows, ignore_index=True), on=["target_date", "ticker"], how="left").fillna(
        {"hit5": 0}
    )


def remove_anomalous_origins(
    frame: pd.DataFrame,
    select: int,
    min_universe_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove a whole origin when any horizon has partial coverage or too few candidates."""
    if not 0 < min_universe_ratio <= 1:
        raise ValueError("--min-universe-ratio harus berada dalam interval (0, 1].")

    coverage = (
        frame.groupby(["origin", "horizon"], as_index=False)
        .agg(
            forecast_tickers=("ticker", "nunique"),
            positive_candidates=("pred_close_gain_mean", lambda values: int(values.gt(0).sum())),
        )
    )
    reference = coverage.groupby("horizon")["forecast_tickers"].median().rename("median_tickers")
    coverage = coverage.merge(reference, on="horizon", how="left")
    coverage["minimum_tickers"] = np.ceil(
        coverage["median_tickers"] * min_universe_ratio
    ).astype(int)
    coverage["low_coverage"] = coverage["forecast_tickers"].lt(coverage["minimum_tickers"])
    coverage["too_few_positive"] = coverage["positive_candidates"].lt(select)
    coverage["excluded"] = coverage["low_coverage"] | coverage["too_few_positive"]
    coverage["reason"] = np.select(
        [
            coverage["low_coverage"] & coverage["too_few_positive"],
            coverage["low_coverage"],
            coverage["too_few_positive"],
        ],
        ["low_coverage_and_too_few_positive", "low_coverage", "too_few_positive"],
        default="",
    )

    # One broken horizon makes that rolling origin incomparable, so all its
    # horizons are removed together in the 5D experiment.
    excluded_origins = set(coverage.loc[coverage["excluded"], "origin"])
    cleaned = frame.loc[~frame["origin"].isin(excluded_origins)].copy()
    if cleaned.empty:
        raise RuntimeError("Semua origin dianggap anomali; periksa data dan threshold coverage.")
    return cleaned, coverage


def prepare_candidate_panel(frame: pd.DataFrame, top_positive: int, select: int) -> pd.DataFrame:
    groups = []
    for _, group in frame.groupby(["origin", "horizon"], sort=True):
        group = group[group["pred_close_gain_mean"].gt(0)].copy()
        if len(group) < select:
            continue
        group = group.nlargest(top_positive, "pred_close_gain_mean").copy()
        for feature in SCORE_FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce")
            values = values.fillna(values.median()).fillna(0)
            group[feature] = values
            group[f"r_{feature}"] = values.rank(pct=True, method="average")
        groups.append(group)
    if not groups:
        raise RuntimeError("Tidak ada positive candidate groups.")
    return pd.concat(groups, ignore_index=True)


def group_precision(frame: pd.DataFrame, scores: np.ndarray, select: int) -> tuple[float, pd.DataFrame]:
    scored = frame.copy()
    scored["secondary_score"] = scores
    selections = []
    precisions = []
    for _, group in scored.groupby(["origin", "horizon"], sort=False):
        count = min(select, len(group))
        chosen = group.nlargest(count, "secondary_score").copy()
        chosen["selected_rank"] = np.arange(1, len(chosen) + 1)
        selections.append(chosen)
        precisions.append(float(chosen["hit5"].mean()))
    return float(np.mean(precisions)), pd.concat(selections, ignore_index=True)


def optimize_weights(
    train: pd.DataFrame,
    trials: int,
    select: int,
    seed: int,
) -> tuple[dict[str, float], optuna.Study]:
    rank_columns = [f"r_{feature}" for feature in SCORE_FEATURES]
    matrix = train[rank_columns].to_numpy(np.float64)

    def objective(trial: optuna.Trial) -> float:
        raw = np.array(
            [trial.suggest_float(feature, -1.0, 1.0) for feature in SCORE_FEATURES],
            dtype=np.float64,
        )
        norm = np.abs(raw).sum()
        if norm < 1e-9:
            return 0.0
        scores = matrix @ (raw / norm)
        precision, _ = group_precision(train, scores, select)
        return precision

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    raw = np.array([study.best_params[feature] for feature in SCORE_FEATURES], dtype=np.float64)
    raw /= np.abs(raw).sum()
    return dict(zip(SCORE_FEATURES, raw.tolist())), study


def score_with_weights(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    matrix = frame[[f"r_{feature}" for feature in SCORE_FEATURES]].to_numpy(np.float64)
    vector = np.array([weights[feature] for feature in SCORE_FEATURES], dtype=np.float64)
    return matrix @ vector


def baseline_precision(frame: pd.DataFrame, column: str, select: int) -> float:
    values = []
    for _, group in frame.groupby(["origin", "horizon"], sort=False):
        values.append(group.nlargest(min(select, len(group)), column)["hit5"].mean())
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    device = configure_runtime(args.seed)
    repo = args.repo.resolve()
    output_root = Path("/kaggle/working/backtest_kronos_screener")
    output_dir = output_root / args.run_name
    cache_dir = output_dir / "forecast_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    kronos_dir = repo / "Kronos IDX FineTune" / "Kronos"
    if not (kronos_dir / "model" / "kronos.py").exists():
        kronos_dir = Path("/kaggle/working/Kronos")
        if not kronos_dir.exists():
            subprocess.run(
                ["git", "clone", "https://github.com/shiyu-coder/Kronos.git", str(kronos_dir)],
                check=True,
            )
        subprocess.run(
            [
                "git", "-C", str(kronos_dir), "checkout",
                "67b630e67f6a18c9e9be918d9b4337c960db1e9a",
            ],
            check=True,
        )
    sys.path.insert(0, str(kronos_dir))
    from model import Kronos, KronosPredictor, KronosTokenizer

    prices = load_prices(repo)
    featured = add_stock_features(prices)
    calendar = pd.DatetimeIndex(sorted(prices["date"].unique()))
    horizons = [1] if args.horizon_mode == 1 else [1, 2, 3, 4, 5]
    origins = backtest_origins(calendar, args.backtest_sessions, max(horizons))

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(device).eval()
    model = Kronos.from_pretrained(str(args.model_path.resolve())).to(device).eval()
    predictor = KronosPredictor(model, tokenizer, device=str(device), max_context=512)

    origin_frames = []
    for origin in origins:
        cache_path = cache_dir / f"forecast_{origin.date()}.parquet"
        if cache_path.exists():
            origin_frames.append(pd.read_parquet(cache_path))
            continue
        origin_position = calendar.get_loc(origin)
        future_dates = list(calendar[origin_position + 1 : origin_position + 1 + max(horizons)])
        result = forecast_origin(
            predictor,
            featured,
            origin,
            future_dates,
            horizons,
            args.paths,
            args.batch_size,
            args.lookback,
            args.seed,
        )
        result.to_parquet(cache_path, index=False)
        origin_frames.append(result)

    forecasts = pd.concat(origin_frames, ignore_index=True)
    forecasts, origin_audit = remove_anomalous_origins(
        forecasts,
        select=args.select,
        min_universe_ratio=args.min_universe_ratio,
    )
    origin_audit.to_csv(output_dir / "origin_quality_audit.csv", index=False)
    excluded = origin_audit.loc[origin_audit["excluded"]]
    if not excluded.empty:
        print("Origin anomali yang otomatis dikeluarkan:")
        print(
            excluded[
                [
                    "origin", "horizon", "forecast_tickers", "median_tickers",
                    "positive_candidates", "reason",
                ]
            ].to_string(index=False)
        )

    labeled = attach_actual_labels(forecasts, prices, calendar)
    candidates = prepare_candidate_panel(labeled, args.top_positive, args.select)
    candidates.to_parquet(output_dir / "daily_positive_top100_panel.parquet", index=False)

    unique_origins = sorted(candidates["origin"].unique())
    split_index = max(1, int(len(unique_origins) * 0.70))
    embargo = max(horizons)
    train_end = max(1, split_index - embargo)
    train_origins = set(unique_origins[:train_end])
    test_origins = set(unique_origins[split_index:])
    train = candidates[candidates["origin"].isin(train_origins)].copy()
    test = candidates[candidates["origin"].isin(test_origins)].copy()
    if train.empty or test.empty:
        raise RuntimeError("Train/test split kosong; tambah backtest sessions.")

    weights, study = optimize_weights(train, args.trials, args.select, args.seed)
    train_precision, train_selected = group_precision(train, score_with_weights(train, weights), args.select)
    test_precision, test_selected = group_precision(test, score_with_weights(test, weights), args.select)
    selected = pd.concat(
        [train_selected.assign(split="optimization"), test_selected.assign(split="holdout")],
        ignore_index=True,
    )
    selected.to_csv(output_dir / "selected_top30_daily.csv", index=False)
    pd.DataFrame(study.trials_dataframe()).to_csv(output_dir / "optuna_trials.csv", index=False)

    horizon_metrics = (
        selected.groupby(["split", "horizon"])
        .agg(selections=("hit5", "size"), hits=("hit5", "sum"), win_rate=("hit5", "mean"))
        .reset_index()
    )
    horizon_metrics.to_csv(output_dir / "win_rate_by_horizon.csv", index=False)

    summary = {
        "run_name": args.run_name,
        "model_path": str(args.model_path),
        "horizon_mode": args.horizon_mode,
        "horizons": horizons,
        "backtest_sessions": len(origins),
        "valid_backtest_sessions": int(candidates["origin"].nunique()),
        "excluded_anomalous_sessions": int(origin_audit.loc[origin_audit["excluded"], "origin"].nunique()),
        "min_universe_ratio": args.min_universe_ratio,
        "origin_start": str(pd.Timestamp(origins[0]).date()),
        "origin_end": str(pd.Timestamp(origins[-1]).date()),
        "target": "actual_high / previous_actual_close - 1 >= 0.05",
        "top_positive": args.top_positive,
        "selected_per_group": args.select,
        "forecast_paths": args.paths,
        "optuna_trials": args.trials,
        "optimization_win_rate": train_precision,
        "holdout_win_rate": test_precision,
        "holdout_kronos_gain_baseline": baseline_precision(test, "pred_close_gain_mean", args.select),
        "holdout_predicted_hit5_baseline": baseline_precision(test, "pred_hit5_probability", args.select),
        "best_weights": weights,
    }
    (output_dir / "best_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    (output_dir / "backtest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(horizon_metrics.to_string(index=False))
    print("Outputs:", output_dir)


if __name__ == "__main__":
    main()
