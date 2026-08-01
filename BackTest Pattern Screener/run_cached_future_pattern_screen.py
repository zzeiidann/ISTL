from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from pattern_screener.config import PatternConfig
from pattern_screener.pattern_ranker import (
    build_pattern_snapshot,
    calculate_secondary_score,
    enrich_candidates,
    rank_enriched_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
ORIGIN = pd.Timestamp("2026-07-31")
TARGETS = pd.date_range("2026-08-03", "2026-08-07", freq="D")
FORECAST_FEATURES = [
    "pred_high_gain_mean", "pred_high_gain_median", "pred_hit5_probability",
    "pred_close_gain_mean", "pred_close_up_probability", "pred_range_mean",
    "pred_high_gain_dispersion", "pred_low_gain_mean",
]
STOCK_FEATURES = [
    "hit5_rate_20", "hit5_rate_60", "mom_5", "mom_20", "volatility_20",
    "volume_ratio_5_60", "turnover_accel", "range_20", "drawdown_60",
]
MODELS = {
    "validated_no_refit_e15": {
        "forecast": "validated-no-refit-e15/all_forecast_paths.parquet",
        "result": "validated_no_refit_e15_1d",
    },
    "production_refit_e4": {
        "forecast": "refit-run-e4/all_forecast_paths.parquet",
        "result": "production_refit_e4_1d",
    },
}


def add_stock_features(prices: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in prices.groupby("ticker", sort=False):
        group = group.sort_values("date").copy()
        close = group["close"]
        returns = close.pct_change()
        high_gain = group["high"] / close.shift(1) - 1
        turnover = close * group["volume"]
        hit5 = high_gain.ge(0.05).astype(float)
        group["hit5_rate_20"] = hit5.rolling(20, min_periods=10).mean()
        group["hit5_rate_60"] = hit5.rolling(60, min_periods=20).mean()
        group["mom_5"] = close.pct_change(5)
        group["mom_20"] = close.pct_change(20)
        group["volatility_20"] = returns.rolling(20).std() * math.sqrt(252)
        group["volume_ratio_5_60"] = group["volume"].rolling(5).mean() / group["volume"].rolling(60).median().replace(0, np.nan)
        group["turnover_accel"] = turnover.rolling(5).mean() / turnover.rolling(60).median().replace(0, np.nan)
        group["range_20"] = group["high"].rolling(20).max() / group["low"].rolling(20).min() - 1
        group["drawdown_60"] = close / close.rolling(60).max() - 1
        parts.append(group)
    return pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def aggregate_cached_paths(paths: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    paths = paths.copy()
    paths["date"] = pd.to_datetime(paths["date"]).dt.tz_localize(None).dt.normalize()
    paths = paths.loc[paths["date"].isin(TARGETS)].sort_values(["ticker", "path_id", "date"])
    actual_close = prices.loc[prices["date"].eq(ORIGIN), ["ticker", "close"]].rename(columns={"close": "origin_close"})
    paths = paths.merge(actual_close, on="ticker", how="inner")
    paths["previous_close"] = paths.groupby(["ticker", "path_id"])["close"].shift(1)
    paths.loc[paths["date"].eq(TARGETS[0]), "previous_close"] = paths["origin_close"]
    paths = paths.loc[paths["previous_close"].gt(0)].copy()
    paths["pred_high_gain"] = paths["high"] / paths["previous_close"] - 1
    paths["pred_close_gain"] = paths["close"] / paths["previous_close"] - 1
    paths["pred_low_gain"] = paths["low"] / paths["previous_close"] - 1
    paths["pred_range"] = (paths["high"] - paths["low"]) / paths["previous_close"]
    paths["pred_hit5"] = paths["pred_high_gain"].ge(0.05)
    paths["pred_close_up"] = paths["pred_close_gain"].gt(0)
    aggregate = paths.groupby(["date", "ticker"], as_index=False).agg(
        pred_high_gain_mean=("pred_high_gain", "mean"),
        pred_high_gain_median=("pred_high_gain", "median"),
        pred_hit5_probability=("pred_hit5", "mean"),
        pred_close_gain_mean=("pred_close_gain", "mean"),
        pred_close_up_probability=("pred_close_up", "mean"),
        pred_range_mean=("pred_range", "mean"),
        pred_high_gain_dispersion=("pred_high_gain", "std"),
        pred_low_gain_mean=("pred_low_gain", "mean"),
    ).rename(columns={"date": "target_date"})
    aggregate["origin"] = ORIGIN
    aggregate["horizon"] = aggregate["target_date"].map({date: i + 1 for i, date in enumerate(TARGETS)})
    return aggregate


def prepare_candidates(forecast: pd.DataFrame, origin_features: pd.DataFrame) -> pd.DataFrame:
    merged = forecast.merge(origin_features, on="ticker", how="inner")
    groups: list[pd.DataFrame] = []
    for _, group in merged.groupby("target_date", sort=True):
        group = group.loc[group["pred_close_gain_mean"].gt(0)].nlargest(100, "pred_close_gain_mean").copy()
        if len(group) < 30:
            raise RuntimeError(f"Only {len(group)} positive candidates for {group['target_date'].iloc[0]}")
        for feature in FORECAST_FEATURES + STOCK_FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            group[feature] = values.fillna(values.median()).fillna(0)
            group[f"r_{feature}"] = group[feature].rank(pct=True, method="average")
        groups.append(group)
    return pd.concat(groups, ignore_index=True)


def main() -> None:
    prices = pd.read_parquet(ROOT / "Kronos IDX FineTune/data/idx_kronos_all_daily.parquet")
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.normalize()
    if prices["date"].max() < ORIGIN:
        raise RuntimeError("Master OHLCV belum memiliki sesi 31 Juli 2026.")
    featured = add_stock_features(prices)
    origin_features = featured.loc[featured["date"].eq(ORIGIN), ["ticker", *STOCK_FEATURES]]
    snapshot = build_pattern_snapshot(prices, PatternConfig())
    output = ROOT / "BackTest Pattern Screener/future_screens/2026-08-03_to_07"
    output.mkdir(parents=True, exist_ok=True)
    all_scored: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    for model, paths in MODELS.items():
        archived = ROOT / "BackTest Kronos Screener/BackTest Results" / paths["result"]
        weights = json.loads((archived / "summary/best_weights.json").read_text())
        historical = json.loads((ROOT / "BackTest Pattern Screener/results" / paths["result"] / "pattern_config.json").read_text())
        config = PatternConfig(
            base_score_weight=historical["base_score_weight"],
            pattern_weight=historical["pattern_weight"],
            penalty_weight=historical["penalty_weight"],
        )
        cached = pd.read_parquet(ROOT / "Kronos IDX FineTune/results/2026-07-30" / paths["forecast"])
        candidates = prepare_candidates(aggregate_cached_paths(cached, prices), origin_features)
        candidates["secondary_score"] = calculate_secondary_score(candidates, weights)
        candidates["model"] = model
        enriched = enrich_candidates(candidates, snapshot)
        scored, selected = rank_enriched_candidates(enriched, config, select=30)
        scored["final_score_percentile"] = scored.groupby("target_date")["final_ranking_score"].rank(pct=True)
        selected["final_score_percentile"] = selected.groupby("target_date")["final_ranking_score"].rank(pct=True)
        scored.to_csv(output / f"{model}_all_candidates_pattern.csv", index=False)
        selected.to_csv(output / f"{model}_top30_pattern.csv", index=False)
        all_scored.append(scored)
        all_selected.append(selected)
    scored = pd.concat(all_scored, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    consensus = scored.groupby(["target_date", "ticker"], as_index=False).agg(
        models_available=("model", "nunique"),
        mean_final_score_percentile=("final_score_percentile", "mean"),
        mean_net_pattern_score=("net_pattern_score", "mean"),
    )
    selected_counts = selected.groupby(["target_date", "ticker"])["model"].nunique().rename("models_selected")
    consensus = consensus.merge(selected_counts, on=["target_date", "ticker"], how="left").fillna({"models_selected": 0})
    consensus = consensus.sort_values(
        ["target_date", "models_selected", "mean_final_score_percentile"],
        ascending=[True, False, False],
    )
    consensus["consensus_rank"] = consensus.groupby("target_date").cumcount() + 1
    consensus = consensus.loc[consensus["consensus_rank"].le(30)]
    consensus.to_csv(output / "consensus_top30_pattern.csv", index=False)
    archive = shutil.make_archive(str(output), "zip", output)
    print(consensus.loc[consensus["consensus_rank"].le(5)].to_string(index=False))
    print("Archive:", archive)


if __name__ == "__main__":
    main()
