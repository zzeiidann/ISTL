from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .bullish_patterns import detect_bullish_patterns
from .config import PatternConfig
from .pattern_scoring import score_patterns
from .price_action_features import add_price_action_features


OUTPUT_COLUMNS = [
    "selected_rank",
    "base_selected_rank",
    "ticker",
    "secondary_score",
    "pattern_quality_score",
    "pattern_penalty_score",
    "net_pattern_score",
    "final_ranking_score",
    "top_bullish_pattern",
    "top_pattern_score",
    "pattern_support",
    "pattern_penalty",
    "pattern_signal_count",
]


def calculate_secondary_score(candidates: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Recreate the existing weighted percentile-rank score."""

    missing = [f"r_{feature}" for feature in weights if f"r_{feature}" not in candidates]
    if missing:
        raise ValueError(f"Candidate panel lacks rank columns: {missing}")
    matrix = candidates[[f"r_{feature}" for feature in weights]].to_numpy(dtype=float)
    vector = np.array([weights[feature] for feature in weights], dtype=float)
    return pd.Series(matrix @ vector, index=candidates.index, name="secondary_score")


def build_pattern_snapshot(
    prices: pd.DataFrame,
    config: PatternConfig | None = None,
) -> pd.DataFrame:
    """Calculate all causal pattern outputs for every ticker-date row."""

    config = config or PatternConfig()
    featured = add_price_action_features(prices, config)
    detected = detect_bullish_patterns(featured, config)
    return score_patterns(detected, config)


def enrich_candidates(
    candidates: pd.DataFrame,
    pattern_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """Merge origin-day pattern information into an existing candidate panel."""

    required = {"origin", "horizon", "ticker", "secondary_score"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate columns missing: {sorted(missing)}")

    left = candidates.copy()
    left["origin"] = pd.to_datetime(left["origin"]).dt.tz_localize(None).dt.normalize()
    snapshot = pattern_snapshot.copy()
    snapshot["date"] = pd.to_datetime(snapshot["date"]).dt.tz_localize(None).dt.normalize()

    always = {
        "date", "ticker", "history_sufficient", "support_distance", "resistance_distance",
        "atr_14", "atr_ratio", "range_compression", "close_position_in_range", "body_ratio",
        "lower_wick_ratio", "upper_wick_ratio", "rsi_14", "ema_10", "ema_20", "ema_50",
        "average_traded_value_20", "median_traded_value_20", "zero_volume_frequency_20",
        "current_volume_ratio", "extension_from_ema20", "reversal_pattern_score",
        "continuation_pattern_score", "structure_score", "volume_confirmation_score",
        "pattern_quality_score", "pattern_penalty_score", "net_pattern_score",
        "top_bullish_pattern", "top_pattern_score", "pattern_support", "pattern_penalty",
        "pattern_signal_count",
    }
    export = [
        column for column in snapshot.columns
        if column in always or column.startswith("pattern_") or column.startswith("penalty_")
    ]
    snapshot = snapshot[export].rename(columns={"date": "origin"})
    duplicates = [column for column in snapshot.columns if column in left.columns and column not in {"origin", "ticker"}]
    snapshot = snapshot.rename(columns={column: f"pattern_feature_{column}" for column in duplicates})
    enriched = left.merge(snapshot, on=["origin", "ticker"], how="left", validate="many_to_one")
    if enriched["pattern_quality_score"].isna().any():
        count = int(enriched["pattern_quality_score"].isna().sum())
        raise ValueError(f"Missing pattern snapshot for {count} candidate rows")
    return enriched


def rank_enriched_candidates(
    enriched: pd.DataFrame,
    config: PatternConfig | None = None,
    select: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend base and pattern percentile ranks, returning scored and selected rows."""

    config = config or PatternConfig()
    frame = enriched.copy()
    groups = ["origin", "horizon"]
    frame["base_percentile_rank"] = frame.groupby(groups)["secondary_score"].rank(pct=True, method="average")
    frame["pattern_percentile_rank"] = frame.groupby(groups)["net_pattern_score"].rank(pct=True, method="average")
    frame["final_ranking_score"] = (
        config.base_score_weight * frame["base_percentile_rank"]
        + config.pattern_weight * frame["pattern_percentile_rank"]
    )
    frame["base_selected_rank"] = frame.groupby(groups)["secondary_score"].rank(
        ascending=False, method="first"
    ).astype(int)

    selected_parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(groups, sort=True):
        chosen = group.nlargest(min(select, len(group)), "final_ranking_score").copy()
        chosen["selected_rank"] = np.arange(1, len(chosen) + 1)
        selected_parts.append(chosen)
    selected = pd.concat(selected_parts, ignore_index=True)
    return frame, selected


def enrich_and_rank_candidates(
    candidates: pd.DataFrame,
    prices: pd.DataFrame,
    weights: dict[str, float],
    config: PatternConfig | None = None,
    select: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end pattern enrichment without changing the base screener."""

    config = config or PatternConfig()
    base = candidates.copy()
    if "secondary_score" not in base:
        base["secondary_score"] = calculate_secondary_score(base, weights)
    snapshot = build_pattern_snapshot(prices, config)
    enriched = enrich_candidates(base, snapshot)
    return rank_enriched_candidates(enriched, config, select)


def selection_precision(selected: pd.DataFrame, top_k: int) -> float:
    subset = selected.loc[selected["selected_rank"].le(top_k)]
    if subset.empty or "hit5" not in subset:
        return float("nan")
    return float(pd.to_numeric(subset["hit5"], errors="coerce").mean())


def optimize_pattern_weight(
    enriched: pd.DataFrame,
    base_config: PatternConfig | None = None,
    weights: tuple[float, ...] = tuple(np.linspace(0, 0.50, 11)),
    select: int = 30,
) -> tuple[PatternConfig, pd.DataFrame]:
    """Grid-search pattern blend weight on the configured full timeframe."""

    base_config = base_config or PatternConfig()
    rows: list[dict[str, float]] = []
    for pattern_weight in weights:
        config = replace(
            base_config,
            pattern_weight=float(pattern_weight),
            base_score_weight=float(1 - pattern_weight),
        )
        _, selected = rank_enriched_candidates(enriched, config, select)
        p5 = selection_precision(selected, 5)
        p10 = selection_precision(selected, 10)
        p30 = selection_precision(selected, 30)
        objective = 0.45 * p5 + 0.35 * p10 + 0.20 * p30
        rows.append(
            {
                "pattern_weight": float(pattern_weight),
                "base_score_weight": float(1 - pattern_weight),
                "precision_top5": p5,
                "precision_top10": p10,
                "precision_top30": p30,
                "objective": objective,
            }
        )
    trials = pd.DataFrame(rows).sort_values(["objective", "precision_top5"], ascending=False)
    best_weight = float(trials.iloc[0]["pattern_weight"])
    return replace(base_config, pattern_weight=best_weight, base_score_weight=1 - best_weight), trials
