from __future__ import annotations

import numpy as np
import pandas as pd

from .bullish_patterns import PATTERN_NAMES
from .config import PatternConfig


REVERSAL_PATTERNS = [
    "bullish_rejection",
    "bullish_engulfing",
    "failed_breakdown",
    "double_bottom_breakout",
    "bearish_structure_break",
    "higher_low",
]
CONTINUATION_PATTERNS = [
    "breakout_with_volume",
    "bull_flag",
    "ascending_triangle",
    "compression_breakout",
    "inside_bar_breakout",
    "breakout_retest",
    "bullish_hh_hl_structure",
]


def _clip(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1)


def score_patterns(frame: pd.DataFrame, config: PatternConfig | None = None) -> pd.DataFrame:
    """Aggregate correlated patterns, calculate penalties, and explain scores."""

    config = config or PatternConfig()
    result = frame.copy()
    reversal_columns = [f"pattern_{name}_score" for name in REVERSAL_PATTERNS]
    continuation_columns = [f"pattern_{name}_score" for name in CONTINUATION_PATTERNS]
    structure_columns = [
        "pattern_bearish_structure_break_score",
        "pattern_higher_low_score",
        "pattern_bullish_hh_hl_structure_score",
    ]

    result["reversal_pattern_score"] = result[reversal_columns].max(axis=1).clip(0, 1)
    result["continuation_pattern_score"] = result[continuation_columns].max(axis=1).clip(0, 1)
    result["structure_score"] = result[structure_columns].max(axis=1).clip(0, 1)
    result["volume_confirmation_score"] = _clip((result["current_volume_ratio"] - 0.8) / 1.7)
    result["pattern_signal_count"] = result[[f"pattern_{name}" for name in PATTERN_NAMES]].sum(axis=1).astype(int)

    primary = result[["reversal_pattern_score", "continuation_pattern_score"]].max(axis=1)
    independent_confirmation = (
        0.07 * result["structure_score"]
        + 0.05 * result["volume_confirmation_score"]
        + 0.04 * _clip(result["pattern_rsi_bullish_divergence_score"])
    )
    result["pattern_quality_score"] = (primary + independent_confirmation).clip(0, 1)

    result["penalty_extended"] = _clip(
        (result["extension_from_ema20"] - config.extension_threshold)
        / max(config.extension_threshold, 1e-9)
    )
    result["penalty_gap_up"] = _clip(
        (result["gap_return"] - config.gap_up_threshold) / max(config.gap_up_threshold, 1e-9)
    )
    result["penalty_weak_breakout_volume"] = (
        result["close"].gt(result["prior_high_20"])
        * _clip((config.weak_volume_ratio - result["current_volume_ratio"]) / config.weak_volume_ratio)
    )
    result["penalty_upper_wick"] = _clip(
        (result["upper_wick_ratio"] - config.upper_wick_penalty_ratio)
        / max(1 - config.upper_wick_penalty_ratio, 1e-9)
    )
    result["penalty_close_below_high"] = _clip((0.65 - result["close_position_in_range"]) / 0.65)
    result["penalty_low_liquidity"] = _clip(
        (config.liquidity_threshold - result["median_traded_value_20"])
        / config.liquidity_threshold
    )
    result["penalty_excess_volatility"] = _clip(
        (result["atr_pct"] - config.volatility_penalty) / config.volatility_penalty
    )
    result["penalty_chasing"] = _clip((result["recent_bullish_count_5"] - 2) / 3)
    result["penalty_near_resistance"] = (
        result["resistance_distance"].between(0, 0.35).astype(float)
        * _clip(1 - result["resistance_distance"] / 0.35)
    )
    result["penalty_insufficient_history"] = (~result["history_sufficient"]).astype(float)
    result["penalty_zero_volume"] = _clip(result["zero_volume_frequency_20"] / config.zero_volume_penalty)

    penalty_columns = [column for column in result.columns if column.startswith("penalty_")]
    # Max captures the dominant entry-risk problem; a small mean component
    # represents multiple independent weaknesses without double counting them.
    result["pattern_penalty_score"] = (
        0.75 * result[penalty_columns].max(axis=1)
        + 0.25 * result[penalty_columns].mean(axis=1)
    ).clip(0, 1)
    result["net_pattern_score"] = (
        result["pattern_quality_score"] - config.penalty_weight * result["pattern_penalty_score"]
    ).clip(0, 1)

    pattern_score_columns = [f"pattern_{name}_score" for name in PATTERN_NAMES]
    result["top_bullish_pattern"] = (
        result[pattern_score_columns].idxmax(axis=1).str.removeprefix("pattern_").str.removesuffix("_score")
    )
    result["top_pattern_score"] = result[pattern_score_columns].max(axis=1)
    result.loc[result["top_pattern_score"].eq(0), "top_bullish_pattern"] = "none"

    support_evidence = pd.DataFrame(
        {
            "volume_expansion": result["volume_confirmation_score"],
            "bullish_structure": result["structure_score"],
            "reversal_setup": result["reversal_pattern_score"],
            "continuation_setup": result["continuation_pattern_score"],
            "rsi_confirmation": result["pattern_rsi_bullish_divergence_score"],
        },
        index=result.index,
    )
    result["pattern_support"] = support_evidence.idxmax(axis=1)
    result.loc[support_evidence.max(axis=1).eq(0), "pattern_support"] = "none"
    result["pattern_penalty"] = result[penalty_columns].idxmax(axis=1).str.removeprefix("penalty_")
    result.loc[result["pattern_penalty_score"].eq(0), "pattern_penalty"] = "none"
    return result.replace([np.inf, -np.inf], np.nan)
