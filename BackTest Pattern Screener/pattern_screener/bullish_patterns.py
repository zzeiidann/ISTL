from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PatternConfig


PATTERN_NAMES = [
    "bullish_rejection",
    "bullish_engulfing",
    "failed_breakdown",
    "double_bottom_breakout",
    "bearish_structure_break",
    "higher_low",
    "rsi_bullish_divergence",
    "breakout_with_volume",
    "bull_flag",
    "ascending_triangle",
    "compression_breakout",
    "inside_bar_breakout",
    "breakout_retest",
    "bullish_hh_hl_structure",
]


def _clip(value: pd.Series) -> pd.Series:
    return pd.to_numeric(value, errors="coerce").fillna(0).clip(0, 1)


def _mean_score(*values: pd.Series) -> pd.Series:
    return _clip(pd.concat([_clip(value) for value in values], axis=1).mean(axis=1))


def detect_bullish_patterns(
    frame: pd.DataFrame,
    config: PatternConfig | None = None,
) -> pd.DataFrame:
    """Add deterministic bullish pattern flags and confidence scores."""

    config = config or PatternConfig()
    result = frame.sort_values(["ticker", "date"]).copy()
    grouped = result.groupby("ticker", sort=False)
    previous_open = grouped["open"].shift(1)
    previous_close = grouped["close"].shift(1)
    previous_high = grouped["high"].shift(1)
    previous_low = grouped["low"].shift(1)
    two_back_high = grouped["high"].shift(2)
    two_back_low = grouped["low"].shift(2)

    near_support = result["low"].le(result["prior_low_20"] + config.support_atr_distance * result["atr_14"])
    close_strength = _clip((result["close_position_in_range"] - 0.5) / 0.5)
    volume_strength = _clip((result["current_volume_ratio"] - 1) / max(config.volume_multiplier - 1, 1e-9))
    breakout_distance = _clip(
        ((result["close"] / result["prior_high_20"] - 1) - config.breakout_buffer) / 0.04
    )
    wick_strength = _clip(result["lower_wick_ratio"] / (config.wick_ratio_threshold * 2))

    rejection = (
        near_support
        & result["lower_wick_ratio"].ge(config.wick_ratio_threshold)
        & result["close_position_in_range"].ge(config.close_near_high)
    )
    result["pattern_bullish_rejection"] = rejection
    result["pattern_bullish_rejection_score"] = _mean_score(
        wick_strength,
        close_strength,
        _clip(1 - result["support_distance"].abs() / 2),
    ).where(rejection, 0)

    engulfing = (
        previous_close.lt(previous_open)
        & result["close"].gt(result["open"])
        & result["open"].le(previous_close)
        & result["close"].ge(previous_open)
        & near_support
    )
    engulf_size = _clip(
        (result["close"] - result["open"]) / (previous_open - previous_close).abs().replace(0, np.nan) / 2
    )
    result["pattern_bullish_engulfing"] = engulfing
    result["pattern_bullish_engulfing_score"] = _mean_score(engulf_size, close_strength, volume_strength).where(engulfing, 0)

    failed = (
        result["low"].lt(result["prior_low_20"])
        & result["close"].gt(result["prior_low_20"])
        & result["close_position_in_range"].ge(0.65)
    )
    reclaim = _clip((result["close"] / result["prior_low_20"] - 1) / 0.04)
    result["pattern_failed_breakdown"] = failed
    result["pattern_failed_breakdown_score"] = _mean_score(reclaim, wick_strength, close_strength).where(failed, 0)

    two_lows = (
        result["double_bottom_tolerance"].le(config.double_bottom_tolerance)
        & result["double_bottom_separation"].between(
            config.double_bottom_min_separation,
            config.double_bottom_max_separation,
        )
    )
    double_bottom = two_lows & result["close"].gt(result["prior_high_20"] * (1 + config.breakout_buffer))
    low_similarity = _clip(1 - result["double_bottom_tolerance"] / config.double_bottom_tolerance)
    result["pattern_double_bottom_breakout"] = double_bottom
    result["pattern_double_bottom_breakout_score"] = _mean_score(low_similarity, breakout_distance, volume_strength).where(double_bottom, 0)

    structure_break = (
        result["confirmed_swing_high"].notna()
        & result["close"].gt(result["confirmed_swing_high"] * (1 + config.breakout_buffer))
        & result["ema_10"].ge(result["ema_20"])
    )
    structure_distance = _clip((result["close"] / result["confirmed_swing_high"] - 1) / 0.05)
    result["pattern_bearish_structure_break"] = structure_break
    result["pattern_bearish_structure_break_score"] = _mean_score(structure_distance, close_strength, volume_strength).where(structure_break, 0)

    higher_low = (
        result["confirmed_swing_low"].gt(result["previous_swing_low"] * 1.005)
        & result["swing_low_separation"].between(3, config.double_bottom_max_separation)
        & result["close"].gt(result["ema_10"])
    )
    higher_low_distance = _clip((result["confirmed_swing_low"] / result["previous_swing_low"] - 1) / 0.10)
    result["pattern_higher_low"] = higher_low
    result["pattern_higher_low_score"] = _mean_score(higher_low_distance, close_strength).where(higher_low, 0)

    prior_rsi_low = grouped["rsi_14"].transform(lambda values: values.shift(1).rolling(20, min_periods=10).min())
    divergence = (
        result["low"].le(result["prior_low_20"])
        & result["rsi_14"].gt(prior_rsi_low + 5)
        & result["rsi_14"].lt(55)
    )
    divergence_strength = _clip((result["rsi_14"] - prior_rsi_low) / 20)
    result["pattern_rsi_bullish_divergence"] = divergence
    result["pattern_rsi_bullish_divergence_score"] = divergence_strength.where(divergence, 0)

    breakout = (
        result["close"].gt(result["prior_high_20"] * (1 + config.breakout_buffer))
        & result["current_volume_ratio"].ge(config.volume_multiplier)
        & result["close_position_in_range"].ge(config.close_near_high)
    )
    result["pattern_breakout_with_volume"] = breakout
    result["pattern_breakout_with_volume_score"] = _mean_score(
        breakout_distance,
        volume_strength,
        close_strength,
        result["range_compression"],
    ).where(breakout, 0)

    prior_impulse = grouped["close"].pct_change(10).groupby(result["ticker"]).shift(config.flag_window)
    consolidation_high = grouped["high"].transform(
        lambda values: values.shift(1).rolling(config.flag_window, min_periods=config.flag_window).max()
    )
    consolidation_low = grouped["low"].transform(
        lambda values: values.shift(1).rolling(config.flag_window, min_periods=config.flag_window).min()
    )
    pullback_width = (consolidation_high / consolidation_low - 1).replace([np.inf, -np.inf], np.nan)
    flag = (
        prior_impulse.ge(config.impulse_return)
        & pullback_width.le(config.flag_max_pullback)
        & result["close"].gt(consolidation_high)
        & result["current_volume_ratio"].gt(1)
    )
    result["pattern_bull_flag"] = flag
    result["pattern_bull_flag_score"] = _mean_score(
        _clip(prior_impulse / (config.impulse_return * 2)),
        _clip(1 - pullback_width / config.flag_max_pullback),
        volume_strength,
    ).where(flag, 0)

    rising_lows = result["prior_low_10"].gt(result["prior_low_20"] * 1.01)
    tight_highs = result["prior_high_cv_10"].lt(0.025)
    triangle = rising_lows & tight_highs & result["close"].gt(result["prior_high_10"] * (1 + config.breakout_buffer))
    result["pattern_ascending_triangle"] = triangle
    result["pattern_ascending_triangle_score"] = _mean_score(
        _clip((result["prior_low_10"] / result["prior_low_20"] - 1) / 0.08),
        _clip(1 - result["prior_high_cv_10"] / 0.025),
        close_strength,
    ).where(triangle, 0)

    compression = (
        result["range_compression"].ge(1 - config.compression_ratio)
        & result["close"].gt(result["prior_high_20"] * (1 + config.breakout_buffer))
        & result["current_volume_ratio"].gt(1)
    )
    result["pattern_compression_breakout"] = compression
    result["pattern_compression_breakout_score"] = _mean_score(
        result["range_compression"], breakout_distance, volume_strength, close_strength
    ).where(compression, 0)

    prior_inside = previous_high.lt(two_back_high) & previous_low.gt(two_back_low)
    inside_breakout = prior_inside & result["close"].gt(previous_high) & result["close_position_in_range"].ge(0.65)
    result["pattern_inside_bar_breakout"] = inside_breakout
    result["pattern_inside_bar_breakout_score"] = _mean_score(
        _clip((result["close"] / previous_high - 1) / 0.04), close_strength, volume_strength
    ).where(inside_breakout, 0)

    raw_breakout = result["close"].gt(result["prior_high_20"] * (1 + config.breakout_buffer))
    recent_breakout_level = result["prior_high_20"].where(raw_breakout)
    recent_level = recent_breakout_level.groupby(result["ticker"]).transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).max()
    )
    retest = (
        recent_level.notna()
        & result["low"].le(recent_level * 1.015)
        & result["close"].gt(recent_level)
        & result["close_position_in_range"].ge(0.60)
    )
    result["pattern_breakout_retest"] = retest
    result["pattern_breakout_retest_score"] = _mean_score(
        _clip(1 - (result["low"] / recent_level - 1).abs() / 0.03), close_strength
    ).where(retest, 0)

    bullish_structure = (
        result["confirmed_swing_low"].gt(result["previous_swing_low"])
        & result["close"].gt(result["confirmed_swing_high"])
        & result["ema_10"].gt(result["ema_20"])
        & result["ema_20"].gt(result["ema_50"])
    )
    result["pattern_bullish_hh_hl_structure"] = bullish_structure
    result["pattern_bullish_hh_hl_structure_score"] = _mean_score(
        higher_low_distance,
        structure_distance,
        _clip((result["ema_10"] / result["ema_20"] - 1) / 0.05),
    ).where(bullish_structure, 0)

    for name in PATTERN_NAMES:
        result[f"pattern_{name}"] = result[f"pattern_{name}"].fillna(False).astype(bool)
        result[f"pattern_{name}_score"] = _clip(result[f"pattern_{name}_score"])
    return result
