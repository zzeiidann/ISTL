from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PatternConfig


REQUIRED_COLUMNS = {"date", "ticker", "open", "high", "low", "close", "volume"}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = _safe_divide(gain, loss)
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(loss.ne(0), 100.0).where(gain.ne(0) | loss.ne(0), 50.0)


def _confirmed_swings(group: pd.DataFrame, config: PatternConfig) -> pd.DataFrame:
    """Return causal swing levels, exposed only after right bars have closed."""

    window = config.swing_window
    span = 2 * window + 1
    candidate_low = group["low"].shift(window)
    candidate_high = group["high"].shift(window)
    confirmed_low = candidate_low.eq(group["low"].rolling(span, min_periods=span).min())
    confirmed_high = candidate_high.eq(group["high"].rolling(span, min_periods=span).max())

    positions = pd.Series(np.arange(len(group), dtype=float), index=group.index)
    low_event = candidate_low.where(confirmed_low)
    low_index_event = positions.shift(window).where(confirmed_low)
    previous_low_at_event = low_event.ffill().shift(1).where(confirmed_low)
    previous_low_index_at_event = low_index_event.ffill().shift(1).where(confirmed_low)

    result = pd.DataFrame(index=group.index)
    result["confirmed_swing_low"] = low_event.ffill()
    result["previous_swing_low"] = previous_low_at_event.ffill()
    result["confirmed_swing_low_index"] = low_index_event.ffill()
    result["previous_swing_low_index"] = previous_low_index_at_event.ffill()
    result["confirmed_swing_high"] = candidate_high.where(confirmed_high).ffill()
    result["swing_low_confirmed_now"] = confirmed_low.fillna(False)
    result["swing_high_confirmed_now"] = confirmed_high.fillna(False)

    # A double bottom need not be formed by the latest two swing lows. Keep the
    # closest comparable earlier low, while exposing it only when the newer
    # swing has received its right-side confirmation bars.
    matched_low = pd.Series(np.nan, index=group.index)
    matched_index = pd.Series(np.nan, index=group.index)
    swing_events: list[tuple[float, float]] = []
    for event_index in group.index[confirmed_low.fillna(False)]:
        current_position = float(low_index_event.loc[event_index])
        current_low = float(low_event.loc[event_index])
        eligible = [
            (position, value)
            for position, value in swing_events
            if config.double_bottom_min_separation
            <= current_position - position
            <= config.double_bottom_max_separation
        ]
        if eligible:
            best_position, best_low = min(
                eligible,
                key=lambda item: abs(current_low - item[1]) / max((current_low + item[1]) / 2, 1e-12),
            )
            matched_low.loc[event_index] = best_low
            matched_index.loc[event_index] = best_position
        swing_events.append((current_position, current_low))
    result["double_bottom_reference_low"] = matched_low.ffill()
    result["double_bottom_reference_index"] = matched_index.ffill()
    return result


def add_price_action_features(
    frame: pd.DataFrame,
    config: PatternConfig | None = None,
) -> pd.DataFrame:
    """Calculate causal OHLCV features independently for each ticker."""

    config = config or PatternConfig()
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    source = frame.copy()
    source["date"] = pd.to_datetime(source["date"]).dt.tz_localize(None).dt.normalize()
    source = source.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    numeric = ["open", "high", "low", "close", "volume"]
    for column in numeric:
        source[column] = pd.to_numeric(source[column], errors="coerce")

    parts: list[pd.DataFrame] = []
    for _, group in source.groupby("ticker", sort=False):
        group = group.sort_values("date").copy()
        previous_close = group["close"].shift(1)
        candle_range = (group["high"] - group["low"]).clip(lower=0)
        body = (group["close"] - group["open"]).abs()
        lower_body = group[["open", "close"]].min(axis=1)
        upper_body = group[["open", "close"]].max(axis=1)
        lower_wick = (lower_body - group["low"]).clip(lower=0)
        upper_wick = (group["high"] - upper_body).clip(lower=0)
        true_range = pd.concat(
            [candle_range, (group["high"] - previous_close).abs(), (group["low"] - previous_close).abs()],
            axis=1,
        ).max(axis=1)

        group["bar_number"] = np.arange(1, len(group) + 1)
        group["history_sufficient"] = group["bar_number"].ge(config.min_history)
        for window in (5, 10, 20, 60):
            group[f"prior_high_{window}"] = group["high"].shift(1).rolling(window, min_periods=window).max()
            group[f"prior_low_{window}"] = group["low"].shift(1).rolling(window, min_periods=window).min()

        group["prior_volume_mean_20"] = group["volume"].shift(1).rolling(20, min_periods=10).mean()
        group["prior_volume_median_60"] = group["volume"].shift(1).rolling(60, min_periods=20).median()
        group["volume_ratio_5_20"] = _safe_divide(
            group["volume"].shift(1).rolling(5, min_periods=3).mean(),
            group["volume"].shift(1).rolling(20, min_periods=10).mean(),
        )
        group["volume_ratio_5_60"] = _safe_divide(
            group["volume"].shift(1).rolling(5, min_periods=3).mean(),
            group["prior_volume_median_60"],
        )
        group["current_volume_ratio"] = _safe_divide(group["volume"], group["prior_volume_mean_20"])
        group["traded_value"] = group["close"] * group["volume"]
        group["average_traded_value_20"] = group["traded_value"].shift(1).rolling(20, min_periods=10).mean()
        group["median_traded_value_20"] = group["traded_value"].shift(1).rolling(20, min_periods=10).median()
        group["zero_volume_frequency_20"] = group["volume"].shift(1).eq(0).rolling(20, min_periods=10).mean()
        group["volume_stability_20"] = 1 - _safe_divide(
            group["volume"].shift(1).rolling(20, min_periods=10).std(),
            group["prior_volume_mean_20"],
        ).clip(0, 1)

        group["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        group["prior_atr_mean_20"] = group["atr_14"].shift(1).rolling(20, min_periods=10).mean()
        group["atr_ratio"] = _safe_divide(group["atr_14"], group["prior_atr_mean_20"])
        group["atr_pct"] = _safe_divide(group["atr_14"], group["close"])
        group["range_compression"] = (
            1 - _safe_divide(
                candle_range.shift(1).rolling(config.compression_window, min_periods=5).mean(),
                candle_range.shift(1).rolling(20, min_periods=10).mean(),
            )
        ).clip(0, 1)
        group["close_position_in_range"] = _safe_divide(group["close"] - group["low"], candle_range).fillna(0.5)
        group["body_ratio"] = _safe_divide(body, candle_range).fillna(0)
        group["lower_wick_ratio"] = _safe_divide(lower_wick, body.clip(lower=1e-12)).fillna(0)
        group["upper_wick_ratio"] = _safe_divide(upper_wick, candle_range).fillna(0)
        group["gap_return"] = _safe_divide(group["open"], previous_close) - 1
        group["return_1"] = group["close"].pct_change()
        group["return_5"] = group["close"].pct_change(5)
        group["return_10"] = group["close"].pct_change(10)
        group["return_20"] = group["close"].pct_change(20)
        group["rsi_14"] = _rsi(group["close"])
        for window in (10, 20, 50):
            group[f"ema_{window}"] = group["close"].ewm(span=window, adjust=False, min_periods=window).mean()

        group["support_distance"] = _safe_divide(group["close"] - group["prior_low_20"], group["atr_14"])
        group["resistance_distance"] = _safe_divide(group["prior_high_20"] - group["close"], group["atr_14"])
        group["extension_from_ema20"] = _safe_divide(group["close"], group["ema_20"]) - 1
        group["prior_high_cv_10"] = _safe_divide(
            group["high"].shift(1).rolling(10, min_periods=7).std(),
            group["high"].shift(1).rolling(10, min_periods=7).mean(),
        )
        group["recent_bullish_count_5"] = group["return_1"].shift(1).gt(0.03).rolling(5, min_periods=3).sum()

        swings = _confirmed_swings(group, config)
        group = pd.concat([group, swings], axis=1)
        group["swing_low_separation"] = group["confirmed_swing_low_index"] - group["previous_swing_low_index"]
        group["swing_low_tolerance"] = (
            _safe_divide(
                (group["confirmed_swing_low"] - group["previous_swing_low"]).abs(),
                group[["confirmed_swing_low", "previous_swing_low"]].mean(axis=1),
            )
        )
        group["double_bottom_separation"] = (
            group["confirmed_swing_low_index"] - group["double_bottom_reference_index"]
        )
        group["double_bottom_tolerance"] = _safe_divide(
            (group["confirmed_swing_low"] - group["double_bottom_reference_low"]).abs(),
            group[["confirmed_swing_low", "double_bottom_reference_low"]].mean(axis=1),
        )
        parts.append(group)

    result = pd.concat(parts, ignore_index=True)
    return result.replace([np.inf, -np.inf], np.nan)
