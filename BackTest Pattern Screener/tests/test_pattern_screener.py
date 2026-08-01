from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pattern_screener.bullish_patterns import detect_bullish_patterns
from pattern_screener.config import PatternConfig
from pattern_screener.pattern_scoring import score_patterns
from pattern_screener.price_action_features import add_price_action_features


def make_prices(ticker: str = "TEST", rows: int = 90) -> pd.DataFrame:
    x = np.arange(rows, dtype=float)
    close = 98 + 0.025 * x + 0.5 * np.sin(x / 4)
    open_ = close - 0.15
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="D"),
            "ticker": ticker,
            "open": open_,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 1_000_000.0),
        }
    )


def calculate(frame: pd.DataFrame) -> pd.DataFrame:
    features = add_price_action_features(frame, PatternConfig())
    patterns = detect_bullish_patterns(features, PatternConfig())
    return score_patterns(patterns, PatternConfig())


class PatternScreenerTests(unittest.TestCase):
    def test_breakout_with_strong_volume(self) -> None:
        frame = make_prices()
        prior_resistance = frame.loc[69:88, "high"].max()
        frame.loc[89, ["open", "low", "close", "high", "volume"]] = [prior_resistance, prior_resistance - 0.2, prior_resistance + 4, prior_resistance + 4.3, 3_000_000]
        row = calculate(frame).iloc[-1]
        self.assertTrue(row["pattern_breakout_with_volume"])
        self.assertGreater(row["pattern_breakout_with_volume_score"], 0.45)

    def test_breakout_with_weak_volume_scores_lower(self) -> None:
        strong = make_prices("STRONG")
        weak = make_prices("WEAK")
        for frame, volume in ((strong, 3_000_000), (weak, 700_000)):
            resistance = frame.loc[69:88, "high"].max()
            frame.loc[89, ["open", "low", "close", "high", "volume"]] = [resistance, resistance - 0.2, resistance + 4, resistance + 4.3, volume]
        strong_row = calculate(strong).iloc[-1]
        weak_row = calculate(weak).iloc[-1]
        self.assertFalse(weak_row["pattern_breakout_with_volume"])
        self.assertGreater(strong_row["pattern_breakout_with_volume_score"], weak_row["pattern_breakout_with_volume_score"])
        self.assertGreater(weak_row["penalty_weak_breakout_volume"], 0)

    def test_failed_breakdown(self) -> None:
        frame = make_prices()
        support = frame.loc[69:88, "low"].min()
        frame.loc[89, ["open", "low", "close", "high", "volume"]] = [support - 0.2, support - 5, support + 2, support + 2.5, 1_800_000]
        row = calculate(frame).iloc[-1]
        self.assertTrue(row["pattern_failed_breakdown"])
        self.assertGreater(row["pattern_failed_breakdown_score"], 0)

    def test_bullish_rejection_near_support(self) -> None:
        frame = make_prices()
        support = frame.loc[69:88, "low"].min()
        frame.loc[89, ["open", "low", "close", "high"]] = [support + 1, support - 4, support + 2, support + 2.2]
        row = calculate(frame).iloc[-1]
        self.assertTrue(row["pattern_bullish_rejection"])

    def test_double_bottom_breakout(self) -> None:
        frame = make_prices(rows=100)
        frame.loc[60, ["open", "high", "low", "close"]] = [97, 98, 90, 96]
        frame.loc[61:64, "low"] = np.maximum(frame.loc[61:64, "low"], 94)
        frame.loc[76, ["open", "high", "low", "close"]] = [97, 98, 90.8, 96]
        frame.loc[77:98, "low"] = np.linspace(94, 99, 22)
        frame.loc[77:98, "high"] = np.linspace(99, 101, 22)
        resistance = frame.loc[79:98, "high"].max()
        frame.loc[99, ["open", "low", "close", "high", "volume"]] = [resistance, resistance - 0.2, resistance + 4, resistance + 4.5, 3_000_000]
        row = calculate(frame).iloc[-1]
        self.assertTrue(row["pattern_double_bottom_breakout"])
        self.assertGreater(row["pattern_double_bottom_breakout_score"], 0)

    def test_bull_flag_breakout(self) -> None:
        frame = make_prices(rows=100)
        frame.loc[84:94, "close"] = np.linspace(90, 105, 11)
        frame.loc[84:94, "open"] = frame.loc[84:94, "close"] - 0.3
        frame.loc[84:94, "high"] = frame.loc[84:94, "close"] + 0.6
        frame.loc[84:94, "low"] = frame.loc[84:94, "close"] - 0.6
        frame.loc[95:98, "open"] = 104
        frame.loc[95:98, "close"] = 104.2
        frame.loc[95:98, "high"] = 105
        frame.loc[95:98, "low"] = 103
        frame.loc[99, ["open", "low", "close", "high", "volume"]] = [105, 104.8, 108, 108.3, 2_500_000]
        row = calculate(frame).iloc[-1]
        self.assertTrue(row["pattern_bull_flag"])

    def test_compression_breakout(self) -> None:
        frame = make_prices(rows=100)
        frame.loc[70:88, "high"] = frame.loc[70:88, "close"] + 3
        frame.loc[70:88, "low"] = frame.loc[70:88, "close"] - 3
        frame.loc[89:98, "high"] = frame.loc[89:98, "close"] + 0.3
        frame.loc[89:98, "low"] = frame.loc[89:98, "close"] - 0.3
        resistance = frame.loc[79:98, "high"].max()
        frame.loc[99, ["open", "low", "close", "high", "volume"]] = [resistance, resistance - 0.2, resistance + 3, resistance + 3.2, 2_500_000]
        row = calculate(frame).iloc[-1]
        self.assertTrue(row["pattern_compression_breakout"])

    def test_extended_price_penalty_reduces_net_score(self) -> None:
        normal = calculate(make_prices("NORMAL")).iloc[-1].copy()
        extended_frame = make_prices("EXTENDED")
        extended_frame.loc[89, ["open", "low", "close", "high"]] = [115, 114, 120, 121]
        extended = calculate(extended_frame).iloc[-1]
        self.assertGreater(extended["penalty_extended"], normal["penalty_extended"])
        self.assertLessEqual(extended["net_pattern_score"], extended["pattern_quality_score"])

    def test_large_upper_wick_penalty(self) -> None:
        frame = make_prices()
        frame.loc[89, ["open", "low", "close", "high"]] = [100, 99, 101, 115]
        row = calculate(frame).iloc[-1]
        self.assertGreater(row["penalty_upper_wick"], 0.5)

    def test_insufficient_history(self) -> None:
        result = calculate(make_prices(rows=30))
        row = result.iloc[-1]
        self.assertFalse(row["history_sufficient"])
        self.assertEqual(row["penalty_insufficient_history"], 1)
        self.assertGreaterEqual(row["net_pattern_score"], 0)

    def test_multiple_tickers_are_independent(self) -> None:
        a = make_prices("AAA")
        b = make_prices("BBB")
        a.loc[89, "close"] = 500
        combined = calculate(pd.concat([b.sample(frac=1, random_state=2), a.sample(frac=1, random_state=1)]))
        b_only = calculate(b)
        combined_b = combined.loc[combined["ticker"].eq("BBB")].reset_index(drop=True)
        pd.testing.assert_series_equal(combined_b["prior_high_20"], b_only["prior_high_20"], check_names=False)

    def test_chronological_sorting(self) -> None:
        frame = make_prices().sample(frac=1, random_state=42)
        result = add_price_action_features(frame)
        self.assertTrue(result["date"].is_monotonic_increasing)

    def test_future_data_does_not_change_past_features(self) -> None:
        frame = make_prices(rows=80)
        before = add_price_action_features(frame)
        future = make_prices(rows=81).iloc[[-1]].copy()
        future[["open", "high", "low", "close", "volume"]] = [1, 10_000, 0.01, 8_000, 999_000_000]
        after = add_price_action_features(pd.concat([frame, future], ignore_index=True))
        columns = ["prior_high_20", "prior_low_20", "prior_volume_mean_20", "atr_14", "rsi_14"]
        pd.testing.assert_frame_equal(
            before[columns].reset_index(drop=True),
            after.iloc[: len(before)][columns].reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
