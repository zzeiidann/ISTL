from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pattern_screener.config import PatternConfig
from pattern_screener.evaluation import add_forward_outcomes, summarize_forward_outcomes
from pattern_screener.pattern_ranker import rank_enriched_candidates


class RankingAndEvaluationTests(unittest.TestCase):
    def test_pattern_blend_reranks_without_replacing_secondary_score(self) -> None:
        frame = pd.DataFrame(
            {
                "origin": pd.to_datetime(["2026-07-31"] * 3),
                "horizon": [1, 1, 1],
                "ticker": ["BASE", "BALANCED", "PATTERN"],
                "secondary_score": [0.90, 0.80, 0.70],
                "net_pattern_score": [0.05, 0.60, 0.95],
            }
        )
        original = frame["secondary_score"].copy()
        _, selected = rank_enriched_candidates(
            frame,
            PatternConfig(base_score_weight=0.40, pattern_weight=0.60),
            select=3,
        )
        pd.testing.assert_series_equal(frame["secondary_score"], original)
        self.assertEqual(selected.iloc[0]["ticker"], "PATTERN")
        self.assertEqual(set(selected["selected_rank"]), {1, 2, 3})

    def test_forward_outcomes_are_separate_and_numerically_correct(self) -> None:
        dates = pd.date_range("2026-01-01", periods=12, freq="D")
        prices = pd.DataFrame(
            {
                "date": dates,
                "ticker": "AAA",
                "open": np.arange(100, 112),
                "high": np.arange(101, 113),
                "low": np.arange(99, 111),
                "close": np.arange(100, 112),
                "volume": 1_000_000,
            }
        )
        signals = pd.DataFrame({"origin": [dates[5]], "ticker": ["AAA"], "net_pattern_score": [0.8]})
        evaluated = add_forward_outcomes(signals, prices, horizons=(1, 3, 5))
        self.assertAlmostEqual(evaluated.loc[0, "forward_return_1d"], 1 / 105)
        self.assertAlmostEqual(evaluated.loc[0, "forward_return_3d"], 3 / 105)
        self.assertNotIn("forward_return_1d", signals.columns)
        summary = summarize_forward_outcomes(evaluated, horizons=(1, 3, 5))
        self.assertEqual(summary.loc[summary["horizon"].eq(3), "signal_count"].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
