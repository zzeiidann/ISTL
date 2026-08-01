from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatternConfig:
    """Thresholds for causal price-pattern detection and rank blending."""

    min_history: int = 60
    support_lookback: int = 20
    resistance_lookback: int = 20
    volume_window: int = 20
    volume_multiplier: float = 1.5
    weak_volume_ratio: float = 0.9
    wick_ratio_threshold: float = 2.0
    support_atr_distance: float = 0.75
    breakout_buffer: float = 0.002
    close_near_high: float = 0.70
    swing_window: int = 2
    double_bottom_tolerance: float = 0.035
    double_bottom_min_separation: int = 5
    double_bottom_max_separation: int = 45
    impulse_return: float = 0.08
    flag_window: int = 5
    flag_max_pullback: float = 0.06
    compression_window: int = 10
    compression_ratio: float = 0.75
    extension_threshold: float = 0.12
    gap_up_threshold: float = 0.06
    upper_wick_penalty_ratio: float = 0.35
    volatility_penalty: float = 0.08
    liquidity_threshold: float = 1_000_000_000.0
    zero_volume_penalty: float = 0.10
    penalty_weight: float = 0.50
    base_score_weight: float = 0.75
    pattern_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.min_history < 20:
            raise ValueError("min_history must be at least 20")
        if not 0 <= self.pattern_weight <= 1:
            raise ValueError("pattern_weight must be in [0, 1]")
        if abs(self.base_score_weight + self.pattern_weight - 1.0) > 1e-9:
            raise ValueError("base_score_weight + pattern_weight must equal 1")
        if not 0 <= self.penalty_weight <= 1:
            raise ValueError("penalty_weight must be in [0, 1]")
