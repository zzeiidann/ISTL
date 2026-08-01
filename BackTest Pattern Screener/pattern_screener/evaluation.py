from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward_outcomes(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Attach forward outcomes after signal generation; never call from live features."""

    required = {"origin", "ticker"}
    if missing := required.difference(signals.columns):
        raise ValueError(f"Signal columns missing: {sorted(missing)}")
    price = prices.copy()
    price["date"] = pd.to_datetime(price["date"]).dt.tz_localize(None).dt.normalize()
    price = price.sort_values(["ticker", "date"])
    output = signals.copy()
    output["origin"] = pd.to_datetime(output["origin"]).dt.tz_localize(None).dt.normalize()

    records: list[dict[str, float | str | pd.Timestamp]] = []
    for ticker, ticker_signals in output.groupby("ticker", sort=False):
        history = price.loc[price["ticker"].eq(ticker)].reset_index(drop=True)
        positions = {date: index for index, date in enumerate(history["date"])}
        for signal_index, signal in ticker_signals.iterrows():
            origin = signal["origin"]
            position = positions.get(origin)
            if position is None:
                continue
            origin_close = float(history.loc[position, "close"])
            record: dict[str, float | str | pd.Timestamp] = {"_signal_index": signal_index}
            for horizon in horizons:
                future = history.iloc[position + 1 : position + 1 + horizon]
                if len(future) < horizon or origin_close <= 0:
                    record[f"forward_return_{horizon}d"] = np.nan
                    record[f"mfe_{horizon}d"] = np.nan
                    record[f"mae_{horizon}d"] = np.nan
                    continue
                record[f"forward_return_{horizon}d"] = float(future.iloc[-1]["close"] / origin_close - 1)
                record[f"mfe_{horizon}d"] = float(future["high"].max() / origin_close - 1)
                record[f"mae_{horizon}d"] = float(future["low"].min() / origin_close - 1)
            records.append(record)
    outcomes = pd.DataFrame(records).set_index("_signal_index") if records else pd.DataFrame()
    return output.join(outcomes)


def summarize_forward_outcomes(
    evaluated: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Summarize returns, hit rates, MFE, and MAE for offline analysis."""

    rows: list[dict[str, float | int]] = []
    for horizon in horizons:
        returns = pd.to_numeric(evaluated[f"forward_return_{horizon}d"], errors="coerce").dropna()
        mfe = pd.to_numeric(evaluated[f"mfe_{horizon}d"], errors="coerce").dropna()
        mae = pd.to_numeric(evaluated[f"mae_{horizon}d"], errors="coerce").dropna()
        rows.append(
            {
                "horizon": horizon,
                "signal_count": int(len(returns)),
                "mean_forward_return": float(returns.mean()) if len(returns) else np.nan,
                "median_forward_return": float(returns.median()) if len(returns) else np.nan,
                "hit_rate_above_0": float(returns.gt(0).mean()) if len(returns) else np.nan,
                "hit_rate_above_3": float(mfe.ge(0.03).mean()) if len(mfe) else np.nan,
                "hit_rate_above_5": float(mfe.ge(0.05).mean()) if len(mfe) else np.nan,
                "maximum_favorable_excursion": float(mfe.mean()) if len(mfe) else np.nan,
                "maximum_adverse_excursion": float(mae.mean()) if len(mae) else np.nan,
            }
        )
    return pd.DataFrame(rows)
