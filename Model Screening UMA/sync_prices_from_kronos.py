from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Kronos IDX FineTune" / "data" / "idx_kronos_all_daily.parquet"
TARGET = Path(__file__).resolve().parent / "data" / "raw" / "prices_20210616_20260731.parquet"


def main() -> None:
    """Synchronize the UMA raw-price cache from the canonical Kronos OHLCV file."""

    source = pd.read_parquet(SOURCE)
    source = source[["date", "open", "high", "low", "close", "volume", "ticker"]].copy()
    source.insert(5, "adj_close", source["close"])
    source["date"] = pd.to_datetime(source["date"]).dt.tz_localize(None).dt.normalize()

    existing = pd.read_parquet(TARGET)
    existing["date"] = pd.to_datetime(existing["date"]).dt.tz_localize(None).dt.normalize()
    cutoff = source["date"].min()
    combined = pd.concat([existing.loc[existing["date"].lt(cutoff)], source], ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    combined.to_parquet(TARGET, index=False)
    latest = combined["date"].max().date()
    coverage = combined.loc[combined["date"].eq(combined["date"].max()), "ticker"].nunique()
    print(f"Updated {TARGET}: latest={latest}, tickers={coverage}, rows={len(combined):,}")


if __name__ == "__main__":
    main()
