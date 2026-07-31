from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


DATA_DIR = Path(__file__).resolve().parent / "data"
PARQUET_PATH = DATA_DIR / "idx_kronos_all_daily.parquet"
UNIVERSE_PATH = DATA_DIR / "universe_all.csv"
SUMMARY_PATH = DATA_DIR / "dataset_summary.csv"


def write_summary(frame: pd.DataFrame) -> None:
    summary = (
        frame.groupby("ticker", as_index=False)
        .agg(
            rows=("date", "size"),
            start=("date", "min"),
            end=("date", "max"),
            zero_volume=("volume", lambda values: int(values.eq(0).sum())),
            sector=("sector", "last"),
            subsector=("subsector", "last"),
        )
    )
    summary["start"] = pd.to_datetime(summary["start"]).dt.strftime("%Y-%m-%d")
    summary["end"] = pd.to_datetime(summary["end"]).dt.strftime("%Y-%m-%d")
    summary.to_csv(SUMMARY_PATH, index=False)


def download_session(tickers: list[str], session: pd.Timestamp, batch_size: int = 100) -> pd.DataFrame:
    yahoo_to_idx = {f"{ticker}.JK": ticker for ticker in tickers}
    frames: list[pd.DataFrame] = []

    symbols = list(yahoo_to_idx)
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        result = yf.download(
            batch,
            start=session.strftime("%Y-%m-%d"),
            end=(session + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if result.empty:
            continue

        for symbol in batch:
            try:
                item = result[symbol] if isinstance(result.columns, pd.MultiIndex) else result
            except KeyError:
                continue
            item = item.dropna(subset=["Open", "High", "Low", "Close"], how="any")
            if item.empty:
                continue
            row = item.iloc[-1]
            frames.append(
                pd.DataFrame(
                    {
                        "date": [session],
                        "ticker": [yahoo_to_idx[symbol]],
                        "open": [float(row["Open"])],
                        "high": [float(row["High"])],
                        "low": [float(row["Low"])],
                        "close": [float(row["Close"])],
                        "volume": [float(row["Volume"])],
                    }
                )
            )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("ticker", keep="last")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh one completed IDX session in the Kronos parquet.")
    parser.add_argument("--date", help="Completed trading session in YYYY-MM-DD format")
    parser.add_argument("--refresh-summary-only", action="store_true")
    args = parser.parse_args()

    old = pd.read_parquet(PARQUET_PATH)
    old["date"] = pd.to_datetime(old["date"]).dt.tz_localize(None).dt.normalize()
    if args.refresh_summary_only:
        write_summary(old)
        print(f"Updated {SUMMARY_PATH}.")
        return
    if not args.date:
        parser.error("--date is required unless --refresh-summary-only is used")
    session = pd.Timestamp(args.date).normalize()

    universe = pd.read_csv(UNIVERSE_PATH)
    eligible = universe.loc[universe["has_yfinance_data"].astype(bool), "ticker"].dropna().astype(str).tolist()
    metadata = universe.set_index("ticker")[["sector", "subsector"]]

    fresh = download_session(eligible, session)
    if fresh.empty:
        raise RuntimeError(f"No Yahoo Finance rows returned for {session.date()}.")

    fresh = fresh.join(metadata, on="ticker")
    fresh["amount"] = fresh[["open", "high", "low", "close"]].mean(axis=1) * fresh["volume"]
    fresh["source"] = "yfinance"
    fresh["frequency"] = "1d"
    fresh = fresh[old.columns]

    old_session_count = old.loc[old["date"].eq(session), "ticker"].nunique()
    new_session_count = fresh["ticker"].nunique()
    print(f"{session.date()}: existing={old_session_count}, downloaded={new_session_count}")
    if new_session_count < old_session_count:
        raise RuntimeError("Downloaded coverage is worse than the existing parquet; refusing to overwrite.")

    merged = pd.concat([old.loc[~old["date"].eq(session)], fresh], ignore_index=True)
    merged = merged.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    merged.to_parquet(PARQUET_PATH, index=False)
    write_summary(merged)
    print(f"Updated {PARQUET_PATH} with {new_session_count} rows for {session.date()}.")


if __name__ == "__main__":
    main()
