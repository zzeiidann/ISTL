"""Update the canonical IDX 15-minute parquet without discarding history."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "Kronos IDX FineTune 15 Minutes/data/idx_kronos_all_15m.parquet"
DEFAULT_UNIVERSE = ROOT / "Kronos IDX FineTune/data/universe_all.csv"
TZ = ZoneInfo("Asia/Jakarta")
KEY = ["ticker", "date"]


def normalize_ticker(value: str) -> str:
    ticker = str(value).strip().upper().removesuffix(".JK")
    return f"{ticker}.JK"


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip().str.removesuffix(".JK")
    dt = pd.to_datetime(frame["date"])
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert(TZ).dt.tz_localize(None)
    frame["date"] = dt
    numeric = ["open", "high", "low", "close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["volume"] = frame["volume"].fillna(0).clip(lower=0)
    if "amount" not in frame:
        frame["amount"] = frame[["open", "high", "low", "close"]].mean(axis=1) * frame["volume"]
    else:
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
        fallback = frame[["open", "high", "low", "close"]].mean(axis=1) * frame["volume"]
        frame["amount"] = frame["amount"].fillna(fallback)
    clock = frame["date"].dt.hour * 60 + frame["date"].dt.minute
    frame = frame[(clock >= 9 * 60) & (clock < 16 * 60)]
    return frame.dropna(subset=["ticker", "date", "open", "high", "low", "close"])[
        ["ticker", "date", "open", "high", "low", "close", "volume", "amount"]
    ]


def download_one(symbol: str, period: str) -> pd.DataFrame:
    raw = yf.download(
        symbol,
        period=period,
        interval="15m",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        prepost=False,
    )
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    date_column = "Datetime" if "Datetime" in raw else "Date"
    return normalize_frame(
        pd.DataFrame(
            {
                "ticker": symbol.removesuffix(".JK"),
                "date": raw[date_column],
                "open": raw["Open"],
                "high": raw["High"],
                "low": raw["Low"],
                "close": raw["Close"],
                "volume": raw["Volume"],
            }
        )
    )


def completed_bars_only(frame: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    now_jakarta = now.astimezone(TZ) if now else datetime.now(TZ)
    naive_now = pd.Timestamp(now_jakarta.replace(tzinfo=None))
    # Yahoo timestamps mark the beginning of each 15-minute candle.
    return frame[frame["date"] + pd.Timedelta(minutes=15) <= naive_now]


def atomic_parquet(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--period", default="60d")
    parser.add_argument("--pause", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None, help="Debug: update only the first N symbols")
    args = parser.parse_args()

    old = normalize_frame(pd.read_parquet(args.data)) if args.data.exists() else pd.DataFrame()
    universe = pd.read_csv(args.universe)
    tickers = universe["ticker"].dropna().astype(str).drop_duplicates().tolist()
    if args.limit:
        tickers = tickers[: args.limit]

    downloaded: list[pd.DataFrame] = []
    missing: list[str] = []
    for number, ticker in enumerate(tickers, 1):
        symbol = normalize_ticker(ticker)
        try:
            fresh = download_one(symbol, args.period)
            if fresh.empty:
                missing.append(symbol.removesuffix(".JK"))
            else:
                downloaded.append(fresh)
        except Exception as exc:  # one bad symbol must not discard the other updates
            missing.append(f"{symbol.removesuffix('.JK')}: {exc}")
        if number % 25 == 0 or number == len(tickers):
            print(f"{number}/{len(tickers)} | updated={len(downloaded)} | missing={len(missing)}")
        time.sleep(args.pause)

    if not downloaded:
        raise RuntimeError("Yahoo Finance did not return any usable completed TF15 bars.")
    fresh_all = completed_bars_only(pd.concat(downloaded, ignore_index=True))
    combined = pd.concat([old, fresh_all], ignore_index=True) if not old.empty else fresh_all
    combined = normalize_frame(combined).drop_duplicates(KEY, keep="last").sort_values(KEY).reset_index(drop=True)
    atomic_parquet(combined, args.data)
    pd.DataFrame({"missing": missing}).to_csv(args.data.with_name("missing_15m.csv"), index=False)

    old_rows = len(old)
    print(
        f"Saved {len(combined):,} rows ({len(combined) - old_rows:+,}) / "
        f"{combined.ticker.nunique()} tickers / {combined.date.min()} -> {combined.date.max()}\n{args.data}"
    )


if __name__ == "__main__":
    main()
