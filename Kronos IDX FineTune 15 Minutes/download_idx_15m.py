"""Download and normalize the recent IDX 15-minute universe from Yahoo Finance."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent
DEFAULT_UNIVERSE = ROOT.parent / "Kronos IDX FineTune" / "data" / "universe_all.csv"
DEFAULT_OUTPUT = ROOT / "data" / "idx_kronos_all_15m.parquet"


def normalize_ticker(value: str) -> str:
    ticker = str(value).strip().upper()
    return ticker if ticker.endswith(".JK") else f"{ticker}.JK"


def download_one(symbol: str, period: str) -> pd.DataFrame:
    frame = yf.download(
        symbol, period=period, interval="15m", auto_adjust=False,
        actions=False, progress=False, threads=False, prepost=False,
    )
    if frame.empty:
        return frame
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    date_col = "Datetime" if "Datetime" in frame.columns else "Date"
    dt = pd.to_datetime(frame[date_col])
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
    out = pd.DataFrame({
        "ticker": symbol.removesuffix(".JK"),
        "date": dt,
        "open": frame["Open"], "high": frame["High"],
        "low": frame["Low"], "close": frame["Close"],
        "volume": frame["Volume"],
    })
    out["amount"] = out[["open", "high", "low", "close"]].mean(axis=1) * out["volume"]
    # Yahoo's IDX stream is already regular-session only; this guard removes
    # accidental pre/post-market records while retaining Friday's split session.
    clock = out["date"].dt.hour * 60 + out["date"].dt.minute
    return out[(clock >= 9 * 60) & (clock < 16 * 60)].dropna(subset=["open", "high", "low", "close"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--period", default="60d", help="Yahoo intraday period, normally 60d")
    parser.add_argument("--pause", type=float, default=0.15)
    args = parser.parse_args()

    universe = pd.read_csv(args.universe)
    tickers = universe["ticker"].dropna().astype(str).drop_duplicates().tolist()
    frames, missing = [], []
    for index, ticker in enumerate(tickers, 1):
        symbol = normalize_ticker(ticker)
        try:
            frame = download_one(symbol, args.period)
            (frames if len(frame) else missing).append(frame if len(frame) else ticker)
        except Exception as exc:
            missing.append(f"{ticker}: {exc}")
        if index % 25 == 0:
            print(f"{index}/{len(tickers)} processed | usable: {len(frames)} | missing: {len(missing)}")
        time.sleep(args.pause)

    if not frames:
        raise RuntimeError("Yahoo Finance tidak mengembalikan data 15 menit untuk universe ini.")
    data = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    data = data.drop_duplicates(["ticker", "date"], keep="last")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.output, index=False)
    pd.DataFrame({"missing": missing}).to_csv(args.output.with_name("missing_15m.csv"), index=False)
    print(f"Saved {len(data):,} rows, {data.ticker.nunique()} tickers -> {args.output}")


if __name__ == "__main__":
    main()
