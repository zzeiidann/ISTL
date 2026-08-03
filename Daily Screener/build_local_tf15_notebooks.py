"""Generate the two local TF15 workflow notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


HERE = Path(__file__).resolve().parent


def source(text: str) -> list[str]:
    return dedent(text).strip().splitlines(True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source(text)}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Daily Screener (uv)", "language": "python", "name": "daily-screener"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


update_cells = [
    markdown("""
    # 01 — Update & compound IDX TF15 parquet

    Download rolling TF15 Yahoo Finance, keep only completed candles, then merge it
    into the existing historical parquet by `(ticker, date)`. The write is atomic,
    so an interrupted run does not replace the good parquet.
    """),
    code("""
    from pathlib import Path
    import subprocess, sys

    HERE = Path.cwd().resolve()
    if HERE.name != "Daily Screener":
        HERE = HERE / "Daily Screener"
    assert (HERE / "update_tf15_parquet.py").exists(), f"Open this notebook from the ISTL repository: {HERE}"

    command = [sys.executable, str(HERE / "update_tf15_parquet.py"), "--period", "60d", "--pause", "0.15"]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    """),
    code("""
    import pandas as pd
    data_path = HERE.parent / "Kronos IDX FineTune 15 Minutes/data/idx_kronos_all_15m.parquet"
    prices = pd.read_parquet(data_path)
    print({
        "rows": len(prices), "tickers": prices.ticker.nunique(),
        "first_bar": str(prices.date.min()), "last_completed_bar": str(prices.date.max()),
        "duplicates": int(prices.duplicated(["ticker", "date"]).sum()),
    })
    prices.tail()
    """),
]

project_cells = [
    markdown("""
    # 02 — Project next IDX session with the TF15 model

    Uses actual TF15 context through the newest completed candle. By default it
    screens the 30 most liquid eligible stocks and ranks the predicted first
    15-minute candle of the next weekday. CPU is supported; it is slower than CUDA.

    Run notebook 01 first. If tomorrow is an IDX holiday, set `TARGET_DATE` manually.
    """),
    code("""
    from pathlib import Path
    import importlib.util

    HERE = Path.cwd().resolve()
    if HERE.name != "Daily Screener":
        HERE = HERE / "Daily Screener"
    module_path = HERE / "project_tf15_next_session.py"
    assert module_path.exists(), f"Open this notebook from the ISTL repository: {HERE}"
    spec = importlib.util.spec_from_file_location("tf15_projection", module_path)
    tf15 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tf15)
    """),
    markdown("## Configuration"),
    code("""
    CANDIDATE_COUNT = 30
    LOOKBACK_BARS = 240
    SAMPLE_PATHS = 3       # use 1 for a quick CPU smoke test; 5 for a stronger estimate
    BATCH_SIZE = None      # automatic: CPU=2, CUDA=16
    TARGET_DATE = None     # example: "2026-08-04"; None = next weekday after latest actual bar
    """),
    code("""
    ranking, forecast_paths, metadata = tf15.run_projection(
        candidate_count=CANDIDATE_COUNT,
        lookback=LOOKBACK_BARS,
        paths=SAMPLE_PATHS,
        batch_size=BATCH_SIZE,
        target_date=TARGET_DATE,
    )
    metadata
    """),
    code("""
    from IPython.display import display
    styled = ranking.style.format({
        "anchor_close": "{:,.2f}", "expected_opening_bar_close": "{:,.2f}",
        "expected_return": "{:+.2%}", "median_return": "{:+.2%}",
        "probability_up": "{:.0%}", "downside_p10": "{:+.2%}",
    }).background_gradient(subset=["expected_return"], cmap="RdYlGn")
    display(styled)
    """),
]


for filename, cells in (
    ("01_update_compound_tf15.ipynb", update_cells),
    ("02_project_next_session_tf15.ipynb", project_cells),
):
    (HERE / filename).write_text(json.dumps(notebook(cells), indent=1) + "\n")
    print("Wrote", HERE / filename)
