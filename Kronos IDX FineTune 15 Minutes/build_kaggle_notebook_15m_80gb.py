"""Build the 15-minute 80 GB notebook from the validated daily template."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
DAILY_ROOT = ROOT.parent / "Kronos IDX FineTune"
SOURCE_BUILDER = DAILY_ROOT / "build_kaggle_notebook_80gb.py"
SOURCE_NOTEBOOK = DAILY_ROOT / "kronos_idx_kaggle_finetune_80gb.ipynb"
OUTPUT_NOTEBOOK = ROOT / "kronos_idx_kaggle_finetune_15m_80gb.ipynb"


def source(text: str) -> list[str]:
    return dedent(text).strip().splitlines(True)


subprocess.run([sys.executable, str(SOURCE_BUILDER)], check=True)
notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
notebook["cells"][0]["source"] = source(
    """
    # Kronos-base — IDX 15-Minute Production Fine-Tuning (80 GB GPU)

    Pipeline intraday terpisah dengan profil optimizer yang sama seperti varian
    harian 80 GB. Context 240 bar mewakili kira-kira 12 sesi IDX dan forecast
    20 bar mewakili kira-kira satu sesi. Tokenizer pretrained tetap dibekukan.
    July validation diikuti clean production refit, sama seperti template harian.
    """
)

for cell in notebook["cells"]:
    text = "".join(cell.get("source", []))

    text = text.replace("idx_kronos_all_daily.parquet", "idx_kronos_all_15m.parquet")
    text = text.replace('root / "Kronos IDX FineTune" / "data"', 'root / "Kronos IDX FineTune 15 Minutes" / "data"')
    text = text.replace('PROJECT_DIR = DATA_PATH.parent.parent\nKRONOS_DIR = PROJECT_DIR / "Kronos"',
                        'PROJECT_DIR = DATA_PATH.parent.parent\nDAILY_PROJECT_DIR = PROJECT_DIR.parent / "Kronos IDX FineTune"\nKRONOS_DIR = DAILY_PROJECT_DIR / "Kronos"')
    text = text.replace('RUNTIME_ROOT / "kronos_idx_outputs"', 'RUNTIME_ROOT / "kronos_idx_15m_outputs"')
    text = text.replace("LOOKBACK = 120", "BARS_PER_SESSION = 20\nLOOKBACK = 240")
    text = text.replace('TRAIN_END = pd.Timestamp("2026-07-30")', 'TRAIN_END = pd.Timestamp("2026-07-30 23:59:59")')
    text = text.replace('AS_OF_DATE = pd.Timestamp("2026-07-30")', 'AS_OF_DATE = pd.Timestamp("2026-07-30 23:59:59")')
    text = text.replace('RECENT_START = pd.Timestamp("2024-01-01")', 'RECENT_START = pd.Timestamp("2026-06-01")')
    text = text.replace("varian harian", "varian 15-menit")
    text = text.replace("OHLCV harian", "OHLCV 15-menit")
    text = text.replace("20d", "20bar").replace("20 hari", "20 bar")
    text = text.replace("horizon_day", "horizon_bar")
    text = text.replace("forecast day", "forecast bar").replace("Forecast day", "Forecast bar")
    text = text.replace("day1_to_day5", "bar1_to_bar5")
    text = text.replace("DAY {day}", "BAR {day}")
    text = text.replace('"training_profile": "80gb_dynamic_refit"', '"training_profile": "80gb_15m_dynamic_refit"')
    text = text.replace('"lookback": LOOKBACK,', '"interval": "15m",\n    "bars_per_session_assumption": BARS_PER_SESSION,\n    "lookback": LOOKBACK,')
    cell["source"] = text.splitlines(True)

# Intraday timestamps must preserve hours/minutes and future timestamps must use
# an observed IDX bar schedule instead of a business-day calendar.
for cell in notebook["cells"]:
    text = "".join(cell.get("source", []))
    if "raw = pd.read_parquet(DATA_PATH)" in text:
        text = text.replace(
            'raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)',
            'raw["date"] = pd.to_datetime(raw["date"])\n'
            'if raw["date"].dt.tz is not None:\n'
            '    raw["date"] = raw["date"].dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)',
        )
        text = text.replace("raw.date.min().date()} → {raw.date.max().date()}", "raw.date.min()} → {raw.date.max()}")
        cell["source"] = text.splitlines(True)
    elif "global_future_dates = pd.Series(pd.bdate_range" in text:
        text = text.replace(
            "global_future_dates = pd.Series(pd.bdate_range(AS_OF_DATE + pd.offsets.BDay(1), periods=PRED_LEN))",
            """# Reuse the most common observed intraday clock. Twenty bars normally
# span one IDX session; Friday/session anomalies remain data-driven.
session_clock = (
    raw.assign(clock=raw[\"date\"].dt.strftime(\"%H:%M\"))
    .groupby([raw[\"date\"].dt.date, \"clock\"]).size().reset_index(name=\"n\")
    .groupby(\"clock\")[\"n\"].count().sort_values(ascending=False)
)
clock_values = sorted(session_clock.head(BARS_PER_SESSION).index)
future_values, future_day = [], (AS_OF_DATE + pd.offsets.BDay(1)).normalize()
while len(future_values) < PRED_LEN:
    if future_day.weekday() < 5:
        future_values.extend(
            future_day + pd.Timedelta(hours=int(clock[:2]), minutes=int(clock[3:]))
            for clock in clock_values
        )
    future_day += pd.offsets.BDay(1)
global_future_dates = pd.Series(future_values[:PRED_LEN])""",
        )
        cell["source"] = text.splitlines(True)

OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT_NOTEBOOK)
