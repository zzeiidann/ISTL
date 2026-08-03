"""Build the 30-minute 80 GB notebook from the 15-minute pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TF15_ROOT = ROOT.parent / "Kronos IDX FineTune 15 Minutes"
SOURCE_BUILDER = TF15_ROOT / "build_kaggle_notebook_15m_80gb.py"
SOURCE_NOTEBOOK = TF15_ROOT / "kronos_idx_kaggle_finetune_15m_80gb.ipynb"
OUTPUT_NOTEBOOK = ROOT / "kronos_idx_kaggle_finetune_30m_80gb.ipynb"


subprocess.run([sys.executable, str(SOURCE_BUILDER)], check=True)
notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

replacements = {
    "15-Minute": "30-Minute", "15-minute": "30-minute",
    "15-menit": "30-menit", "15 menit": "30 menit",
    "idx_kronos_all_15m.parquet": "idx_kronos_all_30m.parquet",
    "Kronos IDX FineTune 15 Minutes": "Kronos IDX FineTune 30 Minutes",
    "kronos_idx_15m_outputs": "kronos_idx_30m_outputs",
    'BARS_PER_SESSION = 20': 'BARS_PER_SESSION = 10',
    'LOOKBACK = 240': 'LOOKBACK = 120',
    'PRED_LEN = 20': 'PRED_LEN = 10',
    "20bar": "10bar", "Twenty bars": "Ten bars",
    '"interval": "15m"': '"interval": "30m"',
    '"training_profile": "80gb_15m_dynamic_refit"': '"training_profile": "80gb_30m_dynamic_refit"',
}

for cell in notebook["cells"]:
    text = "".join(cell.get("source", []))
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Keep weight updates capped at 30 July, but use the completed 31 July bars
    # as real inference context so the next forecast session is 3 August.
    text = text.replace(
        'AS_OF_DATE = pd.Timestamp("2026-07-30 23:59:59")',
        'AS_OF_DATE = pd.Timestamp("2026-07-31 23:59:59")',
    )
    cell["source"] = text.splitlines(True)

OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT_NOTEBOOK)
