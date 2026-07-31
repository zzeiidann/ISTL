"""Build the 80 GB Blackwell notebook that forecasts from the validated model."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
SOURCE_BUILDER = ROOT / "build_kaggle_notebook_80gb.py"
SOURCE_NOTEBOOK = ROOT / "kronos_idx_kaggle_finetune_80gb.ipynb"
OUTPUT_NOTEBOOK = ROOT / "kronos_idx_kaggle_finetune_80gb_no_refit.ipynb"


def source(text: str) -> list[str]:
    return dedent(text).strip().splitlines(True)


subprocess.run([sys.executable, str(SOURCE_BUILDER)], check=True)
notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

notebook["cells"][0]["source"] = source(
    """
    # Kronos-base — IDX Validated Production Forecast (80 GB, No Refit)

    Blackwell/A100/H100 high-throughput variant with no post-validation refit.
    Predictor gradients use targets ending before 1 July 2026. The complete
    1–30 July period is deterministic validation, and the best validated
    checkpoint forecasts from live context through the 30 July close.

    This preserves an independently measured final model while keeping inference
    current. The tokenizer remains frozen and the causal next-token objective is
    unchanged.
    """
)

for cell in notebook["cells"]:
    text = "".join(cell.get("source", []))

    if "## 7. Probabilistic forecast seluruh emiten" in text:
        cell["source"] = source(
            """
            ## 7. Load the best validated model and prepare live context

            Tidak ada refit setelah model selection. Checkpoint dengan loss Juli
            terbaik langsung digunakan untuk inference. Data sampai 30 Juli tetap
            masuk sebagai causal context, tetapi tidak mengubah bobot model.
            """
        )

    elif "REFIT_WINDOWS_PER_EPOCH =" in text:
        text = text.replace("REFIT_WINDOWS_PER_EPOCH = 200_000\n", "")
        cell["source"] = text.splitlines(True)

    elif "class DynamicPanelKlineDataset(Dataset):" in text:
        text = text.replace('{"train", "val", "refit"}', '{"train", "val"}')
        text = text.replace("split must be train, val, or refit", "split must be train or val")
        text = text.replace(
            '\n                    or split == "refit" and target_end <= TRAIN_END',
            "",
        )
        text = text.replace('{"train": 1, "refit": 2}.get(self.split, 0)', '{"train": 1}.get(self.split, 0)')
        cell["source"] = text.splitlines(True)

    elif "# Clean production refit:" in text:
        cell["source"] = source(
            """
            # Final model = independently validated checkpoint. No July gradients.
            best_model = Kronos.from_pretrained(str(CHECKPOINT_DIR)).to(DEVICE).eval()
            predictor = KronosPredictor(best_model, tokenizer, device=str(DEVICE), max_context=MAX_CONTEXT)
            print(f"Loaded best validated checkpoint from {CHECKPOINT_DIR}")
            print(f"Best epoch: {best_epoch} | July validation loss: {best_val:.6f}")

            contexts, x_times, y_times, valid_tickers, last_close_map, skipped = [], [], [], [], {}, []
            global_future_dates = pd.Series(pd.bdate_range(AS_OF_DATE + pd.offsets.BDay(1), periods=PRED_LEN))
            for ticker in TICKERS:
                g = raw[(raw.ticker.eq(ticker)) & (raw.date <= AS_OF_DATE)].sort_values("date").tail(LOOKBACK).copy()
                if len(g) < LOOKBACK:
                    skipped.append({
                        "ticker": ticker, "reason": "insufficient_history", "bars": len(g),
                        "last_date": g.date.max() if len(g) else pd.NaT,
                    })
                    continue
                stale_days = int((AS_OF_DATE - g["date"].max()).days)
                if stale_days > MAX_STALE_CALENDAR_DAYS:
                    skipped.append({
                        "ticker": ticker, "reason": "stale_or_suspended", "bars": len(g),
                        "last_date": g.date.max(),
                    })
                    continue
                contexts.append(g[FEATURES].copy())
                x_times.append(pd.Series(pd.to_datetime(g["date"]).to_numpy()))
                y_times.append(global_future_dates.copy())
                valid_tickers.append(ticker)
                last_close_map[ticker] = float(g["close"].iloc[-1])

            skipped_df = pd.DataFrame(skipped)
            skipped_df.to_csv(OUTPUT_DIR / "skipped_tickers.csv", index=False)
            print(f"Eligible forecast: {len(valid_tickers)} / {len(TICKERS)} | skipped: {len(skipped_df)}")
            display(skipped_df["reason"].value_counts().rename("count").to_frame() if len(skipped_df) else pd.DataFrame())
            """
        )

    elif '"training_profile": "80gb_dynamic_refit"' in text:
        text = text.replace(
            '"training_profile": "80gb_dynamic_refit"',
            '"training_profile": "80gb_dynamic_no_refit"',
        )
        text = text.replace(
            '    "refit_windows_per_epoch": REFIT_WINDOWS_PER_EPOCH,\n',
            '    "final_model_source": "best_july_validation_checkpoint",\n'
            '    "post_validation_refit": False,\n',
        )
        cell["source"] = text.splitlines(True)

OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT_NOTEBOOK)
