"""Build the high-memory GPU variant without changing the standard notebook."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
BASE_BUILDER = ROOT / "build_kaggle_notebook.py"
BASE_NOTEBOOK = ROOT / "kronos_idx_kaggle_finetune.ipynb"
OUTPUT_NOTEBOOK = ROOT / "kronos_idx_kaggle_finetune_80gb.ipynb"


def source(text: str) -> list[str]:
    return dedent(text).strip().splitlines(True)


subprocess.run([sys.executable, str(BASE_BUILDER)], check=True)
notebook = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))

notebook["cells"][0]["source"] = source(
    """
    # Kronos-base — IDX Production Fine-Tuning (80 GB GPU)

    High-throughput production variant for an A100/H100-class 80 GB GPU.
    It keeps the original Kronos causal next-token objective and frozen tokenizer,
    but replaces the fixed 20,000-window subset with fresh, recency-aware sampling
    every epoch. July 2026 is used as full deterministic validation, followed by a
    clean production refit through 30 July using the selected epoch count.

    The standard notebook remains available separately.
    """
)

for cell in notebook["cells"]:
    text = "".join(cell.get("source", []))

    if "torch==2.3.1" in text and "cu118" in text:
        cell["source"] = source(
            """
            # Blackwell (sm_120) requires a PyTorch binary built with CUDA 12.8+.
            # After this install completes, restart the Kaggle kernel once, then Run All.
            %pip install -q --upgrade torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
            %pip install -q einops==0.8.1 huggingface_hub==0.33.1 safetensors==0.6.2 pyarrow plotly kaleido tqdm
            """
        )

    elif "from pathlib import Path" in text and "CUDA compatibility smoke test" in text:
        text = text.replace(
            "# Kronos memakai scaled_dot_product_attention. Fused SDPA dapat memilih\n"
            "    # kernel yang tidak tersedia pada P100; math SDPA bekerja pada P100/T4.",
            "# Blackwell memakai CUDA 12.8 build; izinkan PyTorch memilih SDPA tercepat.",
        )
        text = text.replace("torch.backends.cuda.enable_flash_sdp(False)", "torch.backends.cuda.enable_flash_sdp(True)")
        text = text.replace("torch.backends.cuda.enable_mem_efficient_sdp(False)", "torch.backends.cuda.enable_mem_efficient_sdp(True)")
        text = text.replace(
            '"Restart Kaggle session, lalu Run All agar torch 2.3.1+cu118 dimuat sebelum import."',
            '"Install torch 2.7.1+cu128 dari sel pertama, restart kernel, lalu Run All."',
        )
        text = text.replace(
            'print("✓ CUDA compatibility smoke test passed; math SDPA enabled.")',
            'print("✓ CUDA compatibility smoke test passed; Blackwell SDPA enabled.")',
        )
        cell["source"] = text.splitlines(True)

    elif "balanced sample 20.000 windows" in text:
        cell["source"] = source(
            """
            ## 2. Configuration — 80 GB profile

            Setiap epoch mengambil 200.000 window baru dari candidate pool dengan
            probabilitas lebih tinggi untuk rezim terbaru. Validation Juli tidak
            disampling. BF16, batch 128, TF32, dan worker paralel ditujukan untuk
            A100/H100 80 GB; batch dapat dinaikkan ke 256 setelah benchmark.
            """
        )

    elif 'MODEL_ID = "NeoQuasar/Kronos-base"' in text:
        cell["source"] = source(
            """
            MODEL_ID = "NeoQuasar/Kronos-base"
            LOOKBACK = 120
            PRED_LEN = 20
            MAX_CONTEXT = 512
            TRAIN_END = pd.Timestamp("2026-07-30")
            VAL_START = pd.Timestamp("2026-07-01")
            RECENT_START = pd.Timestamp("2024-01-01")
            AS_OF_DATE = pd.Timestamp("2026-07-30")
            MAX_STALE_CALENDAR_DAYS = 10

            BATCH_SIZE = 128
            EPOCHS = 4
            LEARNING_RATE = 5e-6
            WEIGHT_DECAY = 0.05
            WARMUP_RATIO = 0.05
            TRAIN_WINDOWS_PER_EPOCH = 200_000
            REFIT_WINDOWS_PER_EPOCH = 200_000
            RECENT_SAMPLE_WEIGHT = 3.0
            NUM_WORKERS = 12
            MIN_ACTIVE_RATIO = 0.60
            PATIENCE = 2

            N_FORECAST_PATHS = 5
            INFERENCE_ASSET_BATCH = 32
            TEMPERATURE = 0.8
            TOP_P = 0.9
            SEED = 42

            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            AMP_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            print("Device:", DEVICE, "| AMP dtype:", AMP_DTYPE)
            """
        )

    elif "class PanelKlineDataset(Dataset):" in text:
        cell["source"] = source(
            """
            FEATURES = ["open", "high", "low", "close", "volume", "amount"]
            TIME_FEATURES = ["minute", "hour", "weekday", "day", "month"]

            class DynamicPanelKlineDataset(Dataset):
                def __init__(self, frame, split, samples_per_epoch=None):
                    if split not in {"train", "val", "refit"}:
                        raise ValueError("split must be train, val, or refit")
                    self.split = split
                    self.samples_per_epoch = samples_per_epoch
                    self.series, self.candidates = {}, []
                    window = LOOKBACK + PRED_LEN + 1

                    for ticker, g in frame.groupby("ticker", sort=False):
                        g = g.sort_values("date").reset_index(drop=True).copy()
                        g["minute"] = g["date"].dt.minute
                        g["hour"] = g["date"].dt.hour
                        g["weekday"] = g["date"].dt.weekday
                        g["day"] = g["date"].dt.day
                        g["month"] = g["date"].dt.month
                        self.series[ticker] = g

                        for start in range(len(g) - window + 1):
                            target_start = g.loc[start + LOOKBACK, "date"]
                            target_end = g.loc[start + window - 1, "date"]
                            active_ratio = (g.loc[start:start + LOOKBACK - 1, "volume"] > 0).mean()
                            if active_ratio < MIN_ACTIVE_RATIO:
                                continue
                            eligible = (
                                split == "train" and target_end < VAL_START
                                or split == "val" and target_start >= VAL_START and target_end <= TRAIN_END
                                or split == "refit" and target_end <= TRAIN_END
                            )
                            if eligible:
                                self.candidates.append((ticker, start, target_end))

                    self.resample(0)
                    print(f"{split}: {len(self.candidates):,} candidates; {len(self.indices):,} active windows")

                def resample(self, epoch):
                    if not self.samples_per_epoch or len(self.candidates) <= self.samples_per_epoch:
                        self.indices = [(ticker, start) for ticker, start, _ in self.candidates]
                        return
                    rng = np.random.default_rng(SEED + 10_000 * ({"train": 1, "refit": 2}.get(self.split, 0)) + epoch)
                    weights = np.fromiter(
                        (RECENT_SAMPLE_WEIGHT if target_end >= RECENT_START else 1.0 for _, _, target_end in self.candidates),
                        dtype=np.float64,
                    )
                    weights /= weights.sum()
                    chosen = rng.choice(len(self.candidates), self.samples_per_epoch, replace=False, p=weights)
                    self.indices = [(self.candidates[i][0], self.candidates[i][1]) for i in chosen]

                def __len__(self):
                    return len(self.indices)

                def __getitem__(self, idx):
                    ticker, start = self.indices[idx]
                    window = self.series[ticker].iloc[start:start + LOOKBACK + PRED_LEN + 1]
                    x = window[FEATURES].to_numpy(np.float32)
                    stamps = window[TIME_FEATURES].to_numpy(np.float32)
                    mean, std = x[:LOOKBACK].mean(axis=0), x[:LOOKBACK].std(axis=0)
                    x = np.clip((x - mean) / (std + 1e-5), -5, 5)
                    return torch.from_numpy(x), torch.from_numpy(stamps)

            def make_loader(dataset, shuffle):
                return DataLoader(
                    dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                    num_workers=NUM_WORKERS, pin_memory=True, drop_last=shuffle,
                    persistent_workers=NUM_WORKERS > 0, prefetch_factor=4,
                )

            train_ds = DynamicPanelKlineDataset(raw[raw.date <= TRAIN_END], "train", TRAIN_WINDOWS_PER_EPOCH)
            val_ds = DynamicPanelKlineDataset(raw[raw.date <= TRAIN_END], "val")
            if not len(train_ds) or not len(val_ds):
                raise RuntimeError("Train/validation windows kosong; periksa tanggal dan coverage data.")
            val_loader = make_loader(val_ds, False)
            """
        )

    elif "optimizer = torch.optim.AdamW(" in text and "history = []" in text:
        cell["source"] = source(
            """
            def causal_epoch(model, dataset, optimizer, scheduler, scaler, epoch, label):
                dataset.resample(epoch)
                loader = make_loader(dataset, True)
                model.train()
                total = 0.0
                optimizer.zero_grad(set_to_none=True)
                progress = tqdm(loader, desc=f"{label} epoch {epoch}", leave=False)
                for step, (batch_x, batch_stamp) in enumerate(progress, 1):
                    batch_x = batch_x.to(DEVICE, non_blocking=True)
                    batch_stamp = batch_stamp.to(DEVICE, non_blocking=True)
                    with torch.no_grad():
                        token_0, token_1 = tokenizer.encode(batch_x, half=True)
                    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
                        logits = model(token_0[:, :-1], token_1[:, :-1], batch_stamp[:, :-1, :])
                        loss, _, _ = model.head.compute_loss(
                            logits[0], logits[1], token_0[:, 1:], token_1[:, 1:]
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    total += loss.item()
                    progress.set_postfix(loss=f"{total / step:.4f}")
                return total / len(loader)

            @torch.no_grad()
            def validation_loss(model):
                model.eval()
                total = 0.0
                for step, (batch_x, batch_stamp) in enumerate(tqdm(val_loader, desc="full July validation", leave=False), 1):
                    batch_x = batch_x.to(DEVICE, non_blocking=True)
                    batch_stamp = batch_stamp.to(DEVICE, non_blocking=True)
                    token_0, token_1 = tokenizer.encode(batch_x, half=True)
                    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
                        logits = model(token_0[:, :-1], token_1[:, :-1], batch_stamp[:, :-1, :])
                        loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_0[:, 1:], token_1[:, 1:])
                    total += loss.item()
                return total / step

            steps_per_epoch = math.ceil(len(train_ds) / BATCH_SIZE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=DEVICE.type == "cuda")
            total_steps = steps_per_epoch * EPOCHS
            warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: min(1.0, (step + 1) / warmup_steps) * 0.5 *
                (1.0 + math.cos(math.pi * max(0, step - warmup_steps) / max(1, total_steps - warmup_steps))),
            )
            scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda" and AMP_DTYPE == torch.float16)

            history, best_val, best_epoch, stale_epochs = [], float("inf"), 0, 0
            for epoch in range(1, EPOCHS + 1):
                train_loss = causal_epoch(model, train_ds, optimizer, scheduler, scaler, epoch, "domain adaptation")
                val_loss = validation_loss(model)
                row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]}
                history.append(row)
                print(row)
                if val_loss < best_val:
                    best_val, best_epoch, stale_epochs = val_loss, epoch, 0
                    model.save_pretrained(CHECKPOINT_DIR)
                else:
                    stale_epochs += 1
                    if stale_epochs >= PATIENCE:
                        break

            history_df = pd.DataFrame(history)
            history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
            print(f"Selected epoch count: {best_epoch}; best July validation loss: {best_val:.6f}")
            """
        )

    elif "best_model = Kronos.from_pretrained(str(CHECKPOINT_DIR))" in text:
        cell["source"] = source(
            """
            # Clean production refit: restart from pretrained weights and train through 30 July.
            # The epoch count was selected without allowing July into the selection-stage gradients.
            refit_ds = DynamicPanelKlineDataset(raw[raw.date <= TRAIN_END], "refit", REFIT_WINDOWS_PER_EPOCH)
            best_model = Kronos.from_pretrained(MODEL_ID).to(DEVICE)
            refit_optimizer = torch.optim.AdamW(best_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=DEVICE.type == "cuda")
            refit_steps = math.ceil(len(refit_ds) / BATCH_SIZE) * best_epoch
            refit_warmup = max(1, int(refit_steps * WARMUP_RATIO))
            refit_scheduler = torch.optim.lr_scheduler.LambdaLR(
                refit_optimizer,
                lambda step: min(1.0, (step + 1) / refit_warmup) * 0.5 *
                (1.0 + math.cos(math.pi * max(0, step - refit_warmup) / max(1, refit_steps - refit_warmup))),
            )
            refit_scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda" and AMP_DTYPE == torch.float16)
            refit_history = []
            for epoch in range(1, best_epoch + 1):
                loss = causal_epoch(best_model, refit_ds, refit_optimizer, refit_scheduler, refit_scaler, epoch, "production refit")
                refit_history.append({"epoch": epoch, "train_loss": loss})
            pd.DataFrame(refit_history).to_csv(OUTPUT_DIR / "production_refit_history.csv", index=False)
            production_dir = OUTPUT_DIR / "production_model"
            best_model.save_pretrained(production_dir)
            predictor = KronosPredictor(best_model, tokenizer, device=str(DEVICE), max_context=MAX_CONTEXT)

            contexts, x_times, y_times, valid_tickers, last_close_map, skipped = [], [], [], [], {}, []
            global_future_dates = pd.Series(pd.bdate_range(AS_OF_DATE + pd.offsets.BDay(1), periods=PRED_LEN))
            for ticker in TICKERS:
                g = raw[(raw.ticker.eq(ticker)) & (raw.date <= AS_OF_DATE)].sort_values("date").tail(LOOKBACK).copy()
                if len(g) < LOOKBACK:
                    skipped.append({"ticker": ticker, "reason": "insufficient_history", "bars": len(g), "last_date": g.date.max() if len(g) else pd.NaT})
                    continue
                stale_days = int((AS_OF_DATE - g["date"].max()).days)
                if stale_days > MAX_STALE_CALENDAR_DAYS:
                    skipped.append({"ticker": ticker, "reason": "stale_or_suspended", "bars": len(g), "last_date": g.date.max()})
                    continue
                x_df = g[FEATURES].copy()
                contexts.append(x_df)
                x_times.append(pd.Series(pd.to_datetime(g["date"]).to_numpy()))
                y_times.append(global_future_dates.copy())
                valid_tickers.append(ticker)
                last_close_map[ticker] = float(g["close"].iloc[-1])
            skipped_df = pd.DataFrame(skipped)
            skipped_df.to_csv(OUTPUT_DIR / "skipped_tickers.csv", index=False)
            print(f"Eligible forecast: {len(valid_tickers)} / {len(TICKERS)} | skipped: {len(skipped_df)}")
            """
        )

# Add profile-specific metadata to the existing export cell.
for cell in notebook["cells"]:
    text = "".join(cell.get("source", []))
    if '"pretrained_model": MODEL_ID' in text:
        text = text.replace(
            '"pretrained_model": MODEL_ID,',
            '"pretrained_model": MODEL_ID,\n    "training_profile": "80gb_dynamic_refit",\n    "selected_epochs": int(best_epoch),\n    "train_windows_per_epoch": TRAIN_WINDOWS_PER_EPOCH,\n    "refit_windows_per_epoch": REFIT_WINDOWS_PER_EPOCH,',
        )
        cell["source"] = text.splitlines(True)

OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT_NOTEBOOK)
