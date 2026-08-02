"""Build the leakage-safe 80 GB notebook with IDX tokenizer adaptation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
SOURCE_BUILDER = ROOT / "build_kaggle_notebook_80gb_no_refit.py"
SOURCE_NOTEBOOK = ROOT / "kronos_idx_kaggle_finetune_80gb_no_refit.ipynb"
OUTPUT_NOTEBOOK = ROOT / "kronos_idx_kaggle_finetune_80gb_tokenizer_idx.ipynb"


def source(text: str) -> list[str]:
    return dedent(text).strip().splitlines(True)


subprocess.run([sys.executable, str(SOURCE_BUILDER)], check=True)
notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))

notebook["cells"][0]["source"] = source(
    """
    # Kronos-base — IDX Tokenizer + Predictor Adaptation (80 GB, No Leakage)

    Tokenizer dan predictor sama-sama dimulai dari checkpoint pretrained. Semua
    gradient memakai window dengan target yang berakhir sebelum 1 Juli 2026.
    Periode 1–31 Juli hanya digunakan sebagai validation; tidak ada refit dengan
    data Juli. Best validated tokenizer dibekukan sebelum predictor dilatih.

    Forecast memakai causal context sampai close 31 Juli dan dimulai 3 Agustus 2026.
    """
)

for cell in notebook["cells"]:
    text = "".join(cell.get("source", []))

    if 'MODEL_ID = "NeoQuasar/Kronos-base"' in text:
        text = text.replace(
            'MODEL_ID = "NeoQuasar/Kronos-base"\n',
            'MODEL_ID = "NeoQuasar/Kronos-base"\n'
            'TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"\n',
        )
        text = text.replace(
            "BATCH_SIZE = 128\n",
            "BATCH_SIZE = 128\n"
            "TOKENIZER_BATCH_SIZE = 128\n"
            "TOKENIZER_EPOCHS = 3\n"
            "TOKENIZER_LEARNING_RATE = 1e-5\n"
            "TOKENIZER_WEIGHT_DECAY = 0.01\n"
            "TOKENIZER_WINDOWS_PER_EPOCH = 100_000\n"
            "TOKENIZER_PATIENCE = 1\n",
        )
        cell["source"] = text.splitlines(True)

    elif "## 5. Load pretrained Kronos-base" in text:
        cell["source"] = source(
            """
            ## 5. Fine-tune the tokenizer on pre-July IDX data

            Tokenizer dimulai dari pretrained weights dan diadaptasi secara ringan.
            Candidate training identik dengan predictor: `target_end < 2026-07-01`.
            Juli hanya mengukur reconstruction loss untuk memilih checkpoint.
            Setelah dipilih, tokenizer dibekukan agar representasi token stabil
            selama predictor training.
            """
        )

    elif 'tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")' in text:
        cell["source"] = source(
            """
            TOKENIZER_CHECKPOINT_DIR = OUTPUT_DIR / "tokenizer_idx_best"
            tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID).to(DEVICE)

            def tokenizer_loader(dataset, shuffle):
                kwargs = dict(
                    dataset=dataset, batch_size=TOKENIZER_BATCH_SIZE, shuffle=shuffle,
                    num_workers=NUM_WORKERS, pin_memory=True, drop_last=shuffle,
                )
                if NUM_WORKERS > 0:
                    kwargs.update(persistent_workers=False, prefetch_factor=4)
                return DataLoader(**kwargs)

            def tokenizer_epoch(dataset, optimizer, scaler, epoch):
                dataset.resample(epoch)
                loader = tokenizer_loader(dataset, True)
                tokenizer.train()
                total = 0.0
                progress = tqdm(loader, desc=f"tokenizer IDX epoch {epoch}", leave=False)
                for step, (batch_x, _) in enumerate(progress, 1):
                    batch_x = batch_x.to(DEVICE, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
                        (z_pre, z), bsq_loss, _, _ = tokenizer(batch_x)
                        recon_pre = torch.nn.functional.mse_loss(z_pre, batch_x)
                        recon_full = torch.nn.functional.mse_loss(z, batch_x)
                        loss = (recon_pre + recon_full + bsq_loss) / 2
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total += loss.item()
                    progress.set_postfix(loss=f"{total / step:.4f}")
                return total / len(loader)

            @torch.no_grad()
            def tokenizer_validation_loss():
                tokenizer.eval()
                total, samples = 0.0, 0
                for batch_x, _ in tqdm(tokenizer_loader(val_ds, False), desc="tokenizer July validation", leave=False):
                    batch_x = batch_x.to(DEVICE, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
                        (_, z), _, _, _ = tokenizer(batch_x)
                        loss = torch.nn.functional.mse_loss(z, batch_x)
                    total += loss.item() * batch_x.size(0)
                    samples += batch_x.size(0)
                return total / samples

            tokenizer_train_ds = DynamicPanelKlineDataset(
                raw[raw.date <= TRAIN_END], "train", TOKENIZER_WINDOWS_PER_EPOCH
            )
            tokenizer_optimizer = torch.optim.AdamW(
                tokenizer.parameters(), lr=TOKENIZER_LEARNING_RATE,
                weight_decay=TOKENIZER_WEIGHT_DECAY, fused=DEVICE.type == "cuda",
            )
            tokenizer_scaler = torch.cuda.amp.GradScaler(
                enabled=DEVICE.type == "cuda" and AMP_DTYPE == torch.float16
            )
            tokenizer_history, tokenizer_best_val, tokenizer_stale = [], float("inf"), 0
            for epoch in range(1, TOKENIZER_EPOCHS + 1):
                train_loss = tokenizer_epoch(tokenizer_train_ds, tokenizer_optimizer, tokenizer_scaler, epoch)
                val_loss = tokenizer_validation_loss()
                row = {"epoch": epoch, "train_loss": train_loss, "val_reconstruction_loss": val_loss}
                tokenizer_history.append(row)
                print(row)
                if val_loss < tokenizer_best_val:
                    tokenizer_best_val, tokenizer_stale = val_loss, 0
                    tokenizer.save_pretrained(TOKENIZER_CHECKPOINT_DIR)
                else:
                    tokenizer_stale += 1
                    if tokenizer_stale >= TOKENIZER_PATIENCE:
                        break

            pd.DataFrame(tokenizer_history).to_csv(OUTPUT_DIR / "tokenizer_training_history.csv", index=False)
            tokenizer = KronosTokenizer.from_pretrained(str(TOKENIZER_CHECKPOINT_DIR)).to(DEVICE).eval()
            for parameter in tokenizer.parameters():
                parameter.requires_grad = False

            model = Kronos.from_pretrained(MODEL_ID).to(DEVICE)
            print(f"Best tokenizer July reconstruction loss: {tokenizer_best_val:.6f}")
            print(f"Trainable predictor parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            """
        )

    elif "## 6. Fine-tune predictor" in text:
        cell["source"] = source(
            """
            ## 6. Fine-tune predictor with the frozen IDX tokenizer

            Predictor kembali dimulai dari `Kronos-base`. Gradient predictor juga
            hanya memakai target sebelum Juli. Full July validation memilih model
            final, tanpa post-validation refit.
            """
        )

    elif '"training_profile": "80gb_dynamic_no_refit"' in text:
        text = text.replace(
            '"training_profile": "80gb_dynamic_no_refit"',
            '"training_profile": "80gb_idx_tokenizer_and_predictor_no_refit"',
        )
        text = text.replace(
            '    "pretrained_tokenizer": "NeoQuasar/Kronos-Tokenizer-base",\n',
            '    "pretrained_tokenizer": TOKENIZER_ID,\n'
            '    "final_tokenizer_source": "best_july_validation_checkpoint",\n'
            '    "tokenizer_epochs_run": len(tokenizer_history),\n'
            '    "tokenizer_windows_per_epoch": TOKENIZER_WINDOWS_PER_EPOCH,\n'
            '    "best_tokenizer_validation_reconstruction_loss": tokenizer_best_val,\n',
        )
        cell["source"] = text.splitlines(True)

for cell in notebook["cells"]:
    text = "".join(cell.get("source", []))
    text = text.replace('pd.Timestamp("2026-07-30")', 'pd.Timestamp("2026-07-31")')
    text = text.replace("1–30 July", "1–31 July").replace("1–30 Juli", "1–31 Juli")
    text = text.replace("through the 30 July close", "through the 31 July close")
    text = text.replace("sampai 30 Juli", "sampai 31 Juli")
    cell["source"] = text.splitlines(True)

OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT_NOTEBOOK)
