import json
from pathlib import Path
from textwrap import dedent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(True),
    }


cells = [
    md("""
    # Kronos-small — IDX Multi-Asset Fine-Tuning & Forecasting

    Fine-tune **Kronos-small** pada 50 emiten IDX, menggunakan OHLCV harian,
    split waktu tanpa leakage, dan GPU Kaggle. Notebook menyimpan forecast
    seluruh emiten, menampilkan ranking 30 kandidat, serta dashboard profesional
    untuk lima saham teratas.

    **Research guardrail**

    - Tokenizer pretrained dibekukan.
    - Predictor di-fine-tune hanya sampai `2025-12-31`.
    - Data 2026 boleh menjadi context saat membuat forecast live, tetapi tidak
      pernah masuk gradient update.
    - Forecast bersifat probabilistik dan bukan rekomendasi investasi.
    """),
    md("""
    ## 1. Kaggle setup

    Aktifkan **GPU** dan **Internet** pada Kaggle. Jika source Kronos tidak ikut
    ter-clone bersama repository utama, sel setup akan mengambil source resmi dan
    mengunci commit yang telah diuji.
    """),
    code("""
    %pip install -q einops==0.8.1 huggingface_hub==0.33.1 safetensors==0.6.2 pyarrow plotly kaleido tqdm
    """),
    code("""
    from pathlib import Path
    import os, sys, math, json, random, shutil, subprocess, warnings
    from contextlib import nullcontext

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import Dataset, DataLoader
    from tqdm.auto import tqdm
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from IPython.display import display

    warnings.filterwarnings("ignore", category=FutureWarning)
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    def find_data():
        roots = [Path.cwd(), Path("/kaggle/working"), Path("/kaggle/input")]
        for root in roots:
            if not root.exists():
                continue
            direct = root / "Kronos IDX FineTune" / "data" / "idx_kronos_50_daily.parquet"
            if direct.exists():
                return direct
            hits = list(root.glob("**/idx_kronos_50_daily.parquet"))
            if hits:
                return hits[0]
        raise FileNotFoundError("idx_kronos_50_daily.parquet tidak ditemukan. Clone repo ISTL ke /kaggle/working.")

    DATA_PATH = find_data().resolve()
    PROJECT_DIR = DATA_PATH.parent.parent
    KRONOS_DIR = PROJECT_DIR / "Kronos"
    KRONOS_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"

    if not (KRONOS_DIR / "model" / "kronos.py").exists():
        KRONOS_DIR = Path("/kaggle/working/Kronos")
        if not KRONOS_DIR.exists():
            subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos.git", str(KRONOS_DIR)], check=True)
        subprocess.run(["git", "-C", str(KRONOS_DIR), "checkout", KRONOS_COMMIT], check=True)

    sys.path.insert(0, str(KRONOS_DIR))
    from model import Kronos, KronosTokenizer, KronosPredictor

    OUTPUT_DIR = Path("/kaggle/working/kronos_idx_outputs")
    CHECKPOINT_DIR = OUTPUT_DIR / "kronos_small_idx" / "best_model"
    CHART_DIR = OUTPUT_DIR / "charts"
    for p in [OUTPUT_DIR, CHECKPOINT_DIR, CHART_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print({"device": str(DEVICE), "data": str(DATA_PATH), "kronos_source": str(KRONOS_DIR)})
    if DEVICE.type != "cuda":
        warnings.warn("GPU tidak aktif. Fine-tuning akan sangat lambat; aktifkan Kaggle GPU Accelerator.")
    """),
    md("""
    ## 2. Configuration

    Lookback 120 sesi dipilih agar emiten termuda tetap dapat di-forecast.
    `PRED_LEN=20` mengikuti target riset 20 hari bursa. Untuk eksperimen pertama,
    tiga epoch dan maksimum 20.000 training windows cukup aman pada Kaggle GPU.
    """),
    code("""
    TICKERS = [
        'MCAS','MGNA','MPRO','KDTN','DMMX','BAJA','MLPT','COCO','ZONE','MDIA',
        'KOKA','DWGL','INAI','ECII','DOOH','KOBX','PSDN','NTBK','FUTR','LUCY',
        'JGLE','DEPO','SQMI','TAMA','FLMC','TNCA','KBLV','KLIN','GDST','KOTA',
        'PEGE','SULI','MMIX','TFAS','ZATA','WOOD','LAND','KOPI','HOPE','FILM',
        'PTMP','OILS','TRUS','ALDO','DIVA','LION','RLCO','NANO','ELPI','RONY'
    ]

    LOOKBACK = 120
    PRED_LEN = 20
    MAX_CONTEXT = 512
    TRAIN_END = pd.Timestamp("2025-12-31")
    VAL_START = pd.Timestamp("2025-07-01")
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 2
    EPOCHS = 3
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.05
    MAX_TRAIN_SAMPLES = 20_000
    MAX_VAL_SAMPLES = 4_000
    MIN_ACTIVE_RATIO = 0.60
    PATIENCE = 2

    # Inference uncertainty: setiap run menghasilkan satu sampled forecast path.
    N_FORECAST_PATHS = 5
    INFERENCE_ASSET_BATCH = 8
    TEMPERATURE = 0.8
    TOP_P = 0.9

    assert LOOKBACK + PRED_LEN + 1 <= MAX_CONTEXT
    """),
    md("## 3. Load and audit the Parquet dataset"),
    code("""
    raw = pd.read_parquet(DATA_PATH)
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    raw = raw[raw["ticker"].isin(TICKERS)].copy()
    raw = raw.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])

    price_cols = ["open", "high", "low", "close"]
    raw = raw.dropna(subset=price_cols)
    raw["volume"] = raw["volume"].fillna(0).clip(lower=0)
    raw["amount"] = raw.get("amount", raw[price_cols].mean(axis=1) * raw["volume"])
    raw["amount"] = raw["amount"].fillna(raw[price_cols].mean(axis=1) * raw["volume"])

    audit = (
        raw.groupby("ticker")
        .agg(rows=("date", "size"), start=("date", "min"), end=("date", "max"),
             zero_volume=("volume", lambda x: int((x <= 0).sum())))
        .reindex(TICKERS)
        .reset_index()
    )
    audit["active_pct"] = 1 - audit["zero_volume"] / audit["rows"]
    display(audit)

    fig = px.timeline(
        audit.sort_values("start"), x_start="start", x_end="end", y="ticker",
        color="active_pct", color_continuous_scale="Tealgrn",
        title="IDX Fine-Tuning Universe — Data Coverage"
    )
    fig.update_layout(template="plotly_white", height=900, coloraxis_colorbar_title="Active %")
    fig.show()

    print(f"{raw.ticker.nunique()} tickers | {len(raw):,} rows | {raw.date.min().date()} → {raw.date.max().date()}")
    """),
    md("""
    ## 4. Multi-asset chronological dataset

    Setiap sample dinormalisasi menggunakan mean/std **hanya dari lookback**.
    Window yang terlalu banyak sesi volume nol tidak digunakan untuk training,
    tetapi ticker tersebut tetap masuk tahap forecast.
    """),
    code("""
    FEATURES = ["open", "high", "low", "close", "volume", "amount"]
    TIME_FEATURES = ["minute", "hour", "weekday", "day", "month"]

    class PanelKlineDataset(Dataset):
        def __init__(self, frame, split, max_samples=None):
            if split not in {"train", "val"}:
                raise ValueError("split must be train or val")
            self.split = split
            self.series = {}
            self.indices = []
            window = LOOKBACK + PRED_LEN + 1

            for ticker, g in frame.groupby("ticker", sort=False):
                g = g.sort_values("date").reset_index(drop=True).copy()
                g["minute"] = g["date"].dt.minute
                g["hour"] = g["date"].dt.hour
                g["weekday"] = g["date"].dt.weekday
                g["day"] = g["date"].dt.day
                g["month"] = g["date"].dt.month
                self.series[ticker] = g

                for start in range(0, len(g) - window + 1):
                    target_start = g.loc[start + LOOKBACK, "date"]
                    target_end = g.loc[start + window - 1, "date"]
                    past_volume = g.loc[start:start + LOOKBACK - 1, "volume"]
                    if (past_volume > 0).mean() < MIN_ACTIVE_RATIO:
                        continue
                    if split == "train" and target_end < VAL_START:
                        self.indices.append((ticker, start))
                    elif split == "val" and target_start >= VAL_START and target_end <= TRAIN_END:
                        self.indices.append((ticker, start))

            if max_samples and len(self.indices) > max_samples:
                rng = np.random.default_rng(SEED + (0 if split == "train" else 1))
                selected = np.sort(rng.choice(len(self.indices), max_samples, replace=False))
                self.indices = [self.indices[i] for i in selected]
            print(f"{split}: {len(self.indices):,} windows across {len(self.series)} tickers")

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            ticker, start = self.indices[idx]
            g = self.series[ticker]
            window = g.iloc[start:start + LOOKBACK + PRED_LEN + 1]
            x = window[FEATURES].to_numpy(np.float32)
            stamps = window[TIME_FEATURES].to_numpy(np.float32)
            mean = x[:LOOKBACK].mean(axis=0)
            std = x[:LOOKBACK].std(axis=0)
            x = np.clip((x - mean) / (std + 1e-5), -5, 5)
            return torch.from_numpy(x), torch.from_numpy(stamps)

    train_ds = PanelKlineDataset(raw[raw.date <= TRAIN_END], "train", MAX_TRAIN_SAMPLES)
    val_ds = PanelKlineDataset(raw[raw.date <= TRAIN_END], "val", MAX_VAL_SAMPLES)
    if not len(train_ds) or not len(val_ds):
        raise RuntimeError("Train/validation windows kosong; periksa tanggal dan coverage data.")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        pin_memory=True, drop_last=False, persistent_workers=True
    )
    """),
    md("""
    ## 5. Load pretrained Kronos-small

    Tokenizer tidak diubah. Hanya bobot predictor 24.7M parameter yang
    di-fine-tune agar token dynamics menyesuaikan karakter saham IDX.
    """),
    code("""
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(DEVICE).eval()
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(DEVICE)
    for p in tokenizer.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable predictor parameters: {trainable:,}")
    """),
    md("## 6. Fine-tune predictor"),
    code("""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95)
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, optimizer_steps_per_epoch * EPOCHS), eta_min=LEARNING_RATE / 20
    )
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    history = []
    best_val = float("inf")
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train", leave=False)
        for step, (batch_x, batch_stamp) in enumerate(progress, 1):
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_stamp = batch_stamp.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                token_0, token_1 = tokenizer.encode(batch_x, half=True)

            token_in = [token_0[:, :-1], token_1[:, :-1]]
            token_out = [token_0[:, 1:], token_1[:, 1:]]
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if DEVICE.type == "cuda" else nullcontext()
            with amp_context:
                logits = model(token_in[0], token_in[1], batch_stamp[:, :-1, :])
                loss, s1_loss, s2_loss = model.head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )
                scaled_loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(scaled_loss).backward()
            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            train_loss += loss.item()
            train_batches += 1
            progress.set_postfix(loss=f"{train_loss/train_batches:.4f}")

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_x, batch_stamp in tqdm(val_loader, desc="validation", leave=False):
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                batch_stamp = batch_stamp.to(DEVICE, non_blocking=True)
                token_0, token_1 = tokenizer.encode(batch_x, half=True)
                logits = model(token_0[:, :-1], token_1[:, :-1], batch_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(
                    logits[0], logits[1], token_0[:, 1:], token_1[:, 1:]
                )
                val_loss += loss.item()
                val_batches += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_batches,
            "val_loss": val_loss / val_batches,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            stale_epochs = 0
            model.save_pretrained(CHECKPOINT_DIR)
            print(f"✓ Best checkpoint saved: {CHECKPOINT_DIR}")
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print("Early stopping.")
                break

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    """),
    code("""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df.epoch, y=history_df.train_loss, mode="lines+markers", name="Train"))
    fig.add_trace(go.Scatter(x=history_df.epoch, y=history_df.val_loss, mode="lines+markers", name="Validation"))
    fig.update_layout(
        title="Kronos-small Fine-Tuning Loss", xaxis_title="Epoch", yaxis_title="Token Cross-Entropy",
        template="plotly_white", height=430, hovermode="x unified"
    )
    fig.show()
    """),
    md("""
    ## 7. Probabilistic forecast seluruh emiten

    Model terbaik dimuat kembali. Lima sampled paths dibuat untuk setiap saham.
    Seluruh path disimpan; ranking menggunakan expected 20-day return,
    probabilitas return positif, peak return, dan downside percentile.
    """),
    code("""
    best_model = Kronos.from_pretrained(str(CHECKPOINT_DIR))
    predictor = KronosPredictor(best_model, tokenizer, device=str(DEVICE), max_context=MAX_CONTEXT)

    contexts, x_times, y_times, valid_tickers, last_close_map = [], [], [], [], {}
    for ticker in TICKERS:
        g = raw[raw.ticker.eq(ticker)].sort_values("date").tail(LOOKBACK).copy()
        if len(g) < LOOKBACK:
            print(f"Skip {ticker}: hanya {len(g)} bar, butuh {LOOKBACK}.")
            continue
        x_df = g[FEATURES].copy()
        x_ts = pd.Series(pd.to_datetime(g["date"]).to_numpy())
        # Approximation: weekdays; libur resmi IDX mendatang perlu calendar khusus.
        future = pd.Series(pd.bdate_range(g["date"].max() + pd.offsets.BDay(1), periods=PRED_LEN))
        contexts.append(x_df)
        x_times.append(x_ts)
        y_times.append(future)
        valid_tickers.append(ticker)
        last_close_map[ticker] = float(g["close"].iloc[-1])

    all_paths = []
    for path_id in range(N_FORECAST_PATHS):
        torch.manual_seed(SEED + 10_000 + path_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + 10_000 + path_id)
        for start in tqdm(range(0, len(valid_tickers), INFERENCE_ASSET_BATCH), desc=f"Forecast path {path_id+1}"):
            stop = start + INFERENCE_ASSET_BATCH
            preds = predictor.predict_batch(
                df_list=contexts[start:stop],
                x_timestamp_list=x_times[start:stop],
                y_timestamp_list=y_times[start:stop],
                pred_len=PRED_LEN,
                T=TEMPERATURE,
                top_p=TOP_P,
                top_k=0,
                sample_count=1,
                verbose=False,
            )
            for ticker, pred in zip(valid_tickers[start:stop], preds):
                x = pred.reset_index().rename(columns={"index": "date"})
                x["ticker"] = ticker
                x["path_id"] = path_id
                x["horizon_day"] = np.arange(1, len(x) + 1)
                all_paths.append(x)

    forecasts = pd.concat(all_paths, ignore_index=True)
    forecasts["date"] = pd.to_datetime(forecasts["date"])
    forecasts.to_parquet(OUTPUT_DIR / "all_forecast_paths.parquet", index=False)
    print(f"Saved {len(forecasts):,} forecast rows for {forecasts.ticker.nunique()} tickers.")
    """),
    md("## 8. Rank all stocks and display the 30 strongest expected returns"),
    code("""
    final_day = forecasts[forecasts.horizon_day.eq(PRED_LEN)].copy()
    peak_by_path = forecasts.groupby(["ticker", "path_id"])["close"].max().rename("peak_close").reset_index()
    final_by_path = final_day[["ticker", "path_id", "close"]].rename(columns={"close": "final_close"})
    path_stats = final_by_path.merge(peak_by_path, on=["ticker", "path_id"])
    path_stats["last_close"] = path_stats["ticker"].map(last_close_map)
    path_stats["return_20d"] = path_stats["final_close"] / path_stats["last_close"] - 1
    path_stats["peak_return_20d"] = path_stats["peak_close"] / path_stats["last_close"] - 1

    ranking = (
        path_stats.groupby("ticker")
        .agg(
            last_close=("last_close", "first"),
            expected_close_20d=("final_close", "mean"),
            expected_return_20d=("return_20d", "mean"),
            median_return_20d=("return_20d", "median"),
            probability_up=("return_20d", lambda x: float((x > 0).mean())),
            downside_p10=("return_20d", lambda x: float(np.quantile(x, 0.10))),
            expected_peak_return_20d=("peak_return_20d", "mean"),
            forecast_dispersion=("return_20d", "std"),
        )
        .reset_index()
        .sort_values(["expected_return_20d", "probability_up"], ascending=False)
    )
    ranking["predicted_up"] = (
        (ranking["expected_return_20d"] > 0) &
        (ranking["probability_up"] >= 0.60)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(OUTPUT_DIR / "all_ticker_ranking.csv", index=False)
    ranking.to_parquet(OUTPUT_DIR / "all_ticker_ranking.parquet", index=False)

    top30 = ranking[ranking["predicted_up"]].head(30).copy()
    if len(top30) < 30:
        print(f"Model hanya menemukan {len(top30)} ticker dengan expected return > 0 dan P(up) ≥ 60%; hasil tidak dipaksakan menjadi 30.")
    display(
        top30.style
        .format({
            "last_close": "{:,.0f}", "expected_close_20d": "{:,.0f}",
            "expected_return_20d": "{:+.2%}", "median_return_20d": "{:+.2%}",
            "probability_up": "{:.0%}", "downside_p10": "{:+.2%}",
            "expected_peak_return_20d": "{:+.2%}", "forecast_dispersion": "{:.2%}",
        })
        .background_gradient(subset=["expected_return_20d"], cmap="RdYlGn")
        .background_gradient(subset=["probability_up"], cmap="Blues")
    )
    """),
    code("""
    chart = top30.sort_values("expected_return_20d")
    colors = chart["probability_up"]
    fig = go.Figure(go.Bar(
        x=chart["expected_return_20d"], y=chart["ticker"], orientation="h",
        marker=dict(color=colors, colorscale="Tealgrn", cmin=0, cmax=1,
                    colorbar=dict(title="P(Return > 0)")),
        customdata=np.c_[chart["probability_up"], chart["downside_p10"]],
        hovertemplate="<b>%{y}</b><br>Expected return: %{x:.2%}<br>P(up): %{customdata[0]:.0%}<br>Downside P10: %{customdata[1]:.2%}<extra></extra>"
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
    fig.update_layout(
        title="Kronos-small — Top 30 Expected 20-Day Returns",
        xaxis_tickformat=".1%", xaxis_title="Expected return", yaxis_title="",
        template="plotly_white", height=850, margin=dict(l=70, r=40, t=80, b=60)
    )
    fig.write_html(CHART_DIR / "top30_expected_returns.html")
    fig.show()
    """),
    md("## 9. Professional dashboards for the top five stocks"),
    code("""
    top5_pool = top30 if len(top30) >= 5 else ranking[ranking["expected_return_20d"] > 0]
    top5 = top5_pool.head(5)["ticker"].tolist()
    print("Top 5:", top5)

    for ticker in top5:
        hist = raw[raw.ticker.eq(ticker)].sort_values("date").tail(70)
        fc = forecasts[forecasts.ticker.eq(ticker)]
        band = (
            fc.groupby("date")
            .agg(
                mean_close=("close", "mean"),
                p10_close=("close", lambda x: np.quantile(x, 0.10)),
                p90_close=("close", lambda x: np.quantile(x, 0.90)),
                mean_volume=("volume", "mean"),
            )
            .reset_index()
        )
        stats = ranking.set_index("ticker").loc[ticker]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.72, 0.28],
            subplot_titles=[
                f"{ticker} — expected {stats.expected_return_20d:+.2%} | P(up) {stats.probability_up:.0%}",
                "Historical and Forecast Volume"
            ]
        )
        fig.add_trace(go.Candlestick(
            x=hist.date, open=hist.open, high=hist.high, low=hist.low, close=hist.close,
            name="Historical OHLC", increasing_line_color="#059669", decreasing_line_color="#dc2626"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=band.date, y=band.p90_close, line=dict(width=0), showlegend=False,
            hoverinfo="skip", name="P90"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=band.date, y=band.p10_close, line=dict(width=0), fill="tonexty",
            fillcolor="rgba(14,116,144,0.16)", name="80% forecast interval",
            hovertemplate="P10: %{y:,.0f}<extra></extra>"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=band.date, y=band.mean_close, mode="lines+markers",
            line=dict(color="#0e7490", width=3), marker=dict(size=4),
            name="Mean forecast", hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>"
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=hist.date, y=hist.volume, marker_color="#94a3b8", name="Historical volume"
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=band.date, y=band.mean_volume, marker_color="#0e7490", name="Forecast volume"
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_white", height=720,
            title=dict(text=f"Kronos IDX Forecast Dashboard — {ticker}", x=0.5),
            legend=dict(orientation="h", y=1.03, x=0),
            xaxis_rangeslider_visible=False, hovermode="x unified",
            margin=dict(l=60, r=40, t=105, b=50)
        )
        fig.update_yaxes(title_text="Price (IDR)", tickformat=",", row=1, col=1)
        fig.update_yaxes(title_text="Volume", tickformat=".2s", row=2, col=1)
        fig.write_html(CHART_DIR / f"{ticker}_forecast_dashboard.html")
        fig.show()
    """),
    md("## 10. Package Kaggle outputs"),
    code("""
    metadata = {
        "kronos_commit": KRONOS_COMMIT,
        "pretrained_model": "NeoQuasar/Kronos-small",
        "pretrained_tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "train_gradient_cutoff": str(TRAIN_END.date()),
        "validation_start": str(VAL_START.date()),
        "lookback": LOOKBACK,
        "prediction_horizon": PRED_LEN,
        "forecast_paths": N_FORECAST_PATHS,
        "tickers_requested": TICKERS,
        "tickers_forecast": valid_tickers,
        "best_validation_loss": float(best_val),
        "warning": "Statistical forecast, not investment advice."
    }
    with open(OUTPUT_DIR / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    archive = shutil.make_archive("/kaggle/working/kronos_idx_outputs", "zip", OUTPUT_DIR)
    print("Output folder:", OUTPUT_DIR)
    print("Download archive:", archive)
    print("\\nFiles:")
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file():
            print(" -", p.relative_to(OUTPUT_DIR))
    """),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).with_name("kronos_idx_kaggle_finetune.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(out)
