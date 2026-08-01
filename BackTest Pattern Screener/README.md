# BackTest Pattern Screener

Eksperimen causal bullish-pattern reranking untuk kandidat yang sudah lolos
Kronos screener. Pipeline lama tidak diganti:

```text
All stocks
→ Kronos positive-close top 100
→ existing secondary_score
→ causal bullish-pattern layer
→ rank blending
→ final top 30
```

## Integrasi dengan pipeline existing

OHLCV dibaca dari `Kronos IDX FineTune/data/idx_kronos_all_daily.parquet` dengan
kolom `date, ticker, open, high, low, close, volume, amount`. Candidate panel dan
global base weights dibaca dari dua run full-timeframe di
`BackTest Kronos Screener/BackTest Results`.

Pattern dihitung pada tanggal `origin`. Actual `target_date`, `actual_high_gain`,
dan `hit5` hanya digunakan setelah ranking untuk evaluasi. Inference Kronos tidak
perlu dijalankan ulang.

## Modul

- `config.py`: seluruh threshold dan blend weights.
- `price_action_features.py`: causal OHLCV, liquidity, RSI, EMA, ATR, dan swing features.
- `bullish_patterns.py`: individual boolean detections dan confidence 0–1.
- `pattern_scoring.py`: group aggregation, penalties, net score, dan explanation.
- `pattern_ranker.py`: merge ke candidate panel, base score, rank blending, dan weight search.
- `evaluation.py`: forward returns/MFE/MAE terpisah dari live feature generation.
- `run_pattern_backtest.py`: base-versus-pattern backtest dua model existing.

## Pattern yang dideteksi

### Reversal

1. `bullish_rejection`: lower wick besar dan close kuat di dekat prior support.
2. `bullish_engulfing`: bullish body menelan prior bearish body dekat support.
3. `failed_breakdown`: low mematahkan support, tetapi close merebut kembali level tersebut.
4. `double_bottom_breakout`: dua confirmed swing lows serupa lalu breakout resistance.
5. `bearish_structure_break`: close menembus confirmed swing high dengan EMA10 ≥ EMA20.
6. `higher_low`: confirmed swing low terbaru lebih tinggi dan close berada di atas EMA10.
7. `rsi_bullish_divergence`: supporting evidence saat low baru disertai RSI yang membaik.

### Continuation

1. `breakout_with_volume`: resistance breakout, volume expansion, close dekat high.
2. `bull_flag`: impulse kuat, konsolidasi sempit, lalu breakout.
3. `ascending_triangle`: prior highs rapat, lows naik, kemudian breakout.
4. `compression_breakout`: prior range contraction diikuti breakout dan volume expansion.
5. `inside_bar_breakout`: prior inside bar kemudian close menembus high-nya.
6. `breakout_retest`: breakout historis diikuti retest level yang bertahan.
7. `bullish_hh_hl_structure`: higher low, break swing high, dan EMA bullish stack.

## Aggregation dan penalty

Pattern correlated tidak dijumlahkan mentah. Score reversal dan continuation
masing-masing memakai maksimum confidence dalam kelompoknya. Structure, volume,
dan RSI hanya memberi confirmation bonus kecil:

```python
pattern_quality_score = clip(
    max(reversal_pattern_score, continuation_pattern_score)
    + 0.07 * structure_score
    + 0.05 * volume_confirmation_score
    + 0.04 * rsi_divergence_score,
    0,
    1,
)
```

Penalty meliputi extension dari EMA20, gap-up, weak-volume breakout, upper wick,
close jauh dari high, low liquidity, excessive ATR, chasing, nearby resistance,
zero-volume history, dan insufficient history.

```python
pattern_penalty_score = clip(
    0.75 * max(individual_penalties)
    + 0.25 * mean(individual_penalties),
    0,
    1,
)

net_pattern_score = clip(
    pattern_quality_score - penalty_weight * pattern_penalty_score,
    0,
    1,
)
```

## Ranking

Rank blending dipakai karena skala raw `secondary_score` berubah antar-origin:

```python
final_ranking_score = (
    base_score_weight * base_percentile_rank
    + pattern_weight * pattern_percentile_rank
)
```

Default adalah 75% base dan 25% pattern. Runner dapat mencari pattern weight
0–50% menggunakan objective:

```text
0.45 × top-5 precision + 0.35 × top-10 precision + 0.20 × top-30 precision
```

## Anti-leakage

- Rolling support, resistance, volume, dan reference ATR selalu memakai `shift(1)`.
- Current candle boleh dibandingkan dengan level historis, tetapi tidak ikut membentuk level itu.
- Swing baru diekspos setelah configurable right-window selesai.
- Tidak ada future candle untuk konfirmasi signal.
- Forward returns hanya berada di `evaluation.py` dan tidak dipanggil oleh feature/ranking code.

## Menjalankan

```bash
python3 -m pip install -r "BackTest Pattern Screener/requirements.txt"
python3 "BackTest Pattern Screener/run_pattern_backtest.py"
python3 -m unittest discover -s "BackTest Pattern Screener/tests" -v
```

Untuk fixed 25% pattern blend:

```bash
python3 "BackTest Pattern Screener/run_pattern_backtest.py" --fixed-pattern-weight 0.25
```

## Hasil full-timeframe saat ini

Grid search pada seluruh 41 origin memilih bobot 65% base + 35% pattern untuk
kedua model. Ini sesuai objective proyek yang memakai seluruh timeframe, tanpa
holdout split.

| Model | Ranking | Top 1 | Top 5 | Top 10 | Top 30 |
|---|---|---:|---:|---:|---:|
| validated no-refit e15 | base | 34.15% | 49.27% | 47.56% | 47.48% |
| validated no-refit e15 | + pattern | 53.66% | 54.15% | 53.90% | 46.42% |
| production refit e4 | base | 43.90% | 52.68% | 49.02% | 47.15% |
| production refit e4 | + pattern | 56.10% | 55.61% | 48.78% | 45.77% |

Pattern layer paling berguna untuk memadatkan kandidat teratas. Ia belum
menaikkan precision top-30; karena itu hasil per cutoff disimpan agar trade-off
tidak tertutup oleh satu angka agregat.

Contoh output nyata dari origin 2026-07-29:

| Model | Rank | Ticker | Net pattern | Final rank score | Top pattern | Penalty |
|---|---:|---|---:|---:|---|---|
| validated | 1 | TNCA | 0.208 | 0.949 | bearish structure break | low liquidity |
| validated | 2 | WMPP | 0.062 | 0.945 | higher low | gap up |
| validated | 3 | BULL | 0.346 | 0.875 | higher low | close below high |
| production | 1 | ZONE | 0.346 | 0.966 | bearish structure break | extended |
| production | 2 | TNCA | 0.208 | 0.952 | bearish structure break | low liquidity |
| production | 3 | CASH | 0.212 | 0.942 | higher low | low liquidity |

## Output minimum

`selected_top30_pattern.csv` mempertahankan prediction columns existing dan menambahkan:

```text
selected_rank
base_selected_rank
ticker
secondary_score
pattern_quality_score
pattern_penalty_score
net_pattern_score
final_ranking_score
top_bullish_pattern
top_pattern_score
pattern_support
pattern_penalty
pattern_signal_count
```

## Batasan

- Win rate full-timeframe adalah in-sample optimization objective.
- Pattern tidak menjamin kenaikan harga.
- Double bottom, flag, dan triangle adalah numerical approximations, bukan visual chart matching.
- Threshold awal belum dikalibrasi khusus per regime atau sektor IDX.
- Pattern weights 1D tidak otomatis valid untuk horizon lebih panjang.
