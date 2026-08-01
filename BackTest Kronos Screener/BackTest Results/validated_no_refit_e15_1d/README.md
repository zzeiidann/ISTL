# Validated No-Refit E15 — 1D

Arsip hasil `validated_no_refit_e15_1d`, dibuat dari
`validated_no_refit_e15_1d.zip` pada 1 Agustus 2026.

## Ringkasan

| Metrik | Hasil |
|---|---:|
| Horizon | 1 hari |
| Periode origin | 2026-06-02 s.d. 2026-07-29 |
| Sesi backtest | 42 |
| Sesi valid | 41 |
| Sesi dikecualikan | 1 |
| Pilihan per sesi | Top 30 |
| Optimization selections | 810 |
| Optimization hits | 417 |
| Optimization win rate | 51.48% |
| Holdout selections | 390 |
| Holdout hits | 140 |
| Holdout win rate | 35.90% |
| Kronos gain baseline | 30.77% |
| Predicted hit-5 baseline | 33.85% |
| Optuna trials | 1,500 |

Target hit adalah `actual_high / previous_actual_close - 1 >= 0.05`, yaitu
harga tertinggi hari berikutnya mencapai sekurang-kurangnya 5% di atas actual
close sebelumnya.

Sesi origin 2026-06-16 dikecualikan karena forecast hanya mencakup 62 ticker
dan menghasilkan 14 kandidat positif. Cakupan normal run berada di sekitar
914–916 ticker.

## Struktur arsip

- `archive/`: ZIP asli.
- `summary/`: ringkasan backtest, bobot terbaik, dan win rate.
- `optimization/`: seluruh trial Optuna.
- `selections/`: pilihan top-30 per hari beserta actual result.
- `audit/`: audit kualitas dan kelengkapan setiap origin.
- `data/`: panel kandidat positif top-100.
- `forecast_cache/`: forecast parquet per origin.

Untuk evaluasi utama, gunakan hasil holdout. Hasil optimization adalah data yang
digunakan untuk mencari bobot sehingga tidak mewakili performa out-of-sample.
