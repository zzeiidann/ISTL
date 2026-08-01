# Production Refit E4 — 1D Full-Timeframe

Arsip hasil `production_refit_e4_1d` yang dioptimalkan terhadap seluruh sesi
valid. Dibuat dari `production_refit_e4_1d.zip` pada 1 Agustus 2026.

## Ringkasan

| Metrik | Hasil |
|---|---:|
| Horizon | 1 hari |
| Periode origin | 2026-06-02 s.d. 2026-07-29 |
| Sesi backtest | 42 |
| Sesi valid | 41 |
| Sesi dikecualikan | 1 |
| Scope optimasi | Seluruh sesi valid |
| Pilihan per sesi | Top 30 |
| Total selections | 1,230 |
| Total hits | 580 |
| Overall win rate | 47.15% |
| Kronos gain baseline | 37.97% |
| Predicted hit-5 baseline | 40.16% |
| Optuna trials | 1,500 |

Target hit adalah `actual_high / previous_actual_close - 1 >= 0.05`, yaitu
harga tertinggi hari berikutnya mencapai sekurang-kurangnya 5% di atas actual
close sebelumnya.

Overall win rate adalah objective in-sample full-timeframe: bobot dicari dan
dinilai pada seluruh 41 sesi valid yang sama. Angka ini menunjukkan kecocokan
bobot terhadap timeframe optimasi, bukan estimasi out-of-sample.

Sesi origin 2026-06-16 dikecualikan karena forecast hanya mencakup 62 ticker
dan menghasilkan 14 kandidat positif. Cakupan normal berada di sekitar 914–916
ticker.

## Struktur arsip

- `archive/`: ZIP asli.
- `summary/`: ringkasan backtest, bobot global terbaik, dan win rate.
- `optimization/`: seluruh 1.500 trial Optuna.
- `selections/`: top-30 setiap sesi beserta actual result dan rank.
- `audit/`: audit kualitas dan kelengkapan setiap origin.
- `data/`: panel kandidat positif top-100.
- `forecast_cache/`: forecast parquet per origin.
