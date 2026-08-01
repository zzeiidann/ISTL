# BackTest Results

Folder ini menyimpan arsip hasil backtest Kronos Screener. Setiap run ditempatkan
di subfolder terpisah agar output, konfigurasi, dan cache forecast tidak tercampur.

## Runs

## Runs

| Run | Horizon | Periode origin | Overall win rate | Scope |
|---|---:|---|---:|---|
| [validated_no_refit_e15_1d](validated_no_refit_e15_1d/) | 1 hari | 2026-06-02 s.d. 2026-07-29 | 47.48% | All valid sessions |
| [production_refit_e4_1d](production_refit_e4_1d/) | 1 hari | 2026-06-02 s.d. 2026-07-29 | 47.15% | All valid sessions |

Seluruh hasil baru menggunakan satu weight global yang dioptimalkan terhadap
semua origin valid dalam timeframe backtest.
