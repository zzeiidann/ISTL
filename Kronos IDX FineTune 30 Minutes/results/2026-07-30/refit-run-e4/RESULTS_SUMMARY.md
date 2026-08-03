# TF30 production refit — run summary

## Configuration

- Interval: 30 minutes.
- Training targets end before 1 July 2026.
- Validation: 1–30 July 2026.
- Selected predictor epoch: 4.
- Training/refit windows: 200,000 per epoch.
- Lookback: 120 bars (approximately 12 sessions).
- Forecast horizon: 10 bars (approximately one session).
- Forecast paths: 5.

## Training result

- Validation loss improved every epoch: 3.144625 → 3.133189 → 3.129141 →
  3.128533.
- Production refit loss improved from 3.041378 to 2.936491.
- Both the independently selected checkpoint and production-refit checkpoint
  are archived in this run directory.

## Forecast coverage

- 828 tickers ranked.
- 84 tickers skipped.
- 267 tickers satisfy positive-return and probability-up screening.

## Top 10 expected 10-bar returns

| Rank | Ticker | Expected return | Probability up | P10 downside |
|---:|:---|---:|---:|---:|
| 1 | FAST | 8.54% | 100% | 4.46% |
| 2 | SINI | 8.33% | 80% | 0.28% |
| 3 | TIFA | 8.05% | 60% | -0.90% |
| 4 | MSIE | 7.60% | 100% | 4.68% |
| 5 | IFSH | 6.42% | 80% | -8.17% |
| 6 | ELTY | 6.34% | 80% | -0.86% |
| 7 | BAPI | 5.83% | 100% | 4.74% |
| 8 | HADE | 5.77% | 100% | 3.02% |
| 9 | NSSS | 5.37% | 100% | 3.03% |
| 10 | MAPB | 5.34% | 100% | 1.17% |

## Provenance warning

The executed notebook recorded `forecast_as_of_date = 2026-07-30`. Therefore
this archive's generated forecast is anchored on 30 July and starts on 31 July;
it is **not** the corrected 3 August forecast using real 31 July context. The
updated notebook in the repository keeps weight updates capped at 30 July but
uses actual bars through 31 July for inference. Rerun that version when a clean
3 August forecast is required.

Statistical forecast only; not investment advice.
