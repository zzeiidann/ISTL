# IDX tokenizer + predictor run summary

## Split and configuration

- Training targets end strictly before 2026-07-01.
- Validation covers 2026-07-01 through 2026-07-31.
- Forecast context ends at the 2026-07-31 close.
- Tokenizer: 10 epochs, 100,000 sampled windows per epoch.
- Predictor: best checkpoint at epoch 4, 200,000 sampled windows per epoch.
- No post-validation refit.

## Training result

- Best tokenizer July reconstruction loss: 0.0107690 (epoch 10).
- Tokenizer validation reconstruction improved every epoch, from 0.0115997
  to 0.0107690 (7.16% reduction).
- Best predictor July token cross-entropy: 2.671720 (epoch 4).
- Predictor validation loss improved each epoch from 2.701262 to 2.671720.

The tokenizer adaptation itself converged cleanly. However, the predictor loss
is worse than the earlier frozen-tokenizer no-refit run (2.565153), although
that run ended validation on 2026-07-30 rather than 2026-07-31. This run should
therefore remain experimental until an identical-cutoff A/B test and walk-forward
backtest show an advantage.

## Forecast coverage

- 916 tickers ranked.
- 43 tickers skipped for insufficient history.
- 134 tickers have positive expected 20-session return.
- Forecast uses five sampled paths over a 20-session horizon.

## Top 10 expected 20-session returns

| Rank | Ticker | Expected return | Probability up | P10 downside |
|---:|:---|---:|---:|---:|
| 1 | VIVA | 25.65% | 60% | -10.83% |
| 2 | BWPT | 20.46% | 100% | 4.91% |
| 3 | HILL | 20.01% | 100% | 20.01% |
| 4 | NSSS | 17.85% | 80% | -8.31% |
| 5 | BAPI | 16.06% | 80% | -2.98% |
| 6 | MDIA | 15.96% | 80% | -1.78% |
| 7 | TAMA | 14.05% | 60% | -2.88% |
| 8 | PKPK | 12.37% | 60% | -5.44% |
| 9 | WGSH | 12.25% | 100% | 2.90% |
| 10 | TIFA | 11.67% | 60% | -8.01% |

These are statistical model outputs, not investment recommendations.
