from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from pattern_screener.config import PatternConfig
from pattern_screener.pattern_ranker import (
    build_pattern_snapshot,
    calculate_secondary_score,
    enrich_candidates,
    optimize_pattern_weight,
    rank_enriched_candidates,
)


RUNS = {
    "validated_no_refit_e15_1d": "validated_no_refit_e15_1d",
    "production_refit_e4_1d": "production_refit_e4_1d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest causal bullish-pattern reranking.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--select", type=int, default=30)
    parser.add_argument("--fixed-pattern-weight", type=float)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _select_base(frame: pd.DataFrame, select: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(["origin", "horizon"], sort=True):
        chosen = group.nlargest(min(select, len(group)), "secondary_score").copy()
        chosen["selected_rank"] = range(1, len(chosen) + 1)
        parts.append(chosen)
    return pd.concat(parts, ignore_index=True)


def _metrics(selected: pd.DataFrame, label: str) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for top_k in (1, 3, 5, 10, 20, 30):
        subset = selected.loc[selected["selected_rank"].le(top_k)]
        daily = subset.groupby(["origin", "horizon"])["hit5"].max()
        rows.append(
            {
                "ranking": label,
                "top_k": top_k,
                "selections": int(len(subset)),
                "hits": int(subset["hit5"].sum()),
                "precision": float(subset["hit5"].mean()),
                "daily_basket_success": float(daily.mean()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    results_root = repo / "BackTest Kronos Screener" / "BackTest Results"
    prices_path = repo / "Kronos IDX FineTune" / "data" / "idx_kronos_all_daily.parquet"
    output_root = args.output_dir or repo / "BackTest Pattern Screener" / "results"
    output_root.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(prices_path)
    base_config = PatternConfig()
    print("Calculating one causal pattern snapshot for all models...")
    snapshot = build_pattern_snapshot(prices, base_config)

    combined_metrics: list[pd.DataFrame] = []
    for run_name, folder in RUNS.items():
        source = results_root / folder
        panel_path = source / "data" / "daily_positive_top100_panel.parquet"
        weights_path = source / "summary" / "best_weights.json"
        if not panel_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"Missing archived result for {run_name}: {source}")

        candidates = pd.read_parquet(panel_path)
        weights = json.loads(weights_path.read_text())
        candidates["secondary_score"] = calculate_secondary_score(candidates, weights)
        enriched = enrich_candidates(candidates, snapshot)

        if args.fixed_pattern_weight is None:
            config, trials = optimize_pattern_weight(enriched, base_config, select=args.select)
        else:
            if not 0 <= args.fixed_pattern_weight <= 1:
                raise ValueError("--fixed-pattern-weight must be in [0, 1]")
            config = PatternConfig(
                pattern_weight=args.fixed_pattern_weight,
                base_score_weight=1 - args.fixed_pattern_weight,
            )
            _, trials = optimize_pattern_weight(
                enriched,
                base_config,
                weights=(args.fixed_pattern_weight,),
                select=args.select,
            )

        scored, pattern_selected = rank_enriched_candidates(enriched, config, args.select)
        base_selected = _select_base(candidates, args.select)
        metrics = pd.DataFrame(_metrics(base_selected, "base") + _metrics(pattern_selected, "base_plus_pattern"))
        metrics["run_name"] = run_name
        metrics["lift"] = metrics.groupby("top_k")["precision"].transform(lambda values: values - values.iloc[0])

        run_output = output_root / run_name
        run_output.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(run_output / "pattern_candidate_panel.parquet", index=False)
        pattern_selected.to_csv(run_output / "selected_top30_pattern.csv", index=False)
        base_selected.to_csv(run_output / "selected_top30_base.csv", index=False)
        metrics.to_csv(run_output / "ranking_comparison.csv", index=False)
        trials.to_csv(run_output / "pattern_weight_trials.csv", index=False)
        (run_output / "pattern_config.json").write_text(json.dumps(asdict(config), indent=2))
        summary = {
            "run_name": run_name,
            "pattern_weight": config.pattern_weight,
            "base_score_weight": config.base_score_weight,
            "objective": "0.45 * precision_top5 + 0.35 * precision_top10 + 0.20 * precision_top30",
            "metrics": metrics.to_dict(orient="records"),
        }
        (run_output / "pattern_backtest_summary.json").write_text(json.dumps(summary, indent=2))
        combined_metrics.append(metrics)
        print(run_name, metrics.to_string(index=False))

    pd.concat(combined_metrics, ignore_index=True).to_csv(output_root / "all_model_comparison.csv", index=False)
    print("Outputs:", output_root)


if __name__ == "__main__":
    main()
