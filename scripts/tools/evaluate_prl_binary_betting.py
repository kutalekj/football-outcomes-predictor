from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from football_outcomes.data.snapshots import load_snapshot
from football_outcomes.experiments.bookmaker import (
    build_betting_opportunities,
    select_best_opportunity_per_match,
)
from football_outcomes.experiments.publication_betting import (
    BlockBootstrapConfig,
    PublicationBettingConfig,
    build_fixed_threshold_comparison,
    build_matched_coverage_comparison,
    build_publication_edge_bins,
)
from football_outcomes.experiments.publication_binary import (
    PUBLICATION_BINARY_MODEL_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Evaluate PRL binary publication predictions at fixed edge thresholds " "and matched coverage.")
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    return parser.parse_args()


def _safe_odds(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result <= 1.0:
        return None
    return result


def _odds_by_match_id(snapshot_path: Path) -> dict[int, dict]:
    bundle = load_snapshot(snapshot_path)
    output: dict[int, dict] = {}
    for match in bundle.matches:
        odds = getattr(match, "odds", None) or {}
        under = _safe_odds(odds.get("under25"))
        over = _safe_odds(odds.get("over25"))
        if under is None or over is None:
            continue
        raw_under = 1.0 / under
        raw_over = 1.0 / over
        total = raw_under + raw_over
        if total <= 0.0:
            continue
        output[int(match.id)] = {
            "under25_odds": under,
            "over25_odds": over,
            "book_raw_p_under": raw_under,
            "book_raw_p_over": raw_over,
            "book_fair_p_under": raw_under / total,
            "book_fair_p_over": raw_over / total,
            "book_margin": total - 1.0,
        }
    return output


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _joined_rows(
    rows: list[dict[str, str]],
    odds_by_id: dict[int, dict],
) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        match_id = int(row["match_id"])
        odds = odds_by_id.get(match_id)
        if odds is None:
            continue
        y_under = int(row["y_true"])
        p_under = float(row["probability_under_2_5"])
        output.append(
            {
                "round_idx": int(row["round_index"]),
                "match_id": match_id,
                "match_datetime": row["match_datetime"],
                "season": int(row["season"]),
                "competition": row["competition"],
                "y_true_under25": y_under,
                "y_true_over25": 1 - y_under,
                "p_model_under": p_under,
                "p_model_over": 1.0 - p_under,
                **odds,
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    predictions_path = args.run_directory / "predictions.csv"
    predictions = _read_predictions(predictions_path)
    odds_by_id = _odds_by_match_id(args.snapshot)

    best_by_model: dict[str, list[dict]] = {}
    for model_name in PUBLICATION_BINARY_MODEL_NAMES:
        model_rows = [row for row in predictions if row["model_name"] == model_name]
        joined = _joined_rows(model_rows, odds_by_id)
        opportunities = build_betting_opportunities(joined)
        best_by_model[model_name] = select_best_opportunity_per_match(opportunities)

    config = PublicationBettingConfig(
        thresholds=(0.09, 0.10, 0.11),
        reference_model="proposed-v1",
        bootstrap=BlockBootstrapConfig(
            replicates=args.bootstrap_replicates,
            confidence_level=0.95,
            seed=args.bootstrap_seed,
            block_field="round_idx",
        ),
    )
    fixed = build_fixed_threshold_comparison(best_by_model, config=config)
    matched = build_matched_coverage_comparison(best_by_model, config=config)
    proposed_bins = build_publication_edge_bins(
        best_by_model["proposed-v1"],
        bootstrap=config.bootstrap,
    )

    _write_csv(args.run_directory / "betting_fixed_threshold.csv", fixed)
    _write_csv(args.run_directory / "betting_matched_coverage.csv", matched)
    _write_csv(args.run_directory / "proposed_edge_bins_with_ci.csv", proposed_bins)

    valid_odds_count = len(best_by_model["proposed-v1"])
    print(f"[prl-betting] common matches with valid odds: {valid_odds_count}")
    print(f"[prl-betting] fixed-threshold rows: {len(fixed)}")
    print(f"[prl-betting] matched-coverage rows: {len(matched)}")
    print("[prl-betting] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
