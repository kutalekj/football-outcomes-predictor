from __future__ import annotations

import pytest

from football_outcomes.experiments.publication_betting import (
    BlockBootstrapConfig,
    PublicationBettingConfig,
    build_fixed_threshold_comparison,
    build_matched_coverage_comparison,
    build_publication_edge_bins,
)


def _best_row(
    match_id: int,
    *,
    round_idx: int,
    edge: float,
    profit: float,
    won: int | None = None,
) -> dict:
    if won is None:
        won = int(profit > 0.0)
    return {
        "round_idx": round_idx,
        "match_id": match_id,
        "match_datetime": f"2024-01-{1 + round_idx:02d}T12:00:00",
        "season": 2024,
        "competition": "League",
        "side": "under",
        "model_probability": 0.60,
        "bookmaker_break_even_probability": 0.60 - edge,
        "bookmaker_fair_probability": 0.55,
        "odds": 1.80,
        "bookmaker_margin": 0.05,
        "probability_edge": edge,
        "market_disagreement": 0.05,
        "estimated_ev": 1.80 * edge,
        "won": won,
        "flat_profit": profit,
    }


def _model_rows(edge_shift: float = 0.0) -> list[dict]:
    rows: list[dict] = []
    for match_id in range(1, 41):
        rows.append(
            _best_row(
                match_id,
                round_idx=(match_id - 1) // 4,
                edge=0.075 + 0.002 * match_id + edge_shift,
                profit=0.80 if match_id % 3 else -1.0,
            )
        )
    return rows


def test_publication_edge_bins_include_win_loss_and_ci_fields() -> None:
    rows = _model_rows()
    result = build_publication_edge_bins(
        rows,
        bootstrap=BlockBootstrapConfig(replicates=100, seed=11),
    )
    assert len(result) == 9
    populated = [row for row in result if int(row["num_bets"]) > 0]
    assert populated
    assert all(int(row["wins"]) + int(row["losses"]) == int(row["num_bets"]) for row in populated)
    assert all(row["roi_ci_lower"] is not None for row in populated)
    assert all(row["roi_ci_upper"] is not None for row in populated)


def test_block_bootstrap_is_reproducible_for_same_seed() -> None:
    rows = {"proposed": _model_rows()}
    config = PublicationBettingConfig(
        thresholds=(0.09,),
        bootstrap=BlockBootstrapConfig(replicates=100, seed=123),
    )
    first = build_fixed_threshold_comparison(rows, config=config)
    second = build_fixed_threshold_comparison(rows, config=config)
    assert first[0]["roi_ci_lower"] == pytest.approx(second[0]["roi_ci_lower"])
    assert first[0]["roi_ci_upper"] == pytest.approx(second[0]["roi_ci_upper"])


def test_fixed_threshold_comparison_uses_9_10_11_percent_by_default() -> None:
    rows = {"proposed": _model_rows(), "logistic": _model_rows(-0.005)}
    result = build_fixed_threshold_comparison(
        rows,
        config=PublicationBettingConfig(
            bootstrap=BlockBootstrapConfig(replicates=100, seed=5),
        ),
    )
    assert {row["threshold"] for row in result} == {0.09, 0.10, 0.11}
    assert {row["model"] for row in result} == {"proposed", "logistic"}
    assert {row["selection_mode"] for row in result} == {"fixed-threshold"}


def test_matched_coverage_uses_reference_model_bet_count() -> None:
    proposed = _model_rows()
    logistic = _model_rows(-0.02)
    rows = {"proposed": proposed, "logistic": logistic}
    config = PublicationBettingConfig(
        thresholds=(0.10,),
        reference_model="proposed",
        bootstrap=BlockBootstrapConfig(replicates=100),
    )
    result = build_matched_coverage_comparison(rows, config=config)
    expected_count = sum(float(row["probability_edge"]) >= 0.10 for row in proposed)
    assert {int(row["num_bets"]) for row in result} == {expected_count}
    assert {int(row["reference_num_bets"]) for row in result} == {expected_count}
    assert all(row["roi_ci_lower"] is not None for row in result)
    assert all(row["roi_ci_upper"] is not None for row in result)


def test_model_population_mismatch_is_rejected() -> None:
    proposed = _model_rows()
    logistic = _model_rows()[:-1]
    with pytest.raises(ValueError, match="same match_id population"):
        build_matched_coverage_comparison(
            {"proposed": proposed, "logistic": logistic},
            config=PublicationBettingConfig(
                thresholds=(0.10,),
                bootstrap=BlockBootstrapConfig(replicates=100),
            ),
        )
