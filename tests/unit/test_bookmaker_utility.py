from __future__ import annotations

import pytest

from football_outcomes.experiments.bookmaker import (
    TemporalThresholdConfig,
    build_betting_opportunities,
    build_competition_summary,
    build_coverage_curve,
    build_fixed_edge_bins,
    build_margin_band_summary,
    build_odds_band_summary,
    build_probability_method_betting_summary,
    build_probability_method_metrics,
    build_quantile_ev_bins,
    build_season_summary,
    build_temporal_threshold_summary,
    build_threshold_summary,
    build_walk_forward_probability_rows,
    select_best_opportunity_per_match,
    simulate_threshold,
)


def _joined_row(
    match_id: int,
    *,
    season: int = 2021,
    p_under: float = 0.60,
    under_odds: float = 1.80,
    over_odds: float = 2.10,
    y_under: int = 1,
) -> dict:
    raw_under = 1.0 / under_odds
    raw_over = 1.0 / over_odds
    total = raw_under + raw_over
    return {
        "round_idx": match_id,
        "match_id": match_id,
        "match_datetime": f"202{season - 2020}-01-01T12:00:00",
        "season": season,
        "competition": "League",
        "y_true_under25": y_under,
        "y_true_over25": 1 - y_under,
        "p_model_under": p_under,
        "p_model_over": 1.0 - p_under,
        "under25_odds": under_odds,
        "over25_odds": over_odds,
        "book_raw_p_under": raw_under,
        "book_raw_p_over": raw_over,
        "book_fair_p_under": raw_under / total,
        "book_fair_p_over": raw_over / total,
        "book_margin": total - 1.0,
    }


def test_opportunity_financial_arithmetic() -> None:
    opportunities = build_betting_opportunities([_joined_row(1)])
    under = next(row for row in opportunities if row["side"] == "under")

    expected_edge = 0.60 - 1.0 / 1.80
    assert under["probability_edge"] == pytest.approx(expected_edge)
    assert under["estimated_ev"] == pytest.approx(1.80 * expected_edge)
    assert under["estimated_ev"] == pytest.approx(0.60 * 1.80 - 1.0)
    assert under["flat_profit"] == pytest.approx(0.80)


def test_losing_flat_profit_is_minus_one() -> None:
    opportunities = build_betting_opportunities([_joined_row(1, y_under=0)])
    under = next(row for row in opportunities if row["side"] == "under")
    assert under["won"] == 0
    assert under["flat_profit"] == pytest.approx(-1.0)


def test_two_opportunities_are_created_per_match() -> None:
    opportunities = build_betting_opportunities([_joined_row(1), _joined_row(2)])
    assert len(opportunities) == 4
    assert {(row["match_id"], row["side"]) for row in opportunities} == {
        (1, "under"),
        (1, "over"),
        (2, "under"),
        (2, "over"),
    }


def test_best_opportunity_is_one_side_per_match() -> None:
    opportunities = build_betting_opportunities(
        [
            _joined_row(1, p_under=0.70),
            _joined_row(2, p_under=0.30),
        ]
    )
    selected = select_best_opportunity_per_match(opportunities)
    assert len(selected) == 2
    assert selected[0]["side"] == "under"
    assert selected[1]["side"] == "over"


def test_positive_edge_cannot_exist_on_both_sides_with_overround() -> None:
    opportunities = build_betting_opportunities([_joined_row(1, p_under=0.55)])
    positive = [row for row in opportunities if row["probability_edge"] >= 0.0]
    assert len(positive) <= 1


def test_threshold_simulation_matches_flat_stake_definition() -> None:
    opportunities = build_betting_opportunities(
        [
            _joined_row(1, p_under=0.70, y_under=1),
            _joined_row(2, p_under=0.70, y_under=0),
        ]
    )
    best = select_best_opportunity_per_match(opportunities)
    result = simulate_threshold(best, 0.0)
    expected_profit = 0.80 - 1.0
    assert result["num_bets"] == 2
    assert result["total_profit"] == pytest.approx(expected_profit)
    assert result["roi"] == pytest.approx(expected_profit / 2.0)


def test_threshold_summary_has_all_under_and_over_rows() -> None:
    best = select_best_opportunity_per_match(build_betting_opportunities([_joined_row(1), _joined_row(2)]))
    rows = build_threshold_summary(best, thresholds=(0.0, 0.05))
    assert len(rows) == 6
    assert {(row["threshold"], row["side"]) for row in rows} == {
        (0.0, "all"),
        (0.0, "under"),
        (0.0, "over"),
        (0.05, "all"),
        (0.05, "under"),
        (0.05, "over"),
    }


def test_edge_bins_cover_every_non_negative_best_candidate_exactly_once() -> None:
    joined = [
        _joined_row(1, p_under=0.45),
        _joined_row(2, p_under=0.55),
        _joined_row(3, p_under=0.65),
        _joined_row(4, p_under=0.75),
    ]
    best = select_best_opportunity_per_match(build_betting_opportunities(joined))
    bins = build_fixed_edge_bins(best)
    expected = sum(float(row["probability_edge"]) >= 0.0 for row in best)
    assert sum(int(row["num_bets"]) for row in bins) == expected
    assert all(float(row["edge_lower"]) >= 0.0 for row in bins)
    assert len(bins) == 9


def test_grouped_summaries_include_all_under_and_over() -> None:
    best = select_best_opportunity_per_match(build_betting_opportunities([_joined_row(1), _joined_row(2)]))

    builders = (
        build_odds_band_summary,
        build_margin_band_summary,
        build_season_summary,
        build_competition_summary,
    )
    for builder in builders:
        rows = builder(best, thresholds=(0.0,))
        assert {row["side"] for row in rows} == {"all", "under", "over"}


def test_coverage_curve_is_more_selective_at_lower_requested_coverage() -> None:
    joined = [_joined_row(i, p_under=min(0.95, 0.55 + i * 0.01)) for i in range(1, 30)]
    best = select_best_opportunity_per_match(build_betting_opportunities(joined))
    rows = build_coverage_curve(best, coverage_levels=(1.0, 0.5, 0.1))
    counts = [int(row["num_bets"]) for row in rows]
    assert counts[0] >= counts[1] >= counts[2]


def test_temporal_threshold_selection_uses_only_earlier_seasons() -> None:
    joined = []
    match_id = 1
    for season in (2021, 2022, 2023):
        for index in range(120):
            joined.append(
                _joined_row(
                    match_id,
                    season=season,
                    p_under=0.70,
                    y_under=1 if index % 2 == 0 else 0,
                )
            )
            match_id += 1

    best = select_best_opportunity_per_match(build_betting_opportunities(joined))
    rows = build_temporal_threshold_summary(
        best,
        TemporalThresholdConfig(thresholds=(0.0, 0.05), minimum_development_bets=50),
    )

    assert [row["test_season"] for row in rows] == [2022, 2023]
    assert rows[0]["development_seasons"] == "2021"
    assert rows[1]["development_seasons"] == "2021,2022"


def test_temporal_config_rejects_invalid_minimum_support() -> None:
    with pytest.raises(ValueError):
        TemporalThresholdConfig(minimum_development_bets=0).validate()


def test_ev_quantiles_use_same_positive_candidate_pool_as_edge() -> None:
    joined = [_joined_row(i, p_under=min(0.90, 0.50 + i * 0.01)) for i in range(1, 30)]
    best = select_best_opportunity_per_match(build_betting_opportunities(joined))
    rows = build_quantile_ev_bins(best, quantile_count=5)
    expected = sum(float(row["probability_edge"]) >= 0.0 for row in best)
    assert sum(int(row["num_bets"]) for row in rows) == expected


def test_temporal_threshold_selection_can_be_under_only() -> None:
    joined = []
    match_id = 1
    for season in (2021, 2022, 2023):
        for index in range(120):
            joined.append(
                _joined_row(
                    match_id,
                    season=season,
                    p_under=0.70,
                    y_under=1 if index % 2 == 0 else 0,
                )
            )
            match_id += 1
    best = select_best_opportunity_per_match(build_betting_opportunities(joined))
    rows = build_temporal_threshold_summary(
        best,
        TemporalThresholdConfig(thresholds=(0.0, 0.05), minimum_development_bets=50),
        side="under",
    )
    assert rows
    assert {row["side"] for row in rows} == {"under"}


def test_walk_forward_probability_methods_are_future_only() -> None:
    joined = []
    match_id = 1
    for season in (2021, 2022, 2023):
        for index in range(40):
            joined.append(
                _joined_row(
                    match_id,
                    season=season,
                    p_under=0.40 + 0.20 * (index % 4) / 3.0,
                    y_under=index % 2,
                )
            )
            match_id += 1

    rows = build_walk_forward_probability_rows(joined)
    assert {int(row["test_season"]) for row in rows} == {2022, 2023}
    assert {str(row["method"]) for row in rows} == {
        "model-raw",
        "model-platt",
        "model-isotonic",
        "bookmaker-fair",
        "bookmaker-model-hybrid",
    }
    assert all("2023" not in str(row["development_seasons"]) for row in rows if int(row["test_season"]) == 2023)

    metrics = build_probability_method_metrics(rows)
    assert any(row["method"] == "bookmaker-model-hybrid" and row["scope"] == "pooled" for row in metrics)

    betting = build_probability_method_betting_summary(rows, thresholds=(0.0,))
    assert {row["side"] for row in betting} == {"all", "under"}
