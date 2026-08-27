from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

DEFAULT_THRESHOLDS = (0.00, 0.02, 0.03, 0.05, 0.07, 0.10)
DEFAULT_COVERAGE_LEVELS = (1.00, 0.50, 0.25, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01)

EDGE_BINS = (
    (0.000, 0.015, "0–1.5%"),
    (0.015, 0.030, "1.5–3%"),
    (0.030, 0.050, "3–5%"),
    (0.050, 0.070, "5–7%"),
    (0.070, 0.090, "7–9%"),
    (0.090, 0.110, "9–11%"),
    (0.110, 0.130, "11–13%"),
    (0.130, 0.150, "13–15%"),
    (0.150, float("inf"), "≥15%"),
)

ODDS_BANDS = (
    (1.00, 1.50, "1.00–1.50"),
    (1.50, 1.75, "1.50–1.75"),
    (1.75, 2.00, "1.75–2.00"),
    (2.00, 2.25, "2.00–2.25"),
    (2.25, 2.50, "2.25–2.50"),
    (2.50, 3.00, "2.50–3.00"),
    (3.00, float("inf"), "≥3.00"),
)

MARGIN_BANDS = (
    (float("-inf"), 0.03, "<3%"),
    (0.03, 0.05, "3–5%"),
    (0.05, 0.07, "5–7%"),
    (0.07, 0.10, "7–10%"),
    (0.10, float("inf"), "≥10%"),
)


@dataclass(frozen=True)
class TemporalThresholdConfig:
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    minimum_development_bets: int = 100

    def validate(self) -> None:
        if not self.thresholds:
            raise ValueError("At least one threshold is required.")
        if any(threshold < 0.0 for threshold in self.thresholds):
            raise ValueError("Thresholds must be non-negative.")
        if self.minimum_development_bets < 1:
            raise ValueError("minimum_development_bets must be positive.")


def _as_datetime_key(value: object) -> tuple[int, str]:
    if value is None:
        return (1, "")
    if isinstance(value, datetime):
        return (0, value.isoformat())
    return (0, str(value))


def _safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def build_betting_opportunities(joined_rows: Sequence[Mapping[str, object]]) -> list[dict]:
    """Expand each match into Under and Over opportunities.

    ``book_raw_p_*`` is the break-even probability implied by the quoted odds.
    ``book_fair_p_*`` is the normalized two-way bookmaker probability.
    """

    opportunities: list[dict] = []

    for row in joined_rows:
        common = {
            "round_idx": int(row["round_idx"]),
            "match_id": int(row["match_id"]),
            "match_datetime": row.get("match_datetime"),
            "season": int(row["season"]),
            "competition": str(row["competition"]),
            "bookmaker_margin": float(row["book_margin"]),
        }

        for side in ("under", "over"):
            if side == "under":
                model_probability = float(row["p_model_under"])
                bookmaker_raw_probability = float(row["book_raw_p_under"])
                bookmaker_fair_probability = float(row["book_fair_p_under"])
                odds = float(row["under25_odds"])
                won = int(row["y_true_under25"]) == 1
            else:
                model_probability = float(row["p_model_over"])
                bookmaker_raw_probability = float(row["book_raw_p_over"])
                bookmaker_fair_probability = float(row["book_fair_p_over"])
                odds = float(row["over25_odds"])
                won = int(row["y_true_over25"]) == 1

            probability_edge = model_probability - bookmaker_raw_probability
            market_disagreement = model_probability - bookmaker_fair_probability
            estimated_ev = model_probability * odds - 1.0
            flat_profit = odds - 1.0 if won else -1.0

            opportunities.append(
                {
                    **common,
                    "side": side,
                    "model_probability": model_probability,
                    "bookmaker_break_even_probability": bookmaker_raw_probability,
                    "bookmaker_fair_probability": bookmaker_fair_probability,
                    "odds": odds,
                    "probability_edge": probability_edge,
                    "market_disagreement": market_disagreement,
                    "estimated_ev": estimated_ev,
                    "won": int(won),
                    "flat_profit": float(flat_profit),
                }
            )

    return opportunities


def select_best_opportunity_per_match(
    opportunities: Sequence[Mapping[str, object]],
    *,
    ranking_field: str = "probability_edge",
) -> list[dict]:
    """Choose one deterministic candidate side per match.

    Ties prefer Under purely as a deterministic tie breaker.
    """

    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in opportunities:
        grouped[int(row["match_id"])].append(row)

    selected: list[dict] = []
    for match_id in sorted(grouped):
        candidates = grouped[match_id]
        if len(candidates) != 2:
            raise ValueError(f"Expected exactly two opportunities for match {match_id}.")

        best = max(
            candidates,
            key=lambda row: (
                float(row[ranking_field]),
                1 if str(row["side"]) == "under" else 0,
            ),
        )
        selected.append(dict(best))

    selected.sort(
        key=lambda row: (
            _as_datetime_key(row.get("match_datetime")),
            int(row["round_idx"]),
            int(row["match_id"]),
        )
    )
    return selected


def summarize_bets(
    rows: Sequence[Mapping[str, object]],
    *,
    eligible_match_count: int,
) -> dict:
    ordered = sorted(
        rows,
        key=lambda row: (
            _as_datetime_key(row.get("match_datetime")),
            int(row["round_idx"]),
            int(row["match_id"]),
        ),
    )

    profit_sum = 0.0
    peak = 0.0
    max_drawdown = 0.0
    cumulative_profit: list[float] = []

    for row in ordered:
        profit_sum += float(row["flat_profit"])
        cumulative_profit.append(profit_sum)
        peak = max(peak, profit_sum)
        max_drawdown = max(max_drawdown, peak - profit_sum)

    num_bets = len(ordered)
    wins = sum(int(row["won"]) for row in ordered)
    mean_odds = _safe_mean([float(row["odds"]) for row in ordered])
    mean_edge = _safe_mean([float(row["probability_edge"]) for row in ordered])
    mean_disagreement = _safe_mean([float(row["market_disagreement"]) for row in ordered])
    mean_estimated_ev = _safe_mean([float(row["estimated_ev"]) for row in ordered])
    mean_bookmaker_fair_probability = _safe_mean([float(row["bookmaker_fair_probability"]) for row in ordered])
    observed_hit_rate = _safe_ratio(float(wins), float(num_bets))
    realized_market_residual = None
    if observed_hit_rate is not None and mean_bookmaker_fair_probability is not None:
        realized_market_residual = observed_hit_rate - mean_bookmaker_fair_probability

    return {
        "num_bets": num_bets,
        "coverage": float(num_bets / max(1, eligible_match_count)),
        "wins": wins,
        "hit_rate": observed_hit_rate,
        "mean_odds": mean_odds,
        "mean_probability_edge": mean_edge,
        "mean_market_disagreement": mean_disagreement,
        "mean_estimated_ev": mean_estimated_ev,
        "mean_bookmaker_fair_probability": mean_bookmaker_fair_probability,
        "realized_market_residual": realized_market_residual,
        "total_staked": float(num_bets),
        "total_profit": float(profit_sum),
        "roi": _safe_ratio(float(profit_sum), float(num_bets)),
        "max_drawdown": float(max_drawdown),
        "final_cumulative_profit": cumulative_profit[-1] if cumulative_profit else 0.0,
    }


def simulate_threshold(
    best_opportunities: Sequence[Mapping[str, object]],
    threshold: float,
    *,
    side: str | None = None,
    threshold_field: str = "probability_edge",
    eligible_match_count: int | None = None,
) -> dict:
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    if side not in (None, "under", "over"):
        raise ValueError("side must be None, 'under', or 'over'.")

    selected = [
        row
        for row in best_opportunities
        if float(row[threshold_field]) >= threshold and (side is None or str(row["side"]) == side)
    ]
    denominator = len(best_opportunities) if eligible_match_count is None else eligible_match_count
    result = summarize_bets(selected, eligible_match_count=denominator)
    result.update(
        {
            "threshold": float(threshold),
            "threshold_field": threshold_field,
            "side": side or "all",
        }
    )
    return result


def build_threshold_summary(
    best_opportunities: Sequence[Mapping[str, object]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict]:
    rows: list[dict] = []
    for threshold in thresholds:
        for side in (None, "under", "over"):
            rows.append(
                simulate_threshold(
                    best_opportunities,
                    float(threshold),
                    side=side,
                )
            )
    return rows


def _find_band(value: float, bands: Sequence[tuple[float, float, str]]) -> str | None:
    for lower, upper, label in bands:
        if lower <= value < upper:
            return label
    return None


def _summarize_groups(
    rows: Sequence[Mapping[str, object]],
    group_values: Iterable[tuple[str, Sequence[Mapping[str, object]]]],
    *,
    eligible_match_count: int,
    group_field: str,
) -> list[dict]:
    output: list[dict] = []
    for group_name, group_rows in group_values:
        summary = summarize_bets(group_rows, eligible_match_count=eligible_match_count)
        summary[group_field] = group_name
        output.append(summary)
    return output


def build_fixed_edge_bins(
    best_opportunities: Sequence[Mapping[str, object]],
) -> list[dict]:
    positive_candidate_count = sum(float(row["probability_edge"]) >= 0.0 for row in best_opportunities)
    positive_candidate_share = float(positive_candidate_count / max(1, len(best_opportunities)))

    output: list[dict] = []
    for lower, upper, label in EDGE_BINS:
        group = [row for row in best_opportunities if lower <= float(row["probability_edge"]) < upper]
        summary = summarize_bets(group, eligible_match_count=len(best_opportunities))
        summary.update(
            {
                "edge_bin": label,
                "edge_lower": lower,
                "edge_upper": upper,
                "positive_candidate_count": positive_candidate_count,
                "positive_candidate_share": positive_candidate_share,
            }
        )
        output.append(summary)
    return output


def build_quantile_edge_bins(
    best_opportunities: Sequence[Mapping[str, object]],
    *,
    quantile_count: int = 10,
) -> list[dict]:
    if quantile_count < 2:
        raise ValueError("quantile_count must be at least 2.")

    positive = [row for row in best_opportunities if float(row["probability_edge"]) >= 0.0]
    positive.sort(key=lambda row: float(row["probability_edge"]))
    if not positive:
        return []

    index_groups = np.array_split(np.arange(len(positive)), quantile_count)
    output: list[dict] = []
    for idx, indices in enumerate(index_groups, start=1):
        group = [positive[int(i)] for i in indices.tolist()]
        summary = summarize_bets(group, eligible_match_count=len(best_opportunities))
        edges = [float(row["probability_edge"]) for row in group]
        summary.update(
            {
                "quantile": idx,
                "quantile_label": f"Q{idx}",
                "edge_min": min(edges) if edges else None,
                "edge_max": max(edges) if edges else None,
                "positive_candidate_count": len(positive),
            }
        )
        output.append(summary)
    return output


def build_coverage_curve(
    best_opportunities: Sequence[Mapping[str, object]],
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> list[dict]:
    positive = [row for row in best_opportunities if float(row["probability_edge"]) >= 0.0]
    positive.sort(
        key=lambda row: (
            float(row["probability_edge"]),
            float(row["estimated_ev"]),
        ),
        reverse=True,
    )

    output: list[dict] = []
    seen_counts: set[int] = set()
    for requested_coverage in coverage_levels:
        if not 0.0 < requested_coverage <= 1.0:
            raise ValueError("Coverage levels must be in (0, 1].")
        if not positive:
            count = 0
        else:
            count = max(1, ceil(len(positive) * requested_coverage))
        if count in seen_counts:
            continue
        seen_counts.add(count)
        selected = positive[:count]
        summary = summarize_bets(selected, eligible_match_count=len(best_opportunities))
        summary.update(
            {
                "requested_positive_candidate_coverage": float(requested_coverage),
                "actual_positive_candidate_coverage": float(count / max(1, len(positive))),
                "positive_candidate_count": len(positive),
                "minimum_selected_edge": (
                    min(float(row["probability_edge"]) for row in selected) if selected else None
                ),
            }
        )
        output.append(summary)
    return output


def _threshold_side_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    threshold: float,
    side: str | None,
) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if float(row["probability_edge"]) >= float(threshold) and (side is None or str(row["side"]) == side)
    ]


def build_odds_band_summary(
    best_opportunities: Sequence[Mapping[str, object]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict]:
    output: list[dict] = []
    for threshold in thresholds:
        for side in (None, "under", "over"):
            selected = _threshold_side_rows(
                best_opportunities,
                threshold=float(threshold),
                side=side,
            )
            for lower, upper, label in ODDS_BANDS:
                group = [row for row in selected if lower <= float(row["odds"]) < upper]
                summary = summarize_bets(
                    group,
                    eligible_match_count=len(best_opportunities),
                )
                summary.update(
                    {
                        "threshold": float(threshold),
                        "side": side or "all",
                        "odds_band": label,
                        "odds_lower": lower,
                        "odds_upper": upper,
                    }
                )
                output.append(summary)
    return output


def build_margin_band_summary(
    best_opportunities: Sequence[Mapping[str, object]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict]:
    output: list[dict] = []
    for threshold in thresholds:
        for side in (None, "under", "over"):
            selected = _threshold_side_rows(
                best_opportunities,
                threshold=float(threshold),
                side=side,
            )
            for lower, upper, label in MARGIN_BANDS:
                group = [row for row in selected if lower <= float(row["bookmaker_margin"]) < upper]
                summary = summarize_bets(
                    group,
                    eligible_match_count=len(best_opportunities),
                )
                summary.update(
                    {
                        "threshold": float(threshold),
                        "side": side or "all",
                        "margin_band": label,
                        "margin_lower": lower,
                        "margin_upper": upper,
                    }
                )
                output.append(summary)
    return output


def build_season_summary(
    best_opportunities: Sequence[Mapping[str, object]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict]:
    seasons = sorted({int(row["season"]) for row in best_opportunities})
    output: list[dict] = []
    for threshold in thresholds:
        for season in seasons:
            eligible = [row for row in best_opportunities if int(row["season"]) == season]
            for side in (None, "under", "over"):
                selected = _threshold_side_rows(
                    eligible,
                    threshold=float(threshold),
                    side=side,
                )
                summary = summarize_bets(selected, eligible_match_count=len(eligible))
                summary.update(
                    {
                        "threshold": float(threshold),
                        "side": side or "all",
                        "season": season,
                    }
                )
                output.append(summary)
    return output


def build_competition_summary(
    best_opportunities: Sequence[Mapping[str, object]],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict]:
    competitions = sorted({str(row["competition"]) for row in best_opportunities})
    output: list[dict] = []
    for threshold in thresholds:
        for competition in competitions:
            eligible = [row for row in best_opportunities if str(row["competition"]) == competition]
            for side in (None, "under", "over"):
                selected = _threshold_side_rows(
                    eligible,
                    threshold=float(threshold),
                    side=side,
                )
                summary = summarize_bets(selected, eligible_match_count=len(eligible))
                summary.update(
                    {
                        "threshold": float(threshold),
                        "side": side or "all",
                        "competition": competition,
                    }
                )
                output.append(summary)
    return output


def build_temporal_threshold_summary(
    best_opportunities: Sequence[Mapping[str, object]],
    config: TemporalThresholdConfig = TemporalThresholdConfig(),
    *,
    side: str | None = None,
) -> list[dict]:
    """Choose a threshold on earlier seasons and evaluate the next season.

    Threshold selection maximizes development ROI subject to the configured
    minimum number of development bets. Ties prefer the smaller threshold.
    When ``side`` is supplied, both development selection and future evaluation
    are restricted to that selected side.
    """

    config.validate()
    if side not in (None, "under", "over"):
        raise ValueError("side must be None, 'under', or 'over'.")

    seasons = sorted({int(row["season"]) for row in best_opportunities})
    output: list[dict] = []

    for test_season in seasons[1:]:
        development = [row for row in best_opportunities if int(row["season"]) < test_season]
        test = [row for row in best_opportunities if int(row["season"]) == test_season]

        candidates: list[dict] = []
        for threshold in config.thresholds:
            result = simulate_threshold(
                development,
                float(threshold),
                side=side,
                eligible_match_count=len(development),
            )
            if result["num_bets"] >= config.minimum_development_bets:
                candidates.append(result)

        common = {
            "test_season": test_season,
            "development_seasons": ",".join(str(season) for season in seasons if season < test_season),
            "side": side or "all",
        }

        if not candidates:
            output.append(
                {
                    **common,
                    "selected_threshold": None,
                    "selection_status": "insufficient-development-bets",
                    "development_num_bets": 0,
                    "development_roi": None,
                    "development_total_profit": 0.0,
                    "test_num_bets": 0,
                    "test_roi": None,
                    "test_total_profit": 0.0,
                    "test_max_drawdown": 0.0,
                }
            )
            continue

        candidates.sort(
            key=lambda row: (
                float(row["roi"] if row["roi"] is not None else float("-inf")),
                -float(row["threshold"]),
            ),
            reverse=True,
        )
        chosen = candidates[0]
        test_result = simulate_threshold(
            test,
            float(chosen["threshold"]),
            side=side,
            eligible_match_count=len(test),
        )

        output.append(
            {
                **common,
                "selected_threshold": float(chosen["threshold"]),
                "selection_status": "selected",
                "development_num_bets": int(chosen["num_bets"]),
                "development_roi": chosen["roi"],
                "development_total_profit": chosen["total_profit"],
                "test_num_bets": int(test_result["num_bets"]),
                "test_roi": test_result["roi"],
                "test_total_profit": test_result["total_profit"],
                "test_max_drawdown": test_result["max_drawdown"],
            }
        )

    return output


def build_quantile_ev_bins(
    best_opportunities: Sequence[Mapping[str, object]],
    *,
    quantile_count: int = 10,
) -> list[dict]:
    """Equal-count bins ranked by estimated expected return.

    Positive EV and positive probability edge contain the same opportunities,
    because decimal odds are positive; only the ranking can differ.
    """

    if quantile_count < 2:
        raise ValueError("quantile_count must be at least 2.")
    positive = [row for row in best_opportunities if float(row["estimated_ev"]) >= 0.0]
    positive.sort(key=lambda row: float(row["estimated_ev"]))
    if not positive:
        return []

    index_groups = np.array_split(np.arange(len(positive)), quantile_count)
    output: list[dict] = []
    for idx, indices in enumerate(index_groups, start=1):
        group = [positive[int(i)] for i in indices.tolist()]
        summary = summarize_bets(group, eligible_match_count=len(best_opportunities))
        values = [float(row["estimated_ev"]) for row in group]
        summary.update(
            {
                "quantile": idx,
                "quantile_label": f"Q{idx}",
                "ev_min": min(values) if values else None,
                "ev_max": max(values) if values else None,
                "positive_candidate_count": len(positive),
            }
        )
        output.append(summary)
    return output


def build_ev_coverage_curve(
    best_opportunities: Sequence[Mapping[str, object]],
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> list[dict]:
    positive = [row for row in best_opportunities if float(row["estimated_ev"]) >= 0.0]
    positive.sort(
        key=lambda row: (float(row["estimated_ev"]), float(row["probability_edge"])),
        reverse=True,
    )
    output: list[dict] = []
    seen_counts: set[int] = set()
    for requested_coverage in coverage_levels:
        if not 0.0 < requested_coverage <= 1.0:
            raise ValueError("Coverage levels must be in (0, 1].")
        count = 0 if not positive else max(1, ceil(len(positive) * requested_coverage))
        if count in seen_counts:
            continue
        seen_counts.add(count)
        selected = positive[:count]
        summary = summarize_bets(selected, eligible_match_count=len(best_opportunities))
        summary.update(
            {
                "requested_positive_candidate_coverage": float(requested_coverage),
                "actual_positive_candidate_coverage": float(count / max(1, len(positive))),
                "positive_candidate_count": len(positive),
                "minimum_selected_ev": (min(float(row["estimated_ev"]) for row in selected) if selected else None),
            }
        )
        output.append(summary)
    return output


def _clip_probability(value: float) -> float:
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def build_walk_forward_probability_rows(
    joined_rows: Sequence[Mapping[str, object]],
) -> list[dict]:
    """Create future-only raw, calibrated, bookmaker, and hybrid probabilities.

    For each test season after the first, Platt scaling, isotonic calibration, and
    the two-input hybrid are fitted exclusively on all earlier OOS seasons.
    """

    seasons = sorted({int(row["season"]) for row in joined_rows})
    output: list[dict] = []

    for test_season in seasons[1:]:
        development = [row for row in joined_rows if int(row["season"]) < test_season]
        test = [row for row in joined_rows if int(row["season"]) == test_season]
        if not development or not test:
            continue

        y_dev = np.asarray([int(row["y_true_under25"]) for row in development], dtype=np.int64)
        if len(np.unique(y_dev)) < 2:
            raise ValueError("Calibration development data must contain both classes.")

        p_model_dev = np.asarray(
            [_clip_probability(float(row["p_model_under"])) for row in development],
            dtype=np.float64,
        )
        p_book_dev = np.asarray(
            [_clip_probability(float(row["book_fair_p_under"])) for row in development],
            dtype=np.float64,
        )
        p_model_test = np.asarray(
            [_clip_probability(float(row["p_model_under"])) for row in test],
            dtype=np.float64,
        )
        p_book_test = np.asarray(
            [_clip_probability(float(row["book_fair_p_under"])) for row in test],
            dtype=np.float64,
        )

        platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=123)
        platt.fit(_logit(p_model_dev).reshape(-1, 1), y_dev)
        p_platt = platt.predict_proba(_logit(p_model_test).reshape(-1, 1))[:, 1]

        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(p_model_dev, y_dev)
        p_isotonic = np.asarray(isotonic.predict(p_model_test), dtype=np.float64)

        hybrid = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=123)
        x_dev = np.column_stack((_logit(p_book_dev), _logit(p_model_dev)))
        x_test = np.column_stack((_logit(p_book_test), _logit(p_model_test)))
        hybrid.fit(x_dev, y_dev)
        p_hybrid = hybrid.predict_proba(x_test)[:, 1]

        method_probabilities = {
            "model-raw": p_model_test,
            "model-platt": p_platt,
            "model-isotonic": p_isotonic,
            "bookmaker-fair": p_book_test,
            "bookmaker-model-hybrid": p_hybrid,
        }

        for index, row in enumerate(test):
            common = {
                "test_season": test_season,
                "development_seasons": ",".join(str(season) for season in seasons if season < test_season),
                "round_idx": int(row["round_idx"]),
                "match_id": int(row["match_id"]),
                "match_datetime": row.get("match_datetime"),
                "season": int(row["season"]),
                "competition": str(row["competition"]),
                "y_true_under25": int(row["y_true_under25"]),
                "under25_odds": float(row["under25_odds"]),
                "over25_odds": float(row["over25_odds"]),
                "book_raw_p_under": float(row["book_raw_p_under"]),
                "book_raw_p_over": float(row["book_raw_p_over"]),
                "book_fair_p_under": float(row["book_fair_p_under"]),
                "book_fair_p_over": float(row["book_fair_p_over"]),
                "book_margin": float(row["book_margin"]),
            }
            for method, probabilities in method_probabilities.items():
                output.append(
                    {
                        **common,
                        "method": method,
                        "probability_under25": _clip_probability(float(probabilities[index])),
                    }
                )

    return output


def build_probability_method_metrics(
    probability_rows: Sequence[Mapping[str, object]],
) -> list[dict]:
    methods = sorted({str(row["method"]) for row in probability_rows})
    seasons = sorted({int(row["test_season"]) for row in probability_rows})
    output: list[dict] = []

    for method in methods:
        method_rows = [row for row in probability_rows if str(row["method"]) == method]
        scopes: list[tuple[str, int | None, list[Mapping[str, object]]]] = [("pooled", None, method_rows)]
        scopes.extend(
            (
                "season",
                season,
                [row for row in method_rows if int(row["test_season"]) == season],
            )
            for season in seasons
        )
        for scope, season, rows in scopes:
            if not rows:
                continue
            y = np.asarray([int(row["y_true_under25"]) for row in rows], dtype=np.int64)
            p = np.asarray(
                [_clip_probability(float(row["probability_under25"])) for row in rows],
                dtype=np.float64,
            )
            auc = float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else None
            output.append(
                {
                    "method": method,
                    "scope": scope,
                    "test_season": season,
                    "prediction_count": len(rows),
                    "roc_auc": auc,
                    "accuracy_at_0_5": float(accuracy_score(y, p >= 0.5)),
                    "brier_score": float(brier_score_loss(y, p)),
                    "binary_log_loss": float(log_loss(y, p, labels=[0, 1])),
                }
            )
    return output


def build_probability_method_betting_summary(
    probability_rows: Sequence[Mapping[str, object]],
    *,
    thresholds: Sequence[float] = (0.07, 0.10),
) -> list[dict]:
    """Bet with future-only probabilities produced by each walk-forward method."""

    methods = sorted({str(row["method"]) for row in probability_rows} - {"bookmaker-fair"})
    output: list[dict] = []
    for method in methods:
        rows = [row for row in probability_rows if str(row["method"]) == method]
        joined: list[dict] = []
        for row in rows:
            p_under = _clip_probability(float(row["probability_under25"]))
            joined.append(
                {
                    "round_idx": int(row["round_idx"]),
                    "match_id": int(row["match_id"]),
                    "match_datetime": row.get("match_datetime"),
                    "season": int(row["season"]),
                    "competition": str(row["competition"]),
                    "y_true_under25": int(row["y_true_under25"]),
                    "y_true_over25": 1 - int(row["y_true_under25"]),
                    "p_model_under": p_under,
                    "p_model_over": 1.0 - p_under,
                    "under25_odds": float(row["under25_odds"]),
                    "over25_odds": float(row["over25_odds"]),
                    "book_raw_p_under": float(row["book_raw_p_under"]),
                    "book_raw_p_over": float(row["book_raw_p_over"]),
                    "book_fair_p_under": float(row["book_fair_p_under"]),
                    "book_fair_p_over": float(row["book_fair_p_over"]),
                    "book_margin": float(row["book_margin"]),
                }
            )
        best = select_best_opportunity_per_match(build_betting_opportunities(joined))
        for threshold in thresholds:
            for side in (None, "under"):
                result = simulate_threshold(
                    best,
                    float(threshold),
                    side=side,
                    eligible_match_count=len(best),
                )
                result.update({"method": method})
                output.append(result)
    return output
