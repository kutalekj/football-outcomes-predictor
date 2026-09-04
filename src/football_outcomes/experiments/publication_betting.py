from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from football_outcomes.experiments.bookmaker import EDGE_BINS, summarize_bets

PUBLICATION_EDGE_THRESHOLDS = (0.09, 0.10, 0.11)


@dataclass(frozen=True)
class BlockBootstrapConfig:
    replicates: int = 5000
    confidence_level: float = 0.95
    seed: int = 20260904
    block_field: str = "round_idx"

    def validate(self) -> None:
        if self.replicates < 100:
            raise ValueError("replicates must be at least 100.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1).")
        if not self.block_field:
            raise ValueError("block_field must be non-empty.")


@dataclass(frozen=True)
class PublicationBettingConfig:
    thresholds: tuple[float, ...] = PUBLICATION_EDGE_THRESHOLDS
    reference_model: str = "proposed"
    bootstrap: BlockBootstrapConfig = BlockBootstrapConfig()

    def validate(self) -> None:
        if not self.thresholds:
            raise ValueError("At least one threshold is required.")
        if any(threshold < 0.0 for threshold in self.thresholds):
            raise ValueError("Thresholds must be non-negative.")
        if not self.reference_model:
            raise ValueError("reference_model must be non-empty.")
        self.bootstrap.validate()


def _validate_common_match_population(
    best_by_model: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    if not best_by_model:
        raise ValueError("At least one model is required.")

    expected_ids: set[int] | None = None
    for model_name, rows in best_by_model.items():
        if not model_name:
            raise ValueError("Model names must be non-empty.")
        ids = [int(row["match_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate match_id values for model {model_name!r}.")
        current = set(ids)
        if expected_ids is None:
            expected_ids = current
        elif current != expected_ids:
            raise ValueError("All models must cover exactly the same match_id population.")


def _group_by_block(
    rows: Sequence[Mapping[str, object]],
    *,
    block_field: str,
) -> list[list[Mapping[str, object]]]:
    groups: dict[object, list[Mapping[str, object]]] = {}
    order: list[object] = []
    for row in rows:
        key = row[block_field]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    return [groups[key] for key in order]


def _roi(rows: Sequence[Mapping[str, object]]) -> float | None:
    if not rows:
        return None
    return float(np.sum([float(row["flat_profit"]) for row in rows], dtype=np.float64) / len(rows))


def _bootstrap_percentile_interval(
    values: Sequence[float],
    confidence_level: float,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    alpha = 1.0 - confidence_level
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.quantile(array, alpha / 2.0)),
        float(np.quantile(array, 1.0 - alpha / 2.0)),
    )


def bootstrap_threshold_roi(
    best_opportunities: Sequence[Mapping[str, object]],
    *,
    threshold: float,
    config: BlockBootstrapConfig,
) -> dict:
    """Estimate ROI uncertainty by resampling chronological round blocks.

    Whole blocks of eligible matches are sampled with replacement. The betting
    threshold is then re-applied inside each bootstrap replicate. This preserves
    within-round dependence better than independently resampling individual bets.
    """

    config.validate()
    blocks = _group_by_block(best_opportunities, block_field=config.block_field)
    if not blocks:
        return {
            "roi_ci_lower": None,
            "roi_ci_upper": None,
            "bootstrap_valid_replicates": 0,
            "bootstrap_replicates": config.replicates,
            "bootstrap_confidence_level": config.confidence_level,
            "bootstrap_block_field": config.block_field,
        }

    rng = np.random.default_rng(config.seed)
    roi_values: list[float] = []
    for _ in range(config.replicates):
        block_indices = rng.integers(0, len(blocks), size=len(blocks))
        sampled = [row for block_index in block_indices for row in blocks[int(block_index)]]
        selected = [row for row in sampled if float(row["probability_edge"]) >= threshold]
        value = _roi(selected)
        if value is not None and np.isfinite(value):
            roi_values.append(value)

    lower, upper = _bootstrap_percentile_interval(
        roi_values,
        config.confidence_level,
    )
    return {
        "roi_ci_lower": lower,
        "roi_ci_upper": upper,
        "bootstrap_valid_replicates": len(roi_values),
        "bootstrap_replicates": config.replicates,
        "bootstrap_confidence_level": config.confidence_level,
        "bootstrap_block_field": config.block_field,
    }


def bootstrap_edge_bin_roi(
    best_opportunities: Sequence[Mapping[str, object]],
    *,
    lower: float,
    upper: float,
    config: BlockBootstrapConfig,
) -> dict:
    """Bootstrap ROI for one fixed edge bin using chronological round blocks."""

    config.validate()
    blocks = _group_by_block(best_opportunities, block_field=config.block_field)
    if not blocks:
        return {
            "roi_ci_lower": None,
            "roi_ci_upper": None,
            "bootstrap_valid_replicates": 0,
        }

    rng = np.random.default_rng(config.seed)
    roi_values: list[float] = []
    for _ in range(config.replicates):
        block_indices = rng.integers(0, len(blocks), size=len(blocks))
        sampled = [row for block_index in block_indices for row in blocks[int(block_index)]]
        selected = [row for row in sampled if lower <= float(row["probability_edge"]) < upper]
        value = _roi(selected)
        if value is not None and np.isfinite(value):
            roi_values.append(value)

    ci_lower, ci_upper = _bootstrap_percentile_interval(
        roi_values,
        config.confidence_level,
    )
    return {
        "roi_ci_lower": ci_lower,
        "roi_ci_upper": ci_upper,
        "bootstrap_valid_replicates": len(roi_values),
        "bootstrap_replicates": config.replicates,
        "bootstrap_confidence_level": config.confidence_level,
        "bootstrap_block_field": config.block_field,
    }


def build_publication_edge_bins(
    best_opportunities: Sequence[Mapping[str, object]],
    *,
    bootstrap: BlockBootstrapConfig,
) -> list[dict]:
    """Build the publication edge-bin table with support and ROI uncertainty."""

    output: list[dict] = []
    eligible_count = len(best_opportunities)
    for bin_index, (lower, upper, label) in enumerate(EDGE_BINS):
        selected = [row for row in best_opportunities if lower <= float(row["probability_edge"]) < upper]
        summary = summarize_bets(selected, eligible_match_count=eligible_count)
        losses = int(summary["num_bets"]) - int(summary["wins"])
        summary.update(
            {
                "edge_bin": label,
                "edge_lower": lower,
                "edge_upper": upper,
                "losses": losses,
                "win_share": (float(summary["wins"] / summary["num_bets"]) if int(summary["num_bets"]) > 0 else None),
                "loss_share": (float(losses / summary["num_bets"]) if int(summary["num_bets"]) > 0 else None),
            }
        )
        # Offset the deterministic seed by bin so each bin gets an independent
        # but reproducible bootstrap stream.
        summary.update(
            bootstrap_edge_bin_roi(
                best_opportunities,
                lower=lower,
                upper=upper,
                config=BlockBootstrapConfig(
                    replicates=bootstrap.replicates,
                    confidence_level=bootstrap.confidence_level,
                    seed=bootstrap.seed + bin_index,
                    block_field=bootstrap.block_field,
                ),
            )
        )
        output.append(summary)
    return output


def build_fixed_threshold_comparison(
    best_by_model: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    config: PublicationBettingConfig,
) -> list[dict]:
    """Evaluate all models at the same absolute edge thresholds."""

    config.validate()
    _validate_common_match_population(best_by_model)
    output: list[dict] = []

    for model_index, (model_name, rows) in enumerate(best_by_model.items()):
        for threshold_index, threshold in enumerate(config.thresholds):
            selected = [row for row in rows if float(row["probability_edge"]) >= threshold]
            summary = summarize_bets(selected, eligible_match_count=len(rows))
            summary.update(
                {
                    "model": model_name,
                    "selection_mode": "fixed-threshold",
                    "threshold": float(threshold),
                    "reference_model": config.reference_model,
                }
            )
            summary.update(
                bootstrap_threshold_roi(
                    rows,
                    threshold=float(threshold),
                    config=BlockBootstrapConfig(
                        replicates=config.bootstrap.replicates,
                        confidence_level=config.bootstrap.confidence_level,
                        seed=(config.bootstrap.seed + 1000 * model_index + threshold_index),
                        block_field=config.bootstrap.block_field,
                    ),
                )
            )
            output.append(summary)

    return output


def _rank_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["probability_edge"]),
            float(row.get("estimated_ev", 0.0)),
            -int(row["match_id"]),
        ),
        reverse=True,
    )


def bootstrap_top_n_roi(
    best_opportunities: Sequence[Mapping[str, object]],
    *,
    count: int,
    config: BlockBootstrapConfig,
) -> dict:
    """Bootstrap ROI for a top-N edge-ranked portfolio by chronological blocks.

    Blocks are resampled from the full eligible match population; within every
    replicate the opportunities are ranked again and the strongest N are kept.
    This quantifies historical sampling uncertainty in the matched-coverage view
    without retraining the prediction model.
    """

    config.validate()
    if count < 0:
        raise ValueError("count must be non-negative.")
    blocks = _group_by_block(best_opportunities, block_field=config.block_field)
    if not blocks or count == 0:
        return {
            "roi_ci_lower": None,
            "roi_ci_upper": None,
            "bootstrap_valid_replicates": 0,
            "bootstrap_replicates": config.replicates,
            "bootstrap_confidence_level": config.confidence_level,
            "bootstrap_block_field": config.block_field,
        }

    rng = np.random.default_rng(config.seed)
    roi_values: list[float] = []
    for _ in range(config.replicates):
        block_indices = rng.integers(0, len(blocks), size=len(blocks))
        sampled = [row for block_index in block_indices for row in blocks[int(block_index)]]
        selected = _rank_rows(sampled)[: min(count, len(sampled))]
        value = _roi(selected)
        if value is not None and np.isfinite(value):
            roi_values.append(value)

    lower, upper = _bootstrap_percentile_interval(
        roi_values,
        config.confidence_level,
    )
    return {
        "roi_ci_lower": lower,
        "roi_ci_upper": upper,
        "bootstrap_valid_replicates": len(roi_values),
        "bootstrap_replicates": config.replicates,
        "bootstrap_confidence_level": config.confidence_level,
        "bootstrap_block_field": config.block_field,
    }


def build_matched_coverage_comparison(
    best_by_model: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    config: PublicationBettingConfig,
) -> list[dict]:
    """Give each model the same number of bets as the reference model.

    The reference count is determined separately at each publication threshold.
    Other models then receive their top-N opportunities ranked by probability edge.
    This comparison focuses on opportunity ranking rather than probability scale.
    """

    config.validate()
    _validate_common_match_population(best_by_model)
    if config.reference_model not in best_by_model:
        raise ValueError("reference_model is missing from best_by_model.")

    reference_rows = best_by_model[config.reference_model]
    output: list[dict] = []

    for threshold in config.thresholds:
        reference_count = sum(float(row["probability_edge"]) >= threshold for row in reference_rows)
        for model_name, rows in best_by_model.items():
            selected = _rank_rows(rows)[:reference_count]
            summary = summarize_bets(selected, eligible_match_count=len(rows))
            summary.update(
                {
                    "model": model_name,
                    "selection_mode": "matched-coverage",
                    "reference_threshold": float(threshold),
                    "reference_model": config.reference_model,
                    "reference_num_bets": int(reference_count),
                    "threshold": None,
                }
            )
            summary.update(
                bootstrap_top_n_roi(
                    rows,
                    count=reference_count,
                    config=BlockBootstrapConfig(
                        replicates=config.bootstrap.replicates,
                        confidence_level=config.bootstrap.confidence_level,
                        seed=(
                            config.bootstrap.seed
                            + 10000
                            + 1000 * list(best_by_model).index(model_name)
                            + list(config.thresholds).index(threshold)
                        ),
                        block_field=config.bootstrap.block_field,
                    ),
                )
            )
            output.append(summary)

    return output
