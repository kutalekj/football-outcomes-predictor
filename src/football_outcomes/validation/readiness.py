from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from football_outcomes.data.fs_models import (
    FSMatch,
)
from football_outcomes.datasets.arrays import (
    build_arrays_for_matches,
)
from football_outcomes.datasets.mappings import (
    build_categorical_maps,
)
from football_outcomes.datasets.targets import (
    build_targets_for_matches,
)
from football_outcomes.validation.domain import (
    DomainValidationReport,
)


@dataclass(frozen=True)
class FeatureReadinessConfig:
    competition_names: tuple[str, ...]
    chunk_size: int = 512
    max_goals_class: int = 10
    position_count: int = 4


def _chunks(
    matches: Sequence[FSMatch],
    chunk_size: int,
):
    for start in range(
        0,
        len(matches),
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            len(matches),
        )
        yield start, stop, list(matches[start:stop])


def _add_array_finding(
    report: DomainValidationReport,
    code: str,
    *,
    start: int,
    stop: int,
    message: str,
) -> None:
    report.add(
        code,
        entity_type="match-chunk",
        entity_id=f"{start}:{stop}",
        message=message,
    )


def _validate_exact_shape(
    report: DomainValidationReport,
    *,
    array: np.ndarray,
    expected: tuple[int, ...],
    name: str,
    start: int,
    stop: int,
) -> bool:
    if array.shape == expected:
        return True

    _add_array_finding(
        report,
        f"invalid_{name}_shape",
        start=start,
        stop=stop,
        message=(f"Expected shape {expected}, " f"found {array.shape}."),
    )
    return False


def _validate_dtype(
    report: DomainValidationReport,
    *,
    array: np.ndarray,
    expected: np.dtype,
    name: str,
    start: int,
    stop: int,
) -> None:
    expected_dtype = np.dtype(expected)

    if array.dtype == expected_dtype:
        return

    _add_array_finding(
        report,
        f"invalid_{name}_dtype",
        start=start,
        stop=stop,
        message=(f"Expected dtype " f"{expected_dtype}, found " f"{array.dtype}."),
    )


def _validate_finite(
    report: DomainValidationReport,
    *,
    array: np.ndarray,
    name: str,
    start: int,
    stop: int,
) -> None:
    if np.all(np.isfinite(array)):
        return

    _add_array_finding(
        report,
        f"nonfinite_{name}",
        start=start,
        stop=stop,
        message=("Array contains NaN or " "infinite values."),
    )


def _validate_id_array(
    report: DomainValidationReport,
    *,
    array: np.ndarray,
    upper_bound: int,
    name: str,
    start: int,
    stop: int,
) -> None:
    if array.size == 0:
        return

    minimum = int(array.min())
    maximum = int(array.max())

    if minimum >= 0 and maximum < upper_bound:
        return

    _add_array_finding(
        report,
        f"out_of_range_{name}",
        start=start,
        stop=stop,
        message=(f"Expected IDs in " f"[0, {upper_bound}), found " f"minimum={minimum}, " f"maximum={maximum}."),
    )


def _validate_strength(
    report: DomainValidationReport,
    *,
    strength: np.ndarray,
    start: int,
    stop: int,
) -> tuple[int, int, int]:
    values = strength[
        :,
        (0, 2),
        :,
        :,
    ]
    masks = strength[
        :,
        (1, 3),
        :,
        :,
    ]

    if np.any(values < 0.0) or np.any(values > 1.0):
        _add_array_finding(
            report,
            "strength_values_out_of_range",
            start=start,
            stop=stop,
            message=("Normalized strength values " "must be in [0, 1]."),
        )

    valid_mask = (masks == 0.0) | (masks == 1.0)

    if not np.all(valid_mask):
        _add_array_finding(
            report,
            "nonbinary_strength_mask",
            start=start,
            stop=stop,
            message=("Strength masks must contain " "only 0 and 1."),
        )

    total_cells = int(masks.size)
    observed_cells = int(masks.sum())

    row_observed = masks.sum(axis=-1)
    fully_missing_rows = int(np.count_nonzero(row_observed == 0.0))

    return (
        total_cells,
        observed_cells,
        fully_missing_rows,
    )


def _arrays_are_identical(
    first: tuple[np.ndarray, ...],
    second: tuple[np.ndarray, ...],
) -> bool:
    if len(first) != len(second):
        return False

    return all(
        np.array_equal(
            left,
            right,
        )
        for left, right in zip(
            first,
            second,
        )
    )


def validate_feature_readiness(
    matches: Sequence[FSMatch],
    config: FeatureReadinessConfig,
    *,
    max_examples_per_finding: int = 5,
) -> DomainValidationReport:
    if config.chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if config.position_count <= 0:
        raise ValueError("position_count must be positive.")

    if max_examples_per_finding < 0:
        raise ValueError("max_examples_per_finding " "must be non-negative.")

    report = DomainValidationReport(max_examples_per_finding=(max_examples_per_finding))

    report.metrics["selected_matches"] = len(matches)

    usable_matches: list[FSMatch] = []

    for match in matches:
        feature = getattr(
            match,
            "features_before_match",
            None,
        )

        if feature is None:
            report.add(
                "missing_persisted_features",
                entity_type="match",
                entity_id=(
                    getattr(
                        match,
                        "id",
                        None,
                    )
                ),
                message=(
                    "Selected match has no " "persisted feature object " "and is excluded from " "array validation."
                ),
                severity="warning",
            )
            continue

        usable_matches.append(match)

    report.metrics["usable_feature_matches"] = len(usable_matches)
    report.metrics["missing_feature_matches"] = len(matches) - len(usable_matches)

    if not usable_matches:
        report.add(
            "no_feature_ready_matches",
            entity_type="selection",
            entity_id="all",
            message=("No selected matches have " "feature objects."),
        )
        return report

    try:
        category_maps = build_categorical_maps(
            matches=usable_matches,
            competition_names=(config.competition_names),
        )
    except Exception as error:
        report.add(
            "categorical_map_build_failure",
            entity_type="selection",
            entity_id="all",
            message=repr(error),
        )
        return report

    report.metrics["categorical_team_count"] = len(category_maps.team_id_map)
    report.metrics["categorical_competition_count"] = len(category_maps.comp_id_map)

    processed_matches = 0
    processed_chunks = 0
    under_25_count = 0
    total_strength_cells = 0
    observed_strength_cells = 0
    fully_missing_strength_rows = 0
    numerical_feature_count: int | None = None

    for start, stop, chunk in _chunks(
        usable_matches,
        config.chunk_size,
    ):
        processed_chunks += 1
        chunk_size = len(chunk)

        try:
            first_raw = build_arrays_for_matches(
                matches=chunk,
                cat_maps=category_maps,
                competition_names=(config.competition_names),
                mode="binary_u25",
                max_goals_class=(config.max_goals_class),
            )
            second_raw = build_arrays_for_matches(
                matches=chunk,
                cat_maps=category_maps,
                competition_names=(config.competition_names),
                mode="binary_u25",
                max_goals_class=(config.max_goals_class),
            )
        except Exception as error:
            _add_array_finding(
                report,
                "array_build_failure",
                start=start,
                stop=stop,
                message=repr(error),
            )
            continue

        if len(first_raw) != 8 or len(second_raw) != 8:
            _add_array_finding(
                report,
                "invalid_array_tuple_length",
                start=start,
                stop=stop,
                message=("Array builder must " "return eight arrays."),
            )
            continue

        first = tuple(np.asarray(array) for array in first_raw)
        second = tuple(np.asarray(array) for array in second_raw)

        if not _arrays_are_identical(
            first,
            second,
        ):
            _add_array_finding(
                report,
                "nondeterministic_arrays",
                start=start,
                stop=stop,
                message=("Repeated array builds " "returned different " "values."),
            )

        (
            numerical,
            home_ids,
            away_ids,
            competition_ids,
            strength,
            home_positions,
            away_positions,
            targets,
        ) = first

        numerical_shape_valid = numerical.ndim == 2 and numerical.shape[0] == chunk_size and numerical.shape[1] > 0

        if not numerical_shape_valid:
            _add_array_finding(
                report,
                "invalid_numerical_shape",
                start=start,
                stop=stop,
                message=(
                    "Expected a non-empty "
                    "2D numerical matrix "
                    f"with {chunk_size} rows; "
                    f"found "
                    f"{numerical.shape}."
                ),
            )
        else:
            current_width = int(numerical.shape[1])

            if numerical_feature_count is None:
                numerical_feature_count = current_width
            elif current_width != numerical_feature_count:
                _add_array_finding(
                    report,
                    ("inconsistent_" "numerical_width"),
                    start=start,
                    stop=stop,
                    message=(
                        "Numerical feature " f"width changed from " f"{numerical_feature_count} " f"to {current_width}."
                    ),
                )

        _validate_exact_shape(
            report,
            array=home_ids,
            expected=(chunk_size, 1),
            name="home_ids",
            start=start,
            stop=stop,
        )
        _validate_exact_shape(
            report,
            array=away_ids,
            expected=(chunk_size, 1),
            name="away_ids",
            start=start,
            stop=stop,
        )
        _validate_exact_shape(
            report,
            array=competition_ids,
            expected=(chunk_size, 1),
            name="competition_ids",
            start=start,
            stop=stop,
        )
        strength_shape_valid = _validate_exact_shape(
            report,
            array=strength,
            expected=(
                chunk_size,
                4,
                11,
                34,
            ),
            name="strength",
            start=start,
            stop=stop,
        )
        home_position_shape_valid = _validate_exact_shape(
            report,
            array=home_positions,
            expected=(
                chunk_size,
                11,
            ),
            name="home_positions",
            start=start,
            stop=stop,
        )
        away_position_shape_valid = _validate_exact_shape(
            report,
            array=away_positions,
            expected=(
                chunk_size,
                11,
            ),
            name="away_positions",
            start=start,
            stop=stop,
        )
        target_shape_valid = _validate_exact_shape(
            report,
            array=targets,
            expected=(chunk_size,),
            name="targets",
            start=start,
            stop=stop,
        )

        for array, expected, name in (
            (
                numerical,
                np.float32,
                "numerical",
            ),
            (
                home_ids,
                np.int32,
                "home_ids",
            ),
            (
                away_ids,
                np.int32,
                "away_ids",
            ),
            (
                competition_ids,
                np.int32,
                "competition_ids",
            ),
            (
                strength,
                np.float32,
                "strength",
            ),
            (
                home_positions,
                np.int32,
                "home_positions",
            ),
            (
                away_positions,
                np.int32,
                "away_positions",
            ),
            (
                targets,
                np.float32,
                "targets",
            ),
        ):
            _validate_dtype(
                report,
                array=array,
                expected=expected,
                name=name,
                start=start,
                stop=stop,
            )

        for array, name in (
            (
                numerical,
                "numerical",
            ),
            (
                strength,
                "strength",
            ),
            (
                targets,
                "targets",
            ),
        ):
            _validate_finite(
                report,
                array=array,
                name=name,
                start=start,
                stop=stop,
            )

        _validate_id_array(
            report,
            array=home_ids,
            upper_bound=len(category_maps.team_id_map),
            name="home_ids",
            start=start,
            stop=stop,
        )
        _validate_id_array(
            report,
            array=away_ids,
            upper_bound=len(category_maps.team_id_map),
            name="away_ids",
            start=start,
            stop=stop,
        )
        _validate_id_array(
            report,
            array=competition_ids,
            upper_bound=len(category_maps.comp_id_map),
            name="competition_ids",
            start=start,
            stop=stop,
        )

        if strength_shape_valid:
            (
                chunk_strength_cells,
                chunk_observed_cells,
                chunk_missing_rows,
            ) = _validate_strength(
                report,
                strength=strength,
                start=start,
                stop=stop,
            )

            total_strength_cells += chunk_strength_cells
            observed_strength_cells += chunk_observed_cells
            fully_missing_strength_rows += chunk_missing_rows

        for positions, shape_valid, name in (
            (
                home_positions,
                home_position_shape_valid,
                "home_positions",
            ),
            (
                away_positions,
                away_position_shape_valid,
                "away_positions",
            ),
        ):
            if shape_valid and positions.size > 0:
                minimum = int(positions.min())
                maximum = int(positions.max())

                if minimum < 0 or maximum >= config.position_count:
                    _add_array_finding(
                        report,
                        (f"out_of_range_" f"{name}"),
                        start=start,
                        stop=stop,
                        message=(
                            "Position IDs must "
                            f"be in "
                            f"[0, "
                            f"{config.position_count}); "
                            f"found minimum="
                            f"{minimum}, maximum="
                            f"{maximum}."
                        ),
                    )

        if target_shape_valid:
            valid_targets = (targets == 0.0) | (targets == 1.0)

            if not np.all(valid_targets):
                _add_array_finding(
                    report,
                    "invalid_binary_targets",
                    start=start,
                    stop=stop,
                    message=("Binary targets must " "contain only 0 and 1."),
                )

            expected_targets = build_targets_for_matches(
                matches=chunk,
                mode="binary_u25",
                max_goals_class=(config.max_goals_class),
            )

            if not np.array_equal(
                targets,
                expected_targets,
            ):
                _add_array_finding(
                    report,
                    "target_builder_mismatch",
                    start=start,
                    stop=stop,
                    message=("Targets returned by " "the array builder do " "not match the target " "builder."),
                )

            under_25_count += int(targets.sum())

        processed_matches += chunk_size

    report.metrics["processed_array_matches"] = processed_matches
    report.metrics["processed_array_chunks"] = processed_chunks
    report.metrics["numerical_feature_count"] = numerical_feature_count or 0
    report.metrics["binary_under_25_count"] = under_25_count
    report.metrics["binary_over_25_count"] = processed_matches - under_25_count

    if processed_matches:
        report.metrics["binary_under_25_prevalence"] = under_25_count / processed_matches

    report.metrics["strength_mask_cells"] = total_strength_cells
    report.metrics["observed_strength_cells"] = observed_strength_cells
    report.metrics["missing_strength_cells"] = total_strength_cells - observed_strength_cells
    report.metrics["fully_missing_strength_rows"] = fully_missing_strength_rows

    return report
