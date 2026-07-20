from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from football_outcomes.application.snapshot_selection import (
    resolve_snapshot_path,
)
from football_outcomes.config import fs_settings as sett
from football_outcomes.data.snapshots import load_snapshot
from football_outcomes.data.sofifa_imputation import (
    StrengthImputationConfig,
)
from football_outcomes.data.sofifa_strength import (
    PastOnlyStrengthConfig,
    reconstruct_past_only_match_strength,
)
from football_outcomes.data.sofifa_temporal import (
    SkillProvenance,
)
from football_outcomes.datasets.arrays import (
    build_arrays_for_matches,
)
from football_outcomes.datasets.imputed_strength import (
    StrengthImputationContext,
    build_fold_imputed_arrays,
)
from football_outcomes.datasets.mappings import (
    build_categorical_maps,
)
from football_outcomes.datasets.rounds import (
    distribute_matches_into_rounds,
)
from football_outcomes.utils.fs_feature_utils import (
    match_sort_key,
)
from football_outcomes.validation.imputation import (
    build_step7_validation_report,
    choose_audit_fold_indices,
    render_step7_validation_markdown,
    safe_ratio,
)
from football_outcomes.validation.reporting import (
    sha256_file,
)
from football_outcomes.validation.selection import (
    SelectionValidationConfig,
    select_validation_matches,
)

OBSERVED_PROVENANCE_CODES = (
    int(SkillProvenance.NEAREST_PAST_SOFIFA),
    int(SkillProvenance.OLDER_PAST_SOFIFA),
)

IMPUTED_PROVENANCE_NAMES = (
    "COMPETITION_POSITION_MEDIAN",
    "POSITION_MEDIAN",
    "GLOBAL_SKILL_MEDIAN",
    "NEUTRAL_FALLBACK",
)

SCOPE_FIELDS = (
    "competition",
    "season",
    "matches",
    "team_sides",
    "legacy_observed_cells",
    "legacy_missing_cells",
    "legacy_observation_rate",
    "past_only_nearest_cells",
    "past_only_older_cells",
    "past_only_observed_cells",
    "past_only_unresolved_cells",
    "past_only_observation_rate",
    "legacy_only_observed_cells",
    "past_only_only_observed_cells",
    "both_observed_equal_cells",
    "both_observed_changed_cells",
    "both_missing_cells",
    "matched_player_rows",
    "unmatched_player_rows",
    "maximum_source_age_days",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate full-scope strict past-only SoFIFA coverage " "and audit fold-local strength imputation."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--window-rounds",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--audit-fold-count",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--minimum-support",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--neutral-value",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
    )
    return parser


def _selection_config() -> SelectionValidationConfig:
    return SelectionValidationConfig(
        competitions=tuple(sett.COMPS_LEAGUE),
        first_season=sett.FIRST_SEASON,
        last_season_exclusive=sett.LAST_SEASON,
        excluded_competition_seasons=(frozenset(sett.EXCLUDED_COMP_SEASONS)),
        valid_round_ids_by_season=(sett.LEAGUE_VALID_ROUND_IDS_BY_SEASON),
    )


def _strength_context(bundle) -> StrengthImputationContext:
    return StrengthImputationContext(
        snapshots=bundle.sofifa_snapshots,
        player_occurrences=(bundle.sofifa_player_occurrences),
        fs_to_sofifa_cache=(bundle.fs_to_sofifa_cache),
        reconstruction_config=(
            PastOnlyStrengthConfig(
                player_count=(sett.TEAM_STRENGTH_NUM_PLAYERS),
                skill_count=len(sett.PLAYER_SKILLS),
                max_age_days=(sett.SF_MAX_TIMEDELTA_DAYS),
                max_snapshots=(sett.SF_MAX_SNAPSHOTS_TO_SCAN),
            )
        ),
    )


def _legacy_matrix(
    match,
    side: str,
    *,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, str]:
    feature = getattr(
        match,
        "features_before_match",
        None,
    )

    matrix = (
        getattr(
            feature,
            f"{side}_team_strength",
            None,
        )
        if feature is not None
        else None
    )

    if matrix is None:
        return (
            np.full(
                expected_shape,
                -1.0,
                dtype=np.float64,
            ),
            "missing",
        )

    try:
        values = np.asarray(
            matrix,
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return (
            np.full(
                expected_shape,
                -1.0,
                dtype=np.float64,
            ),
            "invalid",
        )

    if values.shape != expected_shape:
        return (
            np.full(
                expected_shape,
                -1.0,
                dtype=np.float64,
            ),
            "invalid",
        )

    return values, "valid"


def _increment(
    target: Counter,
    key: str,
    value: int,
) -> None:
    target[key] += int(value)


def _update_full_scope_counts(
    *,
    legacy_values: np.ndarray,
    legacy_status: str,
    past_values: np.ndarray,
    past_provenance: np.ndarray,
    past_source_ages: np.ndarray,
    matched_rows: int,
    unmatched_rows: int,
    maximum_age_days: int,
    overall: Counter,
    scope: Counter,
) -> None:
    cell_count = int(legacy_values.size)

    legacy_observed = np.isfinite(legacy_values) & (legacy_values >= 0.0)
    legacy_observed_count = int(legacy_observed.sum())
    legacy_missing_count = cell_count - legacy_observed_count

    if legacy_status == "missing":
        _increment(
            overall,
            "legacy_missing_matrices",
            1,
        )
    elif legacy_status == "invalid":
        _increment(
            overall,
            "legacy_invalid_matrices",
            1,
        )

    if legacy_observed_count == 0:
        _increment(
            overall,
            "legacy_fully_missing_matrices",
            1,
        )

    past_observed = np.isin(
        past_provenance,
        OBSERVED_PROVENANCE_CODES,
    )
    past_unresolved = past_provenance == int(SkillProvenance.UNRESOLVED)
    nearest = past_provenance == int(SkillProvenance.NEAREST_PAST_SOFIFA)
    older = past_provenance == int(SkillProvenance.OLDER_PAST_SOFIFA)

    past_observed_count = int(past_observed.sum())
    past_unresolved_count = int(past_unresolved.sum())

    invalid_provenance = int(cell_count - past_observed_count - past_unresolved_count)

    if invalid_provenance:
        _increment(
            overall,
            "past_invalid_provenance_cells",
            invalid_provenance,
        )

    observed_values = past_values[past_observed]
    invalid_observed_values = int(
        np.count_nonzero(~np.isfinite(observed_values) | (observed_values < 0.0) | (observed_values > 100.0))
    )

    observed_ages = past_source_ages[past_observed]
    future_source_cells = int(np.count_nonzero(observed_ages < 0))
    beyond_max_age = int(np.count_nonzero(observed_ages > maximum_age_days))

    unresolved_age_mismatches = int(np.count_nonzero(past_source_ages[past_unresolved] != -1))

    if unresolved_age_mismatches:
        _increment(
            overall,
            "past_unresolved_age_mismatches",
            unresolved_age_mismatches,
        )

    both_observed = legacy_observed & past_observed
    equal = both_observed & np.isclose(
        legacy_values,
        past_values,
        rtol=0.0,
        atol=1e-6,
    )
    changed = both_observed & ~equal
    legacy_only = legacy_observed & ~past_observed
    past_only = ~legacy_observed & past_observed
    both_missing = ~legacy_observed & ~past_observed

    counts = {
        "legacy_observed_cells": (legacy_observed_count),
        "legacy_missing_cells": (legacy_missing_count),
        "past_only_nearest_cells": int(nearest.sum()),
        "past_only_older_cells": int(older.sum()),
        "past_only_observed_cells": (past_observed_count),
        "past_only_unresolved_cells": (past_unresolved_count),
        "legacy_only_observed_cells": int(legacy_only.sum()),
        "past_only_only_observed_cells": int(past_only.sum()),
        "both_observed_equal_cells": int(equal.sum()),
        "both_observed_changed_cells": int(changed.sum()),
        "both_missing_cells": int(both_missing.sum()),
        "matched_player_rows": matched_rows,
        "unmatched_player_rows": unmatched_rows,
        "future_source_cells": (future_source_cells),
        "source_cells_beyond_max_age": (beyond_max_age),
        "invalid_observed_values": (invalid_observed_values),
    }

    for key, value in counts.items():
        _increment(overall, key, value)
        _increment(scope, key, value)

    nonnegative_ages = observed_ages[observed_ages >= 0]

    if nonnegative_ages.size:
        maximum_age = int(nonnegative_ages.max())
        overall["maximum_source_age_days"] = max(
            overall["maximum_source_age_days"],
            maximum_age,
        )
        scope["maximum_source_age_days"] = max(
            scope["maximum_source_age_days"],
            maximum_age,
        )


def _scope_rows(
    per_scope: Mapping[
        tuple[str, int],
        Counter,
    ],
) -> list[dict[str, Any]]:
    rows = []

    for (competition, season), counts in sorted(per_scope.items()):
        row = {
            "competition": competition,
            "season": season,
            **dict(counts),
        }
        row["legacy_observation_rate"] = safe_ratio(
            counts["legacy_observed_cells"],
            counts["legacy_observed_cells"] + counts["legacy_missing_cells"],
        )
        row["past_only_observation_rate"] = safe_ratio(
            counts["past_only_observed_cells"],
            counts["past_only_observed_cells"] + counts["past_only_unresolved_cells"],
        )
        rows.append(row)

    return rows


def _full_scope_analysis(
    *,
    matches: Sequence,
    context: StrengthImputationContext,
    progress_every: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    expected_shape = (
        context.reconstruction_config.player_count,
        context.reconstruction_config.skill_count,
    )
    overall: Counter = Counter()
    per_scope: dict[
        tuple[str, int],
        Counter,
    ] = defaultdict(Counter)

    for match_index, match in enumerate(
        matches,
        start=1,
    ):
        if progress_every > 0 and match_index % progress_every == 0:
            print("[past-only] processed " f"{match_index}/{len(matches)} matches")

        reconstructed = reconstruct_past_only_match_strength(
            match=match,
            snapshots=context.snapshots,
            player_occurrences=(context.player_occurrences),
            fs_to_sofifa_cache=(context.fs_to_sofifa_cache),
            config=(context.reconstruction_config),
        )

        scope = per_scope[
            (
                str(match.comp_name),
                int(match.season),
            )
        ]
        scope["matches"] += 1

        for side_name, team in (
            ("home", reconstructed.home),
            ("away", reconstructed.away),
        ):
            scope["team_sides"] += 1
            overall["team_sides"] += 1

            legacy_values, legacy_status = _legacy_matrix(
                match,
                side_name,
                expected_shape=(expected_shape),
            )

            try:
                past_values = np.asarray(
                    team.skills,
                    dtype=np.float64,
                )
                past_provenance = np.asarray(
                    team.provenance,
                    dtype=np.int16,
                )
                past_source_ages = np.asarray(
                    team.source_age_days,
                    dtype=np.int32,
                )
            except (TypeError, ValueError):
                overall["past_invalid_matrices"] += 1
                continue

            if (
                past_values.shape != expected_shape
                or past_provenance.shape != expected_shape
                or past_source_ages.shape != expected_shape
            ):
                overall["past_invalid_matrices"] += 1
                continue

            _update_full_scope_counts(
                legacy_values=legacy_values,
                legacy_status=legacy_status,
                past_values=past_values,
                past_provenance=(past_provenance),
                past_source_ages=(past_source_ages),
                matched_rows=(team.matched_player_rows),
                unmatched_rows=(team.unmatched_player_rows),
                maximum_age_days=(context.reconstruction_config.max_age_days),
                overall=overall,
                scope=scope,
            )

    legacy = {
        "observed_cells": overall["legacy_observed_cells"],
        "missing_cells": overall["legacy_missing_cells"],
        "observation_rate": safe_ratio(
            overall["legacy_observed_cells"],
            overall["legacy_observed_cells"] + overall["legacy_missing_cells"],
        ),
        "missing_matrices": overall["legacy_missing_matrices"],
        "invalid_matrices": overall["legacy_invalid_matrices"],
        "fully_missing_matrices": overall["legacy_fully_missing_matrices"],
    }

    past_only = {
        "nearest_past_cells": overall["past_only_nearest_cells"],
        "older_past_cells": overall["past_only_older_cells"],
        "observed_cells": overall["past_only_observed_cells"],
        "unresolved_cells": overall["past_only_unresolved_cells"],
        "observation_rate": safe_ratio(
            overall["past_only_observed_cells"],
            overall["past_only_observed_cells"] + overall["past_only_unresolved_cells"],
        ),
        "matched_player_rows": overall["matched_player_rows"],
        "unmatched_player_rows": overall["unmatched_player_rows"],
        "maximum_source_age_days": overall["maximum_source_age_days"],
        "future_source_cells": overall["future_source_cells"],
        "source_cells_beyond_max_age": overall["source_cells_beyond_max_age"],
        "invalid_observed_values": overall["invalid_observed_values"],
        "invalid_matrices": overall["past_invalid_matrices"],
        "invalid_provenance_cells": overall["past_invalid_provenance_cells"],
        "unresolved_age_mismatches": overall["past_unresolved_age_mismatches"],
    }

    comparison = {
        key: overall[key]
        for key in (
            "legacy_only_observed_cells",
            "past_only_only_observed_cells",
            "both_observed_equal_cells",
            "both_observed_changed_cells",
            "both_missing_cells",
        )
    }

    return (
        legacy,
        past_only,
        comparison,
        _scope_rows(per_scope),
    )


def _arrays_equal(
    left: np.ndarray,
    right: np.ndarray,
) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(
            left,
            right,
            equal_nan=True,
        )
    )


def _audit_folds(
    *,
    rounds: Sequence[Sequence],
    fold_indices: Sequence[int],
    cat_maps,
    competition_names: Sequence[str],
    context: StrengthImputationContext,
    imputation_config: StrengthImputationConfig,
    window_rounds: int,
) -> dict[str, Any]:
    totals: Counter = Counter()
    provenance_totals: Counter = Counter()
    fold_rows = []

    for audit_number, fold_index in enumerate(
        fold_indices,
        start=1,
    ):
        training_matches = [
            match for round_matches in rounds[fold_index - window_rounds : fold_index] for match in round_matches
        ]
        validation_matches = list(rounds[fold_index])

        print(
            "[fold-audit] "
            f"{audit_number}/{len(fold_indices)} "
            f"round={fold_index + 1} "
            f"train={len(training_matches)} "
            f"validation={len(validation_matches)}"
        )

        (
            _,
            validation_arrays,
            diagnostics,
        ) = build_fold_imputed_arrays(
            training_matches=training_matches,
            validation_matches=(validation_matches),
            cat_maps=cat_maps,
            competition_names=(competition_names),
            mode="binary_u25",
            max_goals_class=10,
            context=context,
            imputation_config=(imputation_config),
        )

        base_validation = build_arrays_for_matches(
            matches=validation_matches,
            cat_maps=cat_maps,
            competition_names=(competition_names),
            mode="binary_u25",
            max_goals_class=10,
        )

        provenance = dict(diagnostics.validation_provenance_counts)
        provenance_total = sum(int(value) for value in provenance.values())
        expected_cells = (
            len(validation_matches)
            * 2
            * context.reconstruction_config.player_count
            * context.reconstruction_config.skill_count
        )

        if provenance_total != expected_cells:
            totals["provenance_conservation_failures"] += abs(expected_cells - provenance_total)

        unresolved = int(
            provenance.get(
                "UNRESOLVED",
                0,
            )
        )
        observed = int(
            provenance.get(
                "NEAREST_PAST_SOFIFA",
                0,
            )
        ) + int(
            provenance.get(
                "OLDER_PAST_SOFIFA",
                0,
            )
        )
        imputed = sum(int(provenance.get(name, 0)) for name in (IMPUTED_PROVENANCE_NAMES))

        strength = validation_arrays[4]
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

        invalid_strength_values = int(np.count_nonzero(~np.isfinite(values) | (values < 0.0) | (values > 1.0)))
        invalid_masks = int(
            np.count_nonzero(
                ~np.isin(
                    masks,
                    (0.0, 1.0),
                )
            )
        )
        mask_observed = int(np.count_nonzero(masks == 1.0))
        observed_mask_mismatches = abs(observed - mask_observed)

        nonstructured_mismatches = sum(
            not _arrays_equal(
                validation_arrays[index],
                base_validation[index],
            )
            for index in (
                0,
                1,
                2,
                3,
                5,
                6,
                7,
            )
        )

        totals["validation_matches"] += len(validation_matches)
        totals["validation_cells"] += expected_cells
        totals["observed_provenance_cells"] += observed
        totals["imputed_provenance_cells"] += imputed
        totals["unresolved_provenance_cells"] += unresolved
        totals["observed_mask_mismatches"] += observed_mask_mismatches
        totals["nonstructured_array_mismatches"] += nonstructured_mismatches
        totals["invalid_strength_values"] += invalid_strength_values
        totals["invalid_masks"] += invalid_masks
        totals["neutral_fallback_cells"] += int(
            provenance.get(
                "NEUTRAL_FALLBACK",
                0,
            )
        )
        totals["training_observed_cells"] += int(diagnostics.training_observed_cells)

        for name, count in provenance.items():
            provenance_totals[name] += int(count)

        fold_rows.append(
            {
                "round_index": (fold_index + 1),
                "training_matches": len(training_matches),
                "validation_matches": len(validation_matches),
                "validation_cells": (expected_cells),
                "observed_cells": observed,
                "imputed_cells": imputed,
                "unresolved_cells": (unresolved),
                "neutral_fallback_cells": int(
                    provenance.get(
                        "NEUTRAL_FALLBACK",
                        0,
                    )
                ),
                "training_observed_cells": int(diagnostics.training_observed_cells),
            }
        )

    return {
        "audit_fold_count": len(fold_indices),
        "audit_round_indices": [index + 1 for index in fold_indices],
        "validation_matches": totals["validation_matches"],
        "validation_cells": totals["validation_cells"],
        "observed_provenance_cells": totals["observed_provenance_cells"],
        "imputed_provenance_cells": totals["imputed_provenance_cells"],
        "unresolved_provenance_cells": totals["unresolved_provenance_cells"],
        "neutral_fallback_cells": totals["neutral_fallback_cells"],
        "observed_mask_mismatches": totals["observed_mask_mismatches"],
        "nonstructured_array_mismatches": totals["nonstructured_array_mismatches"],
        "invalid_strength_values": totals["invalid_strength_values"],
        "invalid_masks": totals["invalid_masks"],
        "provenance_conservation_failures": totals["provenance_conservation_failures"],
        "training_observed_cells_across_audits": totals["training_observed_cells"],
        "validation_provenance_counts": dict(sorted(provenance_totals.items())),
        "folds": fold_rows,
    }


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SCOPE_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, 0) for field in SCOPE_FIELDS})


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    if args.progress_every < 0:
        raise ValueError("progress_every must be non-negative.")

    snapshot_path = resolve_snapshot_path(args.snapshot)
    bundle = load_snapshot(snapshot_path)

    selected = select_validation_matches(
        bundle.matches,
        _selection_config(),
    )
    array_ready = sorted(
        (
            match
            for match in selected
            if getattr(
                match,
                "features_before_match",
                None,
            )
            is not None
        ),
        key=match_sort_key,
    )

    if not array_ready:
        raise RuntimeError("No array-ready matches were selected.")

    context = _strength_context(bundle)
    imputation_config = StrengthImputationConfig(
        skill_count=(context.reconstruction_config.skill_count),
        minimum_group_support=(args.minimum_support),
        neutral_value=args.neutral_value,
    )

    rounds = distribute_matches_into_rounds(array_ready)
    audit_fold_indices = choose_audit_fold_indices(
        round_count=len(rounds),
        window_rounds=(args.window_rounds),
        audit_fold_count=(args.audit_fold_count),
    )

    if not audit_fold_indices:
        raise RuntimeError("The selected scope has no eligible " "rolling validation folds.")

    print("[scope] selected=" f"{len(selected)} " "array-ready=" f"{len(array_ready)} " f"rounds={len(rounds)}")

    (
        legacy,
        past_only,
        comparison,
        per_scope_rows,
    ) = _full_scope_analysis(
        matches=array_ready,
        context=context,
        progress_every=(args.progress_every),
    )

    competition_names = tuple(sett.COMPS_LEAGUE)
    cat_maps = build_categorical_maps(
        matches=array_ready,
        competition_names=(competition_names),
    )

    rolling_audit = _audit_folds(
        rounds=rounds,
        fold_indices=audit_fold_indices,
        cat_maps=cat_maps,
        competition_names=(competition_names),
        context=context,
        imputation_config=(imputation_config),
        window_rounds=args.window_rounds,
    )

    total_strength_cells = (
        len(array_ready) * 2 * context.reconstruction_config.player_count * context.reconstruction_config.skill_count
    )

    report = build_step7_validation_report(
        snapshot={
            "filename": snapshot_path.name,
            "sha256": sha256_file(snapshot_path),
            "size_bytes": (snapshot_path.stat().st_size),
        },
        config={
            "window_rounds": (args.window_rounds),
            "audit_fold_count": (args.audit_fold_count),
            "minimum_group_support": (args.minimum_support),
            "neutral_value": (args.neutral_value),
            "player_count": (context.reconstruction_config.player_count),
            "skill_count": (context.reconstruction_config.skill_count),
            "max_age_days": (context.reconstruction_config.max_age_days),
            "max_snapshots": (context.reconstruction_config.max_snapshots),
        },
        scope={
            "selected_matches": len(selected),
            "array_ready_matches": len(array_ready),
            "round_count": len(rounds),
            "competition_seasons": len(per_scope_rows),
            "total_strength_cells": (total_strength_cells),
        },
        legacy=legacy,
        past_only=past_only,
        comparison=comparison,
        rolling_audit=rolling_audit,
    )

    report["competition_seasons"] = per_scope_rows

    args.json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.markdown_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.json_output.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(
        render_step7_validation_markdown(report),
        encoding="utf-8",
    )
    _write_csv(
        args.csv_output,
        per_scope_rows,
    )

    console_summary = {
        "overall_ok": report["overall_ok"],
        "critical_issue_count": report["critical_issue_count"],
        "warning_count": report["warning_count"],
        "selected_matches": len(selected),
        "array_ready_matches": len(array_ready),
        "past_only_observation_rate": (past_only["observation_rate"]),
        "audited_validation_matches": (rolling_audit["validation_matches"]),
        "json_output": str(args.json_output),
        "csv_output": str(args.csv_output),
        "markdown_output": str(args.markdown_output),
    }

    print(
        json.dumps(
            console_summary,
            indent=2,
            sort_keys=True,
        )
    )

    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
