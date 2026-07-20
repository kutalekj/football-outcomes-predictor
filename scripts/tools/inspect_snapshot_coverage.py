from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from football_outcomes.application.snapshot_selection import (
    resolve_snapshot_path,
)
from football_outcomes.config import fs_settings as sett
from football_outcomes.data.snapshots import (
    load_snapshot,
)
from football_outcomes.validation.selection import (
    SelectionValidationConfig,
    select_validation_matches,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Measure lineup, SoFIFA matching " "and strength coverage for the " "selected snapshot scope.")
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
    return parser


def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def _cache_status(
    record: object,
) -> tuple[str, str]:
    if record is None:
        return (
            "missing",
            "missing_cache_entry",
        )

    if (
        not isinstance(
            record,
            (
                tuple,
                list,
            ),
        )
        or len(record) < 5
    ):
        return (
            "invalid",
            "invalid_cache_record",
        )

    sofifa_id = record[0]
    reason = str(record[4])

    if type(sofifa_id) is int and sofifa_id > 0:
        return (
            "matched",
            reason,
        )

    return (
        "failed",
        reason,
    )


def _lineup_size_bucket(
    value: int,
) -> str:
    if value == 0:
        return "empty"

    if value < sett.TEAM_STRENGTH_NUM_PLAYERS:
        return "partial"

    if value == sett.TEAM_STRENGTH_NUM_PLAYERS:
        return "complete"

    return "oversized"


def _scope_key(
    match,
) -> tuple[str, int]:
    return (
        str(match.comp_name),
        int(match.season),
    )


def _analyse_strength_matrix(
    matrix: object,
    *,
    expected_shape: tuple[int, int],
) -> dict[str, int]:
    result = {
        "strength_matrices": 1,
        "valid_strength_matrices": 0,
        "missing_strength_matrices": 0,
        "invalid_strength_matrices": 0,
        "strength_cells": 0,
        "observed_strength_cells": 0,
        "missing_strength_cells": 0,
        "fully_missing_strength_rows": 0,
        "partially_missing_strength_rows": 0,
        "fully_observed_strength_rows": 0,
        "fully_missing_strength_matrices": 0,
        "fully_observed_strength_matrices": 0,
    }

    if matrix is None:
        result["missing_strength_matrices"] = 1
        return result

    try:
        values = np.asarray(
            matrix,
            dtype=np.float64,
        )
    except (
        TypeError,
        ValueError,
    ):
        result["invalid_strength_matrices"] = 1
        return result

    if values.shape != expected_shape:
        result["invalid_strength_matrices"] = 1
        return result

    result["valid_strength_matrices"] = 1

    observed = np.isfinite(values) & (values >= 0.0)

    total_cells = int(observed.size)
    observed_cells = int(observed.sum())
    missing_cells = total_cells - observed_cells

    row_observed = observed.sum(axis=1)
    skill_count = expected_shape[1]

    fully_missing_rows = int(np.count_nonzero(row_observed == 0))
    fully_observed_rows = int(np.count_nonzero(row_observed == skill_count))
    partially_missing_rows = int(np.count_nonzero((row_observed > 0) & (row_observed < skill_count)))

    result["strength_cells"] = total_cells
    result["observed_strength_cells"] = observed_cells
    result["missing_strength_cells"] = missing_cells
    result["fully_missing_strength_rows"] = fully_missing_rows
    result["partially_missing_strength_rows"] = partially_missing_rows
    result["fully_observed_strength_rows"] = fully_observed_rows
    result["fully_missing_strength_matrices"] = int(observed_cells == 0)
    result["fully_observed_strength_matrices"] = int(observed_cells == total_cells)

    return result


def _add_counts(
    target: Counter,
    values: dict[str, int],
) -> None:
    for key, value in values.items():
        target[key] += value


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    snapshot_path = resolve_snapshot_path(args.snapshot)
    bundle = load_snapshot(snapshot_path)

    selection_config = SelectionValidationConfig(
        competitions=tuple(sett.COMPS_LEAGUE),
        first_season=(sett.FIRST_SEASON),
        last_season_exclusive=(sett.LAST_SEASON),
        excluded_competition_seasons=(frozenset(sett.EXCLUDED_COMP_SEASONS)),
        valid_round_ids_by_season=(sett.LEAGUE_VALID_ROUND_IDS_BY_SEASON),
    )

    selected = select_validation_matches(
        bundle.matches,
        selection_config,
    )

    summary: Counter = Counter()
    per_scope: dict[
        tuple[str, int],
        Counter,
    ] = defaultdict(Counter)

    lineup_positions: Counter = Counter()
    cache_reasons: Counter = Counter()

    selected_team_ids: set[int] = set()
    unique_lineup_players: dict[
        int,
        Any,
    ] = {}

    summary["selected_matches"] = len(selected)
    summary["selected_match_sides"] = 2 * len(selected)
    summary["sofifa_snapshot_count"] = len(bundle.sofifa_snapshots)
    summary["player_match_cache_entries"] = len(bundle.fs_to_sofifa_cache)
    summary["team_mapping_entries"] = len(bundle.fs_team_to_sofifa_team)

    sofifa_player_ids: set[int] = set()
    sofifa_snapshot_records = 0

    for _, snapshot_players in bundle.sofifa_snapshots:
        sofifa_snapshot_records += len(snapshot_players)
        sofifa_player_ids.update(snapshot_players.keys())

    summary["sofifa_snapshot_player_records"] = sofifa_snapshot_records
    summary["unique_sofifa_players"] = len(sofifa_player_ids)

    expected_strength_shape = (
        sett.TEAM_STRENGTH_NUM_PLAYERS,
        len(sett.PLAYER_SKILLS),
    )

    for match in selected:
        scope = _scope_key(match)
        scope_counts = per_scope[scope]

        summary["feature_ready_matches"] += int(
            getattr(
                match,
                "features_before_match",
                None,
            )
            is not None
        )
        scope_counts["selected_matches"] += 1

        feature = getattr(
            match,
            "features_before_match",
            None,
        )

        for side in (
            "home",
            "away",
        ):
            team = getattr(
                match,
                f"{side}_team",
                None,
            )
            team_id = getattr(
                team,
                "id",
                None,
            )

            if type(team_id) is int:
                selected_team_ids.add(team_id)

            lineup = getattr(
                match,
                f"{side}_lineup",
                None,
            )

            if not isinstance(
                lineup,
                list,
            ):
                summary["invalid_lineups"] += 1
                scope_counts["invalid_lineups"] += 1
            else:
                lineup_size = len(lineup)
                bucket = _lineup_size_bucket(lineup_size)

                summary[f"{bucket}_lineups"] += 1
                scope_counts[f"{bucket}_lineups"] += 1
                summary["lineup_player_references"] += lineup_size
                scope_counts["lineup_player_references"] += lineup_size

                has_goalkeeper = False

                for player in lineup:
                    player_id = getattr(
                        player,
                        "id",
                        None,
                    )
                    position = getattr(
                        player,
                        "position",
                        None,
                    )

                    position_key = str(position) if position else "missing"
                    lineup_positions[position_key] += 1

                    if position == "Goalkeeper":
                        has_goalkeeper = True

                    if type(player_id) is not int or player_id <= 0:
                        summary[("invalid_lineup_" "player_references")] += 1
                        continue

                    unique_lineup_players[player_id] = player

                    status, reason = _cache_status(bundle.fs_to_sofifa_cache.get(player_id))

                    summary[("lineup_cache_" f"{status}_references")] += 1
                    scope_counts[("lineup_cache_" f"{status}_references")] += 1
                    cache_reasons[reason] += 1

                if not has_goalkeeper:
                    summary[("lineups_without_" "explicit_goalkeeper")] += 1
                    scope_counts[("lineups_without_" "explicit_goalkeeper")] += 1

            matrix = (
                getattr(
                    feature,
                    (f"{side}_" "team_strength"),
                    None,
                )
                if feature is not None
                else None
            )

            matrix_counts = _analyse_strength_matrix(
                matrix,
                expected_shape=(expected_strength_shape),
            )

            _add_counts(
                summary,
                matrix_counts,
            )
            _add_counts(
                scope_counts,
                matrix_counts,
            )

    summary["unique_selected_teams"] = len(selected_team_ids)
    summary["unique_lineup_players"] = len(unique_lineup_players)

    for team_id in selected_team_ids:
        mapping = bundle.fs_team_to_sofifa_team.get(team_id)

        if type(mapping) is int and mapping > 0:
            summary["mapped_selected_teams"] += 1
        elif mapping is None:
            summary[("missing_selected_" "team_mappings")] += 1
        else:
            summary[("explicitly_unmapped_" "selected_teams")] += 1

    for player_id, player in unique_lineup_players.items():
        status, _ = _cache_status(bundle.fs_to_sofifa_cache.get(player_id))

        summary[("unique_lineup_players_" f"cache_{status}")] += 1

        if (
            getattr(
                player,
                "birthday",
                None,
            )
            is None
        ):
            summary[("unique_lineup_players_" "missing_birthdate")] += 1

        position = getattr(
            player,
            "position",
            None,
        )

        if position not in sett.VALID_FS_PLAYER_POSITIONS:
            summary[("unique_lineup_players_" "missing_or_invalid_position")] += 1

    summary["selected_team_mapping_rate"] = _safe_ratio(
        summary["mapped_selected_teams"],
        summary["unique_selected_teams"],
    )
    summary["unique_player_cache_match_rate"] = _safe_ratio(
        summary[("unique_lineup_players_" "cache_matched")],
        summary["unique_lineup_players"],
    )
    summary["lineup_reference_cache_match_rate"] = _safe_ratio(
        summary[("lineup_cache_" "matched_references")],
        summary["lineup_player_references"],
    )
    summary["strength_cell_observation_rate"] = _safe_ratio(
        summary["observed_strength_cells"],
        summary["strength_cells"],
    )

    per_scope_rows: list[dict[str, Any]] = []

    for (
        competition,
        season,
    ), counts in sorted(per_scope.items()):
        lineup_references = counts["lineup_player_references"]
        matched_references = counts[("lineup_cache_" "matched_references")]
        strength_cells = counts["strength_cells"]
        observed_cells = counts["observed_strength_cells"]

        row: dict[str, Any] = {
            "competition": competition,
            "season": season,
            **dict(counts),
            ("lineup_cache_" "match_rate"): _safe_ratio(
                matched_references,
                lineup_references,
            ),
            ("strength_cell_" "observation_rate"): _safe_ratio(
                observed_cells,
                strength_cells,
            ),
        }

        per_scope_rows.append(row)

    payload = {
        "snapshot_filename": (snapshot_path.name),
        "summary": dict(sorted(summary.items())),
        "lineup_positions": dict(
            sorted(
                lineup_positions.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        ),
        "cache_reasons": dict(
            sorted(
                cache_reasons.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        ),
        "competition_seasons": (per_scope_rows),
    }

    args.json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.csv_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )

    args.json_output.write_text(
        rendered + "\n",
        encoding="utf-8",
    )

    csv_fields = sorted({key for row in per_scope_rows for key in row})

    with args.csv_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=csv_fields,
        )
        writer.writeheader()
        writer.writerows(per_scope_rows)

    console_payload = {
        "snapshot_filename": (snapshot_path.name),
        "summary": payload["summary"],
        "lineup_positions": payload["lineup_positions"],
        "cache_reasons": payload["cache_reasons"],
    }

    print(
        json.dumps(
            console_payload,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
