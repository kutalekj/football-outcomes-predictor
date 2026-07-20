from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from football_outcomes.data.fs_models import (
    FSMatch,
    FSMatchFeatures,
    FSTeam,
)
from football_outcomes.validation.readiness import (
    FeatureReadinessConfig,
    validate_feature_readiness,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPETITION = "Test League"


def make_team(
    team_id: int,
    name: str,
) -> FSTeam:
    return FSTeam(
        team_id,
        name,
        name.lower(),
        name,
        name,
        name[:3].upper(),
        "England",
    )


def make_match(
    match_id: int,
    *,
    home_id: int,
    away_id: int,
    home_goals: int = 1,
    away_goals: int = 0,
) -> FSMatch:
    home_team = make_team(
        home_id,
        f"Home {home_id}",
    )
    away_team = make_team(
        away_id,
        f"Away {away_id}",
    )

    feature = FSMatchFeatures(
        comp_id=0,
        season=0.5,
        home_team_id=home_id,
        away_team_id=away_id,
        hours_sin=0.0,
        hours_cos=1.0,
        month_sin=0.0,
        month_cos=1.0,
    )
    feature.home_team_strength = np.full(
        (11, 34),
        80.0,
        dtype=np.float32,
    )
    feature.away_team_strength = np.full(
        (11, 34),
        60.0,
        dtype=np.float32,
    )
    feature.home_player_positions = [
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
    ]
    feature.away_player_positions = [
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
    ]

    match = FSMatch(match_id)
    match.home_team = home_team
    match.away_team = away_team
    match.comp_name = COMPETITION
    match.home_goals = home_goals
    match.away_goals = away_goals
    match.features_before_match = feature

    return match


def make_config(
    *,
    chunk_size: int = 2,
) -> FeatureReadinessConfig:
    return FeatureReadinessConfig(
        competition_names=(COMPETITION,),
        chunk_size=chunk_size,
        max_goals_class=10,
        position_count=4,
    )


def test_readiness_module_is_offline_and_global_free() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "validation" / "readiness.py"
    source = source_path.read_text(encoding="utf-8")

    assert "requests" not in source
    assert "fs_globals" not in source
    assert "Global" not in source
    assert "fs_settings" not in source


def test_valid_arrays_and_targets_pass() -> None:
    matches = [
        make_match(
            1,
            home_id=1,
            away_id=2,
            home_goals=1,
            away_goals=0,
        ),
        make_match(
            2,
            home_id=3,
            away_id=4,
            home_goals=2,
            away_goals=2,
        ),
    ]

    report = validate_feature_readiness(
        matches,
        make_config(),
    )

    assert report.ok
    assert report.critical_issue_count == 0
    assert report.metrics["processed_array_matches"] == 2
    assert report.metrics["binary_under_25_count"] == 1
    assert report.metrics["binary_over_25_count"] == 1
    assert report.metrics["missing_strength_cells"] == 0


def test_missing_feature_object_is_warning() -> None:
    ready = make_match(
        1,
        home_id=1,
        away_id=2,
    )
    missing = make_match(
        2,
        home_id=3,
        away_id=4,
    )
    missing.features_before_match = None

    report = validate_feature_readiness(
        [
            ready,
            missing,
        ],
        make_config(),
    )

    assert report.ok
    assert report.count_for("missing_persisted_features") == 1
    assert report.warning_count == 1
    assert report.metrics["usable_feature_matches"] == 1


def test_nonfinite_numerical_feature_is_critical() -> None:
    match = make_match(
        1,
        home_id=1,
        away_id=2,
    )
    match.features_before_match.season = float("nan")

    report = validate_feature_readiness(
        [match],
        make_config(),
    )

    assert report.count_for("nonfinite_numerical") == 1
    assert not report.ok


def test_invalid_position_id_is_critical() -> None:
    match = make_match(
        1,
        home_id=1,
        away_id=2,
    )
    match.features_before_match.home_player_positions[0] = 99

    report = validate_feature_readiness(
        [match],
        make_config(),
    )

    assert report.count_for("out_of_range_home_positions") == 1
    assert not report.ok


def test_chunk_size_must_be_positive() -> None:
    with pytest.raises(
        ValueError,
        match="chunk_size must be positive",
    ):
        validate_feature_readiness(
            [],
            make_config(chunk_size=0),
        )
