from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from football_outcomes.data.fs_models import (
    FSDataBundle,
)
from football_outcomes.validation.selection import (
    SelectionValidationConfig,
    select_validation_matches,
    validate_bundle_selection,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_match(
    match_id: int,
    *,
    season: int = 2022,
    round_id: int = 10,
    home_id: int,
    away_id: int,
    day: int,
    home_goals: int = 1,
    away_goals: int = 0,
):
    return SimpleNamespace(
        id=match_id,
        comp_name="Test League",
        season=season,
        round_id=round_id,
        datetime=datetime(
            season,
            1,
            day,
        ),
        hour_utc=12,
        home_team=SimpleNamespace(id=home_id),
        away_team=SimpleNamespace(id=away_id),
        home_goals=home_goals,
        away_goals=away_goals,
    )


def make_config() -> SelectionValidationConfig:
    return SelectionValidationConfig(
        competitions=("Test League",),
        first_season=2021,
        last_season_exclusive=2024,
        excluded_competition_seasons=(frozenset()),
        valid_round_ids_by_season={
            (
                "Test League",
                2022,
            ): {10}
        },
    )


def test_selection_module_is_offline_and_global_free() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "validation" / "selection.py"
    source = source_path.read_text(encoding="utf-8")

    assert "requests" not in source
    assert "fs_globals" not in source
    assert "Global" not in source
    assert "fs_settings" not in source


def test_valid_selection_and_rounds_pass() -> None:
    matches = [
        make_match(
            1,
            home_id=1,
            away_id=2,
            day=1,
        ),
        make_match(
            2,
            home_id=3,
            away_id=4,
            day=1,
        ),
        make_match(
            3,
            home_id=1,
            away_id=3,
            day=2,
        ),
    ]

    bundle = FSDataBundle(matches=matches)

    report = validate_bundle_selection(
        bundle,
        make_config(),
    )

    assert report.ok
    assert report.metrics["selected_matches"] == 3
    assert report.metrics["constructed_rounds"] == 2
    assert report.metrics["final_round_size"] == 1


def test_round_filter_is_applied() -> None:
    matches = [
        make_match(
            1,
            round_id=10,
            home_id=1,
            away_id=2,
            day=1,
        ),
        make_match(
            2,
            round_id=99,
            home_id=3,
            away_id=4,
            day=2,
        ),
    ]

    selected = select_validation_matches(
        matches,
        make_config(),
    )

    assert [match.id for match in selected] == [1]


def test_missing_whitelist_is_critical() -> None:
    config = SelectionValidationConfig(
        competitions=("Test League",),
        first_season=2021,
        last_season_exclusive=2024,
        excluded_competition_seasons=(frozenset()),
        valid_round_ids_by_season={},
    )

    bundle = FSDataBundle(
        matches=[
            make_match(
                1,
                home_id=1,
                away_id=2,
                day=1,
            )
        ]
    )

    report = validate_bundle_selection(
        bundle,
        config,
    )

    assert report.count_for("missing_round_whitelist") == 1
    assert not report.ok


def test_missing_configured_competition_is_detected() -> None:
    config = SelectionValidationConfig(
        competitions=(
            "Test League",
            "Missing League",
        ),
        first_season=2021,
        last_season_exclusive=2024,
        excluded_competition_seasons=(frozenset()),
        valid_round_ids_by_season={
            (
                "Test League",
                2022,
            ): {10}
        },
    )

    bundle = FSDataBundle(
        matches=[
            make_match(
                1,
                home_id=1,
                away_id=2,
                day=1,
            )
        ]
    )

    report = validate_bundle_selection(
        bundle,
        config,
    )

    assert report.count_for("configured_competition_missing") == 1


def test_report_is_deterministic() -> None:
    bundle = FSDataBundle(
        matches=[
            make_match(
                2,
                home_id=3,
                away_id=4,
                day=2,
            ),
            make_match(
                1,
                home_id=1,
                away_id=2,
                day=1,
            ),
        ]
    )

    first = validate_bundle_selection(
        bundle,
        make_config(),
    )
    second = validate_bundle_selection(
        bundle,
        make_config(),
    )

    assert first.to_dict() == second.to_dict()


def test_selected_missing_team_is_critical() -> None:
    match = make_match(
        1,
        home_id=1,
        away_id=2,
        day=1,
    )
    match.away_team = None

    bundle = FSDataBundle(matches=[match])

    report = validate_bundle_selection(
        bundle,
        make_config(),
    )

    assert report.count_for(("selected_match_" "missing_away_team")) == 1
    assert not report.ok


def test_selected_invalid_goals_are_critical() -> None:
    match = make_match(
        1,
        home_id=1,
        away_id=2,
        day=1,
        home_goals=-1,
    )

    bundle = FSDataBundle(matches=[match])

    report = validate_bundle_selection(
        bundle,
        make_config(),
    )

    assert report.count_for(("selected_match_" "invalid_home_goals")) == 1
    assert not report.ok
