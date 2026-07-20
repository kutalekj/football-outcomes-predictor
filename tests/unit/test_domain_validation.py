from __future__ import annotations

from datetime import datetime
from pathlib import Path

from football_outcomes.data.fs_models import (
    FSCompSeason,
    FSDataBundle,
    FSMatch,
    FSPlayer,
    FSTeam,
)
from football_outcomes.validation.domain import (
    validate_bundle_domain,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def make_player(
    player_id: int,
    name: str,
) -> FSPlayer:
    player = FSPlayer(
        player_id,
        name,
        name,
        "Player",
        name,
        name,
    )
    player.position = "Midfielder"
    return player


def make_valid_bundle() -> FSDataBundle:
    comp_season = FSCompSeason(
        10,
        2024,
        "England",
        "Test League",
    )

    home_team = make_team(
        1,
        "Home Team",
    )
    away_team = make_team(
        2,
        "Away Team",
    )

    home_player = make_player(
        101,
        "Home Player",
    )
    away_player = make_player(
        102,
        "Away Player",
    )

    home_team.comp_seasons[comp_season.id] = [home_player]
    away_team.comp_seasons[comp_season.id] = [away_player]

    match = FSMatch(1000)
    match.home_team = home_team
    match.away_team = away_team
    match.home_goals = 2
    match.away_goals = 1
    match.datetime = datetime(
        2024,
        8,
        10,
    )
    match.season = 2024
    match.comp_season_id = comp_season.id
    match.comp_name = comp_season.name
    match.country = comp_season.country
    match.home_lineup = [home_player]
    match.away_lineup = [away_player]

    comp_season.matches = [match]

    return FSDataBundle(
        comp_seasons={comp_season.id: (comp_season)},
        teams={
            home_team.id: home_team,
            away_team.id: away_team,
        },
        players={
            home_player.id: home_player,
            away_player.id: away_player,
        },
        matches=[match],
    )


def test_domain_module_is_offline_and_global_free() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "validation" / "domain.py"
    source = source_path.read_text(encoding="utf-8")

    assert "requests" not in source
    assert "fs_globals" not in source
    assert "Global" not in source
    assert "fs_settings" not in source


def test_valid_bundle_passes() -> None:
    bundle = make_valid_bundle()

    report = validate_bundle_domain(bundle)

    assert report.ok
    assert report.critical_issue_count == 0
    assert report.metrics["competition_seasons"] == 1
    assert report.metrics["teams"] == 2
    assert report.metrics["players"] == 2
    assert report.metrics["matches"] == 1
    assert report.metrics["unique_match_ids"] == 1


def test_index_key_mismatches_are_detected() -> None:
    bundle = make_valid_bundle()
    team = bundle.teams.pop(1)
    bundle.teams[999] = team

    report = validate_bundle_domain(bundle)

    assert report.count_for("team_key_id_mismatch") == 1
    assert not report.ok


def test_duplicate_match_ids_are_detected() -> None:
    bundle = make_valid_bundle()
    duplicate = FSMatch(bundle.matches[0].id)
    bundle.matches.append(duplicate)

    report = validate_bundle_domain(bundle)

    assert report.count_for("duplicate_match_id") == 1
    assert not report.ok


def test_invalid_match_core_fields_are_detected() -> None:
    bundle = make_valid_bundle()
    match = bundle.matches[0]

    match.home_team = None
    match.home_goals = -1
    match.datetime = None
    match.season = None
    match.comp_season_id = 999

    report = validate_bundle_domain(bundle)

    assert report.count_for("missing_home_team") == 1
    assert report.count_for("invalid_home_goals") == 1
    assert report.count_for("invalid_match_datetime") == 1
    assert report.count_for("invalid_match_season") == 1
    assert report.count_for("unknown_comp_season_id") == 1


def test_comp_season_membership_is_required() -> None:
    bundle = make_valid_bundle()
    comp_season = next(iter(bundle.comp_seasons.values()))
    comp_season.matches = []

    report = validate_bundle_domain(bundle)

    assert report.count_for("match_missing_from_comp_season") == 1


def test_comp_season_metadata_must_match() -> None:
    bundle = make_valid_bundle()
    match = bundle.matches[0]

    match.season = 2023
    match.comp_name = "Wrong League"
    match.country = "Wrong Country"

    report = validate_bundle_domain(bundle)

    assert report.count_for(("match_comp_season_" "year_mismatch")) == 1
    assert report.count_for("match_comp_name_mismatch") == 1
    assert report.count_for("match_country_mismatch") == 1


def test_noncanonical_player_references_are_detected() -> None:
    bundle = make_valid_bundle()

    canonical = bundle.players[101]
    clone = make_player(
        canonical.id,
        "Clone",
    )

    home_team = bundle.teams[1]
    home_team.comp_seasons[10] = [clone]
    bundle.matches[0].home_lineup = [clone]

    report = validate_bundle_domain(bundle)

    assert report.count_for("noncanonical_roster_player") == 1
    assert report.count_for(("noncanonical_" "home_lineup_player")) == 1


def test_report_is_deterministic_and_caps_examples() -> None:
    bundle = make_valid_bundle()

    for match_id in (
        2000,
        2001,
        2002,
    ):
        match = FSMatch(match_id)
        match.home_goals = 0
        match.away_goals = 0
        match.datetime = datetime(
            2024,
            1,
            1,
        )
        match.season = 2024
        match.comp_season_id = 10
        match.comp_name = "Test League"
        match.country = "England"
        bundle.matches.append(match)

    first = validate_bundle_domain(
        bundle,
        max_examples_per_finding=2,
    )
    second = validate_bundle_domain(
        bundle,
        max_examples_per_finding=2,
    )

    assert first.to_dict() == second.to_dict()

    finding = first.findings["missing_home_team"]

    assert finding.count == 3
    assert len(finding.examples) == 2


def test_raw_missing_team_is_reported_as_warning() -> None:
    bundle = make_valid_bundle()
    bundle.matches[0].home_team = None

    report = validate_bundle_domain(bundle)

    assert report.count_for("missing_home_team") == 1
    assert report.warning_count == 1
    assert report.critical_issue_count == 0
    assert report.ok


def test_detached_comp_season_match_is_warning() -> None:
    bundle = make_valid_bundle()

    comp_season = bundle.comp_seasons[10]

    detached = FSMatch(2000)
    detached.comp_season_id = 10

    comp_season.matches.append(detached)

    report = validate_bundle_domain(bundle)

    assert report.count_for(("detached_comp_season_" "match_reference")) == 1
    assert report.warning_count == 1
    assert report.critical_issue_count == 0
    assert report.ok


def test_unknown_roster_player_is_warning() -> None:
    bundle = make_valid_bundle()

    unknown_player = make_player(
        999,
        "Roster Only Player",
    )

    bundle.teams[1].comp_seasons[10].append(unknown_player)

    report = validate_bundle_domain(bundle)

    assert report.count_for("unknown_roster_player") == 1
    assert report.warning_count == 1
    assert report.critical_issue_count == 0
    assert report.ok


def test_duplicate_lineup_player_is_warning() -> None:
    bundle = make_valid_bundle()

    player = bundle.players[101]
    bundle.matches[0].home_lineup = [
        player,
        player,
    ]

    report = validate_bundle_domain(bundle)

    assert report.count_for(("duplicate_home_" "lineup_player")) == 1
    assert report.warning_count == 1
    assert report.critical_issue_count == 0
    assert report.ok
