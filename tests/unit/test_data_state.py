import ast
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from football_outcomes.data.fs_models import (
    FSDataBundle,
)
from football_outcomes.data.state import (
    FSDataState,
    apply_bundle_to_global,
    apply_state_to_global,
    bundle_from_global,
    bundle_from_state,
    state_from_bundle,
    state_from_global,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_state() -> FSDataState:
    return FSDataState(
        comp_seasons={1: object()},
        teams={2: object()},
        players={3: object()},
        matches=[object()],
        leagues_list=[{"id": 4}],
        sf_avg_team_strength={(2024, 2, "GK"): [70.0]},
        sofifa_snapshots=[
            (
                date(2024, 1, 1),
                {5: {"name": "Player"}},
            )
        ],
        sofifa_player_occurrences={
            5: [
                (
                    0,
                    date(2024, 1, 1),
                )
            ]
        },
        sofifa_players_by_dob={
            date(2000, 1, 1): [
                (
                    5,
                    "Player",
                    "Full Player",
                )
            ]
        },
        fs_to_sofifa_cache={
            3: (
                5,
                0.9,
                0.8,
                True,
                "matched",
            )
        },
        sofifa_team_meta={6: {"name": "Team"}},
        sofifa_players_by_team={
            6: [
                (
                    5,
                    "Player",
                    "Full Player",
                    date(2000, 1, 1),
                )
            ]
        },
        sofifa_teams_by_league={
            ("country", "league"): [
                (
                    6,
                    "Team",
                )
            ]
        },
        fs_team_to_sofifa_team={2: 6},
    )


def test_state_module_has_no_settings_or_network_dependency() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "state.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ImportFrom,
        )
        and node.module is not None
    }

    plain_imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}

    assert "football_outcomes.config." "fs_settings" not in imported_modules
    assert "requests" not in plain_imports


def test_bundle_to_state_preserves_serialized_collections() -> None:
    state = build_state()

    bundle = bundle_from_state(
        state,
        meta={"source": "test"},
    )
    restored = state_from_bundle(bundle)

    assert restored.comp_seasons is state.comp_seasons
    assert restored.teams is state.teams
    assert restored.players is state.players
    assert restored.matches is state.matches
    assert restored.leagues_list is state.leagues_list
    assert restored.sofifa_snapshots is state.sofifa_snapshots
    assert restored.fs_to_sofifa_cache is state.fs_to_sofifa_cache

    assert restored.sf_avg_team_strength == {}
    assert bundle.meta == {"source": "test"}


def test_state_can_be_applied_without_copying() -> None:
    state = build_state()
    target = SimpleNamespace()

    result = apply_state_to_global(
        state,
        target,
    )

    assert result is target
    assert target.all_comp_seasons is state.comp_seasons
    assert target.all_teams is state.teams
    assert target.all_players is state.players
    assert target.all_matches is state.matches
    assert target.sf_avg_team_strength is state.sf_avg_team_strength
    assert target.fs_team_to_sofifa_team is state.fs_team_to_sofifa_team


def test_global_adapters_preserve_all_references() -> None:
    state = build_state()
    target = SimpleNamespace()

    apply_state_to_global(
        state,
        target,
    )

    restored_state = state_from_global(target)
    bundle = bundle_from_global(target)

    assert restored_state.comp_seasons is state.comp_seasons
    assert restored_state.sf_avg_team_strength is state.sf_avg_team_strength

    assert bundle.comp_seasons is state.comp_seasons
    assert bundle.teams is state.teams
    assert bundle.players is state.players
    assert bundle.matches is state.matches
    assert bundle.sofifa_team_meta is state.sofifa_team_meta


def test_main_pipeline_uses_bundle_adapter() -> None:
    source_path = PROJECT_ROOT / "scripts" / "main_footystats.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(
            node.func,
            ast.Name,
        )
    }

    assert "bundle_from_global" in called_names
    assert "FSDataBundle" not in called_names


def test_empty_bundle_optional_fields_are_normalized() -> None:
    bundle = FSDataBundle(
        sofifa_snapshots=[],
        sofifa_player_occurrences={},
        sofifa_players_by_dob={},
        fs_to_sofifa_cache={},
        sofifa_team_meta={},
        sofifa_players_by_team={},
        sofifa_teams_by_league={},
        fs_team_to_sofifa_team={},
    )

    state = state_from_bundle(bundle)

    assert state.sofifa_snapshots == []
    assert state.sofifa_player_occurrences == {}
    assert state.fs_to_sofifa_cache == {}


def test_bundle_application_preserves_average_strength() -> None:
    state = build_state()
    bundle = bundle_from_state(state)

    preserved_average_strength = {
        (
            2023,
            9,
            "GK",
        ): [68.0, 69.0]
    }

    target = SimpleNamespace(sf_avg_team_strength=(preserved_average_strength))

    result = apply_bundle_to_global(
        bundle,
        target,
    )

    assert result is target

    assert target.all_comp_seasons is state.comp_seasons
    assert target.all_teams is state.teams
    assert target.all_players is state.players
    assert target.all_matches is state.matches
    assert target.leagues_list is state.leagues_list

    assert target.sf_avg_team_strength is preserved_average_strength

    assert target.sofifa_snapshots is state.sofifa_snapshots
    assert target.fs_to_sofifa_cache is state.fs_to_sofifa_cache
