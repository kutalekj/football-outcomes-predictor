import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

from football_outcomes.data import (
    fs_retrieve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeClient:
    def __init__(
        self,
        league_rows,
    ):
        self.league_rows = league_rows
        self.calls = []

    def get_data(
        self,
        endpoint,
        params=None,
    ):
        self.calls.append(
            (
                endpoint,
                params,
            )
        )

        return self.league_rows


class FailClient:
    def get_data(
        self,
        endpoint,
        params=None,
    ):
        raise AssertionError("Client must not be called.")


def test_main_pipeline_uses_state_restoration() -> None:
    source_path = PROJECT_ROOT / "scripts" / "main_footystats.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.ImportFrom,
        )
        for alias in node.names
    }

    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Call,
        )
        and isinstance(
            node.func,
            ast.Name,
        )
    }

    assert "apply_bundle_to_global" in imported_names
    assert "fill_globals_with_cache" not in imported_names

    assert "apply_bundle_to_global" in called_names
    assert "fill_globals_with_cache" not in called_names


def test_legacy_fill_delegates_without_network(
    monkeypatch,
) -> None:
    cache = object()
    target = SimpleNamespace(leagues_list=["cached"])
    applied = []

    def fake_apply_bundle(
        supplied_cache,
    ):
        applied.append(supplied_cache)
        return target

    monkeypatch.setattr(
        fs_retrieve,
        "apply_bundle_to_global",
        fake_apply_bundle,
    )

    result = fs_retrieve.fill_globals_with_cache(
        cache,
        update_leagues_list=False,
        client=FailClient(),
    )

    assert result is None
    assert applied == [cache]
    assert target.leagues_list == ["cached"]


def test_legacy_fill_can_refresh_league_list(
    monkeypatch,
) -> None:
    target = SimpleNamespace(leagues_list=["cached"])

    monkeypatch.setattr(
        fs_retrieve,
        "apply_bundle_to_global",
        lambda cache: target,
    )

    client = FakeClient(
        [
            {
                "id": 123,
                "name": "League",
            }
        ]
    )

    result = fs_retrieve.fill_globals_with_cache(
        object(),
        update_leagues_list=True,
        client=client,
    )

    assert result is None

    assert client.calls == [
        (
            "league-list",
            None,
        )
    ]

    assert target.leagues_list == [
        {
            "id": 123,
            "name": "League",
        }
    ]


def test_legacy_fill_no_longer_maps_snapshot_fields() -> None:
    source = inspect.getsource(fs_retrieve.fill_globals_with_cache)
    tree = ast.parse(source)

    assigned_attributes = {
        target.attr
        for node in ast.walk(tree)
        if isinstance(
            node,
            ast.Assign,
        )
        for target in node.targets
        if isinstance(
            target,
            ast.Attribute,
        )
    }

    serialized_attributes = {
        "all_comp_seasons",
        "all_teams",
        "all_players",
        "all_matches",
        "sofifa_snapshots",
        "sofifa_player_occurrences",
        "sofifa_players_by_dob",
        "fs_to_sofifa_cache",
        "sofifa_teams_by_league",
        "sofifa_team_meta",
        "sofifa_players_by_team",
        "fs_team_to_sofifa_team",
    }

    assert assigned_attributes.isdisjoint(serialized_attributes)
