from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from football_outcomes.config.fs_globals import (
    Global,
)
from football_outcomes.data.fs_models import (
    FSCompSeason,
    FSDataBundle,
    FSMatch,
    FSPlayer,
    FSTeam,
)

SofifaSnapshot = tuple[
    date,
    dict[int, dict[str, Any]],
]

PlayerOccurrence = tuple[int, date]

PlayerMatchCacheValue = tuple[
    int | None,
    float,
    float,
    bool,
    str,
]


@dataclass
class FSDataState:
    """Explicit in-memory state used by the data pipeline."""

    comp_seasons: dict[
        int,
        FSCompSeason,
    ] = field(default_factory=dict)

    teams: dict[
        int,
        FSTeam,
    ] = field(default_factory=dict)

    players: dict[
        int,
        FSPlayer,
    ] = field(default_factory=dict)

    matches: list[FSMatch] = field(default_factory=list)

    leagues_list: Any = None

    sf_avg_team_strength: dict[
        tuple[int, int, str],
        list[float],
    ] = field(default_factory=dict)

    sofifa_snapshots: list[SofifaSnapshot] = field(default_factory=list)

    sofifa_player_occurrences: dict[
        int,
        list[PlayerOccurrence],
    ] = field(default_factory=dict)

    sofifa_players_by_dob: dict[
        date,
        list[tuple[int, str, str]],
    ] = field(default_factory=dict)

    fs_to_sofifa_cache: dict[
        int,
        PlayerMatchCacheValue,
    ] = field(default_factory=dict)

    sofifa_team_meta: dict[
        int,
        dict[str, Any],
    ] = field(default_factory=dict)

    sofifa_players_by_team: dict[
        int,
        list[
            tuple[
                int,
                str,
                str,
                date | None,
            ]
        ],
    ] = field(default_factory=dict)

    sofifa_teams_by_league: Any = field(default_factory=dict)

    fs_team_to_sofifa_team: dict[
        int,
        int,
    ] = field(default_factory=dict)


def state_from_bundle(
    bundle: FSDataBundle,
) -> FSDataState:
    """
    Adapt serialized snapshot data to explicit
    in-memory state.

    Average team-strength data is not stored in
    FSDataBundle and therefore starts empty.
    """

    return FSDataState(
        comp_seasons=bundle.comp_seasons,
        teams=bundle.teams,
        players=bundle.players,
        matches=bundle.matches,
        leagues_list=bundle.leagues_list,
        sofifa_snapshots=(
            getattr(
                bundle,
                "sofifa_snapshots",
                None,
            )
            or []
        ),
        sofifa_player_occurrences=(
            getattr(
                bundle,
                "sofifa_player_occurrences",
                None,
            )
            or {}
        ),
        sofifa_players_by_dob=(
            getattr(
                bundle,
                "sofifa_players_by_dob",
                None,
            )
            or {}
        ),
        fs_to_sofifa_cache=(
            getattr(
                bundle,
                "fs_to_sofifa_cache",
                None,
            )
            or {}
        ),
        sofifa_team_meta=(
            getattr(
                bundle,
                "sofifa_team_meta",
                None,
            )
            or {}
        ),
        sofifa_players_by_team=(
            getattr(
                bundle,
                "sofifa_players_by_team",
                None,
            )
            or {}
        ),
        sofifa_teams_by_league=(
            getattr(
                bundle,
                "sofifa_teams_by_league",
                None,
            )
            or {}
        ),
        fs_team_to_sofifa_team=(
            getattr(
                bundle,
                "fs_team_to_sofifa_team",
                None,
            )
            or {}
        ),
    )


def state_from_global(
    global_instance: Any | None = None,
) -> FSDataState:
    """Read the current legacy singleton state."""

    if global_instance is None:
        global_instance = Global.get_instance()

    return FSDataState(
        comp_seasons=(global_instance.all_comp_seasons),
        teams=global_instance.all_teams,
        players=global_instance.all_players,
        matches=global_instance.all_matches,
        leagues_list=(global_instance.leagues_list),
        sf_avg_team_strength=(global_instance.sf_avg_team_strength),
        sofifa_snapshots=(global_instance.sofifa_snapshots),
        sofifa_player_occurrences=(global_instance.sofifa_player_occurrences),
        sofifa_players_by_dob=(global_instance.sofifa_players_by_dob),
        fs_to_sofifa_cache=(global_instance.fs_to_sofifa_cache),
        sofifa_team_meta=(global_instance.sofifa_team_meta),
        sofifa_players_by_team=(global_instance.sofifa_players_by_team),
        sofifa_teams_by_league=(global_instance.sofifa_teams_by_league),
        fs_team_to_sofifa_team=(global_instance.fs_team_to_sofifa_team),
    )


def apply_state_to_global(
    state: FSDataState,
    global_instance: Any | None = None,
) -> Any:
    """
    Replace the legacy singleton collections with
    the supplied explicit state.

    Container identity is preserved.
    """

    if global_instance is None:
        global_instance = Global.get_instance()

    global_instance.all_comp_seasons = state.comp_seasons
    global_instance.all_teams = state.teams
    global_instance.all_players = state.players
    global_instance.all_matches = state.matches
    global_instance.leagues_list = state.leagues_list
    global_instance.sf_avg_team_strength = state.sf_avg_team_strength
    global_instance.sofifa_snapshots = state.sofifa_snapshots
    global_instance.sofifa_player_occurrences = state.sofifa_player_occurrences
    global_instance.sofifa_players_by_dob = state.sofifa_players_by_dob
    global_instance.fs_to_sofifa_cache = state.fs_to_sofifa_cache
    global_instance.sofifa_team_meta = state.sofifa_team_meta
    global_instance.sofifa_players_by_team = state.sofifa_players_by_team
    global_instance.sofifa_teams_by_league = state.sofifa_teams_by_league
    global_instance.fs_team_to_sofifa_team = state.fs_team_to_sofifa_team

    return global_instance


def apply_bundle_to_global(
    bundle: FSDataBundle,
    global_instance: Any | None = None,
) -> Any:
    """
    Restore serialized bundle data into legacy
    process-wide state.

    Average team-strength data is preserved because
    it is loaded independently and is not serialized
    in FSDataBundle.
    """

    if global_instance is None:
        global_instance = Global.get_instance()

    average_team_strength = getattr(
        global_instance,
        "sf_avg_team_strength",
        {},
    )

    state = state_from_bundle(bundle)
    state.sf_avg_team_strength = average_team_strength

    apply_state_to_global(
        state,
        global_instance,
    )

    print(f"{len(state.comp_seasons)} " "comp seasons loaded from snapshot.")
    print(f"{len(state.teams)} " "teams loaded from snapshot.")
    print(f"{len(state.players)} " "players loaded from snapshot.")
    print(f"{len(state.matches)} " "matches loaded from snapshot.")
    print(f"{len(state.sofifa_snapshots)} " "sofifa snapshots loaded from snapshot.")
    print(f"{len(state.fs_to_sofifa_cache)} " "fs->sofifa cached matches " "loaded from snapshot.")

    return global_instance


def bundle_from_state(
    state: FSDataState,
    *,
    meta: dict[str, Any] | None = None,
) -> FSDataBundle:
    """Create a serializable bundle from explicit state."""

    return FSDataBundle(
        comp_seasons=state.comp_seasons,
        teams=state.teams,
        players=state.players,
        matches=state.matches,
        leagues_list=state.leagues_list,
        meta={} if meta is None else meta,
        sofifa_snapshots=(state.sofifa_snapshots),
        sofifa_player_occurrences=(state.sofifa_player_occurrences),
        sofifa_players_by_dob=(state.sofifa_players_by_dob),
        fs_to_sofifa_cache=(state.fs_to_sofifa_cache),
        sofifa_team_meta=(state.sofifa_team_meta),
        sofifa_players_by_team=(state.sofifa_players_by_team),
        sofifa_teams_by_league=(state.sofifa_teams_by_league),
        fs_team_to_sofifa_team=(state.fs_team_to_sofifa_team),
    )


def bundle_from_global(
    global_instance: Any | None = None,
    *,
    meta: dict[str, Any] | None = None,
) -> FSDataBundle:
    """Create a serializable bundle from legacy state."""

    return bundle_from_state(
        state_from_global(global_instance),
        meta=meta,
    )
