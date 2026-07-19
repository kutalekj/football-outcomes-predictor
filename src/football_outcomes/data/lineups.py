from __future__ import annotations

from typing import TYPE_CHECKING

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import (
    FSPlayer,
)
from football_outcomes.utils.fs_common import (
    normalize_fs_player_position,
)

if TYPE_CHECKING:
    from football_outcomes.data.fs_models import (
        FSMatch,
    )

FS_POSITION_ORDER = {
    "Goalkeeper": 0,
    "Defender": 1,
    "Midfielder": 2,
    "Forward": 3,
}


def position_rank(
    player: FSPlayer,
) -> int:
    """Return the stable coarse-position rank."""

    position = normalize_fs_player_position(
        player.position or "",
        player.known_as,
    )

    return FS_POSITION_ORDER.get(
        position,
        99,
    )


def select_and_sort_lineup(
    match: FSMatch,
    team_id: int,
) -> tuple[list[FSPlayer], str]:
    """
    Select, order and pad one side's lineup.

    The result is aligned with the structured
    team-strength and player-position inputs.
    """

    if match.home_team is not None and match.home_team.id == team_id:
        lineup = getattr(
            match,
            "home_lineup",
            None,
        )
        side = "home"

    elif match.away_team is not None and match.away_team.id == team_id:
        lineup = getattr(
            match,
            "away_lineup",
            None,
        )
        side = "away"

    else:
        raise ValueError(f"Team {team_id} not in " f"match {match.id}")

    if lineup is None:
        lineup = []

    elif not isinstance(
        lineup,
        list,
    ):
        raise TypeError("Lineup must be list[FSPlayer], " f"got {type(lineup)}")

    maximum_players = sett.TEAM_STRENGTH_NUM_PLAYERS

    if len(lineup) > maximum_players:
        raise ValueError("Lineup has >" f"{maximum_players} players: " f"{len(lineup)}")

    lineup_sorted = sorted(
        lineup,
        key=position_rank,
    )

    has_goalkeeper = any(
        normalize_fs_player_position(
            player.position or "",
            player.known_as,
        )
        == "Goalkeeper"
        for player in lineup_sorted
    )

    if not has_goalkeeper:
        missing_goalkeeper = FSPlayer(
            -1,
            "MISSING_GK",
            "",
            "",
            "",
            "MISSING_GK",
        )
        missing_goalkeeper.position = "Goalkeeper"
        lineup_sorted.insert(
            0,
            missing_goalkeeper,
        )

    while len(lineup_sorted) < maximum_players:
        missing_player = FSPlayer(
            -1,
            "MISSING",
            "",
            "",
            "",
            "MISSING",
        )
        missing_player.position = "Unknown"
        lineup_sorted.append(missing_player)

    return (
        lineup_sorted[:maximum_players],
        side,
    )


def calculate_team_position_indices(
    match: FSMatch,
    team_id: int,
) -> list[int]:
    """
    Build the categorical position vector aligned
    with the ordered team-strength rows.
    """

    lineup_sorted, _ = select_and_sort_lineup(
        match,
        team_id,
    )

    position_indices = []

    for player in lineup_sorted:
        position = normalize_fs_player_position(
            getattr(
                player,
                "position",
                "",
            )
            or "",
            player.known_as,
        )

        position_indices.append(int(sett.FS_PLAYER_POSITION_TO_IDX[position]))

    return position_indices
