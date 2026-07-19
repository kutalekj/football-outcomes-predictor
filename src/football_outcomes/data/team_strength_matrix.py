from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import (
    FSPlayer,
)
from football_outcomes.data.lineups import (
    position_rank,
)

if TYPE_CHECKING:
    from football_outcomes.data.fs_models import (
        FSMatch,
    )


class PlayerMatchResult(Protocol):
    sofifa_id: int | None
    score_best: float
    score_second: float
    used_dob_gate: bool
    reason: str
    sofifa_best_name: str | None


class PlayerMatcher(Protocol):
    def __call__(
        self,
        player: FSPlayer,
        *,
        fs_team_id: int | None = None,
    ) -> PlayerMatchResult: ...


SkillLoader = Callable[
    [int, "datetime"],
    tuple[list[float], int, int],
]

DebugLogger = Callable[[str], None]
PlayerDisplayName = Callable[[FSPlayer], str]


def goalkeeper_role_score(
    skills: list[float],
) -> float:
    """
    Estimate whether one skill row resembles a
    goalkeeper rather than an outfield player.
    """

    if not skills or len(skills) < sett.GK_SKILL_END_INDEX:
        return 0.0

    goalkeeper_skills = skills[sett.GK_SKILL_START_INDEX : sett.GK_SKILL_END_INDEX]
    outfield_skills = skills[: sett.GK_SKILL_START_INDEX]

    goalkeeper_values = [value for value in goalkeeper_skills if value != -1.0]
    outfield_values = [value for value in outfield_skills if value != -1.0]

    if not goalkeeper_values or not outfield_values:
        return 0.0

    return (sum(goalkeeper_values) / len(goalkeeper_values)) - (sum(outfield_values) / len(outfield_values))


def ensure_one_goalkeeper_row(
    rows: list[
        tuple[
            FSPlayer,
            list[float],
        ]
    ],
) -> list[
    tuple[
        FSPlayer,
        list[float],
    ]
]:
    """
    Preserve the legacy goalkeeper-row policy.

    The selected goalkeeper is moved to the first
    row. When no suitable row exists, a missing
    goalkeeper row is inserted.
    """

    if not getattr(
        sett,
        "FORCE_EXACTLY_ONE_GK_ROW",
        True,
    ):
        return rows

    goalkeeper_rows = [
        (
            index,
            player,
            skills,
        )
        for (
            index,
            (
                player,
                skills,
            ),
        ) in enumerate(rows)
        if player.position == "Goalkeeper"
    ]

    if goalkeeper_rows:
        best_index, _, _ = goalkeeper_rows[0]
        chosen = rows.pop(best_index)
        rows.insert(0, chosen)
        return rows

    scored_rows = [
        (
            index,
            goalkeeper_role_score(skills),
        )
        for (
            index,
            (
                _,
                skills,
            ),
        ) in enumerate(rows)
    ]
    scored_rows.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    minimum_score = getattr(
        sett,
        "GK_ROLE_SCORE_MIN_GAP",
        0.5,
    )

    if scored_rows and scored_rows[0][1] >= minimum_score:
        best_index = scored_rows[0][0]
        chosen = rows.pop(best_index)
        rows.insert(0, chosen)
        return rows

    missing_goalkeeper = FSPlayer(
        -1,
        "MISSING_GK",
        "",
        "",
        "",
        "MISSING_GK",
    )
    missing_goalkeeper.position = "Goalkeeper"

    rows.insert(
        0,
        (
            missing_goalkeeper,
            [-1.0] * len(sett.PLAYER_SKILLS),
        ),
    )

    return rows


def build_team_strength_matrix(
    match: FSMatch,
    team_id: int,
    *,
    match_player: PlayerMatcher,
    merge_skills: SkillLoader,
    debug_log: DebugLogger,
    player_display_name: PlayerDisplayName,
) -> list[list[float]]:
    """
    Assemble one fixed-size team-strength matrix.

    Matching and temporal skill retrieval are
    supplied as dependencies so matrix construction
    remains separate from fuzzy matching.
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

    debug_enabled = getattr(
        sett,
        "DEBUG_TEAM_STRENGTH",
        False,
    )

    if lineup is None:
        if debug_enabled:
            debug_log("[team_strength] lineup=None " f"for team_id={team_id} " f"match={match.id} ({side})")

        lineup = []

    elif isinstance(lineup, list) and len(lineup) == 0:
        if debug_enabled:
            debug_log("[team_strength] lineup=[] " f"for team_id={team_id} " f"match={match.id} ({side})")

    if not isinstance(
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

    rows: list[
        tuple[
            FSPlayer,
            list[float],
        ]
    ] = []

    for player in lineup_sorted:
        match_result = match_player(
            player,
            fs_team_id=team_id,
        )

        if match_result.sofifa_id is None:
            skills = [-1.0] * len(sett.PLAYER_SKILLS)

            if debug_enabled:
                debug_log(
                    "[team_strength] UNMATCHED "
                    "fs='"
                    f"{player_display_name(player)}"
                    "' "
                    "dob="
                    f"{getattr(player, 'birthday', None)} "
                    "sf_name="
                    f"{match_result.sofifa_best_name} "
                    "score="
                    f"{match_result.score_best:.1f} "
                    "reason="
                    f"{match_result.reason}"
                )

        else:
            (
                skills,
                snapshots_used,
                delta_days,
            ) = merge_skills(
                match_result.sofifa_id,
                match.datetime,
            )

            if debug_enabled:
                missing_cells = sum(1 for value in skills if value == -1.0)

                debug_log(
                    "[team_strength] MATCH "
                    "fs='"
                    f"{player_display_name(player)}"
                    "' -> sf_id="
                    f"{match_result.sofifa_id} "
                    "score="
                    f"{match_result.score_best:.1f} "
                    "(2nd="
                    f"{match_result.score_second:.1f}) "
                    "(sf_name="
                    f"{match_result.sofifa_best_name}) "
                    "league="
                    f"{match.comp_name.replace(' ', '_')} "
                    "match_dt="
                    f"{match.datetime.isoformat()} "
                    "missing="
                    f"{missing_cells}/"
                    f"{len(skills)} "
                    "snapshots_used="
                    f"{snapshots_used} "
                    "delta_days="
                    f"{delta_days} "
                    "reason="
                    f"{match_result.reason}"
                )

        rows.append(
            (
                player,
                skills,
            )
        )

    rows = ensure_one_goalkeeper_row(rows)

    while len(rows) < maximum_players:
        missing_player = FSPlayer(
            -1,
            "MISSING",
            "",
            "",
            "",
            "MISSING",
        )
        missing_player.position = "Unknown"

        rows.append(
            (
                missing_player,
                [-1.0] * len(sett.PLAYER_SKILLS),
            )
        )

    rows = rows[:maximum_players]

    return [skills for _, skills in rows]


def calculate_team_strength(
    curr_match: FSMatch,
    team_id: int,
) -> list[list[float]]:
    """
    Compatibility entry point using the current
    matching and skill-retrieval services.

    The local import is temporary until fuzzy
    matching is extracted in the next increment.
    """

    from football_outcomes.utils import fs_player_skill_utils as legacy

    return build_team_strength_matrix(
        match=curr_match,
        team_id=team_id,
        match_player=(legacy._match_fs_to_sofifa),
        merge_skills=(legacy._merge_skills_from_snapshots),
        debug_log=legacy._dbg,
        player_display_name=(legacy._player_display_name),
    )
