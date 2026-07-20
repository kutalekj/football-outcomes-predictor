from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import (
    Global,
)

Occurrence = tuple[int, date]

SnapshotPlayers = Mapping[
    int,
    Mapping[str, Any],
]

SofifaSnapshot = tuple[
    date,
    SnapshotPlayers,
]


def ordered_snapshot_candidates(
    occurrences: Sequence[Occurrence],
    match_date: date,
    *,
    max_days: int | None = None,
    max_snapshots: int | None = None,
) -> list[Occurrence]:
    """
    Order eligible snapshots past-first.

    Past snapshots are ordered from closest to
    furthest. Future snapshots follow, also from
    closest to furthest.
    """

    resolved_max_days = int(sett.SF_MAX_TIMEDELTA_DAYS if max_days is None else max_days)
    resolved_max_snapshots = int(sett.SF_MAX_SNAPSHOTS_TO_SCAN if max_snapshots is None else max_snapshots)

    past = []
    future = []

    for snapshot_index, snapshot_date in occurrences:
        delta_days = (match_date - snapshot_date).days

        if abs(delta_days) > resolved_max_days:
            continue

        candidate = (
            abs(delta_days),
            snapshot_index,
            snapshot_date,
        )

        if delta_days >= 0:
            past.append(candidate)
        else:
            future.append(candidate)

    past.sort(key=lambda candidate: (candidate[0]))
    future.sort(key=lambda candidate: (candidate[0]))

    ordered = [
        (
            snapshot_index,
            snapshot_date,
        )
        for (
            _,
            snapshot_index,
            snapshot_date,
        ) in past
    ]

    ordered.extend(
        (
            snapshot_index,
            snapshot_date,
        )
        for (
            _,
            snapshot_index,
            snapshot_date,
        ) in future
    )

    return ordered[:resolved_max_snapshots]


def merge_skills_from_snapshot_data(
    sofifa_id: int,
    match_datetime: datetime,
    *,
    snapshots: Sequence[SofifaSnapshot],
    player_occurrences: Mapping[
        int,
        Sequence[Occurrence],
    ],
    skill_count: int,
    max_days: int,
    max_snapshots: int,
) -> tuple[
    list[float],
    int,
    int,
]:
    """
    Merge one player's skills from temporal snapshots.

    Returns the merged vector, number of contributing
    snapshots, and signed distance from the match to
    the first contributing snapshot.
    """

    match_date = match_datetime.date()
    occurrences = player_occurrences.get(
        sofifa_id,
        [],
    )

    if not occurrences:
        return (
            [-1.0] * skill_count,
            0,
            0,
        )

    candidates = ordered_snapshot_candidates(
        occurrences,
        match_date,
        max_days=max_days,
        max_snapshots=(max_snapshots),
    )

    if not candidates:
        return (
            [-1.0] * skill_count,
            0,
            0,
        )

    merged = [-1.0] * skill_count

    snapshots_used = 0
    closest_delta_days = None

    for (
        snapshot_index,
        snapshot_date,
    ) in candidates:
        snapshot_players = snapshots[snapshot_index][1]
        record = snapshot_players.get(sofifa_id)

        if record is None:
            continue

        skills = record.get("skills")

        if not skills or len(skills) != skill_count:
            continue

        contributed = False

        for index, value in enumerate(skills):
            if merged[index] == -1.0 and value is not None:
                merged[index] = float(value)
                contributed = True

        if contributed:
            snapshots_used += 1

            if closest_delta_days is None:
                closest_delta_days = (match_date - snapshot_date).days

        if -1.0 not in merged:
            break

    if closest_delta_days is None:
        closest_delta_days = 0

    return (
        merged,
        snapshots_used,
        closest_delta_days,
    )


def merge_skills_from_snapshots(
    sofifa_id: int,
    match_datetime: datetime,
) -> tuple[
    list[float],
    int,
    int,
]:
    """
    Compatibility entry point using legacy state
    and configuration.
    """

    global_instance = Global.get_instance()

    return merge_skills_from_snapshot_data(
        sofifa_id=sofifa_id,
        match_datetime=(match_datetime),
        snapshots=(global_instance.sofifa_snapshots),
        player_occurrences=(global_instance.sofifa_player_occurrences),
        skill_count=len(sett.PLAYER_SKILLS),
        max_days=int(sett.SF_MAX_TIMEDELTA_DAYS),
        max_snapshots=int(sett.SF_MAX_SNAPSHOTS_TO_SCAN),
    )
