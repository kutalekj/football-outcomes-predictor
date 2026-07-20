from __future__ import annotations

import math
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import (
    date,
    datetime,
)
from enum import IntEnum
from numbers import Real
from typing import Any

MISSING_SKILL_VALUE = -1.0


class SkillProvenance(IntEnum):
    UNRESOLVED = 0
    NEAREST_PAST_SOFIFA = 1
    OLDER_PAST_SOFIFA = 2
    COMPETITION_POSITION_MEDIAN = 3
    POSITION_MEDIAN = 4
    GLOBAL_SKILL_MEDIAN = 5
    NEUTRAL_FALLBACK = 6


@dataclass(frozen=True)
class PastSnapshotCandidate:
    snapshot_index: int
    snapshot_date: date
    age_days: int


@dataclass(frozen=True)
class TemporalSkillResult:
    skills: tuple[float, ...]
    provenance: tuple[
        SkillProvenance,
        ...,
    ]
    source_dates: tuple[
        date | None,
        ...,
    ]
    snapshots_used: int
    nearest_snapshot_delta_days: int | None

    @property
    def observed_count(self) -> int:
        return sum(
            provenance
            in (
                SkillProvenance.NEAREST_PAST_SOFIFA,
                SkillProvenance.OLDER_PAST_SOFIFA,
            )
            for provenance in self.provenance
        )

    @property
    def nearest_past_count(self) -> int:
        return self.provenance.count(SkillProvenance.NEAREST_PAST_SOFIFA)

    @property
    def older_past_count(self) -> int:
        return self.provenance.count(SkillProvenance.OLDER_PAST_SOFIFA)

    @property
    def unresolved_count(self) -> int:
        return self.provenance.count(SkillProvenance.UNRESOLVED)


SnapshotPlayers = Mapping[
    int,
    Mapping[str, Any],
]
Snapshot = tuple[
    date,
    SnapshotPlayers,
]
Occurrence = tuple[int, date]


def _as_date(
    value: date | datetime,
) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    raise TypeError("Expected date or datetime, " f"found {type(value)!r}.")


def _validate_limits(
    *,
    max_age_days: int,
    max_snapshots: int,
) -> None:
    if type(max_age_days) is not int or max_age_days < 0:
        raise ValueError("max_age_days must be a " "non-negative integer.")

    if type(max_snapshots) is not int or max_snapshots <= 0:
        raise ValueError("max_snapshots must be a " "positive integer.")


def ordered_past_snapshot_candidates(
    occurrences: Sequence[Occurrence],
    snapshots: Sequence[Snapshot],
    match_datetime: date | datetime,
    *,
    max_age_days: int,
    max_snapshots: int,
) -> tuple[
    PastSnapshotCandidate,
    ...,
]:
    _validate_limits(
        max_age_days=max_age_days,
        max_snapshots=max_snapshots,
    )

    match_date = _as_date(match_datetime)

    candidates_by_index: dict[
        int,
        PastSnapshotCandidate,
    ] = {}

    for occurrence in occurrences:
        if (
            not isinstance(
                occurrence,
                (
                    tuple,
                    list,
                ),
            )
            or len(occurrence) != 2
        ):
            continue

        snapshot_index = occurrence[0]

        if type(snapshot_index) is not int:
            continue

        if not (0 <= snapshot_index < len(snapshots)):
            continue

        snapshot = snapshots[snapshot_index]

        if (
            not isinstance(
                snapshot,
                (
                    tuple,
                    list,
                ),
            )
            or len(snapshot) != 2
        ):
            continue

        try:
            snapshot_date = _as_date(snapshot[0])
        except TypeError:
            continue

        age_days = (match_date - snapshot_date).days

        if age_days < 0:
            continue

        if age_days > max_age_days:
            continue

        candidates_by_index[snapshot_index] = PastSnapshotCandidate(
            snapshot_index=(snapshot_index),
            snapshot_date=(snapshot_date),
            age_days=age_days,
        )

    ordered = sorted(
        candidates_by_index.values(),
        key=lambda candidate: (
            candidate.age_days,
            candidate.snapshot_index,
        ),
    )

    return tuple(ordered[:max_snapshots])


def _coerce_observed_skill(
    value: object,
) -> float | None:
    if isinstance(value, bool):
        return None

    if not isinstance(value, Real):
        return None

    numeric = float(value)

    if not math.isfinite(numeric):
        return None

    if numeric < 0.0:
        return None

    return numeric


def _skills_from_record(
    record: object,
) -> Sequence[object] | None:
    if not isinstance(
        record,
        Mapping,
    ):
        return None

    skills = record.get("skills")

    if not isinstance(
        skills,
        Sequence,
    ) or isinstance(
        skills,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        return None

    return skills


def reconstruct_past_only_skills(
    *,
    sofifa_id: int,
    match_datetime: date | datetime,
    snapshots: Sequence[Snapshot],
    player_occurrences: Mapping[
        int,
        Sequence[Occurrence],
    ],
    skill_count: int,
    max_age_days: int,
    max_snapshots: int,
) -> TemporalSkillResult:
    if type(sofifa_id) is not int or sofifa_id <= 0:
        raise ValueError("sofifa_id must be a positive " "integer.")

    if type(skill_count) is not int or skill_count <= 0:
        raise ValueError("skill_count must be a " "positive integer.")

    _validate_limits(
        max_age_days=max_age_days,
        max_snapshots=max_snapshots,
    )

    occurrences = player_occurrences.get(
        sofifa_id,
        (),
    )

    candidates = ordered_past_snapshot_candidates(
        occurrences,
        snapshots,
        match_datetime,
        max_age_days=max_age_days,
        max_snapshots=max_snapshots,
    )

    skills = [MISSING_SKILL_VALUE] * skill_count
    provenance = [SkillProvenance.UNRESOLVED] * skill_count
    source_dates: list[date | None] = [None] * skill_count

    snapshots_used = 0
    nearest_delta: int | None = None

    for candidate in candidates:
        snapshot = snapshots[candidate.snapshot_index]
        players = snapshot[1]

        if not isinstance(
            players,
            Mapping,
        ):
            continue

        raw_skills = _skills_from_record(players.get(sofifa_id))

        if raw_skills is None:
            continue

        fills: list[tuple[int, float]] = []

        for skill_index in range(skill_count):
            if provenance[skill_index] is not SkillProvenance.UNRESOLVED:
                continue

            if skill_index >= len(raw_skills):
                continue

            value = _coerce_observed_skill(raw_skills[skill_index])

            if value is None:
                continue

            fills.append(
                (
                    skill_index,
                    value,
                )
            )

        if not fills:
            continue

        source_provenance = (
            SkillProvenance.NEAREST_PAST_SOFIFA if snapshots_used == 0 else SkillProvenance.OLDER_PAST_SOFIFA
        )

        for skill_index, value in fills:
            skills[skill_index] = value
            provenance[skill_index] = source_provenance
            source_dates[skill_index] = candidate.snapshot_date

        snapshots_used += 1

        if nearest_delta is None:
            nearest_delta = candidate.age_days

        if all(value is not SkillProvenance.UNRESOLVED for value in provenance):
            break

    return TemporalSkillResult(
        skills=tuple(skills),
        provenance=tuple(provenance),
        source_dates=tuple(source_dates),
        snapshots_used=snapshots_used,
        nearest_snapshot_delta_days=(nearest_delta),
    )
