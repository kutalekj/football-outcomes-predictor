from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from datetime import (
    date,
    datetime,
)

from football_outcomes.data.fs_models import (
    FSMatch,
)
from football_outcomes.data.lineups import (
    calculate_team_position_indices,
    select_and_sort_lineup,
)
from football_outcomes.data.sofifa_temporal import (
    MISSING_SKILL_VALUE,
    SkillProvenance,
    Snapshot,
    reconstruct_past_only_skills,
)

UNRESOLVED_SOURCE_AGE_DAYS = -1


@dataclass(frozen=True)
class PastOnlyStrengthConfig:
    player_count: int
    skill_count: int
    max_age_days: int
    max_snapshots: int

    def __post_init__(self) -> None:
        if type(self.player_count) is not int or self.player_count <= 0:
            raise ValueError("player_count must be a " "positive integer.")

        if type(self.skill_count) is not int or self.skill_count <= 0:
            raise ValueError("skill_count must be a " "positive integer.")

        if type(self.max_age_days) is not int or self.max_age_days < 0:
            raise ValueError("max_age_days must be a " "non-negative integer.")

        if type(self.max_snapshots) is not int or self.max_snapshots <= 0:
            raise ValueError("max_snapshots must be a " "positive integer.")


@dataclass(frozen=True)
class PastOnlyTeamStrengthResult:
    side: str
    team_id: int
    skills: tuple[
        tuple[float, ...],
        ...,
    ]
    provenance: tuple[
        tuple[SkillProvenance, ...],
        ...,
    ]
    source_age_days: tuple[
        tuple[int, ...],
        ...,
    ]
    fs_player_ids: tuple[int, ...]
    sofifa_player_ids: tuple[
        int | None,
        ...,
    ]
    position_indices: tuple[int, ...]

    @property
    def observed_count(self) -> int:
        return sum(
            provenance
            in (
                SkillProvenance.NEAREST_PAST_SOFIFA,
                SkillProvenance.OLDER_PAST_SOFIFA,
            )
            for row in self.provenance
            for provenance in row
        )

    @property
    def nearest_past_count(self) -> int:
        return sum(provenance is SkillProvenance.NEAREST_PAST_SOFIFA for row in self.provenance for provenance in row)

    @property
    def older_past_count(self) -> int:
        return sum(provenance is SkillProvenance.OLDER_PAST_SOFIFA for row in self.provenance for provenance in row)

    @property
    def unresolved_count(self) -> int:
        return sum(provenance is SkillProvenance.UNRESOLVED for row in self.provenance for provenance in row)

    @property
    def matched_player_rows(self) -> int:
        return sum(sofifa_id is not None for sofifa_id in self.sofifa_player_ids)

    @property
    def unmatched_player_rows(self) -> int:
        return len(self.sofifa_player_ids) - self.matched_player_rows


@dataclass(frozen=True)
class PastOnlyMatchStrengthResult:
    match_id: int
    home: PastOnlyTeamStrengthResult
    away: PastOnlyTeamStrengthResult

    @property
    def observed_count(self) -> int:
        return self.home.observed_count + self.away.observed_count

    @property
    def unresolved_count(self) -> int:
        return self.home.unresolved_count + self.away.unresolved_count


def cached_sofifa_id(
    cache: Mapping[int, object],
    fs_player_id: int,
) -> int | None:
    record = cache.get(fs_player_id)

    if (
        not isinstance(
            record,
            (
                tuple,
                list,
            ),
        )
        or len(record) == 0
    ):
        return None

    sofifa_id = record[0]

    if type(sofifa_id) is int and sofifa_id > 0:
        return sofifa_id

    return None


def _as_date(
    value: date | datetime,
) -> date:
    if isinstance(
        value,
        datetime,
    ):
        return value.date()

    if isinstance(
        value,
        date,
    ):
        return value

    raise TypeError("Expected date or datetime, " f"found {type(value)!r}.")


def _missing_skill_row(
    skill_count: int,
) -> tuple[float, ...]:
    return tuple(MISSING_SKILL_VALUE for _ in range(skill_count))


def _missing_provenance_row(
    skill_count: int,
) -> tuple[
    SkillProvenance,
    ...,
]:
    return tuple(SkillProvenance.UNRESOLVED for _ in range(skill_count))


def _missing_age_row(
    skill_count: int,
) -> tuple[int, ...]:
    return tuple(UNRESOLVED_SOURCE_AGE_DAYS for _ in range(skill_count))


def reconstruct_past_only_team_strength(
    *,
    match: FSMatch,
    team_id: int,
    snapshots: Sequence[Snapshot],
    player_occurrences: Mapping[
        int,
        Sequence[tuple[int, date]],
    ],
    fs_to_sofifa_cache: Mapping[
        int,
        object,
    ],
    config: PastOnlyStrengthConfig,
) -> PastOnlyTeamStrengthResult:
    if match.datetime is None:
        raise ValueError(f"Match {match.id} has no " "datetime.")

    match_date = _as_date(match.datetime)

    lineup, side = select_and_sort_lineup(
        match,
        team_id,
    )
    position_indices = tuple(
        calculate_team_position_indices(
            match,
            team_id,
        )
    )

    if len(lineup) != config.player_count:
        raise ValueError("Lineup service returned " f"{len(lineup)} rows; expected " f"{config.player_count}.")

    if len(position_indices) != config.player_count:
        raise ValueError(
            "Position service returned " f"{len(position_indices)} rows; " f"expected " f"{config.player_count}."
        )

    skill_rows: list[tuple[float, ...]] = []
    provenance_rows: list[tuple[SkillProvenance, ...]] = []
    age_rows: list[tuple[int, ...]] = []
    fs_player_ids: list[int] = []
    sofifa_player_ids: list[int | None] = []

    for player in lineup:
        raw_player_id = getattr(
            player,
            "id",
            None,
        )
        fs_player_id = raw_player_id if type(raw_player_id) is int else -1

        fs_player_ids.append(fs_player_id)

        if fs_player_id <= 0:
            sofifa_player_ids.append(None)
            skill_rows.append(_missing_skill_row(config.skill_count))
            provenance_rows.append(_missing_provenance_row(config.skill_count))
            age_rows.append(_missing_age_row(config.skill_count))
            continue

        sofifa_id = cached_sofifa_id(
            fs_to_sofifa_cache,
            fs_player_id,
        )
        sofifa_player_ids.append(sofifa_id)

        if sofifa_id is None:
            skill_rows.append(_missing_skill_row(config.skill_count))
            provenance_rows.append(_missing_provenance_row(config.skill_count))
            age_rows.append(_missing_age_row(config.skill_count))
            continue

        temporal = reconstruct_past_only_skills(
            sofifa_id=sofifa_id,
            match_datetime=(match.datetime),
            snapshots=snapshots,
            player_occurrences=(player_occurrences),
            skill_count=(config.skill_count),
            max_age_days=(config.max_age_days),
            max_snapshots=(config.max_snapshots),
        )

        source_ages = []

        for source_date in temporal.source_dates:
            if source_date is None:
                source_ages.append(UNRESOLVED_SOURCE_AGE_DAYS)
                continue

            source_age = (match_date - source_date).days

            if source_age < 0:
                raise RuntimeError("Past-only reconstruction " "returned a future source " "date.")

            source_ages.append(source_age)

        skill_rows.append(temporal.skills)
        provenance_rows.append(temporal.provenance)
        age_rows.append(tuple(source_ages))

    result = PastOnlyTeamStrengthResult(
        side=side,
        team_id=team_id,
        skills=tuple(skill_rows),
        provenance=tuple(provenance_rows),
        source_age_days=tuple(age_rows),
        fs_player_ids=tuple(fs_player_ids),
        sofifa_player_ids=tuple(sofifa_player_ids),
        position_indices=(position_indices),
    )

    expected_shape = (
        config.player_count,
        config.skill_count,
    )

    for name, matrix in (
        (
            "skills",
            result.skills,
        ),
        (
            "provenance",
            result.provenance,
        ),
        (
            "source ages",
            result.source_age_days,
        ),
    ):
        shape = (
            len(matrix),
            len(matrix[0]) if matrix else 0,
        )

        if shape != expected_shape:
            raise RuntimeError(f"Invalid {name} matrix " f"shape {shape}; expected " f"{expected_shape}.")

    return result


def reconstruct_past_only_match_strength(
    *,
    match: FSMatch,
    snapshots: Sequence[Snapshot],
    player_occurrences: Mapping[
        int,
        Sequence[tuple[int, date]],
    ],
    fs_to_sofifa_cache: Mapping[
        int,
        object,
    ],
    config: PastOnlyStrengthConfig,
) -> PastOnlyMatchStrengthResult:
    if match.home_team is None or match.away_team is None:
        raise ValueError(f"Match {match.id} is missing " "one or both teams.")

    home = reconstruct_past_only_team_strength(
        match=match,
        team_id=match.home_team.id,
        snapshots=snapshots,
        player_occurrences=(player_occurrences),
        fs_to_sofifa_cache=(fs_to_sofifa_cache),
        config=config,
    )
    away = reconstruct_past_only_team_strength(
        match=match,
        team_id=match.away_team.id,
        snapshots=snapshots,
        player_occurrences=(player_occurrences),
        fs_to_sofifa_cache=(fs_to_sofifa_cache),
        config=config,
    )

    return PastOnlyMatchStrengthResult(
        match_id=match.id,
        home=home,
        away=away,
    )
