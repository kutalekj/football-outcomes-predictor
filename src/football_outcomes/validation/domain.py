from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping

from football_outcomes.data.fs_models import (
    FSCompSeason,
    FSDataBundle,
    FSMatch,
    FSPlayer,
    FSTeam,
)

Severity = Literal[
    "critical",
    "warning",
]


@dataclass
class ValidationFinding:
    code: str
    severity: Severity
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "count": self.count,
            "examples": list(self.examples),
        }


@dataclass
class DomainValidationReport:
    metrics: dict[
        str,
        int | float,
    ] = field(default_factory=dict)
    findings: dict[
        str,
        ValidationFinding,
    ] = field(default_factory=dict)
    max_examples_per_finding: int = 5

    def add(
        self,
        code: str,
        *,
        entity_type: str,
        entity_id: object,
        message: str,
        severity: Severity = "critical",
        count: int = 1,
    ) -> None:
        if count <= 0:
            raise ValueError("Finding count must be " "positive.")
        finding = self.findings.get(code)

        if finding is None:
            finding = ValidationFinding(
                code=code,
                severity=severity,
            )
            self.findings[code] = finding
        elif finding.severity != severity:
            raise ValueError("Finding severity changed for " f"{code}: " f"{finding.severity} -> " f"{severity}.")

        finding.count += count

        if len(finding.examples) < self.max_examples_per_finding:
            finding.examples.append(f"{entity_type}" f"[{entity_id}]: " f"{message}")

    def count_for(
        self,
        code: str,
    ) -> int:
        finding = self.findings.get(code)

        if finding is None:
            return 0

        return finding.count

    @property
    def critical_issue_count(
        self,
    ) -> int:
        return sum(finding.count for finding in self.findings.values() if finding.severity == "critical")

    @property
    def warning_count(
        self,
    ) -> int:
        return sum(finding.count for finding in self.findings.values() if finding.severity == "warning")

    @property
    def ok(self) -> bool:
        return self.critical_issue_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "critical_issue_count": (self.critical_issue_count),
            "warning_count": (self.warning_count),
            "metrics": {key: self.metrics[key] for key in sorted(self.metrics)},
            "findings": [self.findings[code].to_dict() for code in sorted(self.findings)],
        }


def _is_valid_primary_id(
    value: object,
) -> bool:
    return type(value) is int and value > 0


def _sorted_items(
    mapping: Mapping[
        object,
        object,
    ],
):
    return sorted(
        mapping.items(),
        key=lambda item: repr(item[0]),
    )


def _validate_indexed_collection(
    report: DomainValidationReport,
    *,
    collection: Mapping[
        object,
        object,
    ],
    collection_name: str,
    entity_name: str,
    expected_type: type,
) -> None:
    report.metrics[collection_name] = len(collection)

    for key, entity in _sorted_items(collection):
        if not _is_valid_primary_id(key):
            report.add(
                f"invalid_{entity_name}_key",
                entity_type=entity_name,
                entity_id=key,
                message=("Dictionary key must be " "a positive integer."),
            )

        if not isinstance(
            entity,
            expected_type,
        ):
            report.add(
                f"invalid_{entity_name}_type",
                entity_type=entity_name,
                entity_id=key,
                message=("Unexpected object type " f"{type(entity).__name__}."),
            )
            continue

        entity_id = getattr(
            entity,
            "id",
            None,
        )

        if not _is_valid_primary_id(entity_id):
            report.add(
                f"invalid_{entity_name}_id",
                entity_type=entity_name,
                entity_id=key,
                message=("Object id must be a " "positive integer."),
            )
            continue

        if key != entity_id:
            report.add(
                (f"{entity_name}_" "key_id_mismatch"),
                entity_type=entity_name,
                entity_id=entity_id,
                message=("Dictionary key " f"{key!r} does not " "match object id " f"{entity_id!r}."),
            )


def _validate_team_reference(
    report: DomainValidationReport,
    *,
    bundle: FSDataBundle,
    match: FSMatch,
    side: str,
) -> None:
    team = getattr(
        match,
        f"{side}_team",
        None,
    )

    if team is None:
        report.add(
            f"missing_{side}_team",
            entity_type="match",
            entity_id=match.id,
            message=(f"{side} team is missing."),
            severity="warning",
        )
        return

    if not isinstance(
        team,
        FSTeam,
    ):
        report.add(
            f"invalid_{side}_team_type",
            entity_type="match",
            entity_id=match.id,
            message=(f"{side} team has type " f"{type(team).__name__}."),
        )
        return

    canonical = bundle.teams.get(team.id)

    if canonical is None:
        report.add(
            f"unknown_{side}_team",
            entity_type="match",
            entity_id=match.id,
            message=(f"Team id {team.id} is " "absent from bundle.teams."),
        )
    elif canonical is not team:
        report.add(
            (f"noncanonical_" f"{side}_team"),
            entity_type="match",
            entity_id=match.id,
            message=(f"Team id {team.id} is " "not the canonical object " "stored in bundle.teams."),
        )


def _validate_match_core(
    report: DomainValidationReport,
    *,
    bundle: FSDataBundle,
    match: FSMatch,
) -> None:
    _validate_team_reference(
        report,
        bundle=bundle,
        match=match,
        side="home",
    )
    _validate_team_reference(
        report,
        bundle=bundle,
        match=match,
        side="away",
    )

    if (
        match.home_team is not None
        and match.away_team is not None
        and getattr(
            match.home_team,
            "id",
            None,
        )
        == getattr(
            match.away_team,
            "id",
            None,
        )
    ):
        report.add(
            "same_home_and_away_team",
            entity_type="match",
            entity_id=match.id,
            message=("Home and away teams have " "the same id."),
        )

    for side in (
        "home",
        "away",
    ):
        goals = getattr(
            match,
            f"{side}_goals",
            None,
        )

        if type(goals) is not int or goals < 0:
            report.add(
                f"invalid_{side}_goals",
                entity_type="match",
                entity_id=match.id,
                message=(f"{side} goals must be " "a non-negative integer; " f"found {goals!r}."),
            )

    if not isinstance(
        match.datetime,
        datetime,
    ):
        report.add(
            "invalid_match_datetime",
            entity_type="match",
            entity_id=match.id,
            message=("Match datetime must be a " "datetime instance."),
        )

    if type(match.season) is not int or not (1900 <= match.season <= 2100):
        report.add(
            "invalid_match_season",
            entity_type="match",
            entity_id=match.id,
            message=("Season must be an integer " "between 1900 and 2100; " f"found {match.season!r}."),
        )

    comp_season_id = getattr(
        match,
        "comp_season_id",
        None,
    )

    if not _is_valid_primary_id(comp_season_id):
        report.add(
            "missing_comp_season_id",
            entity_type="match",
            entity_id=match.id,
            message=("Competition-season id is " "missing or invalid."),
        )
        return

    comp_season = bundle.comp_seasons.get(comp_season_id)

    if not isinstance(
        comp_season,
        FSCompSeason,
    ):
        report.add(
            "unknown_comp_season_id",
            entity_type="match",
            entity_id=match.id,
            message=("Competition-season id " f"{comp_season_id} is absent " "from bundle.comp_seasons."),
        )
        return

    if match.season != comp_season.season:
        report.add(
            "match_comp_season_year_mismatch",
            entity_type="match",
            entity_id=match.id,
            message=(
                f"Match season " f"{match.season!r} does not " "match competition-season " f"{comp_season.season!r}."
            ),
        )

    if (
        getattr(
            match,
            "comp_name",
            None,
        )
        != comp_season.name
    ):
        report.add(
            "match_comp_name_mismatch",
            entity_type="match",
            entity_id=match.id,
            message=(f"Match competition " f"{match.comp_name!r} does " "not match " f"{comp_season.name!r}."),
        )

    if (
        getattr(
            match,
            "country",
            None,
        )
        != comp_season.country
    ):
        report.add(
            "match_country_mismatch",
            entity_type="match",
            entity_id=match.id,
            message=(f"Match country " f"{match.country!r} does not " "match " f"{comp_season.country!r}."),
        )


def _validate_comp_season_membership(
    report: DomainValidationReport,
    *,
    bundle: FSDataBundle,
    canonical_matches: dict[
        int,
        FSMatch,
    ],
) -> None:
    memberships: dict[
        int,
        set[int],
    ] = defaultdict(set)

    reference_count = 0

    for comp_key, comp_season in _sorted_items(bundle.comp_seasons):
        if not isinstance(
            comp_season,
            FSCompSeason,
        ):
            continue

        matches = getattr(
            comp_season,
            "matches",
            None,
        )

        if not isinstance(
            matches,
            list,
        ):
            report.add(
                ("invalid_comp_season_" "matches"),
                entity_type=("competition-season"),
                entity_id=comp_key,
                message=("matches must be a list."),
            )
            continue

        seen_match_ids: set[int] = set()

        for match in matches:
            reference_count += 1

            if not isinstance(
                match,
                FSMatch,
            ):
                report.add(
                    ("invalid_comp_season_" "match_type"),
                    entity_type=("competition-season"),
                    entity_id=comp_key,
                    message=("Competition-season " "contains a non-match " "object."),
                )
                continue

            match_id = getattr(
                match,
                "id",
                None,
            )

            if not _is_valid_primary_id(match_id):
                report.add(
                    ("invalid_comp_season_" "match_id"),
                    entity_type=("competition-season"),
                    entity_id=comp_key,
                    message=("Referenced match has " "an invalid id."),
                )
                continue

            if match_id in seen_match_ids:
                report.add(
                    ("duplicate_comp_season_" "match_reference"),
                    entity_type=("competition-season"),
                    entity_id=comp_key,
                    message=(f"Match {match_id} is " "referenced more than " "once."),
                )

            seen_match_ids.add(match_id)
            memberships[match_id].add(comp_key)

            canonical = canonical_matches.get(match_id)

            if canonical is None:
                report.add(
                    ("detached_comp_season_" "match_reference"),
                    entity_type=("competition-season"),
                    entity_id=comp_key,
                    message=(
                        f"Match {match_id} is "
                        "referenced by the "
                        "competition season but "
                        "is absent from the "
                        "authoritative "
                        "bundle.matches list."
                    ),
                    severity="warning",
                )
            elif canonical is not match:
                report.add(
                    ("noncanonical_comp_" "season_match"),
                    entity_type=("competition-season"),
                    entity_id=comp_key,
                    message=(f"Match {match_id} is " "not the canonical " "bundle.matches object."),
                )

            if (
                getattr(
                    match,
                    "comp_season_id",
                    None,
                )
                != comp_key
            ):
                report.add(
                    ("comp_season_match_" "link_mismatch"),
                    entity_type=("competition-season"),
                    entity_id=comp_key,
                    message=(
                        f"Match {match_id} "
                        "links to competition "
                        "season "
                        f"{getattr(match, 'comp_season_id', None)!r}."
                    ),
                )

    report.metrics["competition_season_match_references"] = reference_count

    for match_id in sorted(canonical_matches):
        linked_comp_seasons = memberships.get(
            match_id,
            set(),
        )

        if not linked_comp_seasons:
            report.add(
                ("match_missing_from_" "comp_season"),
                entity_type="match",
                entity_id=match_id,
                message=("Match is not referenced " "by any competition " "season."),
            )
        elif len(linked_comp_seasons) > 1:
            report.add(
                ("match_in_multiple_" "comp_seasons"),
                entity_type="match",
                entity_id=match_id,
                message=("Match is referenced by " "competition seasons " f"{sorted(linked_comp_seasons)}."),
            )


def _validate_rosters(
    report: DomainValidationReport,
    *,
    bundle: FSDataBundle,
) -> None:
    reference_count = 0

    for team_key, team in _sorted_items(bundle.teams):
        if not isinstance(
            team,
            FSTeam,
        ):
            continue

        rosters = getattr(
            team,
            "comp_seasons",
            None,
        )

        if not isinstance(
            rosters,
            dict,
        ):
            report.add(
                "invalid_team_rosters",
                entity_type="team",
                entity_id=team_key,
                message=("comp_seasons must be " "a dictionary."),
            )
            continue

        for comp_season_id, roster in _sorted_items(rosters):
            if comp_season_id not in bundle.comp_seasons:
                report.add(
                    ("unknown_roster_" "comp_season"),
                    entity_type="team",
                    entity_id=team_key,
                    message=("Roster references " "unknown competition " "season " f"{comp_season_id!r}."),
                )

            if not isinstance(
                roster,
                list,
            ):
                report.add(
                    "invalid_roster_type",
                    entity_type="team",
                    entity_id=team_key,
                    message=("Roster for " f"{comp_season_id!r} " "must be a list."),
                )
                continue

            seen_player_ids: set[int] = set()

            for player in roster:
                reference_count += 1

                if not isinstance(
                    player,
                    FSPlayer,
                ):
                    report.add(
                        ("invalid_roster_" "player_type"),
                        entity_type="team",
                        entity_id=team_key,
                        message=("Roster contains " "a non-player " "object."),
                    )
                    continue

                player_id = player.id

                if player_id in (seen_player_ids):
                    report.add(
                        ("duplicate_roster_" "player"),
                        entity_type="team",
                        entity_id=team_key,
                        message=(f"Player {player_id} " "appears more than " "once in roster " f"{comp_season_id}."),
                    )

                seen_player_ids.add(player_id)

                canonical = bundle.players.get(player_id)

                if canonical is None:
                    report.add(
                        ("unknown_roster_" "player"),
                        entity_type="team",
                        entity_id=team_key,
                        message=(f"Player {player_id} is " "absent from " "bundle.players."),
                        severity="warning",
                    )
                elif canonical is not player:
                    report.add(
                        ("noncanonical_" "roster_player"),
                        entity_type="team",
                        entity_id=team_key,
                        message=(f"Player " f"{player_id} is " "not the canonical " "bundle.players " "object."),
                    )

    report.metrics["team_roster_player_references"] = reference_count


def _validate_lineups(
    report: DomainValidationReport,
    *,
    bundle: FSDataBundle,
    canonical_matches: dict[
        int,
        FSMatch,
    ],
) -> None:
    reference_count = 0

    for match_id in sorted(canonical_matches):
        match = canonical_matches[match_id]
        side_player_ids: dict[
            str,
            set[int],
        ] = {}

        for side in (
            "home",
            "away",
        ):
            lineup = getattr(
                match,
                f"{side}_lineup",
                None,
            )

            if not isinstance(
                lineup,
                list,
            ):
                report.add(
                    (f"invalid_{side}_" "lineup_type"),
                    entity_type="match",
                    entity_id=match_id,
                    message=(f"{side} lineup must " "be a list."),
                )
                continue

            seen_player_ids: set[int] = set()

            for player in lineup:
                reference_count += 1

                if not isinstance(
                    player,
                    FSPlayer,
                ):
                    report.add(
                        (f"invalid_{side}_" "lineup_player_type"),
                        entity_type="match",
                        entity_id=match_id,
                        message=(f"{side} lineup " "contains a " "non-player object."),
                    )
                    continue

                player_id = player.id

                if player_id in (seen_player_ids):
                    report.add(
                        (f"duplicate_{side}_" "lineup_player"),
                        entity_type="match",
                        entity_id=match_id,
                        message=(f"Player " f"{player_id} " "appears more than " "once."),
                        severity="warning",
                    )

                seen_player_ids.add(player_id)

                canonical = bundle.players.get(player_id)

                if canonical is None:
                    report.add(
                        (f"unknown_{side}_" "lineup_player"),
                        entity_type="match",
                        entity_id=match_id,
                        message=(f"Player " f"{player_id} is " "absent from " "bundle.players."),
                    )
                elif canonical is not player:
                    report.add(
                        (f"noncanonical_" f"{side}_lineup_player"),
                        entity_type="match",
                        entity_id=match_id,
                        message=(f"Player " f"{player_id} is " "not the canonical " "bundle.players " "object."),
                    )

            side_player_ids[side] = seen_player_ids

        shared_players = side_player_ids.get(
            "home",
            set(),
        ) & side_player_ids.get(
            "away",
            set(),
        )

        for player_id in sorted(shared_players):
            report.add(
                ("player_in_both_" "lineups"),
                entity_type="match",
                entity_id=match_id,
                message=(f"Player {player_id} " "appears in both lineups."),
                severity="warning",
            )

    report.metrics["lineup_player_references"] = reference_count


def validate_bundle_domain(
    bundle: FSDataBundle,
    *,
    max_examples_per_finding: int = 5,
) -> DomainValidationReport:
    if max_examples_per_finding < 0:
        raise ValueError("max_examples_per_finding " "must be non-negative.")

    report = DomainValidationReport(max_examples_per_finding=(max_examples_per_finding))

    _validate_indexed_collection(
        report,
        collection=bundle.comp_seasons,
        collection_name=("competition_seasons"),
        entity_name="comp_season",
        expected_type=FSCompSeason,
    )
    _validate_indexed_collection(
        report,
        collection=bundle.teams,
        collection_name="teams",
        entity_name="team",
        expected_type=FSTeam,
    )
    _validate_indexed_collection(
        report,
        collection=bundle.players,
        collection_name="players",
        entity_name="player",
        expected_type=FSPlayer,
    )

    report.metrics["matches"] = len(bundle.matches)

    canonical_matches: dict[
        int,
        FSMatch,
    ] = {}

    for index, match in enumerate(bundle.matches):
        if not isinstance(
            match,
            FSMatch,
        ):
            report.add(
                "invalid_match_type",
                entity_type="match-index",
                entity_id=index,
                message=("Unexpected object type " f"{type(match).__name__}."),
            )
            continue

        match_id = getattr(
            match,
            "id",
            None,
        )

        if not _is_valid_primary_id(match_id):
            report.add(
                "invalid_match_id",
                entity_type="match-index",
                entity_id=index,
                message=("Match id must be a " "positive integer."),
            )
            continue

        if match_id in canonical_matches:
            report.add(
                "duplicate_match_id",
                entity_type="match",
                entity_id=match_id,
                message=("Match id occurs more " "than once in " "bundle.matches."),
            )
        else:
            canonical_matches[match_id] = match

        _validate_match_core(
            report,
            bundle=bundle,
            match=match,
        )

    report.metrics["unique_match_ids"] = len(canonical_matches)

    _validate_comp_season_membership(
        report,
        bundle=bundle,
        canonical_matches=(canonical_matches),
    )
    _validate_rosters(
        report,
        bundle=bundle,
    )
    _validate_lineups(
        report,
        bundle=bundle,
        canonical_matches=(canonical_matches),
    )

    return report
