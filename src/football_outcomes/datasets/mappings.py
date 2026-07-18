from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from football_outcomes.data.fs_models import FSMatch


@dataclass
class CatMaps:
    team_id_map: dict[int, int]
    comp_id_map: dict[int, int]


def build_categorical_maps(
    matches: Sequence[FSMatch],
    competition_names: Sequence[str],
) -> CatMaps:
    """Build deterministic dense team and competition maps."""

    team_ids: set[int] = set()
    competition_ids: set[int] = set()

    competition_name_to_id = {name: index for index, name in enumerate(competition_names)}

    for match in matches:
        team_ids.add(match.home_team.id)
        team_ids.add(match.away_team.id)

        if match.comp_name is None:
            raise ValueError(f"Match {match.id} has comp_name=None")

        if match.comp_name not in competition_name_to_id:
            # Preserve the legacy public error wording
            # during the compatibility refactor.
            raise ValueError(f"Match {match.id} has comp_name " f"'{match.comp_name}' which is not " f"in COMPS_LEAGUE")

        competition_ids.add(competition_name_to_id[match.comp_name])

    team_id_map = {team_id: dense_id for dense_id, team_id in enumerate(sorted(team_ids))}
    comp_id_map = {competition_id: dense_id for dense_id, competition_id in enumerate(sorted(competition_ids))}

    return CatMaps(
        team_id_map=team_id_map,
        comp_id_map=comp_id_map,
    )
