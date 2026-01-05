from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import FSMatch, FSMatchFeatures


@dataclass
class CatMaps:
    team_id_map: Dict[int, int]
    comp_id_map: Dict[int, int]


# ---------------------------------------------------------------------
# Categorical mappings
# ---------------------------------------------------------------------


def build_categorical_maps(league_matches_sorted: List[FSMatch]) -> CatMaps:
    """
    Build dense index maps for:
      - team_id
      - comp_id (derived from comp_name via COMPS_LEAGUE)
    """

    team_ids = set()
    comp_ids = set()

    # Build comp_name -> comp_id mapping from settings
    comp_name_to_id = {name: i for i, name in enumerate(sett.COMPS_LEAGUE)}

    for m in league_matches_sorted:
        team_ids.add(m.home_team.id)
        team_ids.add(m.away_team.id)

        if m.comp_name is None:
            raise ValueError(f"Match {m.id} has comp_name=None")

        if m.comp_name not in comp_name_to_id:
            raise ValueError(f"Match {m.id} has comp_name '{m.comp_name}' " f"which is not in COMPS_LEAGUE")

        comp_ids.add(comp_name_to_id[m.comp_name])

    team_id_map = {tid: i for i, tid in enumerate(sorted(team_ids))}
    comp_id_map = {cid: i for i, cid in enumerate(sorted(comp_ids))}

    return CatMaps(team_id_map=team_id_map, comp_id_map=comp_id_map)


# ---------------------------------------------------------------------
# Round distribution
# ---------------------------------------------------------------------


def distribute_matches_into_rounds(sorted_matches: List[FSMatch]) -> List[List[FSMatch]]:
    """
    Chronological rounds where no team appears more than once per round.
    """
    rounds: List[List[FSMatch]] = []
    current_round: List[FSMatch] = []
    teams_in_round = set()

    for match in sorted_matches:
        h = match.home_team.id
        a = match.away_team.id

        if h in teams_in_round or a in teams_in_round:
            rounds.append(current_round)
            current_round = []
            teams_in_round = set()

        current_round.append(match)
        teams_in_round.add(h)
        teams_in_round.add(a)

    if current_round:
        rounds.append(current_round)

    return rounds


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------


def _v(x) -> float:
    return 0.0 if x is None else float(x)


def extract_numerical_features(f: FSMatchFeatures) -> np.ndarray:
    """
    Scalar numerical features only (no categorical, no strengths).
    """
    vals = [
        _v(f.hours_sin),
        _v(f.hours_cos),
        _v(f.month_sin),
        _v(f.month_cos),
        _v(f.home_elo),
        _v(f.away_elo),
        _v(f.match_position_in_season),
        _v(f.home_avg_points_last_5),
        _v(f.home_avg_points_last_20),
        _v(f.away_avg_points_last_5),
        _v(f.away_avg_points_last_20),
        _v(f.home_avg_goals_last_5),
        _v(f.home_avg_goals_last_20),
        _v(f.away_avg_goals_last_5),
        _v(f.away_avg_goals_last_20),
        _v(f.home_avg_xg_last_5),
        _v(f.home_avg_xg_last_20),
        _v(f.away_avg_xg_last_5),
        _v(f.away_avg_xg_last_20),
        _v(f.home_match_load_per_day_last_10_days),
        _v(f.home_match_load_per_day_last_25_days),
        _v(f.away_match_load_per_day_last_10_days),
        _v(f.away_match_load_per_day_last_25_days),
        _v(f.home_curr_position),
        _v(f.away_curr_position),
    ]
    return np.asarray(vals, dtype=np.float32)


def _strength_to_np(mat) -> np.ndarray:
    """
    Convert 11x34 list -> np.ndarray, keep -1 values.
    """
    if mat is None:
        return np.zeros((11, 34), dtype=np.float32)

    arr = np.asarray(mat, dtype=np.float32)

    if arr.shape != (11, 34):
        out = np.zeros((11, 34), dtype=np.float32)
        flat = arr.flatten()
        flat = flat[: 11 * 34]
        out[: flat.size // 34, :34] = flat.reshape(-1, 34)
        arr = out

    if np.nanmax(arr) > 2.0:
        arr = arr / 100.0

    return np.clip(arr, -1.0, 1.0)


def build_arrays_for_matches(
    matches: List[FSMatch],
    cat_maps: CatMaps,
    mode: str,
    max_goals_class: int = 10,
):
    """
    Prepare model inputs and labels.
    """

    X_num, X_h, X_a, X_c, X_s, y = [], [], [], [], [], []

    for m in matches:
        f = getattr(m, "features_before_match", None)
        if f is None:
            raise ValueError(f"Match {m.id} has no features")

        X_num.append(extract_numerical_features(f))
        X_h.append(cat_maps.team_id_map[m.home_team.id])
        X_a.append(cat_maps.team_id_map[m.away_team.id])

        comp_name_to_id = {name: i for i, name in enumerate(sett.COMPS_LEAGUE)}
        X_c.append(comp_name_to_id[m.comp_name])

        hs = _strength_to_np(f.home_team_strength)
        aw = _strength_to_np(f.away_team_strength)
        X_s.append(np.stack([hs, aw], axis=0))

        total_goals = (m.home_goals or 0) + (m.away_goals or 0)
        if mode == "binary_u25":
            y.append(1 if total_goals <= 2 else 0)
        else:
            y.append(min(total_goals, max_goals_class))

    return (
        np.asarray(X_num, np.float32),
        np.asarray(X_h, np.int32)[:, None],
        np.asarray(X_a, np.int32)[:, None],
        np.asarray(X_c, np.int32)[:, None],
        np.asarray(X_s, np.float32),
        np.asarray(y, np.int32 if mode != "binary_u25" else np.float32),
    )
