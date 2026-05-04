from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import FSMatch, FSMatchFeatures
from football_outcomes.utils.fs_player_skill_utils import calculate_team_position_indices


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

    comp_name_to_id = {name: i for i, name in enumerate(sett.COMPS_LEAGUE)}

    for m in league_matches_sorted:
        team_ids.add(m.home_team.id)
        team_ids.add(m.away_team.id)

        if m.comp_name is None:
            raise ValueError(f"Match {m.id} has comp_name=None")

        if m.comp_name not in comp_name_to_id:
            raise ValueError(f"Match {m.id} has comp_name '{m.comp_name}' which is not in COMPS_LEAGUE")

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


def summarize_rounds(rounds: List[List[FSMatch]]) -> dict:
    sizes = np.asarray([len(r) for r in rounds], dtype=np.int32)
    return {
        "num_rounds": int(len(rounds)),
        "min_round_size": int(sizes.min()) if sizes.size else 0,
        "max_round_size": int(sizes.max()) if sizes.size else 0,
        "mean_round_size": float(sizes.mean()) if sizes.size else 0.0,
        "median_round_size": float(np.median(sizes)) if sizes.size else 0.0,
    }


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------


def _v(x) -> float:
    return 0.0 if x is None else float(x)


def extract_numerical_features(f: FSMatchFeatures) -> np.ndarray:
    """
    All scalar numerical features only.
    Excludes categorical IDs and structured team-strength tensors.
    """
    vals = [
        _v(f.season),
        _v(f.hours_sin),
        _v(f.hours_cos),
        _v(f.month_sin),
        _v(f.month_cos),
        _v(f.match_position_in_season),
        _v(f.home_elo),
        _v(f.away_elo),
        _v(f.home_avg_xg_last_5),
        _v(f.home_avg_xg_last_20),
        _v(f.away_avg_xg_last_5),
        _v(f.away_avg_xg_last_20),
        _v(f.home_avg_xg_total_last_5),
        _v(f.home_avg_xg_total_last_20),
        _v(f.away_avg_xg_total_last_5),
        _v(f.away_avg_xg_total_last_20),
        _v(f.home_avg_pre_match_xg_last_5),
        _v(f.home_avg_pre_match_xg_last_20),
        _v(f.away_avg_pre_match_xg_last_5),
        _v(f.away_avg_pre_match_xg_last_20),
        _v(f.home_avg_pre_match_xg_total_last_5),
        _v(f.home_avg_pre_match_xg_total_last_20),
        _v(f.away_avg_pre_match_xg_total_last_5),
        _v(f.away_avg_pre_match_xg_total_last_20),
        _v(f.home_match_load_per_day_last_10_days),
        _v(f.home_match_load_per_day_last_25_days),
        _v(f.away_match_load_per_day_last_10_days),
        _v(f.away_match_load_per_day_last_25_days),
        _v(f.home_avg_points_last_5),
        _v(f.home_avg_points_last_20),
        _v(f.away_avg_points_last_5),
        _v(f.away_avg_points_last_20),
        _v(f.home_avg_goals_last_5),
        _v(f.home_avg_goals_last_20),
        _v(f.away_avg_goals_last_5),
        _v(f.away_avg_goals_last_20),
        _v(f.home_avg_shots_on_target_last_5),
        _v(f.home_avg_shots_on_target_last_20),
        _v(f.away_avg_shots_on_target_last_5),
        _v(f.away_avg_shots_on_target_last_20),
        _v(f.home_avg_shots_off_target_last_5),
        _v(f.home_avg_shots_off_target_last_20),
        _v(f.away_avg_shots_off_target_last_5),
        _v(f.away_avg_shots_off_target_last_20),
        _v(f.home_avg_total_shots_last_5),
        _v(f.home_avg_total_shots_last_20),
        _v(f.away_avg_total_shots_last_5),
        _v(f.away_avg_total_shots_last_20),
        _v(f.home_avg_corner_kicks_last_5),
        _v(f.home_avg_corner_kicks_last_20),
        _v(f.away_avg_corner_kicks_last_5),
        _v(f.away_avg_corner_kicks_last_20),
        _v(f.home_avg_ball_possession_last_5),
        _v(f.home_avg_ball_possession_last_20),
        _v(f.away_avg_ball_possession_last_5),
        _v(f.away_avg_ball_possession_last_20),
        _v(f.home_avg_fouls_last_5),
        _v(f.home_avg_fouls_last_20),
        _v(f.away_avg_fouls_last_5),
        _v(f.away_avg_fouls_last_20),
        _v(f.home_avg_attacks_last_5),
        _v(f.home_avg_attacks_last_20),
        _v(f.away_avg_attacks_last_5),
        _v(f.away_avg_attacks_last_20),
        _v(f.home_avg_dang_attacks_last_5),
        _v(f.home_avg_dang_attacks_last_20),
        _v(f.away_avg_dang_attacks_last_5),
        _v(f.away_avg_dang_attacks_last_20),
        _v(f.home_curr_position),
        _v(f.away_curr_position),
        _v(f.home_avg_goals_scored_home_last_5),
        _v(f.home_avg_goals_scored_home_last_20),
        _v(f.away_avg_goals_scored_away_last_5),
        _v(f.away_avg_goals_scored_away_last_20),
        _v(f.home_avg_goals_conceded_home_last_5),
        _v(f.home_avg_goals_conceded_home_last_20),
        _v(f.away_avg_goals_conceded_away_last_5),
        _v(f.away_avg_goals_conceded_away_last_20),
    ]
    return np.asarray(vals, dtype=np.float32)


@dataclass
class SkillImputationStats:
    position_skill_means: np.ndarray  # shape: (num_positions, 34), raw scale 0..100
    global_skill_means: np.ndarray  # shape: (34,), raw scale 0..100
    position_counts: np.ndarray  # shape: (num_positions, 34)
    global_counts: np.ndarray  # shape: (34,)


def _positions_for_match_side(m: FSMatch, f: FSMatchFeatures, side: str) -> np.ndarray:
    if side == "home":
        pos = getattr(f, "home_player_positions", None) or getattr(f, "home_positions", None)
        if pos is None:
            pos = calculate_team_position_indices(m, m.home_team.id)
    else:
        pos = getattr(f, "away_player_positions", None) or getattr(f, "away_positions", None)
        if pos is None:
            pos = calculate_team_position_indices(m, m.away_team.id)

    return np.asarray(pos, dtype=np.int32)


def fit_position_skill_imputer(matches: List[FSMatch]) -> SkillImputationStats:
    """
    Fit simple position-specific skill means from the current rolling training window only.

    Missing values are represented by -1 in the raw team-strength matrices.
    Means are kept on the raw 0..100 scale and converted to [0,1] later.
    """
    num_positions = len(sett.FS_PLAYER_POSITION_TO_IDX)
    num_skills = sett.TEAM_STRENGTH_NUM_SKILLS

    pos_sums = np.zeros((num_positions, num_skills), dtype=np.float64)
    pos_counts = np.zeros((num_positions, num_skills), dtype=np.float64)
    global_sums = np.zeros((num_skills,), dtype=np.float64)
    global_counts = np.zeros((num_skills,), dtype=np.float64)

    def consume(mat, pos_ids):
        if mat is None:
            return

        arr = np.asarray(mat, dtype=np.float32)
        if arr.shape != (sett.TEAM_STRENGTH_NUM_PLAYERS, num_skills):
            return

        pos_ids = np.asarray(pos_ids, dtype=np.int32)
        if pos_ids.shape[0] != sett.TEAM_STRENGTH_NUM_PLAYERS:
            return

        observed = arr >= 0.0

        for row_idx in range(sett.TEAM_STRENGTH_NUM_PLAYERS):
            p = int(pos_ids[row_idx])
            if p < 0 or p >= num_positions:
                continue

            obs = observed[row_idx]
            vals = arr[row_idx]

            pos_sums[p, obs] += vals[obs]
            pos_counts[p, obs] += 1.0

            global_sums[obs] += vals[obs]
            global_counts[obs] += 1.0

    for m in matches:
        f = getattr(m, "features_before_match", None)
        if f is None:
            continue

        home_pos = _positions_for_match_side(m, f, "home")
        away_pos = _positions_for_match_side(m, f, "away")

        consume(f.home_team_strength, home_pos)
        consume(f.away_team_strength, away_pos)

    # Default neutral value if a skill is never observed in the window.
    global_means = np.divide(
        global_sums,
        np.maximum(global_counts, 1.0),
        out=np.full_like(global_sums, 50.0),
        where=global_counts > 0,
    )

    position_means = np.zeros_like(pos_sums)
    for p in range(num_positions):
        position_means[p] = np.divide(
            pos_sums[p],
            np.maximum(pos_counts[p], 1.0),
            out=global_means.copy(),
            where=pos_counts[p] > 0,
        )

    return SkillImputationStats(
        position_skill_means=position_means.astype(np.float32),
        global_skill_means=global_means.astype(np.float32),
        position_counts=pos_counts.astype(np.float32),
        global_counts=global_counts.astype(np.float32),
    )


def _strength_to_value_and_mask(
    mat,
    position_ids: np.ndarray | None = None,
    skill_imputation_stats: SkillImputationStats | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 11x34 list -> (values, mask).

    values: float32 in [0,1]
    mask:   float32 in {0,1}, 1=observed, 0=missing

    If skill_imputation_stats is provided, missing cells are filled with
    training-window position-specific skill means, while the mask remains
    unchanged. This allows comparing:
      - zero + mask
      - position-mean imputation + mask
    """
    if mat is None:
        values = np.zeros((11, 34), dtype=np.float32)
        mask = np.zeros((11, 34), dtype=np.float32)
        return values, mask

    arr = np.asarray(mat, dtype=np.float32)

    if arr.shape != (11, 34):
        out = np.full((11, 34), -1.0, dtype=np.float32)
        flat = arr.flatten()[: 11 * 34]
        if flat.size > 0:
            reshaped = flat.reshape(-1, 34)
            out[: reshaped.shape[0], : reshaped.shape[1]] = reshaped
        arr = out

    mask = (arr >= 0.0).astype(np.float32)

    values_raw = arr.copy()

    if skill_imputation_stats is not None:
        if position_ids is None:
            position_ids = np.zeros((11,), dtype=np.int32)

        position_ids = np.asarray(position_ids, dtype=np.int32)
        if position_ids.shape[0] != 11:
            position_ids = np.zeros((11,), dtype=np.int32)

        for row_idx in range(11):
            p = int(position_ids[row_idx])
            if p < 0 or p >= skill_imputation_stats.position_skill_means.shape[0]:
                fill = skill_imputation_stats.global_skill_means
            else:
                fill = skill_imputation_stats.position_skill_means[p]

            missing = mask[row_idx] == 0.0
            values_raw[row_idx, missing] = fill[missing]

    values = values_raw.copy()
    values[mask == 1.0] = np.clip(values[mask == 1.0] / 100.0, 0.0, 1.0)

    if skill_imputation_stats is None:
        values[mask == 0.0] = 0.0
    else:
        values[mask == 0.0] = np.clip(values[mask == 0.0] / 100.0, 0.0, 1.0)

    return values.astype(np.float32), mask.astype(np.float32)


def build_arrays_for_matches(
    matches: List[FSMatch],
    cat_maps: CatMaps,
    mode: str,
    max_goals_class: int = 10,
    skill_imputation_stats: SkillImputationStats | None = None,
):
    """
    Prepare model inputs and labels.
    """

    X_num, X_h, X_a, X_c, X_s, X_hp, X_ap, y = [], [], [], [], [], [], [], []

    comp_name_to_id = {name: i for i, name in enumerate(sett.COMPS_LEAGUE)}

    for m in matches:
        f = getattr(m, "features_before_match", None)
        if f is None:
            raise ValueError(f"Match {m.id} has no features")

        X_num.append(extract_numerical_features(f))
        X_h.append(cat_maps.team_id_map[m.home_team.id])
        X_a.append(cat_maps.team_id_map[m.away_team.id])
        X_c.append(comp_name_to_id[m.comp_name])

        home_pos = _positions_for_match_side(m, f, "home")
        away_pos = _positions_for_match_side(m, f, "away")

        hs_val, hs_mask = _strength_to_value_and_mask(
            f.home_team_strength,
            position_ids=home_pos,
            skill_imputation_stats=skill_imputation_stats,
        )
        aw_val, aw_mask = _strength_to_value_and_mask(
            f.away_team_strength,
            position_ids=away_pos,
            skill_imputation_stats=skill_imputation_stats,
        )
        X_s.append(np.stack([hs_val, hs_mask, aw_val, aw_mask], axis=0))

        X_hp.append(np.asarray(home_pos, dtype=np.int32))
        X_ap.append(np.asarray(away_pos, dtype=np.int32))

        total_goals = (m.home_goals or 0) + (m.away_goals or 0)
        if mode == "binary_u25":
            y.append(1.0 if total_goals <= 2 else 0.0)
        elif mode == "goals_dist":
            y.append(int(min(total_goals, max_goals_class)))
        elif mode == "goals_reg":
            y.append(float(total_goals))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    y_dtype = np.float32 if mode in ("binary_u25", "goals_reg") else np.int32

    return (
        np.asarray(X_num, np.float32),
        np.asarray(X_h, np.int32)[:, None],
        np.asarray(X_a, np.int32)[:, None],
        np.asarray(X_c, np.int32)[:, None],
        np.asarray(X_s, np.float32),
        np.asarray(X_hp, np.int32),
        np.asarray(X_ap, np.int32),
        np.asarray(y, y_dtype),
    )


def build_flat_tabular_arrays_for_matches(
    matches: List[FSMatch],
    cat_maps: CatMaps,
    mode: str,
    max_goals_class: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten everything into a single 2D matrix for classical sklearn baselines.
    Includes:
      - scalar numerical features
      - one-hot-ready categorical ids as integer columns
      - flattened strength tensor
      - player position indices
    """
    X_num, X_h, X_a, X_c, X_s, X_hp, X_ap, y = build_arrays_for_matches(
        matches=matches,
        cat_maps=cat_maps,
        mode=mode,
        max_goals_class=max_goals_class,
    )

    X = np.concatenate(
        [
            X_num,
            X_h.astype(np.float32),
            X_a.astype(np.float32),
            X_c.astype(np.float32),
            X_s.reshape(len(matches), -1),
            X_hp.astype(np.float32),
            X_ap.astype(np.float32),
        ],
        axis=1,
    )

    return X.astype(np.float32), y


def build_aux_targets_for_matches(
    matches: List[FSMatch],
    aux_mode: str,
    max_goals_class: int = 10,
) -> np.ndarray:
    """
    Build auxiliary targets from raw match outcomes.

    Supported:
      - "binary_u25" : 1 if total goals <= 2 else 0
      - "goals_reg"  : total goals as float
      - "goals_dist" : clipped total-goals class
    """
    vals = []

    for m in matches:
        total_goals = (m.home_goals or 0) + (m.away_goals or 0)

        if aux_mode == "binary_u25":
            vals.append(1.0 if total_goals <= 2 else 0.0)
        elif aux_mode == "goals_reg":
            vals.append(float(total_goals))
        elif aux_mode == "goals_dist":
            vals.append(int(min(total_goals, max_goals_class)))
        else:
            raise ValueError(f"Unknown aux_mode: {aux_mode}")

    dtype = np.float32 if aux_mode in ("binary_u25", "goals_reg") else np.int32
    return np.asarray(vals, dtype=dtype)


def build_strength_only_arrays_for_matches(
    matches: List[FSMatch],
    mode: str,
    max_goals_class: int = 10,
):
    """
    Prepare only the lineup-based structured inputs:
      - strength tensor: (N, 4, 11, 34)
      - home positions:  (N, 11)
      - away positions:  (N, 11)
      - target y

    This is intended for standalone pretraining of the structured branch.
    """
    X_s, X_hp, X_ap, y = [], [], [], []

    for m in matches:
        f = getattr(m, "features_before_match", None)
        if f is None:
            raise ValueError(f"Match {m.id} has no features")

        hs_val, hs_mask = _strength_to_value_and_mask(f.home_team_strength)
        aw_val, aw_mask = _strength_to_value_and_mask(f.away_team_strength)
        X_s.append(np.stack([hs_val, hs_mask, aw_val, aw_mask], axis=0))

        home_pos = getattr(f, "home_player_positions", None) or getattr(f, "home_positions", None)
        if home_pos is None:
            home_pos = calculate_team_position_indices(m, m.home_team.id)

        away_pos = getattr(f, "away_player_positions", None) or getattr(f, "away_positions", None)
        if away_pos is None:
            away_pos = calculate_team_position_indices(m, m.away_team.id)

        X_hp.append(np.asarray(home_pos, dtype=np.int32))
        X_ap.append(np.asarray(away_pos, dtype=np.int32))

        total_goals = (m.home_goals or 0) + (m.away_goals or 0)
        if mode == "binary_u25":
            y.append(1.0 if total_goals <= 2 else 0.0)
        elif mode == "goals_dist":
            y.append(int(min(total_goals, max_goals_class)))
        elif mode == "goals_reg":
            y.append(float(total_goals))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    y_dtype = np.float32 if mode in ("binary_u25", "goals_reg") else np.int32

    return (
        np.asarray(X_s, np.float32),
        np.asarray(X_hp, np.int32),
        np.asarray(X_ap, np.int32),
        np.asarray(y, y_dtype),
    )
