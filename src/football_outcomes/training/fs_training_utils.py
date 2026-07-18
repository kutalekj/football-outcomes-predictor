from __future__ import annotations

from typing import List, Tuple

import numpy as np

from football_outcomes.config import fs_settings as sett
from football_outcomes.data.fs_models import FSMatch, FSMatchFeatures
from football_outcomes.datasets import mappings as _mappings
from football_outcomes.datasets import rounds as _rounds
from football_outcomes.datasets.targets import (
    target_dtype,
    target_for_match,
)
from football_outcomes.utils.fs_player_skill_utils import calculate_team_position_indices

# Compatibility exports for callers using the legacy module path.
distribute_matches_into_rounds = _rounds.distribute_matches_into_rounds
summarize_rounds = _rounds.summarize_rounds
CatMaps = _mappings.CatMaps


# Categorical mappings


def build_categorical_maps(
    league_matches_sorted: List[FSMatch],
) -> CatMaps:
    """Build mappings using the legacy competition ordering."""

    return _mappings.build_categorical_maps(
        league_matches_sorted,
        sett.COMPS_LEAGUE,
    )


# Feature extraction


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


def _strength_to_value_and_mask(mat) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 11x34 list -> (values, mask)
      values: float32 in [0,1], missing cells filled with 0.0
      mask:   float32 in {0,1}, 1=observed, 0=missing
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

    values = arr.copy()
    values[mask == 1.0] = np.clip(values[mask == 1.0] / 100.0, 0.0, 1.0)
    values[mask == 0.0] = 0.0

    return values.astype(np.float32), mask.astype(np.float32)


def build_arrays_for_matches(
    matches: List[FSMatch],
    cat_maps: CatMaps,
    mode: str,
    max_goals_class: int = 10,
):
    """
    Prepare model inputs and labels.
    """

    X_num, X_h, X_a, X_c, X_s, X_hp, X_ap, y = [], [], [], [], [], [], [], []

    # TODO: Revert this patch (added for submission)
    comp_name_to_global_id = {name: i for i, name in enumerate(sett.COMPS_LEAGUE)}

    for m in matches:
        f = getattr(m, "features_before_match", None)
        if f is None:
            raise ValueError(f"Match {m.id} has no features")

        X_num.append(extract_numerical_features(f))
        X_h.append(cat_maps.team_id_map[m.home_team.id])
        X_a.append(cat_maps.team_id_map[m.away_team.id])

        # TODO: Revert this patch (added for submission)
        global_comp_id = comp_name_to_global_id[m.comp_name]
        X_c.append(cat_maps.comp_id_map[global_comp_id])

        home_strength, home_strength_mask = _strength_to_value_and_mask(f.home_team_strength)
        away_strength, away_strength_mask = _strength_to_value_and_mask(f.away_team_strength)

        X_s.append(
            np.stack(
                [
                    home_strength,
                    home_strength_mask,
                    away_strength,
                    away_strength_mask,
                ],
                axis=0,
            )
        )

        home_pos = getattr(f, "home_player_positions", None) or getattr(f, "home_positions", None)
        if home_pos is None:
            home_pos = calculate_team_position_indices(m, m.home_team.id)

        away_pos = getattr(f, "away_player_positions", None) or getattr(f, "away_positions", None)
        if away_pos is None:
            away_pos = calculate_team_position_indices(m, m.away_team.id)

        X_hp.append(np.asarray(home_pos, dtype=np.int32))
        X_ap.append(np.asarray(away_pos, dtype=np.int32))

        y.append(
            target_for_match(
                m,
                mode,
                max_goals_class,
            )
        )

    y_dtype = target_dtype(mode)

    n = len(matches)  # TODO: Revert this patch (added for submission)

    X_s_arr = np.asarray(X_s, np.float32)
    if X_s_arr.size == 0:
        X_s_arr = np.zeros((n, 4, 11, 34), dtype=np.float32)
    else:
        X_s_arr = X_s_arr.reshape((n, 4, 11, 34)).astype(np.float32)

    return (
        np.asarray(X_num, np.float32),
        np.asarray(X_h, np.int32)[:, None],
        np.asarray(X_a, np.int32)[:, None],
        np.asarray(X_c, np.int32)[:, None],
        X_s_arr,
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

    for match in matches:
        try:
            value = target_for_match(
                match,
                aux_mode,
                max_goals_class,
            )
        except ValueError:
            raise ValueError(f"Unknown aux_mode: {aux_mode}") from None

        vals.append(value)

    return np.asarray(
        vals,
        dtype=target_dtype(aux_mode),
    )


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

        y.append(
            target_for_match(
                m,
                mode,
                max_goals_class,
            )
        )

    y_dtype = target_dtype(mode)

    n = len(matches)  # TODO: Revert this patch (added for submission)

    X_s_arr = np.asarray(X_s, np.float32)
    if X_s_arr.size == 0:
        X_s_arr = np.zeros((n, 4, 11, 34), dtype=np.float32)
    else:
        X_s_arr = X_s_arr.reshape((n, 4, 11, 34)).astype(np.float32)

    return (
        X_s_arr,
        np.asarray(X_hp, np.int32),
        np.asarray(X_ap, np.int32),
        np.asarray(y, y_dtype),
    )
