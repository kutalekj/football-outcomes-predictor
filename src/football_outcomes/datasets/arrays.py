from __future__ import annotations

import numpy as np

from football_outcomes.data.fs_models import (
    FSMatchFeatures,
)


def _value_or_zero(value) -> float:
    return 0.0 if value is None else float(value)


def extract_numerical_features(f: FSMatchFeatures) -> np.ndarray:
    """
    All scalar numerical features only.
    Excludes categorical IDs and structured team-strength tensors.
    """
    vals = [
        _value_or_zero(f.season),
        _value_or_zero(f.hours_sin),
        _value_or_zero(f.hours_cos),
        _value_or_zero(f.month_sin),
        _value_or_zero(f.month_cos),
        _value_or_zero(f.match_position_in_season),
        _value_or_zero(f.home_elo),
        _value_or_zero(f.away_elo),
        _value_or_zero(f.home_avg_xg_last_5),
        _value_or_zero(f.home_avg_xg_last_20),
        _value_or_zero(f.away_avg_xg_last_5),
        _value_or_zero(f.away_avg_xg_last_20),
        _value_or_zero(f.home_avg_xg_total_last_5),
        _value_or_zero(f.home_avg_xg_total_last_20),
        _value_or_zero(f.away_avg_xg_total_last_5),
        _value_or_zero(f.away_avg_xg_total_last_20),
        _value_or_zero(f.home_avg_pre_match_xg_last_5),
        _value_or_zero(f.home_avg_pre_match_xg_last_20),
        _value_or_zero(f.away_avg_pre_match_xg_last_5),
        _value_or_zero(f.away_avg_pre_match_xg_last_20),
        _value_or_zero(f.home_avg_pre_match_xg_total_last_5),
        _value_or_zero(f.home_avg_pre_match_xg_total_last_20),
        _value_or_zero(f.away_avg_pre_match_xg_total_last_5),
        _value_or_zero(f.away_avg_pre_match_xg_total_last_20),
        _value_or_zero(f.home_match_load_per_day_last_10_days),
        _value_or_zero(f.home_match_load_per_day_last_25_days),
        _value_or_zero(f.away_match_load_per_day_last_10_days),
        _value_or_zero(f.away_match_load_per_day_last_25_days),
        _value_or_zero(f.home_avg_points_last_5),
        _value_or_zero(f.home_avg_points_last_20),
        _value_or_zero(f.away_avg_points_last_5),
        _value_or_zero(f.away_avg_points_last_20),
        _value_or_zero(f.home_avg_goals_last_5),
        _value_or_zero(f.home_avg_goals_last_20),
        _value_or_zero(f.away_avg_goals_last_5),
        _value_or_zero(f.away_avg_goals_last_20),
        _value_or_zero(f.home_avg_shots_on_target_last_5),
        _value_or_zero(f.home_avg_shots_on_target_last_20),
        _value_or_zero(f.away_avg_shots_on_target_last_5),
        _value_or_zero(f.away_avg_shots_on_target_last_20),
        _value_or_zero(f.home_avg_shots_off_target_last_5),
        _value_or_zero(f.home_avg_shots_off_target_last_20),
        _value_or_zero(f.away_avg_shots_off_target_last_5),
        _value_or_zero(f.away_avg_shots_off_target_last_20),
        _value_or_zero(f.home_avg_total_shots_last_5),
        _value_or_zero(f.home_avg_total_shots_last_20),
        _value_or_zero(f.away_avg_total_shots_last_5),
        _value_or_zero(f.away_avg_total_shots_last_20),
        _value_or_zero(f.home_avg_corner_kicks_last_5),
        _value_or_zero(f.home_avg_corner_kicks_last_20),
        _value_or_zero(f.away_avg_corner_kicks_last_5),
        _value_or_zero(f.away_avg_corner_kicks_last_20),
        _value_or_zero(f.home_avg_ball_possession_last_5),
        _value_or_zero(f.home_avg_ball_possession_last_20),
        _value_or_zero(f.away_avg_ball_possession_last_5),
        _value_or_zero(f.away_avg_ball_possession_last_20),
        _value_or_zero(f.home_avg_fouls_last_5),
        _value_or_zero(f.home_avg_fouls_last_20),
        _value_or_zero(f.away_avg_fouls_last_5),
        _value_or_zero(f.away_avg_fouls_last_20),
        _value_or_zero(f.home_avg_attacks_last_5),
        _value_or_zero(f.home_avg_attacks_last_20),
        _value_or_zero(f.away_avg_attacks_last_5),
        _value_or_zero(f.away_avg_attacks_last_20),
        _value_or_zero(f.home_avg_dang_attacks_last_5),
        _value_or_zero(f.home_avg_dang_attacks_last_20),
        _value_or_zero(f.away_avg_dang_attacks_last_5),
        _value_or_zero(f.away_avg_dang_attacks_last_20),
        _value_or_zero(f.home_curr_position),
        _value_or_zero(f.away_curr_position),
        _value_or_zero(f.home_avg_goals_scored_home_last_5),
        _value_or_zero(f.home_avg_goals_scored_home_last_20),
        _value_or_zero(f.away_avg_goals_scored_away_last_5),
        _value_or_zero(f.away_avg_goals_scored_away_last_20),
        _value_or_zero(f.home_avg_goals_conceded_home_last_5),
        _value_or_zero(f.home_avg_goals_conceded_home_last_20),
        _value_or_zero(f.away_avg_goals_conceded_away_last_5),
        _value_or_zero(f.away_avg_goals_conceded_away_last_20),
    ]
    return np.asarray(vals, dtype=np.float32)


def strength_to_value_and_mask(mat) -> tuple[np.ndarray, np.ndarray]:
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
