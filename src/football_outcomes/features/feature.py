"""
feature.py
"""


import numpy as np


class MatchFeatures:
    def __init__(self, comp_id, season, home_team_id, away_team_id,
                 hours_sin, hours_cos, month_sin, month_cos):
        self.comp_id = comp_id
        self.season = season

        self.home_team_id = home_team_id
        self.away_team_id = away_team_id

        self.hours_sin = hours_sin
        self.hours_cos = hours_cos
        self.month_sin = month_sin
        self.month_cos = month_cos

        self.home_elo = None
        self.away_elo = None

        self.relative_match_position_in_country_season = None

        self.home_avg_xg_last_5 = None
        self.home_avg_xg_last_20 = None
        self.away_avg_xg_last_5 = None
        self.away_avg_xg_last_20 = None

        self.home_avg_xg_total_last_5 = None
        self.home_avg_xg_total_last_20 = None
        self.away_avg_xg_total_last_5 = None
        self.away_avg_xg_total_last_20 = None

        self.home_avg_pre_match_xg_last_5 = None
        self.home_avg_pre_match_xg_last_20 = None
        self.away_avg_pre_match_xg_last_5 = None
        self.away_avg_pre_match_xg_last_20 = None

        self.home_avg_pre_match_xg_total_last_5 = None
        self.home_avg_pre_match_xg_total_last_20 = None
        self.away_avg_pre_match_xg_total_last_5 = None
        self.away_avg_pre_match_xg_total_last_20 = None

        self.home_match_load_per_day_last_10_days = None
        self.home_match_load_per_day_last_25_days = None
        self.away_match_load_per_day_last_10_days = None
        self.away_match_load_per_day_last_25_days = None

        self.home_avg_points_last_5 = None
        self.home_avg_points_last_20 = None
        self.away_avg_points_last_5 = None
        self.away_avg_points_last_20 = None

        self.home_avg_goals_last_5 = None
        self.home_avg_goals_last_20 = None
        self.away_avg_goals_last_5 = None
        self.away_avg_goals_last_20 = None

        self.home_avg_shots_on_target_last_5 = None
        self.home_avg_shots_on_target_last_20 = None
        self.away_avg_shots_on_target_last_5 = None
        self.away_avg_shots_on_target_last_20 = None

        self.home_avg_total_shots_last_5 = None
        self.home_avg_total_shots_last_20 = None
        self.away_avg_total_shots_last_5 = None
        self.away_avg_total_shots_last_20 = None

        self.home_avg_shots_inside_box_last_5 = None
        self.home_avg_shots_inside_box_last_20 = None
        self.away_avg_shots_inside_box_last_5 = None
        self.away_avg_shots_inside_box_last_20 = None

        self.home_avg_corner_kicks_last_5 = None
        self.home_avg_corner_kicks_last_20 = None
        self.away_avg_corner_kicks_last_5 = None
        self.away_avg_corner_kicks_last_20 = None

        self.home_avg_ball_possession_last_5 = None
        self.home_avg_ball_possession_last_20 = None
        self.away_avg_ball_possession_last_5 = None
        self.away_avg_ball_possession_last_20 = None

        self.home_avg_passes_acc_last_5 = None
        self.home_avg_passes_acc_last_20 = None
        self.away_avg_passes_acc_last_5 = None
        self.away_avg_passes_acc_last_20 = None

        self.home_curr_position = None
        self.away_curr_position = None

        self.home_avg_goals_scored_home_last_5 = 0
        self.home_avg_goals_scored_home_last_20 = 0
        self.away_avg_goals_scored_away_last_5 = 0
        self.away_avg_goals_scored_away_last_20 = 0

        self.home_avg_goals_conceded_home_last_5 = 0
        self.home_avg_goals_conceded_home_last_20 = 0
        self.away_avg_goals_conceded_away_last_5 = 0
        self.away_avg_goals_conceded_away_last_20 = 0

        self.home_team_strength = None
        self.away_team_strength = None

    @staticmethod
    def match_features_to_vector(match_features):
        # Convert the match_features object to a dictionary
        features_dict = vars(match_features)

        features = []
        for key, value in features_dict.items():
            # TODO: Pre-define list of categorical feature names
            if key not in ["home_team_id", "away_team_id", "comp_id", "home_team_strength", "away_team_strength"]:
                features.append(value)

        features = np.array(features)

        # Append team strength vectors
        # features = np.append(features, features_dict["home_team_strength"])
        # features = np.append(features, features_dict["away_team_strength"])

        return features
