"""
feature.py
"""


import numpy as np


class MatchFeatures:
    def __init__(self, comp_id_encoded, season, home_team_id_encoded, away_team_id_encoded,
                 hours_sin, hours_cos, month_sin, month_cos):
        self.comp_id = comp_id_encoded
        self.season = season

        self.home_team_id = home_team_id_encoded
        self.away_team_id = away_team_id_encoded

        self.hours_sin = hours_sin
        self.hours_cos = hours_cos
        self.month_sin = month_sin
        self.month_cos = month_cos

        self.home_elo = None
        self.away_elo = None

        self.relative_match_position = None

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

    # TODO: Once designing model architecture, consider adding Embedding layer to make the vector denser (less zeros)
    @staticmethod
    def match_features_to_vector(match_features):
        # Convert the match_features object to a dictionary
        features_dict = vars(match_features)

        # Separate out one-hot encoded features and numerical features
        one_hot_encoded_features = []
        numerical_features = []

        for key, value in features_dict.items():
            if isinstance(value, np.ndarray):
                # It's a one-hot encoded feature (since it's a NumPy array)
                one_hot_encoded_features.extend(value.flatten().tolist())
            else:
                # It's a numerical feature (or something else), add it to the list
                numerical_features.append(value)

        # Combine all features into one vector
        full_vector = numerical_features + one_hot_encoded_features

        return np.array(full_vector)
