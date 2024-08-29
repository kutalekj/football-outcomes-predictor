"""
feature.py
"""


class MatchFeatures:
    def __init__(self, comp_id, season, relative_match_position, home_team_id_encoded, away_team_id_encoded):
        self.comp_id = comp_id
        self.season = season
        self.relative_match_position = relative_match_position

        self.home_team_id = home_team_id_encoded
        self.away_team_id = away_team_id_encoded

        self.hours_sin = None
        self.hours_cos = None
        self.month_sin = None
        self.month_cos = None

        self.home_elo = None
        self.away_elo = None

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

    @staticmethod
    def match_features_to_vector(match_features):
        # Get a dict of all attributes and their values
        attributes = vars(match_features)

        # Convert the values to a list
        vector = list(attributes.values())

        return vector
