"""
feature.py
"""


# TODO: Relativize features
class MatchFeatures:
    def __init__(self, comp_id, season, round_, home_team_id, away_team_id):
        self.comp_id = comp_id
        self.season = season
        self.round = round_
        self.home_team_id = home_team_id
        self.away_team_id = away_team_id

        self.hours = None
        self.month = None

        self.home_elo = None
        self.away_elo = None

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

        self.home_avg_goals_scored_home_last_5 = None
        self.home_avg_goals_scored_home_last_20 = None
        self.away_avg_goals_scored_home_last_5 = None
        self.away_avg_goals_scored_home_last_20 = None

        self.home_avg_goals_conceded_home_last_5 = None
        self.home_avg_goals_conceded_home_last_20 = None
        self.away_avg_goals_conceded_home_last_5 = None
        self.away_avg_goals_conceded_home_last_20 = None

    @staticmethod
    def match_features_to_vector(match_features):
        # Get a dict of all attributes and their values
        attributes = vars(match_features)

        # Convert the values to a list
        vector = list(attributes.values())

        return vector
