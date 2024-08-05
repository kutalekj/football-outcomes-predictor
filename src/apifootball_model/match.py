"""
match.py
"""

import os
import csv
import http.client
import json
import datetime
import settings
import utils as ut
from feature import MatchFeatures
import features_utils as feature_ut


class Match:
    def __init__(self, id_):
        self.id = id_
        self.status = None

        self.datetime = None
        self.hour = None  # feature
        self.month = None  # feature

        self.country = None
        self.comp_id = None  # feature
        self.comp_name = None
        self.season = None  # feature
        self.round = None  # feature

        self.home_team_id = None  # feature
        self.home_team_name = None
        self.away_team_id = None  # feature
        self.away_team_name = None

        self.winner_team_id = None
        self.home_team_goals = None  # feature src
        self.away_team_goals = None  # feature src
        self.home_team_points = None  # feature src
        self.away_team_points = None  # feature src

        self.home_team_elo = None  # feature
        self.away_team_elo = None  # feature

        self.home_team_shots_on_target = None  # feature src
        self.away_team_shots_on_target = None  # feature src

    @staticmethod
    def load_existing_matches():
        existing_matches = []

        # File exists, load existing matches
        if os.path.exists(settings.MATCHES_FILENAME):
            with open(settings.MATCHES_FILENAME, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    existing_matches.append(row)

        return existing_matches

    @staticmethod
    def get_matches_data_using_api(all_comps, from_season=None, from_round=None):
        matches = []

        seasons = [x for x in range(from_season, settings.LAST_SEASON + 1)] \
            if from_season is not None else [x for x in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)]

        first_round = from_round if from_round is not None else 1

        for comp in all_comps:
            for season in seasons:
                conn = http.client.HTTPSConnection(settings.HOST)

                request_string = "/fixtures?season=" + str(season) + "&league=" + str(comp.id)

                conn.request("GET", request_string, headers=settings.HEADERS)
                res = conn.getresponse()
                data = res.read()
                data_fixtures = json.loads(data)

                # Loop over matches
                for fixture in data_fixtures['response']:
                    new_match = Match(int(fixture['fixture']['id']))

                    new_match.status = fixture['fixture']['status']['short']
                    if new_match.status not in ["FT", "AET", "PEN"]:
                        print(f"WARNING: Match {new_match.id} not finished")  # TODO: Debug a OT/PEN match - how handle?

                    new_match.datetime = datetime.fromisoformat(fixture['fixture']['date'])
                    new_match.hour = int(new_match.datetime.hour)
                    new_match.month = int(new_match.datetime.month)

                    new_match.country = fixture['league']['country']

                    new_match.comp_id = comp.id
                    if int(fixture['league']['id']) != comp.id:
                        raise ValueError(
                            f"Comp ID found [{fixture['league']['id']}] not matching expected value {str(comp.id)}")

                    new_match.comp_name = comp.name
                    if fixture['league']['name'] != comp.name:
                        raise ValueError(
                            f"Comp name found [{fixture['league']['name']}] not matching expected value {comp.name}")

                    new_match.season = season
                    if int(fixture['league']['season']) != season:
                        raise ValueError(
                            f"Season found [{fixture['league']['season']}] not matching expected value {str(season)}")

                    new_match.round = comp.get_round_by_comp_season_round_name(season, fixture['league']['round'])
                    if new_match.round is None:
                        raise ValueError(f"Unable to get round for the match {str(new_match.id)}")

                    # If searching only for matches from a certain round and currently have a lower one, skip the match
                    if new_match.round.total_rank_all_time < first_round:
                        continue

                    new_match.home_team_id = int(fixture['teams']['home']['id'])
                    new_match.home_team_name = fixture['teams']['home']['name']

                    new_match.away_team_id = int(fixture['teams']['away']['id'])
                    new_match.away_team_name = fixture['teams']['away']['name']

                    if bool(fixture['teams']['home']['winner']) and not bool(fixture['teams']['away']['winner']):
                        new_match.winner_team_id = new_match.home_team_id
                    elif not bool(fixture['teams']['home']['winner']) and bool(fixture['teams']['away']['winner']):
                        new_match.winner_team_id = new_match.away_team_id
                    else:
                        new_match.winner_team_id = -1

                    # Matches that did not finish in regular time are skipped
                    if new_match.status != "FT":
                        print(
                            f"INFO: Match {new_match.id} between {new_match.home_team_name} and"
                            f"{new_match.away_team_name} played at {str(new_match.datetime)}"
                            f"did not finish in regular time.")
                        continue

                    new_match.home_team_goals = int(fixture['goals']['home'])
                    new_match.away_team_goals = int(fixture['goals']['away'])

                    if new_match.home_team_goals > new_match.away_team_goals:
                        new_match.home_team_points = 3
                        new_match.away_team_points = 0
                    elif new_match.home_team_goals < new_match.away_team_goals:
                        new_match.home_team_points = 0
                        new_match.away_team_points = 3
                    else:
                        new_match.home_team_points = 1
                        new_match.away_team_points = 1

                    new_match.home_team_elo = settings.INIT_ELO
                    new_match.away_team_elo = settings.INIT_ELO

                    # STATISTICS
                    stats_request_string = "/fixtures?statistics?fixture=" + str(new_match.id)
                    conn.request("GET", stats_request_string, headers=settings.HEADERS)
                    res = conn.getresponse()
                    data = res.read()
                    data_stats = json.loads(data)

                    new_match.home_team_shots_on_target = Match.get_stats_value(data_stats['response'], "Shots on Goal",
                                                                                "home")
                    new_match.away_team_shots_on_target = Match.get_stats_value(data_stats['response'], "Shots on Goal",
                                                                                "away")

                    # Calculate features vector
                    new_match.calculate_match_features()

                    # Add to list
                    matches.append(new_match)

        return matches

    @staticmethod
    def get_stats_value(stats, stat_name, home_away):
        if len(stat_name != 2):
            raise Exception(f"Fixture statistics response expected to contain info for exactly two matches."
                            f"Instead, {str(len(stat_name))} were found.")

        if stat_name == "Shots on Goal":

            # Home team
            if home_away == "home":
                for statistic in stats[0]['statistics']:
                    if statistic['type'] == stat_name:
                        return statistic['value']

            # Away team
            elif home_away == "away":
                for statistic in stats[1]['statistics']:
                    if statistic['type'] == stat_name:
                        return statistic['value']

            else:
                raise ValueError("The \"home_away\" parameter set to a wrong value.")

        else:
            raise ValueError(f"Unsupported statistic value found: {stat_name}")

    def calculate_match_features(self):
        new_match_features = MatchFeatures(self.comp_id, self.season, self.round, self.home_team_id, self.away_team_id)

        new_match_features.hours = self.hour
        new_match_features.month = self.month

        # Note that elo will be updated after training

        new_match_features.home_avg_points_last_5 = feature_ut.get_avg_points_last_n(self, 5, "home")
        new_match_features.home_avg_points_last_20 = feature_ut.get_avg_points_last_n(self, 20, "home")
        new_match_features.away_avg_points_last_5 = feature_ut.get_avg_points_last_n(self, 5, "away")
        new_match_features.away_avg_points_last_20 = feature_ut.get_avg_points_last_n(self, 20, "away")

        new_match_features.home_avg_goals_last_5 = feature_ut.get_avg_goals_last_n(self, 5, "home")
        new_match_features.home_avg_goals_last_20 = feature_ut.get_avg_goals_last_n(self, 20, "home")
        new_match_features.away_avg_goals_last_5 = feature_ut.get_avg_goals_last_n(self, 5, "away")
        new_match_features.away_avg_goals_last_20 = feature_ut.get_avg_goals_last_n(self, 20, "away")

        new_match_features.home_avg_shots_on_target_last_5 = feature_ut.get_avg_shots_on_target_last_n(self, 5, "home")
        new_match_features.home_avg_shots_on_target_last_20 = feature_ut.get_avg_shots_on_target_last_n(self, 20,
                                                                                                        "home")
        new_match_features.away_avg_shots_on_target_last_5 = feature_ut.get_avg_shots_on_target_last_n(self, 5, "away")
        new_match_features.away_avg_shots_on_target_last_20 = feature_ut.get_avg_shots_on_target_last_n(self, 20,
                                                                                                        "away")

        new_match_features.home_avg_goals_scored_home_last_5 = \
            feature_ut.get_avg_goals_scored_conceded_home_or_away_last_n(self, 5, "home", "scored")
        new_match_features.home_avg_goals_scored_home_last_20 = \
            feature_ut.get_avg_goals_scored_conceded_home_or_away_last_n(self, 20, "home", "scored")
        new_match_features.away_avg_goals_scored_away_last_5 = \
            feature_ut.get_avg_goals_scored_conceded_home_or_away_last_n(self, 5, "away", "scored")
        new_match_features.away_avg_goals_scored_away_last_20 = \
            feature_ut.get_avg_goals_scored_conceded_home_or_away_last_n(self, 20, "away", "scored")

        new_match_features.home_avg_goals_conceded_home_last_5 = \
            feature_ut.get_avg_goals_scored_conceded_home_or_away_last_n(self, 5, "home", "conceded")
        new_match_features.home_avg_goals_conceded_home_last_20 = \
            feature_ut.get_avg_goals_scored_conceded_home_or_away_last_n(self, 20, "home", "conceded")
        new_match_features.away_avg_goals_conceded_away_last_5 = \
            feature_ut.get_avg_goals_scored_conceded_home_or_away_last_n(self, 5, "away", "conceded")
        new_match_features.away_avg_goals_conceded_away_last_20 = \
            feature_ut.get_avg_goals_scored_conceded_home_or_away_last_n(self, 20, "away", "conceded")

        # Current position in a table
        table = ut.get_table_by_comp_season(new_match_features.comp_id, new_match_features.season)
        new_match_features.home_curr_position = \
            table.get_team_position_at_round(new_match_features.home_team_id, new_match_features.round)
        new_match_features.away_curr_position = \
            table.get_team_position_at_round(new_match_features.away_team_id, new_match_features.round)

        return MatchFeatures.match_features_to_vector(new_match_features)
