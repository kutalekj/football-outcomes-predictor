"""
match.py
"""

import os
import csv
import http.client
import json
from dateutil.parser import parse
from datetime import datetime
import settings
import utils as ut
import time
from feature import MatchFeatures
import features_utils as feature_ut
from globals import Global


class Match:
    def __init__(self, id_):
        self.id = id_
        self.status = None

        self.datetime = None
        self.hour = None  # feature
        self.month = None  # feature

        self.country = None
        self.comp = None  # feature src
        self.season = None  # feature
        self.round = None  # feature src

        self.home_team_id = None  # feature
        self.home_team_name = None
        self.away_team_id = None  # feature
        self.away_team_name = None

        self.winner_team_id = None
        self.home_team_goals = None  # feature src
        self.away_team_goals = None  # feature src
        self.home_team_points = None  # feature src
        self.away_team_points = None  # feature src

        self.home_team_shots_on_target = None  # feature src
        self.away_team_shots_on_target = None  # feature src

        self.features_before_match_played = None
        self.feature_vector_before_match_played = None

    def __eq__(self, other):
        if isinstance(other, Match):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

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
    def get_new_matches_data_using_api(from_season=None, from_date=None):
        global_instance = Global.get_instance()

        seasons = [x for x in range(from_season, settings.LAST_SEASON + 1)] \
            if from_season is not None else [x for x in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)]

        for comp in global_instance.all_comps:
            for season in seasons:

                # Get historic match data from API - up to current day
                conn = http.client.HTTPSConnection(settings.HOST)

                request_string = "/fixtures?season=" + str(season) + "&league=" + str(comp.id) + \
                                 "&from=" + str(settings.FIRST_SEASON) + "-01-01" + \
                                 "&to=" + datetime.today().strftime("%Y-%m-%d") \
                                 if from_date is None else \
                                 "/fixtures?season=" + str(season) + "&league=" + str(comp.id) + \
                                 "&from=" + from_date.strftime("%Y-%m-%d") + \
                                 "&to=" + datetime.today().strftime("%Y-%m-%d")

                conn.request("GET", request_string, headers=settings.HEADERS)
                res = conn.getresponse()
                data = res.read()
                data_fixtures = json.loads(data)

                print(
                    f"{len(data_fixtures['response'])} matches were found in comp {comp.name} in season {str(season)}")

                # Loop over matches - get match info
                for fixture in data_fixtures['response']:
                    new_match = Match(int(fixture['fixture']['id']))

                    new_match.status = fixture['fixture']['status']['short']
                    if new_match.status not in ["FT", "AET", "PEN"]:
                        if new_match.status in ["Canc", "CANC"]:
                            print(
                                f"Canceled match found between {fixture['teams']['home']['name']} and {fixture['teams']['away']['name']} played at {fixture['fixture']['date']}")
                            continue

                        if new_match.status == "PST":
                            print(
                                f"Postponed match found between {fixture['teams']['home']['name']} and {fixture['teams']['away']['name']} played at {fixture['fixture']['date']}")
                            continue

                        if new_match.status == "NS":
                            print(
                                f"Match between {fixture['teams']['home']['name']} and {fixture['teams']['away']['name']} did not start yet (should be played at {fixture['fixture']['date']})")
                            continue

                        if new_match.status == "TBD":
                            print(
                                f"Match between {fixture['teams']['home']['name']} and {fixture['teams']['away']['name']} not scheduled yet - to be played")
                            continue

                        print(f"WARNING: Match {new_match.id} not finished")

                    new_match.datetime = parse(fixture['fixture']['date'])
                    new_match.hour = int(new_match.datetime.hour)
                    new_match.month = int(new_match.datetime.month)

                    new_match.country = fixture['league']['country']

                    new_match.comp = comp
                    if int(fixture['league']['id']) != comp.id:
                        raise ValueError(
                            f"Comp ID found [{fixture['league']['id']}] not matching expected value {str(comp.id)}")

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

                    # Home team
                    new_match.home_team_id = int(fixture['teams']['home']['id'])
                    new_match.home_team_name = fixture['teams']['home']['name']

                    home_team = ut.get_team_if_exists(new_match.home_team_id)
                    if home_team is None:
                        raise Exception(f"Failed to find a home team {new_match.home_team_name} to assign a match.")
                    home_team.matches.append(new_match)

                    # Away team
                    new_match.away_team_id = int(fixture['teams']['away']['id'])
                    new_match.away_team_name = fixture['teams']['away']['name']

                    away_team = ut.get_team_if_exists(new_match.away_team_id)
                    if away_team is None:
                        raise Exception(f"Failed to find a home team {new_match.away_team_name} to assign a match.")
                    away_team.matches.append(new_match)

                    if bool(fixture['teams']['home']['winner']) and not bool(fixture['teams']['away']['winner']):
                        new_match.winner_team_id = new_match.home_team_id
                    elif not bool(fixture['teams']['home']['winner']) and bool(fixture['teams']['away']['winner']):
                        new_match.winner_team_id = new_match.away_team_id
                    else:
                        new_match.winner_team_id = -1

                    new_match.home_team_goals = int(fixture['score']['fulltime']['home'])
                    new_match.away_team_goals = int(fixture['score']['fulltime']['away'])

                    if new_match.home_team_goals > new_match.away_team_goals:
                        new_match.home_team_points = 3
                        new_match.away_team_points = 0
                    elif new_match.home_team_goals < new_match.away_team_goals:
                        new_match.home_team_points = 0
                        new_match.away_team_points = 3
                    else:
                        new_match.home_team_points = 1
                        new_match.away_team_points = 1

                    # Statistics
                    stats_request_string = "/fixtures/statistics?fixture=" + str(new_match.id)
                    conn.request("GET", stats_request_string, headers=settings.HEADERS)
                    res = conn.getresponse()
                    data = res.read()
                    data_stats = json.loads(data)['response']

                    new_match.home_team_shots_on_target = new_match.get_stats_value(data_stats, "Shots on Goal", "home")
                    new_match.away_team_shots_on_target = new_match.get_stats_value(data_stats, "Shots on Goal", "away")

                    # Add match to list
                    # TODO: Add check that this new match is not already in existing matches (all_matches)
                    global_instance.all_matches.append(new_match)

                    # Delay so that limit of requests per minute is not exceeded
                    time.sleep(0.1)

    def get_stats_value(self, stats, stat_name, home_away):
        # Stats not present
        if len(stats) == 0:
            if home_away == "home":
                print(
                    f"Statistics [{stat_name}] missing for a match between {self.home_team_name} and {self.away_team_name} played at {self.datetime}")
                # TODO: Add debug count for missing statistics for each regular team - for knowing how many missing
            return -1

        if len(stats) != 2:
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

    # TODO: Some round matches are postponed and played after following round at sometime in the future
    # TODO: Calculate match features after loading all matches? Or is features calculation invariant to this?
    # TODO: Debug ELO to find this out. Check rounds orderings
    # TODO: Try to call endpoint for matches not only by comp and season, but also by rounds (three loops, not two)?

    # TODO: Primarily, matches should be sorted by datetime when played, not rounds
    # TODO: Rounds might be played in different order for each team
    # TODO: So, getting previous/next match should not depend on round number - should get new match of a team by date!

    # TODO: DIFFERENT ROUNDS ORDERING FOR EACH TEAM MUST BE SET!!! Each team might play rounds in different order
    # TODO: 1. Init comps and tables and get matches (do not calculate round ranks or table pos. or features)
    # TODO: 2. Sort all matches in global_instance.all_matches by datetime asc
    # TODO: 3. Each match should add pointer to a previous match for both teams (or to a next one as well?)...
    # TODO: 3C. No no, here, distribute matches between teams
    # TODO: ...(maybe to both a  previous match and a previous regular match) and calculate round ranks for both teams
    # TODO: 4C, No no, round ranking not needed to specify - it corresponds to the rank (regular) of match in the list
    # TODO: 4. After round correctly ranked for each team, calculate table position, and features, for each match
    # TODO: Then, when a new match comes some day, it is appended to the sorted matches list all_matches, connected...
    # TODO: ...to a previous match for both teams involved and his round ranks are calculated based on its previous...
    # TODO: ...matches. Finally, features before match played are calculated for it and it is passed to model training.
    # TODO: Note that if bidirectional matches connection, do not forget to update the neighbours pointers too each time
    # TODO: Fortunately, if doing this correctly, getting prev and next matches should be easy, without loops
    # TODO: Note that maybe the round name should be added to the Match ctor so that each match is uniquely initialized
    # TODO: No no, this is ment for features, not matches, and it is not needed.
    # TODO: Note that after implementing get_match_by_comp_season_round_team following this scheme, the...
    # TODO: ...get_teams_involved method for each round will not be needed...
    # TODO: ...(it was used for knowing which round is played by all comp season teams) :)
    # TODO: Note that if will need method get_all_matches_of_round the matches (only the already played ones) might...
    # TODO: ...by found by comp, season and round name (instead of rank) - but risking that some matches not played yet
    # TODO: Add-note - the method get_all_matches_of_round was needed for get_teams_involved - so not needed now :)
    # TODO: Note that similarly if will need method get_all_regular_matches_in_season_up_to_round the matches...
    # TODO: ...(again, only the already played ones - create flag for round_finished?) might be found by iterating...
    # TODO: ...to the past over previous matches of season comp for each team (and then set(list())) - but similarly,...
    # TODO: ...risking that some matches were not played. This method will be probably needed in some shape, because...
    # TODO: ...it is needed for table re-calculation
    def calculate_match_features(self):
        new_match_features = MatchFeatures(self.comp.id, self.season, self.round.name, self.home_team_id,
                                           self.away_team_id)

        new_match_features.hours = self.hour
        new_match_features.month = self.month

        (new_match_features.home_elo, new_match_features.away_elo) = \
            feature_ut.calculate_elo_for_both_teams(self)

        new_match_features.home_match_load_per_day_last_10_days = feature_ut.get_match_load_per_day_last_n(self, 10,
                                                                                                           "home")
        new_match_features.home_match_load_per_day_last_25_days = feature_ut.get_match_load_per_day_last_n(self, 25,
                                                                                                           "home")
        new_match_features.away_match_load_per_day_last_10_days = feature_ut.get_match_load_per_day_last_n(self, 10,
                                                                                                           "away")
        new_match_features.away_match_load_per_day_last_25_days = feature_ut.get_match_load_per_day_last_n(self, 25,
                                                                                                           "away")

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

        # Current position in a table (if regular match, -1 otherwise)
        if self.round.is_regular:
            table = ut.get_table_by_comp_season(self.comp.id, self.season)
            new_match_features.home_curr_position = table.get_curr_team_position_in_season_up_to_date(self.home_team_id,
                                                                                                      self.round)
            new_match_features.away_curr_position = table.get_curr_team_position_in_season_up_to_date(self.away_team_id,
                                                                                                      self.round)
        else:
            new_match_features.home_curr_position = -1
            new_match_features.away_curr_position = -1

        return new_match_features
