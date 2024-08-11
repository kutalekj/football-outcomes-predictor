"""
match.py
"""

import os
import csv
import http.client
import json
from dateutil.parser import parse
import settings
import utils as ut
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
    def get_new_matches_data_using_api(all_comps, from_season=None, from_round=None):
        global_instance = Global.get_instance()

        seasons = [x for x in range(from_season, settings.LAST_SEASON + 1)] \
            if from_season is not None else [x for x in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)]
        seasons = [2021]  # TODO: Temporary

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

                    # STATISTICS
                    stats_request_string = "/fixtures/statistics?fixture=" + str(new_match.id)
                    conn.request("GET", stats_request_string, headers=settings.HEADERS)
                    res = conn.getresponse()
                    data = res.read()
                    data_stats = json.loads(data)['response']

                    # TODO: Debug
                    if new_match.home_team_id == 42 or new_match.away_team_id == 42:
                        print(
                            f"Round {str(new_match.round.total_rank_all_time)} (called [{str(new_match.round.name)}])\t\t\t{str(new_match.datetime.day)}.{str(new_match.datetime.month)}. {str(new_match.datetime.year)}")

                    new_match.home_team_shots_on_target = Match.get_stats_value(data_stats, "Shots on Goal", "home")
                    new_match.away_team_shots_on_target = Match.get_stats_value(data_stats, "Shots on Goal", "away")

                    # Calculate features
                    new_match.features_before_match_played = new_match.calculate_match_features()
                    new_match.feature_vector_before_match_played = MatchFeatures.match_features_to_vector(
                        new_match.features_before_match_played)

                    # Add to list TODO: Add check that this new match is not already in existing matches (all_matches)
                    global_instance.all_matches.append(new_match)

            # Once having matches and their teams, get the information which rounds consist of all teams and which not
            comp.init_teams_involved_in_rounds()

    @staticmethod
    def get_stats_value(stats, stat_name, home_away):
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

    # TODO: DIFFERENT ROUNDS ORDERING FOR EACH MATCH MUST BE SET!!! Each team might play rounds in different order
    # TODO: 1. Init comps and tables and get matches (do not calculate round ranks or table pos. or features)
    # TODO: 2. Sort all matches in global_instance.all_matches by datetime asc
    # TODO: 3. Each match should add pointer to a previous match for both teams (or to a next one as well?)...
    # TODO: ...(maybe to both a  previous match and a previous regular match) and calculate round ranks for both teams
    # TODO: 4. After round correctly ranked for each team, calculate table position, and features, for each match
    # TODO: Then, when a new match comes some day, it is appended to the sorted matches list all_matches, connected...
    # TODO: ...to a previous match for both teams involved and his round ranks are calculated based on its previous...
    # TODO: ...matches. Finally, features before match played are calculated for it and it is passed to model training.
    # TODO: Note that if bidirectional matches connection, do not forget to update the neighbours pointers too each time
    # TODO: Fortunately, if doing this correctly, getting prev and next matches should be easy, without loops
    # TODO: Note that maybe the round name should be added to the Match ctor so that each match is uniquely initialized
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
        new_match_features = MatchFeatures(self.comp.id, self.season, self.round.regular_rank_in_season,
                                           self.home_team_id, self.away_team_id)

        new_match_features.hours = self.hour
        new_match_features.month = self.month

        (new_match_features.home_elo, new_match_features.away_elo) = \
            feature_ut.calculate_elo_for_both_teams(self)

        # TODO: Debug
        if new_match_features.home_team_id == 42:
            print(f"Elo after match {self.round.regular_rank_in_season - 1} = " + str(new_match_features.home_elo))
        if new_match_features.away_team_id == 42:
            print(f"Elo after match {self.round.regular_rank_in_season - 1} = " + str(new_match_features.away_elo))

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
        table = ut.get_table_by_comp_season(self.comp.id, self.season)
        new_match_features.home_curr_position = table.get_curr_team_position_in_season_at_round(self.home_team_id,
                                                                                                self.round)
        new_match_features.away_curr_position = table.get_curr_team_position_in_season_at_round(self.away_team_id,
                                                                                                self.round)

        return new_match_features
