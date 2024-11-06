"""
match.py
"""

import os
import csv
import http.client
import json
from dateutil.parser import parse
from datetime import datetime, timedelta
import numpy as np
from selenium.webdriver.common.devtools.v85.target import send_message_to_target

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
        self.hour = None
        self.month = None

        self.country = None
        self.comp = None
        self.season = None
        self.round = None

        self.home_team = None
        self.away_team = None

        self.winner_team_id = None
        self.home_team_goals = None
        self.away_team_goals = None
        self.home_team_points = None
        self.away_team_points = None

        self.home_team_shots_on_target = None
        self.away_team_shots_on_target = None

        self.home_elo_before_match_not_normalized = None
        self.away_elo_before_match_not_normalized = None

        self.features_before_match_played = None
        self.feature_vector_before_match_played = None

    def __eq__(self, other):
        if isinstance(other, Match):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    @staticmethod
    def get_new_matches_data_using_api(existing=None):
        global_instance = Global.get_instance()
        seasons = [x for x in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)]

        for comp in global_instance.all_comps:
            for season in seasons:

                if existing is not None and (comp.id, season) in existing:
                    existing_matches_sorted = sorted(
                        [x for x in global_instance.all_matches if x.comp.id == comp.id and x.season == season],
                        key=lambda match_: match_.datetime)
                    latest_match_datetime = existing_matches_sorted[-1].datetime
                    print(f"Latest existing match datetime for {comp.name} in {season} is {latest_match_datetime}")

                    request_string = "/fixtures?season=" + str(season) + "&league=" + str(comp.id) + \
                                     "&from=" + latest_match_datetime.strftime("%Y-%m-%d") + \
                                     "&to=" + datetime.today().strftime("%Y-%m-%d")
                else:
                    request_string = "/fixtures?season=" + str(season) + "&league=" + str(comp.id) + \
                                     "&from=" + str(settings.FIRST_SEASON) + "-01-01" + \
                                     "&to=" + datetime.today().strftime("%Y-%m-%d")

                # Get historic match data from API - up to current day
                conn = http.client.HTTPSConnection(settings.HOST)
                conn.request("GET", request_string, headers=settings.HEADERS)
                res = conn.getresponse()
                data = res.read()
                data_fixtures = json.loads(data)

                print(
                    f"{len(data_fixtures['response'])} matches were found in comp {comp.name} in season {str(season)}")

                # Loop over matches - get match info
                for fixture in data_fixtures['response']:

                    # Do not create new match instance if match already present - it would overwrite the instance
                    new_match_id = int(fixture['fixture']['id'])
                    if new_match_id in [x.id for x in global_instance.all_matches]:
                        print(f"Skipping match with ID={new_match_id} (already existing)")
                        continue

                    new_match = Match(new_match_id)

                    new_match.status = fixture['fixture']['status']['short']
                    if new_match.status not in ["FT", "AET", "PEN"]:

                        if new_match.status in ["Canc", "CANC"]:
                            print(
                                f"Canceled match found between {fixture['teams']['home']['name']} and"
                                f" {fixture['teams']['away']['name']} played at {fixture['fixture']['date']}")
                            continue

                        if new_match.status == "PST":
                            print(
                                f"Postponed match found between {fixture['teams']['home']['name']} and"
                                f" {fixture['teams']['away']['name']} played at {fixture['fixture']['date']}")
                            continue

                        if new_match.status == "NS":
                            print(
                                f"Match between {fixture['teams']['home']['name']} and"
                                f" {fixture['teams']['away']['name']} did not start yet"
                                f" (should be played at {fixture['fixture']['date']})")
                            continue

                        if new_match.status == "TBD":
                            print(
                                f"Match between {fixture['teams']['home']['name']} and"
                                f" {fixture['teams']['away']['name']} not scheduled yet - to be played")
                            continue

                        if new_match.status == "ABD":
                            print(
                                f"Match between {fixture['teams']['home']['name']} and"
                                f" {fixture['teams']['away']['name']} was abandoned for some reason")
                            continue

                        if new_match.status == "AWD":
                            print(
                                f"Match between {fixture['teams']['home']['name']} and"
                                f" {fixture['teams']['away']['name']} was not played - TechnicalLoss")
                            continue

                        if new_match.status == "WO":
                            print(
                                f"Match between {fixture['teams']['home']['name']} and"
                                f" {fixture['teams']['away']['name']} was not played - WalkOver")
                            continue

                        if new_match.status in ["1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"]:
                            print(
                                f"Match between {fixture['teams']['home']['name']} and"
                                f" {fixture['teams']['away']['name']} is now in play! Skipping...")
                            continue

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
                    home_team_id = int(fixture['teams']['home']['id'])

                    home_team = ut.get_team_if_exists(home_team_id)
                    if home_team is None:
                        print(f"\t\t\t\t\tFAILED to find a home team with ID {home_team_id} to assign a match.")
                        continue

                    new_match.home_team = home_team
                    home_team.matches.append(new_match)

                    # Away team
                    away_team_id = int(fixture['teams']['away']['id'])

                    away_team = ut.get_team_if_exists(away_team_id)
                    if away_team is None:
                        print(f"\t\t\t\t\tFAILED to find an away team with ID {home_team_id} to assign a match.")
                        continue

                    new_match.away_team = away_team
                    away_team.matches.append(new_match)

                    if bool(fixture['teams']['home']['winner']) and not bool(fixture['teams']['away']['winner']):
                        new_match.winner_team_id = new_match.home_team.id
                    elif not bool(fixture['teams']['home']['winner']) and bool(fixture['teams']['away']['winner']):
                        new_match.winner_team_id = new_match.away_team.id
                    else:
                        new_match.winner_team_id = settings.WINNER_TEAM_ID_CODE_FOR_DRAW

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

                    # Add new match to list
                    global_instance.all_matches.append(new_match)

                    # Delay so that limit of requests per minute is not exceeded
                    time.sleep(0.15)

    def get_stats_value(self, stats, stat_name, home_away):
        # Stats not present
        if len(stats) == 0:
            if home_away == "home" and self.round.is_regular:
                print(
                    f"Statistics [{stat_name}] estimated for a match between {self.home_team.name} "
                    f"and {self.away_team.name} played at {self.datetime}")

            # The value returned (used for features) is already normalized - denormalize it for now and round it
            return round(feature_ut.get_avg_shots_on_target_last_n(self, 5, home_away) * settings.SOG_NORM_COEFFICIENT)

        if len(stats) != 2:
            raise Exception(f"Fixture statistics response expected to contain info for exactly two matches."
                            f"Instead, {str(len(stat_name))} were found.")

        if stat_name == "Shots on Goal":

            # Home team
            if home_away == "home":
                for statistic in stats[0]['statistics']:
                    if statistic['type'] == stat_name:
                        return statistic['value'] if statistic['value'] is not None else -1

            # Away team
            elif home_away == "away":
                for statistic in stats[1]['statistics']:
                    if statistic['type'] == stat_name:
                        return statistic['value'] if statistic['value'] is not None else -1

            else:
                raise ValueError("The \"home_away\" parameter set to a wrong value.")

        else:
            raise ValueError(f"Unsupported statistic value found: {stat_name}")

    def calculate_relative_match_position_in_country_season(self, season):
        global_instance = Global.get_instance()

        start_date = global_instance.start_end_dates_per_country_season[self.country][season]['start']. \
            replace(tzinfo=self.datetime.tzinfo)
        end_date = global_instance.start_end_dates_per_country_season[self.country][season]['end']. \
            replace(tzinfo=self.datetime.tzinfo)

        if start_date <= self.datetime <= end_date:
            season_length = (end_date - start_date).days
            days_into_season = (self.datetime - start_date).days
            relative_position = days_into_season / season_length

            return relative_position

        # Case for matches finishing e.g. one day after the regular season end date
        elif start_date <= self.datetime <= end_date + timedelta(days=14):
            print(f"___WARNING: Found match between {self.home_team.name} and {self.away_team.name} played at "
                  f"{str(self.datetime)} not fitting into the expected timedelta range "
                  f"[{str(start_date)},{str(end_date)}] - too late")
            return settings.ALMOST_ONE

        # Case for matches finishing e.g. one day before the regular season start date
        elif start_date <= self.datetime + timedelta(days=14) <= end_date:
            print(f"___WARNING: Found match between {self.home_team.name} and {self.away_team.name} played at "
                  f"{str(self.datetime)} not fitting into the expected timedelta range "
                  f"[{str(start_date)},{str(end_date)}] - too early")
            return settings.ALMOST_ZERO

        return None

    def calculate_match_features(self):
        # Normalize season to [0,1]
        normalized_season = feature_ut.normalize_season(self.season)

        # Hour & month
        hours_sin = feature_ut.normalized_hour_month_cyclic(np.sin(2 * np.pi * self.hour / 24))
        hours_cos = feature_ut.normalized_hour_month_cyclic(np.cos(2 * np.pi * self.hour / 24))
        month_sin = feature_ut.normalized_hour_month_cyclic(np.sin(2 * np.pi * self.month / 12))
        month_cos = feature_ut.normalized_hour_month_cyclic(np.cos(2 * np.pi * self.month / 12))

        # Create new instance
        new_match_features = MatchFeatures(self.comp.id, normalized_season, self.home_team.id, self.away_team.id,
                                           hours_sin, hours_cos, month_sin, month_cos)

        # Elo
        (new_match_features.home_elo, new_match_features.away_elo) = feature_ut.calculate_elo_for_both_teams(self)

        # Relative match position
        new_match_features.relative_match_position_in_country_season = \
            self.calculate_relative_match_position_in_country_season(self.season)
        if new_match_features.relative_match_position_in_country_season is None:
            raise ValueError(f"Unable to get relative position of match in {str(self.season)} {self.comp.name}")

        # Other numerical features
        new_match_features.home_match_load_per_day_last_10_days = \
            1.0 - feature_ut.get_match_load_per_day_last_n(self, 10, "home")
        new_match_features.home_match_load_per_day_last_25_days = \
            1.0 - feature_ut.get_match_load_per_day_last_n(self, 25, "home")
        new_match_features.away_match_load_per_day_last_10_days = \
            1.0 - feature_ut.get_match_load_per_day_last_n(self, 10, "away")
        new_match_features.away_match_load_per_day_last_25_days = \
            1.0 - feature_ut.get_match_load_per_day_last_n(self, 25, "away")

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

        # Take complement (the idea is to have a larger number for a "better" value - many conceded goals is bad)
        new_match_features.home_avg_goals_conceded_home_last_5 = \
            1.0 - feature_ut.get_avg_goals_scored_conceded_home_or_away_last_n(self, 5, "home", "conceded")
        new_match_features.home_avg_goals_conceded_home_last_20 = \
            1.0 - feature_ut.get_avg_goals_scored_conceded_home_or_away_last_n(self, 20, "home", "conceded")
        new_match_features.away_avg_goals_conceded_away_last_5 = \
            1.0 - feature_ut.get_avg_goals_scored_conceded_home_or_away_last_n(self, 5, "away", "conceded")
        new_match_features.away_avg_goals_conceded_away_last_20 = \
            1.0 - feature_ut.get_avg_goals_scored_conceded_home_or_away_last_n(self, 20, "away", "conceded")

        # Current position in a table (if regular match, -1 otherwise)
        if self.round.is_regular:
            table = ut.get_table_by_comp_season(self.comp.id, self.season)
            new_match_features.home_curr_position = table.get_curr_team_position_in_season_up_to_date(self.home_team.id,
                                                                                                      self.datetime)
            new_match_features.away_curr_position = table.get_curr_team_position_in_season_up_to_date(self.away_team.id,
                                                                                                      self.datetime)
        else:
            new_match_features.home_curr_position = -1
            new_match_features.away_curr_position = -1

        # DEBUG PRINTS...
        if self.home_team.name == "Genk" or self.away_team.name == "Genk":
            if self.home_team.name == "Genk":
                print("\n\n\tFeatures before match:")
                print(f"ELO={new_match_features.home_elo}")
                print(
                    f"Relative match position in country season="
                    f"{new_match_features.relative_match_position_in_country_season}")
                print(
                    f"Match load last 10/25 days={new_match_features.home_match_load_per_day_last_10_days}/"
                    f"{new_match_features.home_match_load_per_day_last_25_days}")
                print(
                    f"Avg points last 5/20 matches={new_match_features.home_avg_points_last_5}/"
                    f"{new_match_features.home_avg_points_last_20}")
                print(
                    f"Avg goals last 5/20 matches={new_match_features.home_avg_goals_last_5}/"
                    f"{new_match_features.home_avg_goals_last_20}")
                print(
                    f"Avg shots on goal last 5/20 matches={new_match_features.home_avg_shots_on_target_last_5}/"
                    f"{new_match_features.home_avg_shots_on_target_last_20}")
                print(
                    f"Avg goals scored home last 5/20 matches={new_match_features.home_avg_goals_scored_home_last_5}/"
                    f"{new_match_features.home_avg_goals_scored_home_last_20}")
                print(
                    f"Avg goals conceded home last 5/20 matches="
                    f"{new_match_features.home_avg_goals_conceded_home_last_5}/"
                    f"{new_match_features.home_avg_goals_conceded_home_last_20}")
                print(f"Table position={new_match_features.home_curr_position}")
            elif self.away_team.name == "Genk":
                print("\n\n\tFeatures before match:")
                print(f"ELO={new_match_features.away_elo}")
                print(
                    f"Relative match position in country season="
                    f"{new_match_features.relative_match_position_in_country_season}")
                print(
                    f"Match load last 10/25 days={new_match_features.away_match_load_per_day_last_10_days}/"
                    f"{new_match_features.away_match_load_per_day_last_25_days}")
                print(
                    f"Avg points last 5/20 matches={new_match_features.away_avg_points_last_5}/"
                    f"{new_match_features.away_avg_points_last_20}")
                print(
                    f"Avg goals last 5/20 matches={new_match_features.away_avg_goals_last_5}/"
                    f"{new_match_features.away_avg_goals_last_20}")
                print(
                    f"Avg shots on goal last 5/20 matches={new_match_features.away_avg_shots_on_target_last_5}/"
                    f"{new_match_features.away_avg_shots_on_target_last_20}")
                print(
                    f"Avg goals scored away last 5/20 matches={new_match_features.away_avg_goals_scored_away_last_5}/"
                    f"{new_match_features.away_avg_goals_scored_away_last_20}")
                print(
                    f"Avg goals conceded away last 5/20 matches="
                    f"{new_match_features.away_avg_goals_conceded_away_last_5}/"
                    f"{new_match_features.away_avg_goals_conceded_away_last_20}")
                print(f"Table position={new_match_features.away_curr_position}")

            print("\n\tMATCH_STATISTICS:")
            print(f"{self.datetime}: {self.comp.name}, {self.season}, {self.round.name}")
            print(
                f"{self.home_team.name} {self.home_team_goals} ({self.home_team_shots_on_target}) - "
                f"{self.away_team.name} {self.away_team_goals} ({self.away_team_shots_on_target})")

        return new_match_features
