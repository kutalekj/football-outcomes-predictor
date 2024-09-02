import csv
import datetime
import os
import numpy as np
from dateutil.parser import parse as date_parse
from globals import Global
from match import Match
from feature import MatchFeatures
import utils as ut
import settings
import features_utils as feature_ut


def store_matches(file_name):
    global_instance = Global.get_instance()

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Writing the header
        writer.writerow([
            'id', 'status', 'datetime', 'hour', 'month', 'country', 'comp_id',
            'season', 'round_name', 'home_team_id', 'away_team_id',
            'home_team_goals', 'away_team_goals', 'home_team_points',
            'away_team_points', 'home_team_shots_on_target', 'away_team_shots_on_target',
            'home_elo_before_match_not_normalized', 'away_elo_before_match_not_normalized',
            'relative_position_in_comp_season', 'winner_team_id',
            'features_before_match_played'
        ])

        for match in global_instance.all_matches:
            writer.writerow([
                match.id,
                match.status,
                match.datetime.isoformat(),
                match.hour,
                match.month,
                match.country,
                match.comp.id,
                match.season,
                match.round.name,
                match.home_team.id,
                match.away_team.id,
                match.home_team_goals,
                match.away_team_goals,
                match.home_team_points,
                match.away_team_points,
                match.home_team_shots_on_target,
                match.away_team_shots_on_target,
                match.relative_position_in_comp_season,
                match.winner_team_id,
                repr(match.features_before_match_played.__dict__)  # Store all features
            ])


def load_matches(file_name):
    global_instance = Global.get_instance()

    try:
        with open(file_name, mode='r', newline='') as file:
            reader = csv.DictReader(file)

            for row in reader:
                match = Match(int(row['id']))
                match.status = row['status']
                match.datetime = date_parse(row['datetime'])
                match.hour = int(row['hour'])
                match.month = int(row['month'])

                match.country = row['country']
                match.comp = next((comp for comp in global_instance.all_comps if comp.id == int(row['comp_id'])), None)
                match.season = int(row['season'])
                match.round = match.comp.get_round_by_comp_season_round_name(match.season, row['round_name'])

                # Home team
                home_team = ut.get_team_if_exists(int(row['home_team_id']))
                if home_team is None:
                    raise Exception(f"Failed to find a home team with ID {int(row['home_team_id'])} to assign a match.")

                match.home_team = home_team
                home_team.matches.append(match)

                # Away team
                away_team = ut.get_team_if_exists(int(row['away_team_id']))
                if away_team is None:
                    raise Exception(f"Failed to find an away team with ID {int(row['away_team_id'])} to assign a match.")

                match.away_team = away_team
                away_team.matches.append(match)

                match.home_team_goals = int(row['home_team_goals'])
                match.away_team_goals = int(row['away_team_goals'])
                match.home_team_points = int(row['home_team_points'])
                match.away_team_points = int(row['away_team_points'])
                match.home_team_shots_on_target = int(row['home_team_shots_on_target'])
                match.away_team_shots_on_target = int(row['away_team_shots_on_target'])

                match.relative_position_in_comp_season = float(row['relative_position_in_comp_season'])
                match.winner_team_id = int(row['winner_team_id']) if row['winner_team_id'] else None

                # Add the match to the global instance
                global_instance.all_matches.append(match)

    except FileNotFoundError as e:
        print(f"Error: The file '{file_name}' was not found. Please check the file name and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
