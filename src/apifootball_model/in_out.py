import csv
import datetime
import os
import json
import re
from collections import defaultdict
import numpy as np
from dateutil.parser import parse as date_parse
from globals import Global
from match import Match
from feature import MatchFeatures
import utils as ut
import settings
import features_utils as feature_ut


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)


def store_matches(file_name):
    global_instance = Global.get_instance()

    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Writing the header
        writer.writerow([
            'id', 'status', 'datetime', 'hour', 'month', 'country', 'comp_id',
            'season', 'round_name', 'home_team_id', 'away_team_id',
            'home_team_goals', 'away_team_goals', 'home_team_points',
            'away_team_points', 'home_team_shots_on_target', 'away_team_shots_on_target', 'home_team_total_shots',
            'away_team_total_shots', 'home_team_shots_inside_box', 'away_team_shots_inside_box',
            'home_team_corner_kicks', 'away_team_corner_kicks', 'home_team_ball_possession',
            'away_team_ball_possession', 'home_team_passes_acc', 'away_team_passes_acc', 'winner_team_id',
            'home_team_lineup', 'away_team_lineup', 'home_fs_team_lineup', 'away_fs_team_lineup'
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
                match.home_team_total_shots,
                match.away_team_total_shots,
                match.home_team_shots_inside_box,
                match.away_team_shots_inside_box,
                match.home_team_corner_kicks,
                match.away_team_corner_kicks,
                match.home_team_ball_possession,
                match.away_team_ball_possession,
                match.home_team_passes_acc,
                match.away_team_passes_acc,
                match.winner_team_id,
                json.dumps(match.home_team_lineup),
                json.dumps(match.away_team_lineup),
                json.dumps(match.home_fs_team_lineup, cls=DateTimeEncoder),
                json.dumps(match.away_fs_team_lineup, cls=DateTimeEncoder)
            ])


def load_matches(file_name):
    global_instance = Global.get_instance()

    try:
        with open(file_name, mode='r', newline='', encoding='utf-8') as file:
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
                    raise Exception(f"Failed to find an away team with ID "
                                    f"{int(row['away_team_id'])} to assign a match.")

                match.away_team = away_team
                away_team.matches.append(match)

                match.home_team_goals = int(row['home_team_goals'])
                match.away_team_goals = int(row['away_team_goals'])
                match.home_team_points = int(row['home_team_points'])
                match.away_team_points = int(row['away_team_points'])
                match.home_team_shots_on_target = int(row['home_team_shots_on_target'])
                match.away_team_shots_on_target = int(row['away_team_shots_on_target'])
                match.home_team_total_shots = int(row['home_team_total_shots'])
                match.away_team_total_shots = int(row['away_team_total_shots'])
                match.home_team_shots_inside_box = int(row['home_team_shots_inside_box'])
                match.away_team_shots_inside_box = int(row['away_team_shots_inside_box'])
                match.home_team_corner_kicks = int(row['home_team_corner_kicks'])
                match.away_team_corner_kicks = int(row['away_team_corner_kicks'])
                match.home_team_ball_possession = float(row['home_team_ball_possession'])
                match.away_team_ball_possession = float(row['away_team_ball_possession'])
                match.home_team_passes_acc = float(row['home_team_passes_acc'])
                match.away_team_passes_acc = float(row['away_team_passes_acc'])

                match.winner_team_id = int(row['winner_team_id']) if row['winner_team_id'] else None

                # AF lineups
                home_lineup_json = row.get('home_team_lineup', '[]')
                away_lineup_json = row.get('away_team_lineup', '[]')

                match.home_team_lineup = [(int(player[0]), player[1], player[2]) for player
                                          in json.loads(home_lineup_json)] if home_lineup_json != "null" else []
                match.away_team_lineup = [(int(player[0]), player[1], player[2]) for player
                                          in json.loads(away_lineup_json)] if away_lineup_json != "null" else []

                # FS lineups
                home_fs_lineup_json = row.get('home_fs_team_lineup', '[]')
                away_fs_lineup_json = row.get('away_fs_team_lineup', '[]')

                match.home_fs_team_lineup = [{
                    'fs_id': int(player['fs_id']),
                    'fs_comp_id': int(player['fs_comp_id']),
                    'fs_full_name': player['fs_full_name'],
                    'fs_known_as': player['fs_known_as'],
                    'fs_birthday': date_parse(player['fs_birthday']),
                    'fs_age': int(player['fs_age']),
                    'fs_weight': int(player['fs_weight']),
                    'fs_height': int(player['fs_height']),
                    'fs_league': player['fs_league'],
                    'fs_league_type': player['fs_league_type'],
                    'fs_club_team_id': int(player['fs_club_team_id']),
                    'fs_club_team_2_id': int(player['fs_club_team_2_id']),
                    'fs_position': player['fs_position'],
                    'fs_nationality': player['fs_nationality']
                } for player in json.loads(home_fs_lineup_json)] if home_fs_lineup_json != "null" else []

                match.away_fs_team_lineup = [{
                    'fs_id': int(player['fs_id']),
                    'fs_comp_id': int(player['fs_comp_id']),
                    'fs_full_name': player['fs_full_name'],
                    'fs_known_as': player['fs_known_as'],
                    'fs_birthday': date_parse(player['fs_birthday']),
                    'fs_age': int(player['fs_age']),
                    'fs_weight': int(player['fs_weight']),
                    'fs_height': int(player['fs_height']),
                    'fs_league': player['fs_league'],
                    'fs_league_type': player['fs_league_type'],
                    'fs_club_team_id': int(player['fs_club_team_id']),
                    'fs_club_team_2_id': int(player['fs_club_team_2_id']),
                    'fs_position': player['fs_position'],
                    'fs_nationality': player['fs_nationality']
                } for player in json.loads(away_fs_lineup_json)] if away_fs_lineup_json != "null" else []

                # Add the match to the global instance
                global_instance.all_matches.append(match)

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found. Please check the file name and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_player_stats():
    global_instance = Global.get_instance()

    csv_files = [f for f in os.listdir(settings.CSV_PLAYERS_PATH) if f.endswith('.csv')]  # get CSV files files
    files_with_dates = []  # list to store tuples of (datetime, filename)

    # Extract dates from filenames and sort them
    for filename in csv_files:
        date_str = os.path.splitext(filename)[0]
        file_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')  # assuming filename format 'YYYY-MM-DD.csv'
        files_with_dates.append((file_date, filename))

    files_with_dates.sort()  # sort CSV files by date (asc.)

    attributes = [
        ('player_id', int),
        ('version', 'date'),
        ('name', str),
        ('full_name', str),
        ('height_cm', int),
        ('weight_kg', int),
        ('dob', 'date'),
        ('positions', 'list'),
        ('overall_rating', int),
        ('potential', int),
        ('value', str),
        ('club_id', int),
        ('club_name', str),
        ('club_league_id', int),
        ('club_league_name', str),
        ('club_rating', int),
        ('club_kit_number', int),
        ('club_joined', 'date'),
        ('country_id', int),
        ('country_name', str)
    ]

    player_id_to_names = {}  # mapping from player_id to (name, full_name)

    # Process each CSV file
    for index, (file_date, filename) in enumerate(files_with_dates):
        file_path = os.path.join(settings.CSV_PLAYERS_PATH, filename)
        players_dict = {}

        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)

            num_players_skipped_this_csv = 0

            # Read each player row
            for row_num, row in enumerate(reader, start=2):  # start=2 to account for header

                # Inconsistency between number of cells in a row and the CSV header
                if len(row) != len(headers):
                    # print(f"Row {row_num} in file '{filename}' is misaligned.")

                    # Realign the row (pad it with empty strings to match the header length)
                    missing_cells = len(headers) - len(row)
                    index_to_insert = 24  # insert empty strings  to this position (here usually values missing)
                    if missing_cells > 0:
                        row[index_to_insert:index_to_insert] = [''] * missing_cells

                row_dict = dict(zip(headers, row))  # dict mapping headers to row values

                skip_player = False
                player_data = {}  # init player data dict

                # Load player attributes
                for attr_name, attr_type in attributes:
                    raw_value = row_dict.get(attr_name, '').strip()

                    # Attribute value missing
                    # TODO: Now, only the following four attributes and then the 34 skills attributes are utilized...
                    # TODO: ...meaning that the others can be set as empty value if missing for no harm. But, ...
                    # TODO: ...if will want to work with some of them in the future as well, then this needs changes.
                    if raw_value == '':
                        if attr_name in ['player_id', 'name', 'full_name', 'dob']:  # These can't be missing!
                            raise ValueError(f"Attributes player_id, name, full_name and dob cannot be missing! "
                                             f"(file {filename}, row {row_num})")
                        elif attr_type == int:
                            player_data[attr_name] = 0
                        elif attr_type == str:
                            player_data[attr_name] = ''
                        elif attr_type == 'date':
                            player_data[attr_name] = None
                        elif attr_type == 'list':
                            player_data[attr_name] = []
                        else:
                            player_data[attr_name] = None

                    else:
                        try:
                            if attr_type == int:
                                player_data[attr_name] = int(raw_value)
                            elif attr_type == str:
                                player_data[attr_name] = raw_value
                            elif attr_type == 'date':
                                player_data[attr_name] = datetime.datetime.strptime(raw_value, '%Y-%m-%d')
                            elif attr_type == 'list':
                                player_data[attr_name] = [pos.strip() for pos in raw_value.split(',')]
                            else:
                                player_data[attr_name] = raw_value
                        except Exception:
                            if attr_name in ['player_id', 'name', 'full_name', 'dob']:  # These can't be missing!
                                raise ValueError(f"Attributes player_id, name, full_name and dob must be parsed! "
                                                 f"(file {filename}, row {row_num})")
                            elif attr_type == int:
                                player_data[attr_name] = 0
                            elif attr_type == str:
                                player_data[attr_name] = ''
                            elif attr_type == 'date':
                                player_data[attr_name] = None
                            elif attr_type == 'list':
                                player_data[attr_name] = []
                            else:
                                player_data[attr_name] = None

                # Load player skills
                num_missing_values = 0  # Exp. variable for checking how many rows are missing max five skills values...
                for skill_attr in settings.PLAYER_SKILLS:
                    raw_value = row_dict.get(skill_attr, '').strip()

                    # Skill value missing
                    if raw_value == '':
                        # print(f"Missing value for '{skill_attr}' in file '{filename}', row {row_num}. "
                        #       f"SKIPPING PLAYER")
                        if num_missing_values <= settings.MAX_MISSING_SF_SKILL_VALUES_ALLOWED:
                            player_data[skill_attr] = -1
                            num_missing_values += 1
                        else:
                            skip_player = True
                            break
                    else:
                        try:
                            player_data[skill_attr] = int(raw_value)
                        except ValueError:  # This exception has never occurred so far...
                            # print(f"Invalid integer for '{skill_attr}' in file '{filename}', row {row_num}."
                            #       f"SKIPPING PLAYER")
                            if num_missing_values <= settings.MAX_MISSING_SF_SKILL_VALUES_ALLOWED:
                                player_data[skill_attr] = -1
                                num_missing_values += 1
                            else:
                                skip_player = True
                                break

                if skip_player:  # skip this player if invalid player skill found
                    # print(f"\tSKIPPING AN INVALID PLAYER (file {filename}, row {row_num})")
                    num_players_skipped_this_csv += 1
                    continue

                # Use 'player_id' as the key for player_index_dict
                player_id = player_data['player_id']

                # Update player occurrences dict
                if player_id not in global_instance.sofifa_player_index_dict:
                    global_instance.sofifa_player_index_dict[player_id] = []
                    player_id_to_names[player_id] = set()  # init name set for this player_id

                global_instance.sofifa_player_index_dict[player_id].append((index, file_date))

                # Update name set for the player_id
                name = player_data['name'].split(sep=' - FIFA')[0] if \
                    "FIFA" in player_data['name'] else player_data['name'].split(sep=' -')[0]
                full_name = player_data['full_name']
                player_id_to_names[player_id].add((name, full_name))

                # Add player data to players_dict
                players_dict[player_id] = player_data

                # Update players_by_dob dict
                dob = player_data['dob']
                if dob not in global_instance.sofifa_players_by_dob:
                    global_instance.sofifa_players_by_dob[dob] = []

                # Check if player_id is already in the list for this dob (assertion of possible name inconsistencies)
                if player_id not in {player[0] for player in global_instance.sofifa_players_by_dob[dob]}:
                    global_instance.sofifa_players_by_dob[dob].append((player_id, name, full_name))

            # TODO: Debug print
            print(f"{num_players_skipped_this_csv} player rows were skipped for this CSV ({filename}) "
                  f"because of missing at least {str(settings.MAX_MISSING_SF_SKILL_VALUES_ALLOWED)} skill values")

        # Append to list_of_data
        global_instance.sofifa_players_data.append((file_date, players_dict))

    global_instance.sofifa_player_index_dict = dict(sorted(global_instance.sofifa_player_index_dict.items()))
    global_instance.sofifa_players_by_dob = dict(sorted(global_instance.sofifa_players_by_dob.items()))


def load_avg_gk_skills():
    global_instance = Global.get_instance()

    global_instance.sf_avg_gk_skills = defaultdict(lambda: defaultdict(dict))  # skill -> (team_id, season) -> value
    global_instance.sf_default_gk_skills = {}  # default averages for each skill

    current_skill = None

    with open(settings.AVR_GK_SKILLS, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            # Detect skill section
            if line.startswith("Mean GK"):
                current_skill = line.split(" ")[2].lower()  # get skill name

            elif current_skill and line:

                # Parse team data: "(team_id, 'team_name', season): value"
                match = re.match(r"\((\d+|\-1), '([^']*)', (\d{4})\): ([\d.]+)", line)
                if match:
                    team_id, team_name, season, avg_value = match.groups()
                    team_id, season = int(team_id), int(season)
                    avg_value = float(avg_value)

                    if team_id == -1:  # default value for the season
                        global_instance.sf_default_gk_skills[current_skill] = avg_value
                    else:
                        global_instance.sf_avg_gk_skills[current_skill][(team_id, season)] = avg_value


def load_avg_team_strength_scaled():
    global_instance = Global.get_instance()

    global_instance.sf_avg_team_strength = {}
    global_instance.sf_default_team_strength = []

    with open(settings.AVG_TEAM_STRENGTHS, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # Parse the skill data
            match = re.match(r"\((\d+|-\d+), '([^']*)', (-?\d+)\):\s+(.*)", line)
            if not match:
                raise ValueError("Invalid line (non-matching) found in the average/default team strength data")

            team_id, team_name, season, skill_values = match.groups()
            team_id, season = int(team_id), int(season)

            if "nan" in skill_values:
                raise ValueError(f"NaN values found in average/default team strength data "
                                 f"(for team {team_name} and season {season})")

            # Parse skill values into a list of floats
            skill_values_list = [
                float(val) for val in skill_values.split()
            ]

            if len(skill_values_list) != 4 * len(settings.CSV_CATEGORIES):
                raise ValueError(f"An average team strength scaled vector of unexpected length found when loading: "
                                 f"{skill_values_list}")

            if team_id == -1 and season == -1:  # Default skill vector
                global_instance.sf_default_team_strength = skill_values_list
            else:
                if (team_id, season) not in global_instance.sf_avg_team_strength:
                    global_instance.sf_avg_team_strength[(team_id, season)] = []
                global_instance.sf_avg_team_strength[(team_id, season)] = skill_values_list
