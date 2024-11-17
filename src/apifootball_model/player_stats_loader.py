import os
import glob
import csv
import random
import difflib
import numpy as np
from datetime import datetime
from settings import CSV_CATEGORIES, CSV_PLAYERS_PATH
from globals import Global

NUM_FUZZY_MATCHES = 10
FUZZY_CUTOFF = 0.2


def get_csv_file(match_datetime, directory_path):
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    csv_files_dates = []

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)

        try:
            date_str = filename.replace('.csv', '')
            date = datetime.strptime(date_str, '%Y-%m-%d')
            csv_files_dates.append((csv_file, date))
        except ValueError:
            continue

    files_before_match = [(file, date) for (file, date) in csv_files_dates
                          if date.replace(tzinfo=match_datetime.tzinfo) <= match_datetime]
    if not files_before_match:
        return None

    selected_file = max(files_before_match, key=lambda x: x[1])[0]
    return selected_file


def find_player_row(full_player_name, date_of_birth, csv_file):
    # Open CSV with a corresponding historical version
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        players = []
        rows = []
        for row in reader:
            name = row['full_name']
            players.append(name)
            rows.append(row)

    # Get close matches of players' names
    # TODO: Maybe will need to require same birth dates for match - otherwise a match could be found always, ...
    # TODO: ...even in situation where it shouldn't have been (because the player is actually not in the 18k list)
    matches = difflib.get_close_matches(full_player_name, players, n=NUM_FUZZY_MATCHES, cutoff=FUZZY_CUTOFF)
    # TODO: Check if sorted by match probability (desc.)

    if matches:
        matched_name = matches[0]

        index = players.index(matched_name)
        player_row = rows[index]

        # If date of birth matches, return the most probable match immediately
        dob_csv = datetime.strptime(rows[index]['dob'], '%Y-%m-%d')
        if date_of_birth.date() == dob_csv.date():
            print(f"dob match")
            return player_row

        else:
            print(f'Name {full_player_name} matched to {matched_name}, but dates of birth not matching '
                  f'({date_of_birth} vs. {dob_csv})...')
            return player_row
            # TODO: Might need to implement more complex logic here

    else:
        return None


def extract_stats(player_row):
    stats = {}
    for category, columns in CSV_CATEGORIES.items():
        values = []

        for column in columns:
            value_str = player_row.get(column, '').strip()
            try:
                value = int(value_str)
            except (ValueError, TypeError):
                value = -1

            values.append(value)

        stats[category] = values

    return stats


def get_player_stats_for_team(team_lineup_info, team_rating_comp_season, curr_match, directory_path):
    selected_csv = get_csv_file(curr_match.datetime, directory_path)

    if not selected_csv:
        raise Exception(f"Unable to find CSV file corresponding to match played at {curr_match.datetime}")

    # Loop over players in team roster
    stats = []
    all_ok = True

    for player_info in team_lineup_info:
        player_name, date_of_birth, player_rating_comp_season, usual_position = player_info

        player_row = find_player_row(player_name, date_of_birth, selected_csv)

        if player_row is None:
            # stats_dict = {category: [-1] * len(columns) for category, columns in CSV_CATEGORIES.items()}
            # all_ok = False
            print(f"Failed to retrieve data from CSV for player {player_name} for matched played at "
                  f"{curr_match.datetime}. Imputing...")

            # Impute missing player stats
            stats_dict = estimate_player_stats(curr_match,
                                               player_rating_comp_season, usual_position, team_rating_comp_season)

            stats.append(stats_dict)

        else:
            stats_dict = extract_stats(player_row)
            stats.append(stats_dict)

    # return all_ok, stats
    print(f"Returning stats dict: {stats}")
    return stats


def estimate_player_stats(curr_match, player_rating_comp_season, usual_position, team_rating_comp_season):
    global_instance = Global.get_instance()

    # Define the acceptable rating difference
    rating_threshold = 0.5  # Adjust as needed

    # Mapping of positions to positions in data (if needed)
    # Assuming 'usual_position' directly matches player_stats['position']
    # If not, you may need to map 'Attacker' to 'F', 'Midfielder' to 'M', etc.

    # Get all teams in the same competition and season  # TODO: ???
    similar_players_stats = []
    comp_name = curr_match.comp.name
    season_str = str(curr_match.season)
    for team in global_instance.all_teams:
        # Check if team has stats for the same competition and season
        team_player_stats_comp = team.player_stats_comp_season.get(comp_name, {}).get(season_str, [])
        team_avg_rating = team.rating_comp_season.get(comp_name, {}).get(season_str, None)

        if not team_player_stats_comp or team_avg_rating is None:
            continue  # Skip teams without data

        # Calculate team rating difference
        team_rating_diff = abs(team_avg_rating - team_rating_comp_season)
        # You can consider using team_rating_diff if desired

        for player_stats in team_player_stats_comp:
            # Check if player's position matches the usual_position
            if player_stats['position'] == usual_position:
                player_rating = player_stats['rating']
                # Check if player's rating is within the acceptable range
                if abs(player_rating - player_rating_comp_season) <= rating_threshold:
                    # Try to find the player's stats from the CSV
                    player_name = player_stats['name']
                    date_of_birth = player_stats['birth_date']
                    selected_csv = get_csv_file(curr_match.datetime, CSV_PLAYERS_PATH)
                    if selected_csv:
                        player_row = find_player_row(player_name, date_of_birth, selected_csv)
                        if player_row:
                            stats_dict = extract_stats(player_row)
                            similar_players_stats.append(stats_dict)

    if not similar_players_stats:
        # If no similar players found, estimate stats based on average stats for the position
        print(f"No similar players found for position {usual_position} and rating {player_rating_comp_season}.")
        # Use default values or global averages
        # For each category, assign random values within a reasonable range
        estimated_stats = {}
        for category, columns in CSV_CATEGORIES.items():
            estimated_stats[category] = [random.randint(50, 70) for _ in columns]
        return estimated_stats

    # Aggregate stats from similar players
    estimated_stats = {}
    for category, columns in CSV_CATEGORIES.items():
        # Collect stats for this category from all similar players
        category_values = []
        for stats in similar_players_stats:
            values = stats.get(category, [])
            if values:
                category_values.append(values)
        if category_values:
            # Convert to numpy array for computation
            category_array = np.array(category_values)
            # Compute mean across similar players
            category_mean = np.mean(category_array, axis=0)
            # Convert to integer values
            category_mean_int = [int(round(val)) for val in category_mean]
            # Add some minor randomization
            category_estimated = [min(99, max(1, val + random.randint(-2, 2))) for val in category_mean_int]
            estimated_stats[category] = category_estimated
        else:
            # If no data available for this category, assign default values
            estimated_stats[category] = [random.randint(50, 70) for _ in columns]

    print(f"Returning estimated player stats {estimated_stats}")
    return estimated_stats
