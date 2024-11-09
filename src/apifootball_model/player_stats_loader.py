import os
import glob
import csv
import difflib
from datetime import datetime
from settings import CSV_CATEGORIES

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

    files_before_match = [(file, date) for (file, date) in csv_files_dates if date <= match_datetime]
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


def get_player_stats_for_team(team_lineup_info, match_datetime, directory_path):
    selected_csv = get_csv_file(match_datetime, directory_path)

    if not selected_csv:
        raise Exception(f"Unable to find CSV file corresponding to match played at {match_datetime}")

    # Loop over players in team roster
    stats = []
    all_ok = True

    for player_info in team_lineup_info:
        player_name, date_of_birth, rating_in_comp_season = player_info

        player_row = find_player_row(player_name, date_of_birth, selected_csv)

        if player_row is None:
            stats_dict = {category: [-1] * len(columns) for category, columns in CSV_CATEGORIES.items()}
            all_ok = False
            print(f"Failed to retrieve data from CSV for player {player_name} for matched played at {match_datetime}. "
                  f"Imputing...")

            # TODO: Implement imputing missing player stats (utilizing "rating_in_comp_season")
            stats.append(stats_dict)

        else:
            stats_dict = extract_stats(player_row)
            stats.append(stats_dict)

    return all_ok, stats


"""
player_name = 'Erling Haaland'
match_datetime_str = '2024-10-02'
directory_path = 'C:\\Users\\kutalekj\\PycharmProjects\\sofifa-web-scraper\\output\\test'

found, player_stats = get_player_stats(player_name, match_datetime_str, directory_path)

if found:
    print("Player stats found:")
    for category, values in player_stats.items():
        print(f"{category.capitalize()}: {values}")
else:
    print("Player not found or stats not available.")
    print(player_stats)
"""
