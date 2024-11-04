import os
import glob
import csv
import difflib
from datetime import datetime

categories = {
    "attacking": ["crossing", "finishing", "heading_accuracy", "short_passing", "volleys"],
    "skill": ["dribbling", "curve", "fk_accuracy", "long_passing", "ball_control"],
    "movement": ["acceleration", "sprint_speed", "agility", "reactions", "balance"],
    "power": ["shot_power", "jumping", "stamina", "strength", "long_shots"],
    "mentality": ["aggression", "interceptions", "positioning", "vision", "penalties", "composure"],
    "defending": ["defensive_awareness", "standing_tackle", "sliding_tackle"],
    "goalkeeping": ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]
}


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


def find_player_row(player_name, csv_file):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        players = []
        rows = []
        for row in reader:
            name = row['name']
            players.append(name)
            rows.append(row)
    matches = difflib.get_close_matches(player_name, players, n=1, cutoff=0.2)
    if matches:
        matched_name = matches[0]
        print(f'Name {player_name} matched to {matched_name}.')
        index = players.index(matched_name)
        player_row = rows[index]
        return player_row
    else:
        return None


def extract_stats(player_row):
    stats = {}
    for category, columns in categories.items():
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


def get_player_stats(player_name, match_datetime_str, directory_path):
    match_datetime = datetime.strptime(match_datetime_str, '%Y-%m-%d')
    selected_csv = get_csv_file(match_datetime, directory_path)
    if not selected_csv:
        stats_dict = {category: [-1]*len(columns) for category, columns in categories.items()}
        return False, stats_dict
    player_row = find_player_row(player_name, selected_csv)
    if not player_row:
        stats_dict = {category: [-1]*len(columns) for category, columns in categories.items()}
        return False, stats_dict
    else:
        stats = extract_stats(player_row)
        return True, stats


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
