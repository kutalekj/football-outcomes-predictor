import csv
import json
import datetime
import numpy as np
from team import Team
from comp import Comp
from rounds import Round
from match import Match
from feature import MatchFeatures
from globals import Global
import settings


def convert_datetime_in_lineup(lineup_list):
    """
    Recursively converts all datetime objects in the lineup dictionary to ISO strings.
    lineup_list is e.g. match.home_fs_team_lineup or away_fs_team_lineup.
    """
    for item in lineup_list:
        for k, v in item.items():
            if isinstance(v, datetime.datetime):
                item[k] = v.isoformat()  # convert to ISO string
    return lineup_list


def store_comps(comps_csv_path):
    global_instance = Global.get_instance()

    # 1) Identify all Comp objects from global_instance.all_matches
    #    or however else you collect them
    unique_comps = set()
    for match in global_instance.all_matches:
        if match.comp:
            unique_comps.add(match.comp)

    fieldnames = [
        'comp_id', 'name', 'country',
        # Store complex fields as JSON
        'rounds_per_season_json',
        'all_rounds_sorted_ids_json',
        'regular_round_keywords_json',
        'teams_per_season_json',
        'fs_teams_per_season_json',
        'start_end_dates_per_season_json'
    ]

    with open(comps_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for comp in unique_comps:
            row = {}
            row['comp_id'] = comp.id
            row['name'] = comp.name
            row['country'] = comp.country if comp.country else ''

            # Convert complex lists to JSON. Inside them, refer to Rounds/Teams by ID, not object
            # For example, "rounds_per_season" might contain Round instances.
            # We'll store them by round.name or an explicit round_id if you define it.
            rps_json = []
            for item in comp.rounds_per_season:
                # item is e.g. {'season': int, 'rounds': list of Round}
                # Convert each Round object to ID references, or store round fields
                try:
                    rounds_id_list = [r.name for r in item['rounds']]  # or define a unique round_id
                except:
                    rounds_id_list = [r for r in item['rounds']]
                rps_json.append({
                    'season': item['season'],
                    'rounds': rounds_id_list
                })
            row['rounds_per_season_json'] = json.dumps(rps_json)

            # all_rounds_sorted is a list of Round instances
            all_rounds_ids = [r.name for r in comp.all_rounds_sorted]
            row['all_rounds_sorted_ids_json'] = json.dumps(all_rounds_ids)

            row['regular_round_keywords_json'] = json.dumps(comp.regular_round_keywords)

            # similarly handle teams_per_season, fs_teams_per_season, etc.
            row['teams_per_season_json'] = json.dumps(comp.teams_per_season, default=str)
            row['fs_teams_per_season_json'] = json.dumps(comp.fs_teams_per_season, default=str)
            row['start_end_dates_per_season_json'] = json.dumps(comp.start_end_dates_per_season, default=str)

            writer.writerow(row)

    print(f"Stored {len(unique_comps)} comps in {comps_csv_path}")


def store_rounds(rounds_csv_path):
    global_instance = Global.get_instance()

    unique_rounds = set()
    for match in global_instance.all_matches:
        if match.round:
            unique_rounds.add(match.round)

    fieldnames = [
        'round_id', 'comp_id', 'comp_name', 'season',
        'name', 'is_regular'
    ]

    with open(rounds_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in unique_rounds:
            row = {}
            # define a unique round_id, e.g. f"{r.comp_id}_{r.season}_{r.name}"
            round_id = f"{r.comp_id}_{r.season}_{r.name}"
            row['round_id'] = round_id
            row['comp_id'] = r.comp_id
            row['comp_name'] = r.comp_name
            row['season'] = r.season
            row['name'] = r.name
            row['is_regular'] = (str(r.is_regular).lower() if isinstance(r.is_regular, bool) else '')
            writer.writerow(row)

    print(f"Stored {len(unique_rounds)} rounds in {rounds_csv_path}")


def store_teams(teams_csv_path):
    global_instance = Global.get_instance()

    unique_teams = set()
    for match in global_instance.all_matches:
        if match.home_team:
            unique_teams.add(match.home_team)
        if match.away_team:
            unique_teams.add(match.away_team)

    fieldnames = [
        'team_id', 'name', 'fs_id', 'fs_clean_name',
        'regularity_in_comp_season_json',
        'players_in_regular_comp_season_json',
        'avg_gk_diving_json',
        'avg_gk_handling_json',
        'avg_gk_kicking_json',
        'avg_gk_positioning_json',
        'avg_gk_reflexes_json'
    ]

    with open(teams_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in unique_teams:
            row = {}
            row['team_id'] = t.id
            row['name'] = t.name
            row['fs_id'] = t.fs_id if t.fs_id else ''
            row['fs_clean_name'] = t.fs_clean_name if t.fs_clean_name else ''

            row['regularity_in_comp_season_json'] = json.dumps(t.regularity_in_comp_season, default=str)
            row['players_in_regular_comp_season_json'] = json.dumps(t.players_in_regular_comp_season, default=str)
            row['avg_gk_diving_json'] = json.dumps(t.avg_gk_diving)
            row['avg_gk_handling_json'] = json.dumps(t.avg_gk_handling)
            row['avg_gk_kicking_json'] = json.dumps(t.avg_gk_kicking)
            row['avg_gk_positioning_json'] = json.dumps(t.avg_gk_positioning)
            row['avg_gk_reflexes_json'] = json.dumps(t.avg_gk_reflexes)

            writer.writerow(row)

    print(f"Stored {len(unique_teams)} teams in {teams_csv_path}")


def store_matches(matches_csv_path):
    """
    Store all Match objects in global_instance.all_matches to matches.csv,
    referencing comp_id, round_id, home_team_id, away_team_id, plus nested data as JSON.
    """
    global_instance = Global.get_instance()

    fieldnames = [
        'id', 'status',
        'datetime', 'hour', 'month', 'country',
        'comp_id',       # reference
        'round_id',      # reference
        'season',
        'home_team_id',  # reference
        'away_team_id',  # reference
        'home_team_lineup_json',
        'away_team_lineup_json',
        'home_fs_team_lineup_json',
        'away_fs_team_lineup_json',
        'winner_team_id',
        'home_team_goals',
        'away_team_goals',
        'home_team_points',
        'away_team_points',
        'home_team_xg',
        'away_team_xg',
        'total_xg',
        'home_team_pre_match_xg',
        'away_team_pre_match_xg',
        'total_pre_match_xg',
        'home_team_shots_on_target',
        'away_team_shots_on_target',
        'home_team_total_shots',
        'away_team_total_shots',
        'home_team_shots_inside_box',
        'away_team_shots_inside_box',
        'home_team_corner_kicks',
        'away_team_corner_kicks',
        'home_team_ball_possession',
        'away_team_ball_possession',
        'home_team_passes_acc',
        'away_team_passes_acc',
        'home_elo_before_match_not_normalized',
        'away_elo_before_match_not_normalized',
        # We'll store MatchFeatures fully as JSON
        'features_before_match_played_json',
        'feature_vector_before_match_played_json'
    ]

    with open(matches_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for match in global_instance.all_matches:
            row = {}
            row['id'] = match.id
            row['status'] = match.status or ''
            row['datetime'] = match.datetime.isoformat() if match.datetime else ''
            row['hour'] = match.hour if match.hour is not None else ''
            row['month'] = match.month if match.month is not None else ''
            row['country'] = match.country or ''
            row['comp_id'] = match.comp.id if match.comp else ''
            round_id = ''
            if match.round:
                round_id = f"{match.round.comp_id}_{match.round.season}_{match.round.name}"
            row['round_id'] = round_id
            row['season'] = match.season if match.season is not None else ''
            row['home_team_id'] = match.home_team.id if match.home_team else ''
            row['away_team_id'] = match.away_team.id if match.away_team else ''

            row['home_team_lineup_json'] = json.dumps(match.home_team_lineup or [])
            row['away_team_lineup_json'] = json.dumps(match.away_team_lineup or [])

            home_fs_lineup_prepared = convert_datetime_in_lineup(match.home_fs_team_lineup or [])
            away_fs_lineup_prepared = convert_datetime_in_lineup(match.away_fs_team_lineup or [])

            row['home_fs_team_lineup_json'] = json.dumps(home_fs_lineup_prepared)
            row['away_fs_team_lineup_json'] = json.dumps(away_fs_lineup_prepared)

            row['winner_team_id'] = match.winner_team_id if match.winner_team_id is not None else ''
            row['home_team_goals'] = match.home_team_goals if match.home_team_goals is not None else ''
            row['away_team_goals'] = match.away_team_goals if match.away_team_goals is not None else ''
            row['home_team_points'] = match.home_team_points if match.home_team_points is not None else ''
            row['away_team_points'] = match.away_team_points if match.away_team_points is not None else ''
            row['home_team_xg'] = match.home_team_xg
            row['away_team_xg'] = match.away_team_xg
            row['total_xg'] = match.total_xg
            row['home_team_pre_match_xg'] = match.home_team_pre_match_xg
            row['away_team_pre_match_xg'] = match.away_team_pre_match_xg
            row['total_pre_match_xg'] = match.total_pre_match_xg

            row['home_team_shots_on_target'] = match.home_team_shots_on_target if match.home_team_shots_on_target is not None else ''
            row['away_team_shots_on_target'] = match.away_team_shots_on_target if match.away_team_shots_on_target is not None else ''
            row['home_team_total_shots'] = match.home_team_total_shots if match.home_team_total_shots is not None else ''
            row['away_team_total_shots'] = match.away_team_total_shots if match.away_team_total_shots is not None else ''
            row['home_team_shots_inside_box'] = match.home_team_shots_inside_box if match.home_team_shots_inside_box is not None else ''
            row['away_team_shots_inside_box'] = match.away_team_shots_inside_box if match.away_team_shots_inside_box is not None else ''
            row['home_team_corner_kicks'] = match.home_team_corner_kicks if match.home_team_corner_kicks is not None else ''
            row['away_team_corner_kicks'] = match.away_team_corner_kicks if match.away_team_corner_kicks is not None else ''
            row['home_team_ball_possession'] = match.home_team_ball_possession if match.home_team_ball_possession is not None else ''
            row['away_team_ball_possession'] = match.away_team_ball_possession if match.away_team_ball_possession is not None else ''
            row['home_team_passes_acc'] = match.home_team_passes_acc if match.home_team_passes_acc is not None else ''
            row['away_team_passes_acc'] = match.away_team_passes_acc if match.away_team_passes_acc is not None else ''
            row['home_elo_before_match_not_normalized'] = (match.home_elo_before_match_not_normalized
                                                           if match.home_elo_before_match_not_normalized is not None
                                                           else '')
            row['away_elo_before_match_not_normalized'] = (match.away_elo_before_match_not_normalized
                                                           if match.away_elo_before_match_not_normalized is not None
                                                           else '')

            # MatchFeatures
            if match.features_before_match_played:
                features_dict = vars(match.features_before_match_played)
                row['features_before_match_played_json'] = json.dumps(features_dict, default=str)
            else:
                row['features_before_match_played_json'] = '{}'

            # Feature Vector as JSON (convert np.array to list)
            if match.feature_vector_before_match_played is not None:
                row['feature_vector_before_match_played_json'] = json.dumps(
                    match.feature_vector_before_match_played.tolist()
                )
            else:
                row['feature_vector_before_match_played_json'] = '[]'

            writer.writerow(row)

    print(f"Stored {len(global_instance.all_matches)} matches in {matches_csv_path}")


def load_comps(comps_csv_path):
    global_instance = Global.get_instance()

    dict_of_comps = {}
    with open(comps_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            comp_id = int(row['comp_id'])
            c = Comp(comp_id, row['name'], [])
            c.country = row['country'] or ''

            # parse JSON
            c.rounds_per_season = json.loads(row['rounds_per_season_json'])
            # We'll fill c.all_rounds_sorted later with Round objects
            c.all_rounds_sorted = []
            c.regular_round_keywords = json.loads(row['regular_round_keywords_json'])
            c.teams_per_season = json.loads(row['teams_per_season_json'])
            c.fs_teams_per_season = json.loads(row['fs_teams_per_season_json'])
            c.start_end_dates_per_season = json.loads(row['start_end_dates_per_season_json'])

            dict_of_comps[c.id] = c

    global_instance._dict_of_comps = dict_of_comps  # store for next steps
    print(f"Loaded {len(dict_of_comps)} comps from {comps_csv_path}")


def load_rounds(rounds_csv_path):
    global_instance = Global.get_instance()

    dict_of_rounds = {}
    with open(rounds_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # reconstruct round
            round_id = row['round_id']  # unique round_id
            comp_id = int(row['comp_id']) if row['comp_id'] else 0
            r = Round(comp_id, row['comp_name'], int(row['season']), row['name'])
            r.is_regular = (row['is_regular'] == 'true')
            dict_of_rounds[round_id] = r

    global_instance._dict_of_rounds = dict_of_rounds

    # Now link them to comp
    for rid, rnd in dict_of_rounds.items():
        comp_obj = global_instance._dict_of_comps.get(rnd.comp_id)
        if comp_obj:
            comp_obj.all_rounds_sorted.append(rnd)

    print(f"Loaded {len(dict_of_rounds)} rounds from {rounds_csv_path} and linked to comps.")


def load_teams(teams_csv_path):
    global_instance = Global.get_instance()

    dict_of_teams = {}
    with open(teams_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            team_id = int(row['team_id'])
            t = Team(team_id, row['name'])
            t.fs_id = int(row['fs_id']) if row['fs_id'] else None
            t.fs_clean_name = row['fs_clean_name'] or None
            t.regularity_in_comp_season = json.loads(row['regularity_in_comp_season_json'])
            t.players_in_regular_comp_season = json.loads(row['players_in_regular_comp_season_json'])
            t.avg_gk_diving = json.loads(row['avg_gk_diving_json'])
            t.avg_gk_handling = json.loads(row['avg_gk_handling_json'])
            t.avg_gk_kicking = json.loads(row['avg_gk_kicking_json'])
            t.avg_gk_positioning = json.loads(row['avg_gk_positioning_json'])
            t.avg_gk_reflexes = json.loads(row['avg_gk_reflexes_json'])

            dict_of_teams[t.id] = t

    global_instance._dict_of_teams = dict_of_teams
    print(f"Loaded {len(dict_of_teams)} teams from {teams_csv_path}")


def load_matches(matches_csv_path):
    global_instance = Global.get_instance()

    all_matches = []
    with open(matches_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = Match(int(row['id']))
            m.status = row['status'] or ''
            if row['datetime']:
                m.datetime = datetime.datetime.fromisoformat(row['datetime'])
            m.hour = int(row['hour']) if row['hour'] else None
            m.month = int(row['month']) if row['month'] else None
            m.country = row['country'] or ''
            m.season = int(row['season']) if row['season'] else None

            # Link comp
            if row['comp_id']:
                comp_id = int(row['comp_id'])
                m.comp = global_instance._dict_of_comps.get(comp_id)

            # Link round
            round_id = row['round_id']
            if round_id and round_id in global_instance._dict_of_rounds:
                m.round = global_instance._dict_of_rounds[round_id]

            # Link teams
            if row['home_team_id']:
                ht_id = int(row['home_team_id'])
                m.home_team = global_instance._dict_of_teams.get(ht_id)
            if row['away_team_id']:
                aw_id = int(row['away_team_id'])
                m.away_team = global_instance._dict_of_teams.get(aw_id)

            # lineups
            m.home_team_lineup = json.loads(row['home_team_lineup_json'] or '[]')
            m.away_team_lineup = json.loads(row['away_team_lineup_json'] or '[]')
            m.home_fs_team_lineup = json.loads(row['home_fs_team_lineup_json'] or '[]')
            m.away_fs_team_lineup = json.loads(row['away_fs_team_lineup_json'] or '[]')

            m.winner_team_id = int(row['winner_team_id']) if row['winner_team_id'] else None
            m.home_team_goals = int(row['home_team_goals']) if row['home_team_goals'] else None
            m.away_team_goals = int(row['away_team_goals']) if row['away_team_goals'] else None
            m.home_team_points = int(row['home_team_points']) if row['home_team_points'] else None
            m.away_team_points = int(row['away_team_points']) if row['away_team_points'] else None

            m.home_team_xg = float(row['home_team_xg']) if row['home_team_xg'] else -1
            m.away_team_xg = float(row['away_team_xg']) if row['away_team_xg'] else -1
            m.total_xg = float(row['total_xg']) if row['total_xg'] else -1
            m.home_team_pre_match_xg = float(row['home_team_pre_match_xg']) if row['home_team_pre_match_xg'] else -1
            m.away_team_pre_match_xg = float(row['away_team_pre_match_xg']) if row['away_team_pre_match_xg'] else -1
            m.total_pre_match_xg = float(row['total_pre_match_xg']) if row['total_pre_match_xg'] else -1

            m.home_team_shots_on_target = int(row['home_team_shots_on_target']) if row[
                'home_team_shots_on_target'] else None
            m.away_team_shots_on_target = int(row['away_team_shots_on_target']) if row[
                'away_team_shots_on_target'] else None
            m.home_team_total_shots = int(row['home_team_total_shots']) if row['home_team_total_shots'] else None
            m.away_team_total_shots = int(row['away_team_total_shots']) if row['away_team_total_shots'] else None
            m.home_team_shots_inside_box = int(row['home_team_shots_inside_box']) if row[
                'home_team_shots_inside_box'] else None
            m.away_team_shots_inside_box = int(row['away_team_shots_inside_box']) if row[
                'away_team_shots_inside_box'] else None
            m.home_team_corner_kicks = int(row['home_team_corner_kicks']) if row['home_team_corner_kicks'] else None
            m.away_team_corner_kicks = int(row['away_team_corner_kicks']) if row['away_team_corner_kicks'] else None

            m.home_team_ball_possession = float(row['home_team_ball_possession']) if row[
                'home_team_ball_possession'] else None
            m.away_team_ball_possession = float(row['away_team_ball_possession']) if row[
                'away_team_ball_possession'] else None
            m.home_team_passes_acc = float(row['home_team_passes_acc']) if row['home_team_passes_acc'] else None
            m.away_team_passes_acc = float(row['away_team_passes_acc']) if row['away_team_passes_acc'] else None

            m.home_elo_before_match_not_normalized = float(row['home_elo_before_match_not_normalized']) if row[
                'home_elo_before_match_not_normalized'] else None
            m.away_elo_before_match_not_normalized = float(row['away_elo_before_match_not_normalized']) if row[
                'away_elo_before_match_not_normalized'] else None

            # MatchFeatures
            feat_json = row['features_before_match_played_json']
            if feat_json and feat_json != '{}':
                feat_data = json.loads(feat_json)
                # Reconstruct the MatchFeatures
                mf = MatchFeatures(
                    feat_data.get('comp_id', 0),
                    feat_data.get('season', 0),
                    feat_data.get('home_team_id', 0),
                    feat_data.get('away_team_id', 0),
                    feat_data.get('hours_sin', 0),
                    feat_data.get('hours_cos', 0),
                    feat_data.get('month_sin', 0),
                    feat_data.get('month_cos', 0)
                )
                # Fill all other keys
                for k, v in feat_data.items():
                    setattr(mf, k, v)
                m.features_before_match_played = mf

            # Feature Vector (JSON -> np.array)
            fv_json = row.get('feature_vector_before_match_played_json', '[]')
            if fv_json and fv_json != '[]':
                fv_list = json.loads(fv_json)
                m.feature_vector_before_match_played = np.array(fv_list, dtype=float)
            else:
                m.feature_vector_before_match_played = None

            all_matches.append(m)

    # Link matches back to the teams
    for match_obj in all_matches:
        if match_obj.home_team:
            match_obj.home_team.matches.append(match_obj)
        if match_obj.away_team:
            match_obj.away_team.matches.append(match_obj)

    global_instance.all_matches = all_matches
    print(f"Loaded {len(all_matches)} matches from {matches_csv_path}")


def store_all_matches_data():
    store_comps(settings.ALL_LS_COMPS_CSV)
    store_rounds(settings.ALL_LS_ROUNDS_CSV)
    store_teams(settings.ALL_LS_TEAMS_CSV)
    store_matches(settings.ALL_LS_MATCHES_CSV)


def load_all_matches_data():
    # 1) load comps
    load_comps(settings.ALL_LS_COMPS_CSV)
    # 2) load rounds (which references comps)
    load_rounds(settings.ALL_LS_ROUNDS_CSV)
    # 3) load teams
    load_teams(settings.ALL_LS_TEAMS_CSV)
    # 4) load matches (which references comps, rounds, and teams)
    load_matches(settings.ALL_LS_MATCHES_CSV)
