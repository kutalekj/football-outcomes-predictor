import csv
import datetime
import os

from football_outcomes.config import settings
from football_outcomes.config.globals import Global
from football_outcomes.data import io as in_out

# from football_outcomes.data import io_mega as in_out_mega
from football_outcomes.data.comp import Comp
from football_outcomes.data.match import Match
from football_outcomes.data.season_comp_table import SeasonCompTable
from football_outcomes.data.state_io import export_summary_csvs, load_global_state, save_global_state
from football_outcomes.features.feature import MatchFeatures

# from football_outcomes.training.train_rnn import train
# from football_outcomes.training.train_ann import train
from football_outcomes.utils import common as utils

# import random
# import statistics

# import numpy as np


# from football_outcomes.training.train_compID_encoder import train
# from football_outcomes.training.train_teamID_encoder import train
# from football_outcomes.training.train_team_strength import train
ut = utils

global_instance = Global.get_instance()

if not settings.ALL_LOAD:

    # 0. Load average skills and team strengths (SF)
    in_out.load_sf_avg_team_strength()

    # 1. Init comps (seasons, teams, AF rounds, FS matches)
    Comp.get_fs_leagues_list()

    for comp in settings.COMPS_v2:
        new_comp = Comp(comp["id"], comp["name"], comp["regular_round_keywords"])
        new_comp.init_teams_in_comp()
        new_comp.init_all_rounds()

        global_instance.all_comps.append(new_comp)
    global_instance.all_teams = sorted(global_instance.all_teams, key=lambda team_: team_.id)

    # 2. Init country start/end dates, comp season tables and default GK skills
    for comp in global_instance.all_comps:
        for season in [x for x in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)]:

            # Country start/end dates
            if comp.country not in global_instance.start_end_dates_per_country_season:
                global_instance.start_end_dates_per_country_season[comp.country] = {}

            if season not in global_instance.start_end_dates_per_country_season[comp.country]:
                global_instance.start_end_dates_per_country_season[comp.country][season] = {
                    "start": datetime.datetime.max,
                    "end": datetime.datetime.min,
                }

            if len(comp.regular_round_keywords) == 0:
                continue  # omit the cups - do not create tables for them

            new_table = SeasonCompTable(comp.id, comp.name, season)
            new_table.init_teams_in_season_comp()

            global_instance.all_tables.append(new_table)

    for comp in global_instance.all_comps:
        comp.init_country_start_end_dates_in_seasons()

    # 3a. Load match data from local
    if settings.MATCH_DATA_LOAD:
        in_out.load_matches(settings.M_LOAD_CSV)
    all_loaded_comp_seasons = list(set([(x.comp.id, x.season) for x in global_instance.all_matches]))

    # 3b. Get new match data from API
    Match.get_new_matches_data_using_api(existing=all_loaded_comp_seasons)

    # 4. Correct (set) team regularity and match AF teams with FS teams
    for team in global_instance.all_teams:
        team.matches = sorted(team.matches, key=lambda match_: match_.datetime)  # sort team matches by datetime (asc.)
        team.correct_team_regularity_and_match_af_fs_teams()  # assume each team plays exactly in one reg. comp season!

    # 5a. Exclude irregular teams from table calculations
    SeasonCompTable.exclude_irregular_teams_from_table_calc()

    # 5b. Get FS players from all teams for each comp season (represented by table)
    SeasonCompTable.get_fs_player_rosters_per_regular_comp_season_team()

    # 6. Load individual player stats from sofifa CSV files
    in_out.load_player_stats()

    # 7. Calculate features for each match (must be done chronologically asc.!)
    global_instance.all_matches = sorted(global_instance.all_matches, key=lambda match_: match_.datetime)
    for match in global_instance.all_matches:

        # DEBUG
        if match.home_team.name == "Genk" or match.away_team.name == "Genk":
            stop_here = True

        # Match AF/FS match lineups
        ut.get_fs_match_lineups(match)  # match players in AF match lineup with those in teams' FS comp season roster

        # Get xG match stats (FS)
        if (
            match.round.is_regular
            and match.datetime > settings.GET_XG_IF_MATCH_DATE_NEWER_THAN.replace(tzinfo=match.datetime.tzinfo)
            and (match.total_xg == -1 and match.total_pre_match_xg == -1)
        ):
            ut.get_fs_match_xg(match)

        # Feature calculation
        match.features_before_match_played = match.calculate_match_features()
        match.feature_vector_before_match_played = MatchFeatures.match_features_to_vector(
            match.features_before_match_played
        )
else:
    # in_out_mega.load_all_matches_data()

    load_path = settings.PROCESSED_DIR / "test1.fop"
    load_global_state(load_path)

# 8a. Store matches to local
if settings.MATCH_DATA_STORE:
    in_out.store_matches(settings.M_STORE_CSV)

# 8b. Store all data to local
if settings.ALL_STORE:
    # in_out_mega.store_all_matches_data()

    save_path = settings.PROCESSED_DIR / "test1_full.fop"
    snapshot_path = save_global_state(save_path)
    print(f"Saved snapshot to: {snapshot_path}")

    export_summary_csvs()

# MISSING PLAYERS CHECKING
output_dir = r"C:\Users\kutalekj\Downloads"
os.makedirs(output_dir, exist_ok=True)

print(id(global_instance))

# --- Dump first list (tuples of 2) - unchanged ---
mp2_path = os.path.join(output_dir, "mp2_AF_FS.csv")
with open(mp2_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(getattr(global_instance, "mp2_AF_FS_players_matching_potential_misses_couples", []))

# --- Dump second (dict[comp_id][season] -> list of tuples) ---
mp5_path = os.path.join(output_dir, "mp5_DOB.csv")
with open(mp5_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    mp5_dict = getattr(global_instance, "mp5_DOB_misses_couples", {})
    writer.writerow(
        [
            "comp_id",
            "season",
            "match_datetime",
            "home_team",
            "away_team",
            "fs_known_as",
            "fs_birthday",
        ]
    )
    for comp_id in sorted(mp5_dict.keys()):
        seasons_map = mp5_dict[comp_id]
        for season in sorted(seasons_map.keys()):
            for tup in seasons_map[season]:
                writer.writerow([comp_id, season, *tup])

# --- Dump third (dict[comp_id][season] -> list of tuples) ---
mp6_path = os.path.join(output_dir, "mp6_FS_SF.csv")
with open(mp6_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    mp6_dict = getattr(global_instance, "mp6_FS_SF_players_matching_potential_misses_couples", {})
    # CSV header is optional; if you want one, uncomment:
    # writer.writerow(["comp_id", "season", "fs_name", "sf_name", "score"])
    for comp_id in sorted(mp6_dict.keys()):
        seasons_map = mp6_dict[comp_id]
        for season in sorted(seasons_map.keys()):
            for tup in seasons_map[season]:
                writer.writerow([comp_id, season, *tup])

print(f"CSV files saved to:\n{mp2_path}\n{mp6_path}\n")

# --- Write numerical variables (dict[comp_id][season] -> int) to a text file ---
mp_out_path = os.path.join(output_dir, "mp_out.txt")

numeric_vars = [
    "mp0_all_players_involved_in_AF_FS_checking",
    "mpX_OK_players_AF_FS_matching",
    "mp1a_AF_lineups_missing",
    "mp1b_FS_lineups_missing",
    "mp2_AF_FS_players_matching_potential_misses",  # still a scalar
    "mp3_all_players_involved_in_team_strength_calculation",
    "mp4_team_strength_complete_lineup_imitation",
    "mp5_team_strength_DOB_missing",
    "mp6_team_strength_FS_SF_matching",
    "mp7_team_strength_imitated_skills_as_no_CSV_data",
    "mp7_SKILLS_team_strength_imitated_skills_as_no_data",
    "mp8a_team_strength_imitated_players_as_no_CSV_data",
    "mp8b_team_strength_imitated_players_as_no_CSV_data",
    "mp9_team_strength_balancing_field_to_gk",
    "mp9_team_strength_balancing_gk_to_def",
    "mp9_team_strength_balancing_gk_to_mid",
    "mp9_team_strength_balancing_gk_to_att",
]

with open(mp_out_path, mode="w", encoding="utf-8") as f:
    f.write(f"CSV files saved to:\n{mp2_path}\n{mp6_path}\n\n")
    for var_name in numeric_vars:
        value = getattr(global_instance, var_name, None)
        if isinstance(value, dict):
            f.write(f"{var_name}:\n")
            for cid in sorted(value.keys()):
                inner = value[cid]
                if isinstance(inner, dict):  # per-season map
                    for season in sorted(inner.keys()):
                        f.write(f"  {cid} / {season}: {inner[season]}\n")
                else:
                    # fallback if some older var remains {cid: int}
                    f.write(f"  {cid}: {inner}\n")
        else:
            f.write(f"{var_name}: {value}\n")

print(f"Output written to:\n{mp_out_path}")

"""
# 9a. Distribute regular matches into rounds for training
regular_matches = [x for x in global_instance.all_matches if x.round.is_regular]
regular_matches = sorted(regular_matches, key=lambda match_: match_.datetime)

# 9b. Create mapping of categorical feature values to indices
team_id_map, comp_id_map = ut.get_categorical_features_maps(regular_matches)

# 10. Train
regular_matches_in_rounds = ut.distribute_matches_into_rounds(regular_matches)
train(regular_matches_in_rounds, team_id_map, comp_id_map)
"""

"""
regular_matches_in_rounds = ut.distribute_matches_into_rounds(regular_matches)
# regular_matches_in_rounds = ut.distribute_matches_into_rounds_uniformly(regular_matches)
for i, r in enumerate(regular_matches_in_rounds):
    print(f"{str(len(r))} matches found in round {str(i)}")
# TODO: Maybe ensure that there are at least N matches in each round?

# TODO: Run again without uniformly distributed rounds and check if the total number of matches per comp are as expected
# TODO: ...because now there are always 1250 matches in training data, which is less then without uniform, right?
# TODO: ...so since there are less training data, there are more validation data - more total number of val matches?

# 10. Train
train(regular_matches_in_rounds)

"""
print("breakpoint")
