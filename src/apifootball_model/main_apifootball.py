import datetime
import statistics
import numpy as np
import csv
import utils as ut
import settings
from comp import Comp
from season_comp_table import SeasonCompTable
from match import Match
from feature import MatchFeatures
from globals import Global
import in_out
import in_out_mega
# from train_rnn import train
# from train_ann import train
from train_compID_encoder import train
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.switch_backend('TkAgg')

global_instance = Global.get_instance()

if not settings.ALL_LOAD:

    # 0. Load average skills and team strengths (SF)
    in_out.load_sf_avg_team_strength()

    # 1. Init comps (seasons, teams, AF rounds, FS matches)
    Comp.get_fs_leagues_list()

    for comp in settings.COMPS_v2:
        new_comp = Comp(comp['id'], comp['name'], comp['regular_round_keywords'])
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
                    'start': datetime.datetime.max, 'end': datetime.datetime.min}

            if len(comp.regular_round_keywords) == 0:
                continue  # omit the cups - do not create tables for them

            new_table = SeasonCompTable(comp.id, comp.name, season)
            new_table.init_teams_in_season_comp()

            global_instance.all_tables.append(new_table)

    for comp in global_instance.all_comps:
        comp.init_country_start_end_dates_in_seasons()

    # 3a. Load match data from local
    if settings.MATCH_DATA_LOAD:
        in_out.load_matches(settings.MATCH_DATA_LOAD_FILENAME)
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
        if match.round.is_regular and\
                match.datetime > settings.GET_XG_IF_MATCH_DATE_NEWER_THAN.replace(tzinfo=match.datetime.tzinfo) and\
                (match.total_xg == -1 and match.total_pre_match_xg == -1):
            ut.get_fs_match_xg(match)

        # Feature calculation
        match.features_before_match_played = match.calculate_match_features()
        match.feature_vector_before_match_played = MatchFeatures.match_features_to_vector(
            match.features_before_match_played)
else:
    in_out_mega.load_all_matches_data()

# 8a. Store matches to local
if settings.MATCH_DATA_STORE:
    in_out.store_matches(settings.MATCH_DATA_STORE_FILENAME)

# 8b. Store all data to local
if settings.ALL_STORE:
    in_out_mega.store_all_matches_data()


# 9. Distribute regular matches into rounds for training
# TODO: Fix the error that about 30 regular matches are missing team strength (the following is a temp. solution)!
# regular_matches = [x for x in global_instance.all_matches if x.round.is_regular and x.feature_vector_before_match_played.shape[0] == 126]
regular_matches = [x for x in global_instance.all_matches if x.round.is_regular]
regular_matches = sorted(regular_matches, key=lambda match_: match_.datetime)

# 10a. Train comp ID embeddings
comp_id_to_name = {}
for m in regular_matches:
    comp_name = "Bundesliga AUT" if m.comp.country == "Austria" and m.comp.name == "Bundesliga" else m.comp.name
    comp_id_to_name[m.comp.id] = comp_name
unique_comp_ids = sorted(comp_id_to_name.keys())  # sorting ensures consistent indexing
print(unique_comp_ids)

all_comp_ids = [x.comp.id for x in regular_matches]
comp_id_map = {comp_id: idx for idx, comp_id in enumerate(unique_comp_ids)}  # map to [0, 24)

legend_labels = []
print("Comp IDs mapping:")
for idx, comp_id in enumerate(unique_comp_ids):
    legend_label = f"[{comp_id}: {comp_id_to_name[comp_id]}] -> [{idx}]"
    legend_labels.append(legend_label)
    print(legend_label)

mapped_team_ids = np.array([comp_id_map[comp_id] for comp_id in all_comp_ids])

learned_embeddings = train(mapped_team_ids, batch_size=32, num_epochs=10)

aligned_comp_ids = [comp_id for comp_id, idx in sorted(comp_id_map.items(), key=lambda item: item[1])]  # ensure alignment of learned embeddings with original comp IDs

# Visualize the results
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(learned_embeddings)
plt.figure(figsize=(11, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.8)
plt.title("PCA projection of learned comp embeddings")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")

# Annotations + legend
for i, txt in enumerate(range(len(learned_embeddings))):
    plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9)
for label in legend_labels:
    plt.scatter([], [], label=label)
plt.legend(title="Competitions mapping", fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))  # move to avoid overlap

plt.tight_layout()
plt.show()

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
