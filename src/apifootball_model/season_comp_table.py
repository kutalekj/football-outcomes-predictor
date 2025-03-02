"""
season_comp_table.py
"""

import http.client
import json
import requests
from datetime import datetime
import numpy as np
import settings
import utils as ut
from globals import Global


class SeasonCompTable:
    def __init__(self, comp_id, comp_name, season):
        self.comp_id = comp_id
        self.comp_name = comp_name
        self.season = season
        self.teams = None
        self.team_stats = None

        self.all_fs_players_involved = None

        self.conn = http.client.HTTPSConnection(settings.HOST)

    def init_teams_in_season_comp(self):
        global_instance = Global.get_instance()
        print(f"Initializing table for comp [{self.comp_name}].")

        comp_season_teams = [team for team in global_instance.all_teams if
                             {'comp': ut.get_comp_by_id(self.comp_id), 'season': self.season, 'is_regular': False}
                             in team.regularity_in_comp_season]  # at the moment there are no teams set as regular yet

        teams = []
        for team in comp_season_teams:

            # Assign average/default SOFIFA goalkeeper skills
            team.avg_gk_diving[self.season] = ut.get_avg_gk_skill_value("diving", team.id, self.season) * 100
            team.avg_gk_handling[self.season] = ut.get_avg_gk_skill_value("handling", team.id, self.season) * 100
            team.avg_gk_kicking[self.season] = ut.get_avg_gk_skill_value("kicking", team.id, self.season) * 100
            team.avg_gk_positioning[self.season] = ut.get_avg_gk_skill_value("positioning", team.id, self.season) * 100
            team.avg_gk_reflexes[self.season] = ut.get_avg_gk_skill_value("reflexes", team.id, self.season) * 100

            teams.append(team)

        self.teams = teams
        self.team_stats = {(team.id, team.name): {'points': 0, 'games_played': 0, 'goals_for': 0, 'goals_against': 0,
                                                  'avg_points_per_game': 0} for team in self.teams}

    def update_table(self, matches):
        for match in matches:
            home_team_id = match.home_team.id
            home_team_name = match.home_team.name
            away_team_id = match.away_team.id
            away_team_name = match.away_team.name
            home_goals = match.home_team_goals
            away_goals = match.away_team_goals

            self.team_stats[(home_team_id, home_team_name)]['games_played'] += 1
            self.team_stats[(home_team_id, home_team_name)]['goals_for'] += home_goals
            self.team_stats[(home_team_id, home_team_name)]['goals_against'] += away_goals
            self.team_stats[(away_team_id, away_team_name)]['games_played'] += 1
            self.team_stats[(away_team_id, away_team_name)]['goals_for'] += away_goals
            self.team_stats[(away_team_id, away_team_name)]['goals_against'] += home_goals

            if home_goals > away_goals:
                self.team_stats[(home_team_id, home_team_name)]['points'] += 3.0
            elif home_goals < away_goals:
                self.team_stats[(away_team_id, away_team_name)]['points'] += 3.0
            else:
                self.team_stats[(home_team_id, home_team_name)]['points'] += 1.0
                self.team_stats[(away_team_id, away_team_name)]['points'] += 1.0

        # Normalize points (some teams might have more matches played up to a certain date)
        for team in self.teams:
            self.team_stats[(team.id, team.name)]['avg_points_per_game'] = \
                self.team_stats[(team.id, team.name)]['points'] / \
                self.team_stats[(team.id, team.name)]['games_played'] \
                    if self.team_stats[(team.id, team.name)]['games_played'] > 0 else 0.0

    # TODO: How to handle transitions between individual seasons? - in a new season there might be different teams
    # TODO: Issue1: In a new season there might be a team that had no previous matches...
    # TODO: Issue2: Prev match getting would currently return last match of 2021 as prev match of first match of 2023...
    # TODO: ...if the team was not competing in 2022 - unwanted behavior.
    # TODO: Solution: Probably currently fixing none of these - future features

    # TODO: Since table positions are now calculated after all matches loading, calculation can be done without resets
    # TODO: Solution: Probably currently not fixing - possible future features (it might be faster, but complicated...)
    def calculate_and_get_teams_positions_in_season_up_to_date(self, date):
        # Reset team stats
        self.team_stats = {(team.id, team.name): {'points': 0, 'games_played': 0, 'goals_for': 0, 'goals_against': 0,
                                                  'avg_points_per_game': 0} for team in self.teams}

        # Get all matches up to the wanted date
        regular_matches_up_to_date = ut.get_all_regular_matches_in_season_table_up_to_date(self, date)

        # Update the table with matches up to the wanted date
        self.update_table(regular_matches_up_to_date)

        # TODO: Adj.: Team with equal sorting might get same position (e.g. 1st position for all teams before season)
        sorted_teams = sorted(self.teams, key=lambda team: (
            self.team_stats[(team.id, team.name)]['avg_points_per_game'],
            self.team_stats[(team.id, team.name)]['goals_for'] - self.team_stats[(team.id, team.name)]['goals_against'],
            self.team_stats[(team.id, team.name)]['goals_for']), reverse=True)

        return sorted_teams

    def get_curr_team_position_in_season_up_to_date(self, team_id, date):
        sorted_teams = self.calculate_and_get_teams_positions_in_season_up_to_date(date)

        # Get relative position (1.0 as the best, 0.0 as the worst!)
        for position, team in enumerate(sorted_teams, start=1):
            if team.id == team_id:
                return float(1.0 - (position / len(self.teams)))

        raise Exception(f"Unable to calculate team [{str(team_id)}] position in the current comp season "
                        f"[{self.comp_name}, {str(self.season)}]")

    @staticmethod
    def exclude_irregular_teams_from_table_calc():
        global_instance = Global.get_instance()

        for table in global_instance.all_tables:
            table.teams = [team for team in table.teams if
                           any([season_elem for season_elem in team.regularity_in_comp_season if
                                season_elem['comp'].id == table.comp_id and
                                season_elem['season'] == table.season and
                                season_elem['is_regular']])]

            table.team_stats = {(team_id, team_name): stats for (team_id, team_name), stats in table.team_stats.items()
                                if
                                any([season_elem for season_elem in
                                     ut.get_team_if_exists(team_id).regularity_in_comp_season
                                     if season_elem['comp'].id == table.comp_id and
                                     season_elem['season'] == table.season and
                                     season_elem['is_regular']])}

    @staticmethod
    def get_fs_player_rosters_per_regular_comp_season_team():
        global_instance = Global().get_instance()
        for comp in global_instance.all_comps:
            if len(comp.regular_round_keywords) == 0:
                continue  # skip for irregulars

            for season in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1):
                # First, get all FS players from comp season and assign them to table (to avoid requests for each team)
                table = ut.get_table_by_comp_season(comp.id, season)

                fs_season_id = comp.get_fs_season_id(comp.id, comp.country, season)  # get FS season_id (comp season ID)

                comp_season_players_stats_request_string_fs = settings.FS_HOST + "/league-players?key=" + \
                                                              settings.FS_KEY + "&season_id=" + str(
                    fs_season_id) + "&include=stats"
                res = requests.get(comp_season_players_stats_request_string_fs)
                data_comp_season_players_stats_fs = res.json()  # get data
                num_pages = data_comp_season_players_stats_fs['pager']['max_page']

                all_data_comp_season_players_stats_fs = []
                for page_num in range(1, num_pages + 1):  # iterate over all pages
                    request_url = comp_season_players_stats_request_string_fs + "&page=" + str(page_num)
                    res_json = requests.get(request_url).json()
                    print(f"[5] \t\tFS req. remaining: {res_json['metadata']['request_remaining']}...")

                    all_data_comp_season_players_stats_fs += [{
                        'fs_id': x['id'],
                        'fs_comp_id': x['competition_id'],
                        'fs_full_name': x['full_name'],
                        'fs_known_as': x['known_as'],
                        'fs_birthday': datetime.utcfromtimestamp(x['birthday']),
                        'fs_age': x['age'],
                        'fs_weight': x['weight'],
                        'fs_height': x['height'],
                        'fs_league': x['league'],
                        'fs_league_type': x['league_type'],
                        'fs_club_team_id': x['club_team_id'],
                        'fs_club_team_2_id': x['club_team_2_id'],
                        'fs_position': x['position'],
                        'fs_nationality': x['nationality'],
                    } for x in res_json['data']]

                table.all_fs_players_involved = all_data_comp_season_players_stats_fs

                # Then, when irregular teams should be already excluded from tables, assign comp season players to teams
                for team in table.teams:
                    # Condition as follows, because player might have played for more teams in a comp season
                    selected_fs_players = [x for x in table.all_fs_players_involved if
                                           team.fs_id == x['fs_club_team_id']
                                           or ('fs_club_team_2_id' in x and team.fs_id == x['fs_club_team_2_id'])]
                    # TODO check: Maybe debug check this teams matching condition

                    team.players_in_regular_comp_season.append({'comp': comp, 'season': season,
                                                                'fs_players': selected_fs_players})

                    global_instance.tmp_average_player_skills[(season, team.id, team.name, "goalkeeper")] = \
                        {skill: [] for skill in settings.PLAYER_SKILLS}
                    global_instance.tmp_average_player_skills[(season, team.id, team.name, "defender")] = \
                        {skill: [] for skill in settings.PLAYER_SKILLS}
                    global_instance.tmp_average_player_skills[(season, team.id, team.name, "midfielder")] = \
                        {skill: [] for skill in settings.PLAYER_SKILLS}
                    global_instance.tmp_average_player_skills[(season, team.id, team.name, "attacker")] = \
                        {skill: [] for skill in settings.PLAYER_SKILLS}
