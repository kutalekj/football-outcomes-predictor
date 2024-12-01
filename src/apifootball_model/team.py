import requests
import settings
import utils as ut


class Team:
    def __init__(self, id_, name):
        self.id = id_
        self.name = name

        self.fs_id = None
        self.fs_name = None

        # Exclude lower tier teams that played only relegation playoff match at the end of season from season tables
        self.regularity_in_comp_season = []

        # Store all players being in roster of each team in each its regular comp season
        # (then, in team strength calc., each player from team's match lineup will be matched against one from these)
        self.players_in_regular_comp_season = []

        self.player_stats_comp_season = {}  # TODO: Rename - no stats, just players list
        self.rating_comp_season = {}  # TODO: Remove

        self.matches = []  # just list of all matches of the team sorted by the datetime played asc

    def __eq__(self, other):
        if isinstance(other, Team):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    def get_index_of_match_in_sorted_team_matches_list(self, match):
        if match is None:
            return None

        team_matches = sorted(self.matches, key=lambda match_: match_.datetime)

        for i, match__ in enumerate(team_matches):
            if match__ == match:
                return i

        # This should never happen
        return None

    def correct_team_regularity_and_match_af_fs_teams(self):
        for season in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1):
            team_matches_in_season = [m for m in self.matches if m.season == season]

            # If team participating in the season, check if it played some regular matches
            if len(team_matches_in_season) > 0:

                regular_team_matches_in_season = [match for match in team_matches_in_season if match.round.is_regular]
                if any([match.round.is_regular for match in team_matches_in_season]):

                    # Set the "is_regular" team attribute to True
                    for season_elem in self.regularity_in_comp_season:
                        regular_team_matches_in_comp_season_booleans = \
                            [match.round.is_regular for match in regular_team_matches_in_season
                             if match.comp.id == season_elem['comp'].id]

                        if season_elem['season'] == season and \
                                len(season_elem['comp'].regular_round_keywords) > 0 and \
                                any(regular_team_matches_in_comp_season_booleans):
                            print(f"_DEBUG_: Setting team {self.name} as regular in {season_elem['comp'].name} in {season}.")
                            season_elem['is_regular'] = True

                            # 2. Match the regular AF team with FS team from the same comp_season
                            if self.fs_id is None:  # if not matched yet (might have been done is previous seasons)
                                self.assign_fs_team_id_team_name_by_comp_season(season_elem['comp'], season)

    def assign_fs_team_id_team_name_by_comp_season(self, comp, season):
        fs_teams_comp_season = [x for x in comp.fs_teams_per_season if x['season'] == season]
        if len(fs_teams_comp_season) != 1:
            raise ValueError(f"Unexpected to find none, or multiple FS teams for a single comp season "
                             f"({comp.name}, {str(season)})")
        fs_teams_comp_season = fs_teams_comp_season[0]

        # Match AF team with FS team
        self.fs_id, self.fs_name = ut.match_af_team_to_fs_team(self.name, fs_teams_comp_season)
        # TODO: Minor adj. might be forbidding to match teams already matched before

    # TODO: The following should be already implemented in SCT.init_players_lists_in_regular_comp_season_teams()
    """
    def get_players_in_regular_comp_season(self, regular_season_elem):
        comp = regular_season_elem['comp']
        season = regular_season_elem['season']
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

            all_data_comp_season_players_stats_fs += res_json['data']

        self.players_in_regular_comp_season.append({'comp': comp, 'season': season,
                                                    'fs_players': all_data_comp_season_players_stats_fs})  # assign data
        # TODO.........
    """
