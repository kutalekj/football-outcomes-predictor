import requests
import settings
import utils as ut


class Team:
    def __init__(self, id_, name):
        self.id = id_
        self.name = name

        self.fs_id = None
        self.fs_clean_name = None

        # Exclude lower tier teams that played only relegation playoff match at the end of season from season tables
        self.regularity_in_comp_season = []

        # Store all players being in roster of each team in each its regular comp season
        # (then, in team strength calc., each player from team's match lineup will be matched against one from these)
        self.players_in_regular_comp_season = []

        # Average SOFIFA goalkeeper skills for cases when data is missing - for imputing
        self.avg_gk_diving = {year: None for year in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)}
        self.avg_gk_handling = {year: None for year in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)}
        self.avg_gk_kicking = {year: None for year in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)}
        self.avg_gk_positioning = {year: None for year in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)}
        self.avg_gk_reflexes = {year: None for year in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1)}

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
                            print(f"[4] Setting team {self.name} as regular in {season_elem['comp'].name} in {season}.")
                            season_elem['is_regular'] = True

                            # 2. Match the regular AF team with FS team from the same comp_season
                            if self.fs_id is None:  # if not matched yet (might have been done is previous seasons)
                                self.assign_fs_team_id_team_name_by_comp_season(season_elem['comp'], season)

        # TODO check: Add debug print for number of matches (both all and regulars) for each comp season
        # TODO code: Split in two functions (two different functionalities)? - currently like this because of reg. check

    def assign_fs_team_id_team_name_by_comp_season(self, comp, season):
        fs_teams_comp_season = [x for x in comp.fs_teams_per_season if x['season'] == season]
        if len(fs_teams_comp_season) != 1:
            raise ValueError(f"Unexpected to find none, or multiple FS teams for a single comp season "
                             f"({comp.name}, {str(season)})")
        fs_teams_comp_season = fs_teams_comp_season[0]

        # Match AF team with FS team
        if comp.id in [61, 88, 119, 179, 307] and self.id in [80, 85, 197, 201, 254, 402, 405, 406, 413, 2944]:
            if comp.id == 61 and self.id == 80:
                self.fs_id, self.fs_clean_name = 57, "Olympique Lyonnais"
            if comp.id == 61 and self.id == 85:
                self.fs_id, self.fs_clean_name = 68, "PSG"
            if comp.id == 88 and self.id == 197:
                self.fs_id, self.fs_clean_name = 121, "PSV"
            if comp.id == 88 and self.id == 201:
                self.fs_id, self.fs_clean_name = 378, "AZ"
            if comp.id == 88 and self.id == 413:
                self.fs_id, self.fs_clean_name = 379, "NEC"  # Nijmegen
            if comp.id == 119 and self.id == 402:
                self.fs_id, self.fs_clean_name = 974, "AaB"  # Aalborg
            if comp.id == 119 and self.id == 405:
                self.fs_id, self.fs_clean_name = 2523, "OB"  # Odense
            if comp.id == 119 and self.id == 406:
                self.fs_id, self.fs_clean_name = 2516, "AGF"  # Aarhus
            if comp.id == 179 and self.id == 254:
                self.fs_id, self.fs_clean_name = 32, "Hearts"
            if comp.id == 307 and self.id == 2944:
                self.fs_id, self.fs_clean_name = 5071, "Al Feiha"  # Al-Fayha
            print(f"\t\t\t\t\tAF team matched to FS team: [{self.name}] [{self.fs_clean_name}] (manually)")
            # TODO check: Check this matching once again - this must be 100% accurate

        else:
            self.fs_id, self.fs_clean_name = ut.match_af_team_to_fs_team(self.name, fs_teams_comp_season)
        # TODO adj: Consider to forbid matching of teams already matched before
