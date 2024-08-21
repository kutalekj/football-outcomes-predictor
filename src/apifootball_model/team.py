import settings


class Team:
    def __init__(self, id_, name):
        self.id = id_
        self.name = name

        # Exclude lower tier teams that played only relegation playoff match at the end of season from season tables
        self.regularity_in_comp_season = []

        self.matches = []  # just list of all matches of the team sorted by the datetime played asc
        # {'comp_id': comp_id, 'comp_name': comp_name, 'season': season_num, 'matches': matches_list} - NO!

    def __eq__(self, other):
        if isinstance(other, Team):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    def get_index_of_match_in_sorted_team_matches_list(self, match):
        team_matches = sorted(self.matches, key=lambda match_: match_.datetime)

        for i, match__ in enumerate(team_matches):
            if match__ == match:
                return i

        # This should never happen
        return None

    def correct_team_regularity(self):
        for season in range(settings.FIRST_SEASON, settings.LAST_SEASON + 1):
            team_matches_in_season = [m for m in self.matches if m.season == season]

            # If team participating in the season, check if it played some regular matches
            if len(team_matches_in_season) > 0:

                regular_team_matches_in_season = [match.round.is_regular for match in team_matches_in_season]

                # Team played some matches in the season, but none of them was regular
                if not any(regular_team_matches_in_season):

                    # Set the "is_regular" team attribute to False
                    for season_elem in self.regularity_in_comp_season:
                        if season_elem['season'] == season:
                            season_elem['is_regular'] = False
