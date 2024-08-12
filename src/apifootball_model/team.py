class Team:
    def __init__(self, id_, name):
        self.id = id_
        self.name = name

        self.matches = []  # just list of all matches of the team sorted by the datetime played asc
        # {'comp_id': comp_id, 'comp_name': comp_name, 'season': season_num, 'matches': matches_list} - NO!

    def __eq__(self, other):
        if isinstance(other, Team):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)
