class Global:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Global, cls).__new__(cls, *args, **kwargs)

            cls._instance.all_matches = []  # list[FSMatch]
            cls._instance.all_comp_seasons = {}  # dict[int, FSCompSeason]
            cls._instance.all_players = {}  # dict[int, FSPlayer]
            cls._instance.all_teams = {}  # dict[int, FSTeam]

            cls._instance.leagues_list = []  # list[dict]

            cls._instance.sf_avg_team_strength = {}  # for each team's season's position category

            # Main list with sofifa players' data: list[tuple[datetime, dict[int, dict]]]
            # Tuples: (datetime of CSV with data, dict); Dict: indexed by player IDs, values = dicts
            cls._instance.sofifa_snapshots = []

            # Dict to keep track of player occurrences: dict[int, list[tuple[int, datetime]]]
            # Indexed by SF player IDs; Tuples: (index to main list, datetime of CSV with data)
            cls._instance.sofifa_player_occurrences = {}

            # Dict grouping players by date of birth (for matching with FS players)
            # Indexed by dates of birth; each elem is list of triples (player_id, name, full_name)
            # dict[datetime.date, list[tuple[int, str, str]]]
            cls._instance.sofifa_players_by_dob = {}
        return cls._instance

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
