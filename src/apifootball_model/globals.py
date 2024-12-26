class Global:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Global, cls).__new__(cls, *args, **kwargs)

            cls._instance.all_matches = []
            cls._instance.all_comps = []
            cls._instance.all_tables = []
            cls._instance.all_teams = []

            cls._instance.num_unique_regular_teams_for_training = []
            cls._instance.num_unique_regular_comps_for_training = []

            # Wanted are the start/end dates of season for each country, not for each comp separately
            # For example, for England 2021 there are PL, Championship, UEFA competitions, FA Cup and EFL Trophy
            cls._instance.start_end_dates_per_country_season = {}

            cls._instance.fs_leagues_list = []  # all seasons of all comps
            cls._instance.fs_leagues_matches = {}  # all matches in each comp season

            # TODO: Remove these after imputing
            cls._instance.gk_diving = {}
            cls._instance.gk_handling = {}
            cls._instance.gk_kicking = {}
            cls._instance.gk_positioning = {}
            cls._instance.gk_reflexes = {}

            # Main list with sofifa players' data
            # Each elem is tuple (datetime of CSV with data, big dict); big dict indexed by player IDs, values = dicts
            cls._instance.sofifa_players_data = []

            # Dict to keep track of player occurrences
            # Indexed by sofifa player IDs; each elem is list of tuples (index to main list, datetime of CSV with data)
            cls._instance.sofifa_player_index_dict = {}

            # Dict grouping players by date of birth (for matching with FS players)
            # Indexed by dates of birth; each elem is list of triples (player_id, name, full_name)
            cls._instance.sofifa_players_by_dob = {}

        return cls._instance

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
