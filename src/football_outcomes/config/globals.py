class Global:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Global, cls).__new__(cls, *args, **kwargs)

            cls._instance.all_matches = []
            cls._instance.all_comps = []
            cls._instance.all_tables = []
            cls._instance.all_teams = []

            # Wanted are the start/end dates of season for each country, not for each comp separately
            # For example, for England 2021 there are PL, Championship, UEFA competitions, FA Cup and EFL Trophy
            cls._instance.start_end_dates_per_country_season = {}

            cls._instance.fs_leagues_list = []  # all FS seasons of all comps
            cls._instance.fs_leagues_matches = {}  # all FS matches in each comp season

            cls._instance.sf_avg_team_strength = None  # average SF team strength for each reg. team's season's pos.cat.

            # Main list with sofifa players' data
            # Each elem is tuple (datetime of CSV with data, big dict); big dict indexed by player IDs, values = dicts
            cls._instance.sofifa_players_data = []

            # Dict to keep track of player occurrences
            # Indexed by sofifa player IDs; each elem is list of tuples (index to main list, datetime of CSV with data)
            cls._instance.sofifa_player_index_dict = {}

            # Dict grouping players by date of birth (for matching with FS players)
            # Indexed by dates of birth; each elem is list of triples (player_id, name, full_name)
            cls._instance.sofifa_players_by_dob = {}

            # Missing players checking
            cls._instance.mp0_all_players_involved_in_AF_FS_checking = 0
            cls._instance.mpX_OK_players_AF_FS_matching = 0
            cls._instance.mp1a_AF_lineups_missing = 0
            cls._instance.mp1b_FS_lineups_missing = 0
            cls._instance.mp2_AF_FS_players_matching_potential_misses = 0

            competition_ids = [
                39,
                40,
                41,
                42,
                61,
                62,
                78,
                79,
                88,
                94,
                106,
                119,
                135,
                136,
                140,
                141,
                144,
                179,
                188,
                203,
                207,
                218,
                307,
                323,
            ]

            SEASONS = [2021, 2022, 2023, 2024, 2025]

            def _zeros_by_comp_and_season(competition_ids):
                return {cid: {s: 0 for s in SEASONS} for cid in competition_ids}

            def _lists_by_comp_and_season(competition_ids):
                return {cid: {s: [] for s in SEASONS} for cid in competition_ids}

            # --- counters (dict[comp_id][season] -> int) ---
            cls._instance.mp3_all_players_involved_in_team_strength_calculation = _zeros_by_comp_and_season(
                competition_ids
            )
            cls._instance.mp4_team_strength_complete_lineup_imitation = _zeros_by_comp_and_season(competition_ids)
            cls._instance.mp5_team_strength_DOB_missing = _zeros_by_comp_and_season(competition_ids)
            cls._instance.mp6_team_strength_FS_SF_matching = _zeros_by_comp_and_season(competition_ids)
            cls._instance.mp7_team_strength_imitated_skills_as_no_CSV_data = _zeros_by_comp_and_season(competition_ids)
            cls._instance.mp7_SKILLS_team_strength_imitated_skills_as_no_data = _zeros_by_comp_and_season(
                competition_ids
            )
            cls._instance.mp8a_team_strength_imitated_players_as_no_CSV_data = _zeros_by_comp_and_season(
                competition_ids
            )
            cls._instance.mp8b_team_strength_imitated_players_as_no_CSV_data = _zeros_by_comp_and_season(
                competition_ids
            )

            cls._instance.mp9_team_strength_balancing_field_to_gk = _zeros_by_comp_and_season(competition_ids)
            cls._instance.mp9_team_strength_balancing_gk_to_def = _zeros_by_comp_and_season(competition_ids)
            cls._instance.mp9_team_strength_balancing_gk_to_mid = _zeros_by_comp_and_season(competition_ids)
            cls._instance.mp9_team_strength_balancing_gk_to_att = _zeros_by_comp_and_season(competition_ids)

            # --- couples (dict[comp_id][season] -> list[tuple]) ---
            cls._instance.mp6_FS_SF_players_matching_potential_misses_couples = _lists_by_comp_and_season(
                competition_ids
            )
            cls._instance.mp2_AF_FS_players_matching_potential_misses_couples = []

        return cls._instance

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
