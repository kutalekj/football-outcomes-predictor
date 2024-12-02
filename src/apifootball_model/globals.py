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

            cls._instance.fs_leagues_list = []

        return cls._instance

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
