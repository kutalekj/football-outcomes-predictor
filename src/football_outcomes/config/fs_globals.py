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

        return cls._instance

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
