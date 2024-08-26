class Global:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Global, cls).__new__(cls, *args, **kwargs)

            cls._instance.all_matches = []
            cls._instance.all_comps = []
            cls._instance.all_tables = []
            cls._instance.all_teams = []

            cls._instance.all_avg_points = []
            cls._instance.all_avg_goals = []
            cls._instance.all_avg_shots_on_goal = []

        return cls._instance

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance
