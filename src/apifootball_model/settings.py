"""
settings.py
"""

from datetime import timedelta

KEY = "4a9e20eecbec58c517cb485f31552caf"
HOST = "v3.football.api-sports.io"

HEADERS = {
    'x-rapidapi-host': HOST,
    'x-rapidapi-key': KEY
}

FS_KEY = "9360c5f9b742b0177a1e42b1afee860151cab101673147456e60412da6d46b38"
FS_HOST = "https://api.football-data-api.com"

LOAD_MATCH_DATA_FROM_LOCAL_CSV = True
MEGA_STORE = False
MEGA_LOAD = False

FIRST_SEASON = 2021
LAST_SEASON = 2024

MAX_MATCH_HISTORY_TO_CHECK_LOW = 15
MAX_MATCH_HISTORY_TO_CHECK_HIGH = 50

# {v3API_id, name, regular_round_keywords}
COMPS = [
    {'id': 39, 'name': "Premier League", 'regular_round_keywords': ['Regular Season']},
    {'id': 40, 'name': "Championship", 'regular_round_keywords': ['Regular Season']},
    {'id': 61, 'name': "Ligue 1", 'regular_round_keywords': ['Regular Season']},
    {'id': 62, 'name': "Ligue 2", 'regular_round_keywords': ['Regular Season']},
    {'id': 78, 'name': "Bundesliga", 'regular_round_keywords': ['Regular Season']},
    {'id': 79, 'name': "2. Bundesliga", 'regular_round_keywords': ['Regular Season']},
    {'id': 88, 'name': "Eredivisie", 'regular_round_keywords': ['Regular Season']},
    {'id': 94, 'name': "Primeira Liga", 'regular_round_keywords': ['Regular Season']},
    {'id': 119, 'name': "Superliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round']},  # DEN
    {'id': 135, 'name': "Serie A", 'regular_round_keywords': ['Regular Season']},
    {'id': 136, 'name': "Serie B", 'regular_round_keywords': ['Regular Season']},
    {'id': 140, 'name': "La Liga", 'regular_round_keywords': ['Regular Season']},
    {'id': 141, 'name': "Segunda División", 'regular_round_keywords': ['Regular Season']},
    {'id': 144, 'name': "Jupiler Pro League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Conference League Play-off Group']},
    {'id': 203, 'name': "Süper Lig", 'regular_round_keywords': ['Regular Season']},
    {'id': 207, 'name': "Super League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -']},  # SUI
    {'id': 210, 'name': "HNL", 'regular_round_keywords': ['Regular Season']},
    {'id': 2, 'name': "UEFA Champions League", 'regular_round_keywords': []},
    {'id': 3, 'name': "UEFA Europa League", 'regular_round_keywords': []},
    {'id': 848, 'name': "UEFA Europa Conference League", 'regular_round_keywords': []},
    {'id': 45, 'name': "FA Cup", 'regular_round_keywords': []},
    {'id': 46, 'name': "EFL Trophy", 'regular_round_keywords': []},
    {'id': 81, 'name': "DFB Pokal", 'regular_round_keywords': []},
    {'id': 66, 'name': "Coupe de France", 'regular_round_keywords': []},
    {'id': 137, 'name': "Coppa Italia", 'regular_round_keywords': []},
    {'id': 143, 'name': "Copa del Rey", 'regular_round_keywords': []},
    {'id': 90, 'name': "KNVB Beker", 'regular_round_keywords': []},  # NED
    {'id': 96, 'name': "Taça de Portugal", 'regular_round_keywords': []},
    {'id': 97, 'name': "Taça da Liga", 'regular_round_keywords': []},
    {'id': 209, 'name': "Schweizer Cup", 'regular_round_keywords': []},
    {'id': 212, 'name': "Cup", 'regular_round_keywords': []},  # CRO
    {'id': 206, 'name': "Cup", 'regular_round_keywords': []},  # TUR
    {'id': 121, 'name': "DBU Pokalen", 'regular_round_keywords': []},  # DEN
    {'id': 147, 'name': "Cup", 'regular_round_keywords': []}  # BEL
]

# {v3API_id, name, regular_round_keywords, fs_alias}
COMPS_v2 = [
    {'id': 39, 'name': "Premier League", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Premier League"},
    {'id': 40, 'name': "Championship", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Championship"},
    {'id': 41, 'name': "League One", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "EFL League One"},
    {'id': 42, 'name': "League Two", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "EFL League Two"},
    {'id': 61, 'name': "Ligue 1", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ligue 1"},
    {'id': 62, 'name': "Ligue 2", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ligue 2"},
    {'id': 78, 'name': "Bundesliga", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Bundesliga"},
    {'id': 79, 'name': "2. Bundesliga", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "2. Bundesliga"},
    {'id': 88, 'name': "Eredivisie", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Eredivisie"},
    {'id': 94, 'name': "Primeira Liga", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Liga NOS"},
    {'id': 106, 'name': "Ekstraklasa", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Ekstraklasa"},  # POL
    {'id': 119, 'name': "Superliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round'],
     'fs_alias': "Superliga"},  # DEN
    {'id': 135, 'name': "Serie A", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Serie A"},
    {'id': 136, 'name': "Serie B", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Serie B"},
    {'id': 140, 'name': "La Liga", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "La Liga"},
    {'id': 141, 'name': "Segunda División", 'regular_round_keywords': ['Regular Season'],
     'fs_alias': "Segunda División"},
    {'id': 144, 'name': "Jupiler Pro League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Conference League Play-off Group'],
     'fs_alias': "Pro League"},  # BEL
    {'id': 179, 'name': "Premiership",
     'regular_round_keywords': ['1st Phase', 'Championship Round', 'Relegation Round -'],
     'fs_alias': "Premiership"},  # SCO
    {'id': 188, 'name': "A-League",
     'regular_round_keywords': ['Regular Season', 'Elimination Finals', 'Semi-finals', 'Grand Final'],
     'fs_alias': "A-League"},  # AUS
    {'id': 203, 'name': "Süper Lig", 'regular_round_keywords': ['Regular Season'], 'fs_alias': "Süper Lig"},  # TUR
    {'id': 207, 'name': "Super League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -'],
     'fs_alias': "Super League"},  # SUI
    {'id': 218, 'name': "Bundesliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -'],
     'fs_alias': "Bundesliga"},  # AUT
    {'id': 307, 'name': "Pro League", 'regular_round_keywords': ['Regular Season'],
     'fs_alias': "Professional League"},  # SA
    {'id': 323, 'name': "Indian Super League",
     'regular_round_keywords': ['Regular Season', 'Qualifying Finals', 'Championship -'],
     'fs_alias': "Indian Super League"},  # IND
    {'id': 2, 'name': "UEFA Champions League", 'regular_round_keywords': [], 'fs_alias': "UEFA Champions League"},
    {'id': 3, 'name': "UEFA Europa League", 'regular_round_keywords': [], 'fs_alias': "UEFA Europa League"},
    {'id': 848, 'name': "UEFA Europa Conference League", 'regular_round_keywords': [],
     'fs_alias': "UEFA Europa Conference League"},
    {'id': 45, 'name': "FA Cup", 'regular_round_keywords': [], 'fs_alias': "FA Cup"},
    {'id': 46, 'name': "EFL Trophy", 'regular_round_keywords': [], 'fs_alias': "EFL Trophy"},
    {'id': 81, 'name': "DFB Pokal", 'regular_round_keywords': [], 'fs_alias': "DFB Pokal"},
    {'id': 66, 'name': "Coupe de France", 'regular_round_keywords': [], 'fs_alias': "Coupe de France"},
    {'id': 137, 'name': "Coppa Italia", 'regular_round_keywords': [], 'fs_alias': "Coppa Italia"},
    {'id': 143, 'name': "Copa del Rey", 'regular_round_keywords': [], 'fs_alias': "Copa del Rey"},
    {'id': 90, 'name': "KNVB Beker", 'regular_round_keywords': [], 'fs_alias': "KNVB Cup"},  # NED
    {'id': 96, 'name': "Taça de Portugal", 'regular_round_keywords': [], 'fs_alias': "Taça de Portugal"},
    {'id': 97, 'name': "Taça da Liga", 'regular_round_keywords': [], 'fs_alias': "Portuguese League Cup"},
    {'id': 108, 'name': "Cup", 'regular_round_keywords': [], 'fs_alias': "Polish Cup"},  # POL
    {'id': 209, 'name': "Schweizer Cup", 'regular_round_keywords': [], 'fs_alias': "Swiss Cup"},
    {'id': 206, 'name': "Cup", 'regular_round_keywords': [], 'fs_alias': "Turkish Cup"},  # TUR
    {'id': 121, 'name': "DBU Pokalen", 'regular_round_keywords': [], 'fs_alias': "Danish Cup"},  # DEN
    {'id': 147, 'name': "Cup", 'regular_round_keywords': [], 'fs_alias': "Belgian Cup"},  # BEL
    {'id': 181, 'name': "FA Cup", 'regular_round_keywords': [], 'fs_alias': "Scottish Cup"},  # SCO
    {'id': 185, 'name': "League Cup", 'regular_round_keywords': [], 'fs_alias': "Scottish League Cup"},  # SCO
    {'id': 220, 'name': "Cup", 'regular_round_keywords': [], 'fs_alias': "Austrian Cup"},  # AUT
    {'id': 504, 'name': "King's Cup", 'regular_round_keywords': [], 'fs_alias': "Kings Cup"},  # SA
    {'id': 874, 'name': "Australia Cup", 'regular_round_keywords': [], 'fs_alias': "FFA Cup"}  # AUS
]

INIT_ELO = 1500

WINNER_TEAM_ID_CODE_FOR_DRAW = -1

ZERO = 0.000000
ALMOST_ZERO = 0.001
ALMOST_ONE = 0.999

NUM_NUMERICAL_FEATURES = 34
NUM_CATEGORICAL_FEATURES = 3

SOG_NORM_COEFFICIENT = 12.5
GOALS_NORM_COEFFICIENT = 5.19
MATCH_LOAD_NORM_COEFFICIENT = 0.246
TOTAL_SHOTS_NORM_COEFFICIENT = 28.0
SHOTS_IN_BOX_NORM_COEFFICIENT = 19.5
CORNER_KICKS_NORM_COEFFICIENT = 13.5

TEAM_XG_NORM_COEFFICIENT = 3.25
TOTAL_XG_NORM_COEFFICIENT = 5.25
TEAM_PRE_MATCH_XG_NORM_COEFFICIENT = 3.4
TOTAL_PRE_MATCH_XG_NORM_COEFFICIENT = 5.75

CSV_PLAYERS_PATH = 'C:\\Users\\kutalekj\\PycharmProjects\\sofifa-web-scraper\\output_optimized_phase2\\full'
AVR_GK_SKILLS = 'C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\apifootball_model\\' \
                'avg_sofifa_gk_skills_per_team_comp_season.txt'
AVG_TEAM_STRENGTHS = 'C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\apifootball_model\\' \
                'avg_team_strength_scaled_per_team_comp_season.txt'

PLAYER_SKILLS = ['crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys', 'dribbling', 'curve',
                 'fk_accuracy', 'long_passing', 'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions',
                 'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots', 'aggression', 'interceptions',
                 'positioning', 'vision', 'penalties', 'composure', 'defensive_awareness', 'standing_tackle',
                 'sliding_tackle', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes']

CSV_CATEGORIES = {
    "attacking": ["crossing", "finishing", "heading_accuracy", "short_passing", "volleys"],
    "skill": ["dribbling", "curve", "fk_accuracy", "long_passing", "ball_control"],
    "movement": ["acceleration", "sprint_speed", "agility", "reactions", "balance"],
    "power": ["shot_power", "jumping", "stamina", "strength", "long_shots"],
    "mentality": ["aggression", "interceptions", "positioning", "vision", "penalties", "composure"],
    "defending": ["defensive_awareness", "standing_tackle", "sliding_tackle"],
    "goalkeeping": ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]
}

PLAYER_CATEGORY_RELEVANCE = {
    "goalkeeping": ["goalkeeper"],
    "defending": ["defender", "midfielder"],
    "attacking": ["attacker", "midfielder"],
    "movement": ["defender", "midfielder", "attacker"],
    "power": ["defender", "midfielder", "attacker"],
    "mentality": ["defender", "midfielder", "attacker"],
    "skill": ["midfielder", "attacker"]
}

SKILL_TO_CATEGORY = {}  # dict of CSV_CATEGORIES (hash map)
for cat_name, cat_skills in CSV_CATEGORIES.items():
    for sk in cat_skills:
        SKILL_TO_CATEGORY[sk] = cat_name

ALL_SOFIFA_HEADERS = [
        'player_id', 'version', 'name', 'full_name', 'description', 'image', 'height_cm', 'weight_kg', 'dob',
        'positions', 'overall_rating', 'potential', 'value', 'wage', 'preferred_foot', 'weak_foot', 'skill_moves',
        'international_reputation', 'work_rate', 'body_type', 'real_face', 'release_clause', 'specialities',
        'club_id', 'club_name', 'club_league_id', 'club_league_name', 'club_logo', 'club_rating', 'club_position',
        'club_kit_number', 'club_joined', 'club_contract_valid_until', 'country_id', 'country_name',
        'country_league_id', 'country_league_name', 'country_flag', 'country_rating', 'country_position',
        'country_kit_number', 'crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys', 'dribbling',
        'curve', 'fk_accuracy', 'long_passing', 'ball_control', 'acceleration', 'sprint_speed', 'agility',
        'reactions', 'balance', 'shot_power', 'jumping', 'stamina', 'strength', 'long_shots', 'aggression',
        'interceptions', 'positioning', 'vision', 'penalties', 'composure', 'defensive_awareness', 'standing_tackle',
        'sliding_tackle', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes', 'play_styles'
    ]

SIMILARITY_THRESHOLD_AF_FS = 35  # rapidfuzz ratio ranges from 0 to 100
SIMILARITY_THRESHOLD_FS_SOFIFA = 55  # rapidfuzz ratio ranges from 0 to 100
MINIMUM_MATCHED_LINEUP_PLAYERS = 8
MAX_MISSING_SF_SKILL_VALUES_ALLOWED = 5
MAX_TIMEDELTA_SF_PLAYER_SKILL = timedelta(days=150)  # 5 months

MEGA_LS_COMPS_CSV = "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\mega_comps.csv"
MEGA_LS_TEAMS_CSV = "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\mega_teams.csv"
MEGA_LS_ROUNDS_CSV = "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\mega_rounds.csv"
MEGA_LS_MATCHES_CSV = "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\mega_matches.csv"
