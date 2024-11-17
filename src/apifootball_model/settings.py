"""
settings.py
"""

import datetime

KEY = "4a9e20eecbec58c517cb485f31552caf"
HOST = "v3.football.api-sports.io"

HEADERS = {
    'x-rapidapi-host': HOST,
    'x-rapidapi-key': KEY
}

MATCHES_FILENAME = "api_ftb_matches_.csv"

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

# {v3API_id, name, regular_round_keywords}
COMPS_v2 = [
    {'id': 39, 'name': "Premier League", 'regular_round_keywords': ['Regular Season']},
    {'id': 40, 'name': "Championship", 'regular_round_keywords': ['Regular Season']},
    {'id': 41, 'name': "League One", 'regular_round_keywords': ['Regular Season']},
    {'id': 42, 'name': "League Two", 'regular_round_keywords': ['Regular Season']},
    {'id': 61, 'name': "Ligue 1", 'regular_round_keywords': ['Regular Season']},
    {'id': 62, 'name': "Ligue 2", 'regular_round_keywords': ['Regular Season']},
    {'id': 78, 'name': "Bundesliga", 'regular_round_keywords': ['Regular Season']},
    {'id': 79, 'name': "2. Bundesliga", 'regular_round_keywords': ['Regular Season']},
    {'id': 88, 'name': "Eredivisie", 'regular_round_keywords': ['Regular Season']},
    {'id': 94, 'name': "Primeira Liga", 'regular_round_keywords': ['Regular Season']},
    {'id': 106, 'name': "Ekstraklasa", 'regular_round_keywords': ['Regular Season']},  # POL
    {'id': 119, 'name': "Superliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round']},  # DEN
    {'id': 135, 'name': "Serie A", 'regular_round_keywords': ['Regular Season']},
    {'id': 136, 'name': "Serie B", 'regular_round_keywords': ['Regular Season']},
    {'id': 140, 'name': "La Liga", 'regular_round_keywords': ['Regular Season']},
    {'id': 141, 'name': "Segunda División", 'regular_round_keywords': ['Regular Season']},
    {'id': 144, 'name': "Jupiler Pro League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Conference League Play-off Group']},  # BEL
    {'id': 179, 'name': "Premiership",
     'regular_round_keywords': ['1st Phase', 'Championship Round', 'Relegation Round -']},  # SCO
    {'id': 188, 'name': "A-League",
     'regular_round_keywords': ['Regular Season', 'Elimination Finals', 'Semi-finals', 'Grand Final']},  # AUS
    {'id': 203, 'name': "Süper Lig", 'regular_round_keywords': ['Regular Season']},  # TUR
    {'id': 207, 'name': "Super League",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -']},  # SUI
    {'id': 218, 'name': "Bundesliga",
     'regular_round_keywords': ['Regular Season', 'Championship Round', 'Relegation Round -']},  # AUT
    {'id': 307, 'name': "Pro League", 'regular_round_keywords': ['Regular Season']},  # SA
    {'id': 323, 'name': "Super League",
     'regular_round_keywords': ['Regular Season', 'Qualifying Finals', 'Championship -']},  # IND
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
    {'id': 108, 'name': "Cup", 'regular_round_keywords': []},  # POL
    {'id': 209, 'name': "Schweizer Cup", 'regular_round_keywords': []},
    {'id': 206, 'name': "Cup", 'regular_round_keywords': []},  # TUR
    {'id': 121, 'name': "DBU Pokalen", 'regular_round_keywords': []},  # DEN
    {'id': 147, 'name': "Cup", 'regular_round_keywords': []},  # BEL
    {'id': 181, 'name': "FA Cup", 'regular_round_keywords': []},  # SCO
    {'id': 185, 'name': "League Cup", 'regular_round_keywords': []},  # SCO
    {'id': 220, 'name': "Cup", 'regular_round_keywords': []},  # AUT
    {'id': 504, 'name': "King's Cup", 'regular_round_keywords': []},  # SA
    {'id': 874, 'name': "Australia Cup", 'regular_round_keywords': []}  # AUS
]

INIT_ELO = 1500

WINNER_TEAM_ID_CODE_FOR_DRAW = -1

ZERO = 0.000000
ALMOST_ZERO = 0.001
ALMOST_ONE = 0.999

NUM_NUMERICAL_FEATURES = 34
NUM_CATEGORICAL_FEATURES = 3

SOG_NORM_COEFFICIENT = 11.85
GOALS_NORM_COEFFICIENT = 5.19
MATCH_LOAD_NORM_COEFFICIENT = 0.246
TOTAL_SHOTS_NORM_COEFFICIENT = 29.86
SHOTS_IN_BOX_NORM_COEFFICIENT = 20.29
CORNER_KICKS_NORM_COEFFICIENT = 13.46

CSV_PLAYERS_PATH = 'C:\\Users\\kutalekj\\PycharmProjects\\sofifa-web-scraper\\output\\full'

CSV_CATEGORIES = {
    "attacking": ["crossing", "finishing", "heading_accuracy", "short_passing", "volleys"],
    "skill": ["dribbling", "curve", "fk_accuracy", "long_passing", "ball_control"],
    "movement": ["acceleration", "sprint_speed", "agility", "reactions", "balance"],
    "power": ["shot_power", "jumping", "stamina", "strength", "long_shots"],
    "mentality": ["aggression", "interceptions", "positioning", "vision", "penalties", "composure"],
    "defending": ["defensive_awareness", "standing_tackle", "sliding_tackle"],
    "goalkeeping": ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]
}
