"""
settings.py
"""

from datetime import datetime, timedelta
from pathlib import Path

# API
KEY = "4a9e20eecbec58c517cb485f31552caf"
HOST = "v3.football.api-sports.io"

HEADERS = {"x-rapidapi-host": HOST, "x-rapidapi-key": KEY}

FS_KEY = "9360c5f9b742b0177a1e42b1afee860151cab101673147456e60412da6d46b38"
FS_HOST = "https://api.football-data-api.com"

# Load/Store
MATCH_DATA_LOAD = True
MATCH_DATA_STORE = False

ALL_STORE = False
ALL_LOAD = False

ALL_LS_COMPS_CSV = (
    "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\all-comps_25-09-18.csv"
)
ALL_LS_TEAMS_CSV = (
    "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\all-teams_25-09-18.csv"
)
ALL_LS_ROUNDS_CSV = (
    "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\all-rounds_25-09-18.csv"
)
ALL_LS_MATCHES_CSV = (
    "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\all-matches_25-09-18.csv"
)

TRAINED_MODELS_DIR = "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\apifootball_model\\learned_models"

GET_XG_IF_MATCH_DATE_NEWER_THAN = datetime(2025, 9, 17)  # YYYY-MM-DD

FIRST_SEASON = 2021
LAST_SEASON = 2025

NUM_REGULAR_COMPS = 24
NUM_REGULAR_TEAMS = 518

MAX_MATCH_HISTORY_TO_CHECK_LOW = 15

# {v3API_id, name, regular_round_keywords, fs_alias}
COMPS_v2_TEST = [
    {
        "id": 144,
        "name": "Jupiler Pro League",
        "regular_round_keywords": [
            "Regular Season",
            "Championship Round",
            "Conference League Play-off Group",
        ],
        "fs_alias": "Pro League",
    },  # BEL
    {"id": 147, "name": "Cup", "regular_round_keywords": [], "fs_alias": "Belgian Cup"},  # BEL
    {
        "id": 2,
        "name": "UEFA Champions League",
        "regular_round_keywords": [],
        "fs_alias": "UEFA Champions League",
    },
    {
        "id": 3,
        "name": "UEFA Europa League",
        "regular_round_keywords": [],
        "fs_alias": "UEFA Europa League",
    },
    {
        "id": 848,
        "name": "UEFA Europa Conference League",
        "regular_round_keywords": [],
        "fs_alias": "UEFA Europa Conference League",
    },
]

# {v3API_id, name, regular_round_keywords, fs_alias}
COMPS_v2 = [
    {
        "id": 39,
        "name": "Premier League",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "Premier League",
    },
    {
        "id": 40,
        "name": "Championship",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "Championship",
    },
    {
        "id": 41,
        "name": "League One",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "EFL League One",
    },
    {
        "id": 42,
        "name": "League Two",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "EFL League Two",
    },
    {
        "id": 61,
        "name": "Ligue 1",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "Ligue 1",
    },
    {
        "id": 62,
        "name": "Ligue 2",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "Ligue 2",
    },
    {
        "id": 78,
        "name": "Bundesliga",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "Bundesliga",
    },
    {
        "id": 79,
        "name": "2. Bundesliga",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "2. Bundesliga",
    },
    {
        "id": 88,
        "name": "Eredivisie",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "Eredivisie",
    },
    {
        "id": 94,
        "name": "Primeira Liga",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "Liga NOS",
    },
    {
        "id": 106,
        "name": "Ekstraklasa",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "Ekstraklasa",
    },  # POL
    {
        "id": 119,
        "name": "Superliga",
        "regular_round_keywords": ["Regular Season", "Championship Round", "Relegation Round"],
        "fs_alias": "Superliga",
    },  # DEN
    {
        "id": 135,
        "name": "Serie A",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "Serie A",
    },
    {
        "id": 136,
        "name": "Serie B",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "Serie B",
    },
    {
        "id": 140,
        "name": "La Liga",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "La Liga",
    },
    {
        "id": 141,
        "name": "Segunda DivisiÃ³n",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "Segunda DivisiÃ³n",
    },
    {
        "id": 144,
        "name": "Jupiler Pro League",
        "regular_round_keywords": [
            "Regular Season",
            "Championship Round",
            "Conference League Play-off Group",
        ],
        "fs_alias": "Pro League",
    },  # BEL
    {
        "id": 179,
        "name": "Premiership",
        "regular_round_keywords": ["1st Phase", "Championship Round", "Relegation Round -"],
        "fs_alias": "Premiership",
    },  # SCO
    {
        "id": 188,
        "name": "A-League",
        "regular_round_keywords": [
            "Regular Season",
            "Elimination Finals",
            "Semi-finals",
            "Grand Final",
        ],
        "fs_alias": "A-League",
    },  # AUS
    {
        "id": 203,
        "name": "SÃ¼per Lig",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "SÃ¼per Lig",
    },  # TUR
    {
        "id": 207,
        "name": "Super League",
        "regular_round_keywords": ["Regular Season", "Championship Round", "Relegation Round -"],
        "fs_alias": "Super League",
    },  # SUI
    {
        "id": 218,
        "name": "Bundesliga",
        "regular_round_keywords": ["Regular Season", "Championship Round", "Relegation Round -"],
        "fs_alias": "Bundesliga",
    },  # AUT
    {
        "id": 307,
        "name": "Pro League",
        "regular_round_keywords": ["Regular Season"],
        "fs_alias": "Professional League",
    },  # SA
    {
        "id": 323,
        "name": "Indian Super League",
        "regular_round_keywords": ["Regular Season", "Qualifying Finals", "Championship -"],
        "fs_alias": "Indian Super League",
    },  # IND
    {
        "id": 2,
        "name": "UEFA Champions League",
        "regular_round_keywords": [],
        "fs_alias": "UEFA Champions League",
    },
    {
        "id": 3,
        "name": "UEFA Europa League",
        "regular_round_keywords": [],
        "fs_alias": "UEFA Europa League",
    },
    {
        "id": 848,
        "name": "UEFA Europa Conference League",
        "regular_round_keywords": [],
        "fs_alias": "UEFA Europa Conference League",
    },
    {"id": 45, "name": "FA Cup", "regular_round_keywords": [], "fs_alias": "FA Cup"},
    {"id": 46, "name": "EFL Trophy", "regular_round_keywords": [], "fs_alias": "EFL Trophy"},
    {"id": 81, "name": "DFB Pokal", "regular_round_keywords": [], "fs_alias": "DFB Pokal"},
    {
        "id": 66,
        "name": "Coupe de France",
        "regular_round_keywords": [],
        "fs_alias": "Coupe de France",
    },
    {"id": 137, "name": "Coppa Italia", "regular_round_keywords": [], "fs_alias": "Coppa Italia"},
    {"id": 143, "name": "Copa del Rey", "regular_round_keywords": [], "fs_alias": "Copa del Rey"},
    {"id": 90, "name": "KNVB Beker", "regular_round_keywords": [], "fs_alias": "KNVB Cup"},  # NED
    {
        "id": 96,
        "name": "TaÃ§a de Portugal",
        "regular_round_keywords": [],
        "fs_alias": "TaÃ§a de Portugal",
    },
    {
        "id": 97,
        "name": "TaÃ§a da Liga",
        "regular_round_keywords": [],
        "fs_alias": "Portuguese League Cup",
    },
    {"id": 108, "name": "Cup", "regular_round_keywords": [], "fs_alias": "Polish Cup"},  # POL
    {"id": 209, "name": "Schweizer Cup", "regular_round_keywords": [], "fs_alias": "Swiss Cup"},
    {"id": 206, "name": "Cup", "regular_round_keywords": [], "fs_alias": "Turkish Cup"},  # TUR
    {
        "id": 121,
        "name": "DBU Pokalen",
        "regular_round_keywords": [],
        "fs_alias": "Danish Cup",
    },  # DEN
    {"id": 147, "name": "Cup", "regular_round_keywords": [], "fs_alias": "Belgian Cup"},  # BEL
    {"id": 181, "name": "FA Cup", "regular_round_keywords": [], "fs_alias": "Scottish Cup"},  # SCO
    {
        "id": 185,
        "name": "League Cup",
        "regular_round_keywords": [],
        "fs_alias": "Scottish League Cup",
    },  # SCO
    {"id": 220, "name": "Cup", "regular_round_keywords": [], "fs_alias": "Austrian Cup"},  # AUT
    {"id": 504, "name": "King's Cup", "regular_round_keywords": [], "fs_alias": "Kings Cup"},  # SA
    {
        "id": 874,
        "name": "Australia Cup",
        "regular_round_keywords": [],
        "fs_alias": "FFA Cup",
    },  # AUS
]

INIT_ELO = 1500

WINNER_TEAM_ID_CODE_FOR_DRAW = -1

NUM_NUMERICAL_FEATURES = 70
NUM_MATCHES_PER_ROUND_FOR_TRAINING = 50

COMP_ID_EMBEDDING_SIZE = 5
TEAM_ID_EMBEDDING_SIZE = 8
TEAM_STRENGTH_EMBEDDING_SIZE = 24

COMP_ID_EMBEDDING_MODEL_PATH = "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\apifootball_model\\learned_models\\comp_id_embedding_model.keras"
TEAM_ID_EMBEDDING_MODEL_PATH = "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\apifootball_model\\learned_models\\team_id_embedding_model.keras"
TEAM_STRENGTH_EMBEDDING_MODEL_PATH = "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\apifootball_model\\learned_models\\team_strength_embedding_model_gk_outfield_balanced.keras"
MAIN_MODEL_PATH = "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\apifootball_model\\learned_models\\main_model_ann.keras"

ALMOST_ZERO = 0.001
ALMOST_ONE = 0.999

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

CSV_PLAYERS_PATH = (
    "C:\\Users\\kutalekj\\PycharmProjects\\sofifa-web-scraper\\output_optimized_phase5\\full"
)

PLAYER_SKILLS = [
    "crossing",
    "finishing",
    "heading_accuracy",
    "short_passing",
    "volleys",
    "dribbling",
    "curve",
    "fk_accuracy",
    "long_passing",
    "ball_control",
    "acceleration",
    "sprint_speed",
    "agility",
    "reactions",
    "balance",
    "shot_power",
    "jumping",
    "stamina",
    "strength",
    "long_shots",
    "aggression",
    "interceptions",
    "positioning",
    "vision",
    "penalties",
    "composure",
    "defensive_awareness",
    "standing_tackle",
    "sliding_tackle",
    "gk_diving",
    "gk_handling",
    "gk_kicking",
    "gk_positioning",
    "gk_reflexes",
]

TEAM_STRENGTH_NORM_PERCENTILES = {
    "gk_p1": [
        0.08,
        0.05,
        0.09,
        0.15,
        0.05,
        0.05,
        0.09,
        0.08,
        0.13,
        0.10,
        0.17,
        0.17,
        0.21,
        0.41,
        0.21,
        0.38,
        0.32,
        0.16,
        0.35,
        0.05,
        0.14,
        0.06,
        0.04,
        0.14,
        0.10,
        0.22,
        0.05,
        0.09,
        0.08,
        0.55,
        0.52,
        0.51,
        0.51,
        0.55,
    ],
    "gk_p99": [
        0.29,
        0.20,
        0.26,
        0.60,
        0.20,
        0.30,
        0.33,
        0.29,
        0.62,
        0.42,
        0.63,
        0.61,
        0.68,
        0.85,
        0.67,
        0.65,
        0.80,
        0.45,
        0.80,
        0.20,
        0.45,
        0.30,
        0.20,
        0.69,
        0.46,
        0.69,
        0.29,
        0.23,
        0.24,
        0.86,
        0.84,
        0.87,
        0.86,
        0.89,
    ],
    "outfield_p1": [
        0.25,
        0.20,
        0.32,
        0.45,
        0.20,
        0.31,
        0.23,
        0.22,
        0.33,
        0.42,
        0.34,
        0.34,
        0.33,
        0.47,
        0.33,
        0.30,
        0.36,
        0.43,
        0.34,
        0.20,
        0.30,
        0.14,
        0.22,
        0.28,
        0.28,
        0.43,
        0.16,
        0.15,
        0.14,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
    ],
    "outfield_p99": [
        0.83,
        0.83,
        0.84,
        0.85,
        0.80,
        0.86,
        0.84,
        0.81,
        0.83,
        0.86,
        0.91,
        0.91,
        0.91,
        0.85,
        0.91,
        0.85,
        0.91,
        0.91,
        0.91,
        0.82,
        0.87,
        0.83,
        0.84,
        0.84,
        0.82,
        0.85,
        0.83,
        0.84,
        0.82,
        0.16,
        0.16,
        0.16,
        0.16,
        0.16,
    ],
}

SIMILARITY_THRESHOLD_FS_SOFIFA = 55  # rapidfuzz ratio ranges from 0 to 100
MAX_MISSING_SF_SKILL_VALUES_ALLOWED = 5
MAX_TIMEDELTA_SF_PLAYER_SKILL = timedelta(days=270)  # 9 months

BOARD_QUEUE_PATH = (
    "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\Board\\board_queue_rel.json"
)
FIREBASE_CREDENTIALS_PATH = "C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\apifootball_model\\boardmobile-61491-firebase-adminsdk-fbsvc-5839d80385.json"

# Repo root = .../MyFlashscoreScraper
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

AVG_TEAM_STRENGTHS = PROCESSED_DIR / "avg_team_strengths.csv"
RECORDS_CSV = PROCESSED_DIR / "records.csv"

M_LOAD_CSV = PROCESSED_DIR / "m_25-09-18_BEL.csv"
M_STORE_CSV = PROCESSED_DIR / "m_25-09-18_BEL.csv"
ALL_COMPS_CSV = PROCESSED_DIR / "all-comps_25-09-18.csv"
ALL_MATCHES_CSV = PROCESSED_DIR / "all-matches_25-09-18.csv"
ALL_ROUNDS_CSV = PROCESSED_DIR / "all-rounds_25-09-18.csv"
ALL_TEAMS_CSV = PROCESSED_DIR / "all-teams_25-09-18.csv"
