from pathlib import Path

FS_KEY = "9360c5f9b742b0177a1e42b1afee860151cab101673147456e60412da6d46b38"
FS_HOST = "https://api.football-data-api.com"

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../football-outcomes-predictor
BASE_SRC_DIR = Path(__file__).resolve().parents[1]  # .../src/football_outcomes

DATA_DIR = PROJECT_ROOT / "data"  # .../football-outcomes-predictor/data
PROCESSED_DIR = DATA_DIR / "processed"  # .../data/processed

LOAD_SNAPSHOT_PATH = BASE_SRC_DIR / "data/cache" / "fs_full_26-01-03.pkl"
SAVE_SNAPSHOT_PATH = BASE_SRC_DIR / "data/cache" / "fs_full_26-01-03_v2.pkl"
AVG_TEAM_STRENGTH_PATH = PROCESSED_DIR / "avg_team_strengths.csv"

ALL_LOAD = True
ALL_GET_NEW = False
ALL_STORE = False

FIRST_SEASON = 2021
LAST_SEASON = 2025

SOFIFA_CSV_DIR = "C:\\Users\\kutalekj\\PycharmProjects\\sofifa-web-scraper\\output_optimized_phase6\\full"
SOFIFA_FILENAME_DATE_FORMAT = "%Y-%m-%d"  # e.g. 2024-01-15.csv

COUNTRIES = [
    "Belgium",
    "England",
    "France",
    "Netherlands",
    "Turkey",
    "Germany",
    "Saudi Arabia",
    "India",
    "Australia",
    "Austria",
    "Spain",
    "Italy",
    "Scotland",
    "Poland",
    "Denmark",
    "Portugal",
    "Switzerland",
    "Europe",
]

COMPS_LEAGUE = [
    "Belgium Pro League",
    "England Premier League",
    "England Championship",
    "England EFL League One",
    "England EFL League Two",
    "France Ligue 1",
    "France Ligue 2",
    "Netherlands Eredivisie",
    "Turkey Süper Lig",
    "Germany Bundesliga",
    "Germany 2. Bundesliga",
    "Saudi Arabia Professional League",
    "India Indian Super League",
    "Australia A-League",
    "Austria Bundesliga",
    "Spain La Liga",
    "Spain Segunda División",
    "Italy Serie A",
    "Italy Serie B",
    "Scotland Premiership",
    "Poland Ekstraklasa",
    "Denmark Superliga",
    "Portugal Liga NOS",
    "Switzerland Super League",
]

COMPS_EUROPE = ["Europe UEFA Champions League", "Europe UEFA Europa League", "Europe UEFA Europa Conference League"]

COMPS_CUP = [
    "Spain Copa del Rey",
    "Scotland Scottish League Cup",
    "Scotland Scottish Cup",
    "Poland Polish Cup",
    "Turkey Turkish Cup",
    "Switzerland Swiss Cup",
    "Saudi Arabia Kings Cup",
    "Portugal Taça de Portugal",
    "Portugal Portuguese League Cup",
    "Netherlands KNVB Cup",
    "Austria Austrian Cup",
    "Germany DFB Pokal",
    "Italy Coppa Italia",
    "France Coupe de France",
    "England FA Cup",
    "England EFL Trophy",
    "Denmark Danish Cup",
    "Belgium Belgian Cup",
    "Australia FFA Cup",
]

COMPS_QUICK_TEST = [
    "Belgium Pro League",
    "Europe UEFA Champions League",
    "Europe UEFA Europa League",
    "Europe UEFA Europa Conference League",
    "Belgium Belgian Cup",
]

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
