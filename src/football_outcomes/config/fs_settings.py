from pathlib import Path

FS_KEY = "9360c5f9b742b0177a1e42b1afee860151cab101673147456e60412da6d46b38"
FS_HOST = "https://api.football-data-api.com"

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../football-outcomes-predictor
BASE_SRC_DIR = Path(__file__).resolve().parents[1]  # .../src/football_outcomes

DATA_DIR = PROJECT_ROOT / "data"  # .../football-outcomes-predictor/data
PROCESSED_DIR = DATA_DIR / "processed"  # .../data/processed

LOAD_SNAPSHOT_PATH = BASE_SRC_DIR / "data/cache" / "fs_full_26-01-05.pkl"
SAVE_SNAPSHOT_PATH = BASE_SRC_DIR / "data/cache" / "fs_full_26-01-05.pkl"
AVG_TEAM_STRENGTH_PATH = PROCESSED_DIR / "avg_team_strengths.csv"

ALL_LOAD = True
ALL_GET_NEW = False
ALL_STORE = False

FIRST_SEASON = 2021
LAST_SEASON = 2025

SOFIFA_CSV_DIR = "C:\\Users\\kutalekj\\PycharmProjects\\sofifa-web-scraper\\output_optimized_phase6\\full"
SOFIFA_FILENAME_DATE_FORMAT = "%Y-%m-%d"  # e.g. 2024-01-15.csv

ALMOST_ZERO = 1e-6
ALMOST_ONE = 1.0 - 1e-6

INIT_ELO = 1500.0
ELO_K = 32.0
ELO_SEASON_REGRESSION = 0.75
MIN_ELO_MATCHES = 5  # min number of matches in dataset for opponent's ELO being reliable
ELO_NON_LEAGUE_WEIGHT = 0.25  # down-weight domestic and European cups
WINNER_TEAM_ID_CODE_FOR_DRAW = -1

SF_MATCH_LOWER_THRESHOLD = 55  # names matching (when DOB matches)
SF_MATCH_HIGHER_THRESHOLD = 85  # names matching (when DOB doesn't match)
SF_MAX_TIMEDELTA_DAYS = 120  # snapshot search (within +/- N days of match date)
SF_MAX_SNAPSHOTS_TO_SCAN = 6  # num of snapshots to search (past+future ordered)

TEAM_STRENGTH_NUM_PLAYERS = 11
TEAM_STRENGTH_NUM_SKILLS = 34  # should equal to len(PLAYER_SKILLS)
GK_SKILL_START_INDEX = 29
GK_SKILL_END_INDEX = 34  # Python slice end
FORCE_EXACTLY_ONE_GK_ROW = True
GK_ROLE_SCORE_MIN_GAP = 0.5  # minimal separation to treat as GK-like (TODO: tune)
DEBUG_TEAM_STRENGTH = True  # optional log

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
