from pathlib import Path

FS_KEY = "9360c5f9b742b0177a1e42b1afee860151cab101673147456e60412da6d46b38"
FS_HOST = "https://api.football-data-api.com"

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../football-outcomes-predictor
BASE_SRC_DIR = Path(__file__).resolve().parents[1]  # .../src/football_outcomes

DATA_DIR = PROJECT_ROOT / "data"  # .../football-outcomes-predictor/data
PROCESSED_DIR = DATA_DIR / "processed"  # .../data/processed
LOG_DIR = PROCESSED_DIR / "logs"  # .../data/processed/logs

LOAD_SNAPSHOT_PATH = BASE_SRC_DIR / "data/cache" / "fs_full_26-01-22_fix_snap_v3.pkl"
SAVE_SNAPSHOT_PATH = BASE_SRC_DIR / "data/cache" / "fs_full_26-01-22_fix_snap_v3.pkl"
AVG_TEAM_STRENGTH_PATH = PROCESSED_DIR / "avg_team_strengths.csv"

ALL_LOAD = True
ALL_GET_NEW = False
ALL_STORE = False

FIRST_SEASON = 2021
LAST_SEASON = 2025

SOFIFA_CSV_DIR = "C:\\Users\\kutalekj\\PycharmProjects\\sofifa-web-scraper\\output_optimized_phase7\\full"
SOFIFA_FILENAME_DATE_FORMAT = "%Y-%m-%d"  # e.g. 2024-01-15.csv
REBUILD_SOFIFA_FROM_CSV = False

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

# FS -> SOFIFA cache behavior
USE_FS_TO_SOFIFA_CACHE = True
FS_TO_SOFIFA_CACHE_RETRY_FAILED = True  # retry if cached sofifa_id is None
FS_TO_SOFIFA_CACHE_RETRY_AMBIGUOUS = True  # retry if margin too small
FS_TO_SOFIFA_CACHE_MIN_MARGIN = 8.0
FS_TO_SOFIFA_CACHE_ONLY_TRUST_REASONS = {
    "team_dob_pass",
    "team_only_pass",
    "dob_gate_pass",
}

# FS league name -> SOFIFA club_league_id
FS_LEAGUE_TO_SOFIFA_LEAGUE_ID = {
    "Belgium Pro League": 4,
    "England Premier League": 13,
    "England Championship": 14,
    "England EFL League One": 60,
    "England EFL League Two": 61,
    "France Ligue 1": 16,
    "France Ligue 2": 17,
    "Netherlands Eredivisie": 10,
    "Turkey Süper Lig": 68,
    "Germany Bundesliga": 19,
    "Germany 2. Bundesliga": 20,
    "Saudi Arabia Professional League": 350,
    "India Indian Super League": 2149,
    "Australia A-League": 351,
    "Austria Bundesliga": 80,
    "Spain La Liga": 53,
    "Spain Segunda División": 54,
    "Italy Serie A": 31,
    "Italy Serie B": 32,
    "Scotland Premiership": 50,
    "Poland Ekstraklasa": 66,
    "Denmark Superliga": 1,
    "Portugal Liga NOS": 308,
    "Switzerland Super League": 189,
}

# Manual overrides for FS team -> SOFIFA club_id
# Key by FS team id (stable), value = SOFIFA club_id or -1 for "no mapping".
FS_TEAM_ID_TO_SOFIFA_TEAM_ID = {
    669393: 8001,
    669388: 537,
    445: 72,
    442: 74,
    449: 64,
    486: 294,
    479: 57,
    432: 210,
    1399: 111659,
    447: 1823,
    482: 1813,
    487: 1814,
    121: 247,
    379: 1910,
    10058: 101006,
    6810: 110176,
    5071: 113057,
    5065: 112096,
    5066: 605,
    591: 15040,
    578: 111821,
    293: 1860,
    280: 452,
    298: 453,
    470: 44,
    2341: -1,
    512: 112791,
    2332: -1,
    6366: 1846,
    498: -1,
    32: 80,
    114: 237,
    998595: 131463,
    2523: 272,
    974: 820,
    2516: 271,
}

# --- FS team <-> SOFIFA team matching ---
SF_TEAM_MATCH_MAX_CANDIDATES = 3  # keep top-N candidates for debug

# --- Name-only fallback bucket limit ---
SF_NAME_BUCKET_MAX = 200

# --- Optional thresholds for team-based matching steps ---
SF_MATCH_TEAM_DOB_THRESHOLD = 50  # slightly easier than LOWER (DOB+same-team should be strong)
SF_MATCH_TEAM_ONLY_THRESHOLD = 80  # lower than HIGHER, but still strict

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
