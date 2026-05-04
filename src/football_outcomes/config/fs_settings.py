from pathlib import Path

FS_KEY = "9360c5f9b742b0177a1e42b1afee860151cab101673147456e60412da6d46b38"
FS_HOST = "https://api.football-data-api.com"

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../football-outcomes-predictor
BASE_SRC_DIR = Path(__file__).resolve().parents[1]  # .../src/football_outcomes

DATA_DIR = PROJECT_ROOT / "data"  # .../football-outcomes-predictor/data
PROCESSED_DIR = DATA_DIR / "processed"  # .../data/processed
LOG_DIR = PROCESSED_DIR / "logs"  # .../data/processed/logs

LOAD_SNAPSHOT_PATH = BASE_SRC_DIR / "data/cache" / "fs_full_26-01-26_NO-25-26-MATCHES_v5.pkl"
SAVE_SNAPSHOT_PATH = BASE_SRC_DIR / "data/cache" / "fs_full_26-01-26_NO-25-26-MATCHES_v5.pkl"
AVG_TEAM_STRENGTH_PATH = PROCESSED_DIR / "avg_team_strengths.csv"

ALL_LOAD = True
ALL_GET_NEW = False
ALL_STORE = False

FIRST_SEASON = 2021
LAST_SEASON = 2025

SOFIFA_CSV_DIR = "C:\\Users\\kutalekj\\PycharmProjects\\sofifa-web-scraper\\output_optimized_phase7\\full"
SOFIFA_FILENAME_DATE_FORMAT = "%Y-%m-%d"  # e.g. 2024-01-15.csv
REBUILD_SOFIFA_FROM_CSV = False
REBUILD_REGULAR_SEASON_FLAGS = False

ALMOST_ZERO = 1e-6
ALMOST_ONE = 1.0 - 1e-6

ELO_K = 32.0
ELO_SEASON_REGRESSION = 0.75
MIN_ELO_MATCHES = 5  # min number of matches in dataset for opponent's ELO being reliable
ELO_NON_LEAGUE_WEIGHT = 0.25  # down-weight domestic and European cups
WINNER_TEAM_ID_CODE_FOR_DRAW = -1

SF_MATCH_LOW_THRESHOLD = 40  # names matching (when both team and DOB match)
SF_MATCH_HIGH_THRESHOLD = 69.9  # names matching (when team matches but DOB doesn't)
SF_MATCH_MODERATE_THRESHOLD = 60  # names matching (when DOB matches but team doesn't)
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

TEAM_STRENGTH_NUM_PLAYERS = 11
TEAM_STRENGTH_NUM_SKILLS = 34  # should equal to len(PLAYER_SKILLS)
GK_SKILL_START_INDEX = 29
GK_SKILL_END_INDEX = 34  # Python slice end
FORCE_EXACTLY_ONE_GK_ROW = True
GK_ROLE_SCORE_MIN_GAP = 0.5  # minimal separation to treat as GK-like (TODO: tune)
DEBUG_TEAM_STRENGTH = False  # optional log

SHOTS_ON_G_NORM_COEFFICIENT = 12.0
SHOTS_OFF_G_NORM_COEFFICIENT = 18.0
GOALS_NORM_COEFFICIENT = 5.0
MATCH_LOAD_NORM_COEFFICIENT = 0.24
TOTAL_SHOTS_NORM_COEFFICIENT = 29.0
CORNER_KICKS_NORM_COEFFICIENT = 13.0
FOULS_NORM_COEFFICIENT = 22.0
ATTACKS_NORM_COEFFICIENT = 161.0
DANG_ATTACKS_NORM_COEFFICIENT = 100.0

TEAM_XG_NORM_COEFFICIENT = 2.86
TOTAL_XG_NORM_COEFFICIENT = 4.38
TEAM_PRE_MATCH_XG_NORM_COEFFICIENT = 2.35
TOTAL_PRE_MATCH_XG_NORM_COEFFICIENT = 4.106

ELO_MIN_NORM_COEFFICIENT = 1352.5387
ELO_MAX_NORM_COEFFICIENT = 1734.8060
INIT_ELO = 1543.6724  # (ELO_MIN_NORM_COEFFICIENT + ELO_MAX_NORM_COEFFICIENT) / 2

# Competition seasons to exclude from cleaned league-only analyses/training
EXCLUDED_COMP_SEASONS = {
    ("Italy Serie B", 2021),
    ("Poland Ekstraklasa", 2021),
    ("Turkey Süper Lig", 2022),
    ("India Indian Super League", 2024),
}

# Raw match stats that remain stored in match.stats, but are ignored in cleaned analyses/modeling
IGNORED_MATCH_STATS = {
    "home_offsides",
    "away_offsides",
}

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

VALIDATE_ROUND_IDS = True
LEAGUE_VALID_ROUND_IDS_BY_SEASON = {
    ("Australia A-League", 2021): {76909},
    ("Australia A-League", 2022): {92941},
    ("Australia A-League", 2023): {103342},
    ("Australia A-League", 2024): {114726},
    ("Austria Bundesliga", 2021): {70480, 70483, 70484},
    ("Austria Bundesliga", 2022): {92495, 92498, 92499},
    ("Austria Bundesliga", 2023): {101611, 101612, 101613},
    ("Austria Bundesliga", 2024): {110504, 110505, 110506},
    ("Belgium Pro League", 2021): {71342},
    ("Belgium Pro League", 2022): {91136},
    ("Belgium Pro League", 2023): {100236, 100237, 100238},
    ("Belgium Pro League", 2024): {109364, 109365, 109366},
    ("Denmark Superliga", 2021): {69687, 69689, 69690},
    ("Denmark Superliga", 2022): {90173, 90175, 90176},
    ("Denmark Superliga", 2023): {99948, 99950, 99951},
    ("Denmark Superliga", 2024): {109325, 109326, 109327},
    ("England Championship", 2021): {71403},
    ("England Championship", 2022): {91360},
    ("England Championship", 2023): {100549},
    ("England Championship", 2024): {110437},
    ("England EFL League One", 2021): {70530},
    ("England EFL League One", 2022): {91270},
    ("England EFL League One", 2023): {100257},
    ("England EFL League One", 2024): {110421},
    ("England EFL League Two", 2021): {70518},
    ("England EFL League Two", 2022): {91288},
    ("England EFL League Two", 2023): {100252},
    ("England EFL League Two", 2024): {110360},
    ("England Premier League", 2021): {72035},
    ("England Premier League", 2022): {91757},
    ("England Premier League", 2023): {100543},
    ("England Premier League", 2024): {110026},
    ("France Ligue 1", 2021): {70544},
    ("France Ligue 1", 2022): {90935},
    ("France Ligue 1", 2023): {100611},
    ("France Ligue 1", 2024): {110067},
    ("France Ligue 2", 2021): {70537},
    ("France Ligue 2", 2022): {90937},
    ("France Ligue 2", 2023): {100424},
    ("France Ligue 2", 2024): {110068},
    ("Germany Bundesliga", 2021): {72347},
    ("Germany Bundesliga", 2022): {91639},
    ("Germany Bundesliga", 2023): {100525},
    ("Germany Bundesliga", 2024): {110730},
    ("Germany 2. Bundesliga", 2021): {70548},
    ("Germany 2. Bundesliga", 2022): {90933},
    ("Germany 2. Bundesliga", 2023): {100527},
    ("Germany 2. Bundesliga", 2024): {110729},
    ("India Indian Super League", 2021): {76989},
    ("India Indian Super League", 2022): {94212},
    ("India Indian Super League", 2023): {103943},
    ("Italy Serie A", 2021): {72384},
    ("Italy Serie A", 2022): {91440},
    ("Italy Serie A", 2023): {100740},
    ("Italy Serie A", 2024): {110731},
    ("Italy Serie B", 2022): {92413},
    ("Italy Serie B", 2023): {101135},
    ("Italy Serie B", 2024): {111101},
    ("Netherlands Eredivisie", 2021): {69596},
    ("Netherlands Eredivisie", 2022): {90847},
    ("Netherlands Eredivisie", 2023): {100519},
    ("Netherlands Eredivisie", 2024): {110009},
    ("Poland Ekstraklasa", 2022): {90215},
    ("Poland Ekstraklasa", 2023): {100000},
    ("Poland Ekstraklasa", 2024): {109278},
    ("Portugal Liga NOS", 2021): {71608},
    ("Portugal Liga NOS", 2022): {91901},
    ("Portugal Liga NOS", 2023): {101710},
    ("Portugal Liga NOS", 2024): {111861},
    ("Saudi Arabia Professional League", 2021): {72452},
    ("Saudi Arabia Professional League", 2022): {93293},
    ("Saudi Arabia Professional League", 2023): {101406},
    ("Saudi Arabia Professional League", 2024): {111535},
    ("Scotland Premiership", 2021): {70405, 70406},
    ("Scotland Premiership", 2022): {90927, 90928},
    ("Scotland Premiership", 2023): {100485, 100486},
    ("Scotland Premiership", 2024): {110446, 110447},
    ("Spain La Liga", 2021): {72449},
    ("Spain La Liga", 2022): {91641},
    ("Spain La Liga", 2023): {100561},
    ("Spain La Liga", 2024): {109991},
    ("Spain Segunda División", 2021): {71625},
    ("Spain Segunda División", 2022): {91355},
    ("Spain Segunda División", 2023): {100613},
    ("Spain Segunda División", 2024): {110497},
    ("Switzerland Super League", 2021): {70670},
    ("Switzerland Super League", 2022): {90945},
    ("Switzerland Super League", 2023): {100248, 100249, 100250},
    ("Switzerland Super League", 2024): {110027, 110028, 110029},
    ("Turkey Süper Lig", 2021): {71656},
    ("Turkey Süper Lig", 2023): {101503},
    ("Turkey Süper Lig", 2024): {111167},
}

COMPS_LEAGUE_COLORS = {
    "Belgium Pro League": "aquamarine",
    "England Premier League": "blue",
    "England Championship": "royalblue",
    "England EFL League One": "cornflowerblue",
    "England EFL League Two": "dodgerblue",
    "France Ligue 1": "blueviolet",
    "France Ligue 2": "mediumorchid",
    "Netherlands Eredivisie": "orange",
    "Turkey Süper Lig": "rosybrown",
    "Germany Bundesliga": "gold",
    "Germany 2. Bundesliga": "goldenrod",
    "Saudi Arabia Professional League": "darkseagreen",
    "India Indian Super League": "darkkhaki",
    "Australia A-League": "silver",
    "Austria Bundesliga": "lavenderblush",
    "Spain La Liga": "red",
    "Spain Segunda División": "tomato",
    "Italy Serie A": "limegreen",
    "Italy Serie B": "lightgreen",
    "Scotland Premiership": "lavender",
    "Poland Ekstraklasa": "lightpink",
    "Denmark Superliga": "thistle",
    "Portugal Liga NOS": "honeydew",
    "Switzerland Super League": "paleturquoise",
}

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

VALID_FS_PLAYER_POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

FS_PLAYER_POSITION_TO_IDX = {
    "Goalkeeper": 0,
    "Defender": 1,
    "Midfielder": 2,
    "Forward": 3,
    "Unknown": 4,
}
