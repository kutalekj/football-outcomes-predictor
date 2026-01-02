from pathlib import Path

FS_KEY = "9360c5f9b742b0177a1e42b1afee860151cab101673147456e60412da6d46b38"
FS_HOST = "https://api.football-data-api.com"

BASE_SRC_DIR = Path(__file__).resolve().parents[1]  # .../src/football_outcomes
LOAD_SNAPSHOT_PATH = BASE_SRC_DIR / "data/cache" / "fs_full_26-01-02.pkl"
SAVE_SNAPSHOT_PATH = BASE_SRC_DIR / "data/cache" / "fs_full_26-01-02_v2.pkl"

ALL_LOAD = True
ALL_GET_NEW = True
ALL_STORE = True

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
