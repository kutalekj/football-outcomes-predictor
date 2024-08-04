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

# {v3API_id, name}
COMPS = [
    {'id': 39, 'name': "Premier League"},
    {'id': 40, 'name': "Championship"},
    {'id': 61, 'name': "Ligue 1"},
    {'id': 62, 'name': "Ligue 2"},
    {'id': 78, 'name': "Bundesliga"},
    {'id': 79, 'name': "2. Bundesliga"},
    {'id': 88, 'name': "Eredivisie"},
    {'id': 94, 'name': "Primeira Liga"},
    {'id': 119, 'name': "Superliga"},  # DEN
    {'id': 135, 'name': "Serie A"},
    {'id': 136, 'name': "Serie B"},
    {'id': 140, 'name': "La Liga"},
    {'id': 141, 'name': "Segunda División"},
    {'id': 144, 'name': "Jupiler Pro League"},
    {'id': 203, 'name': "Süper Lig"},
    {'id': 207, 'name': "Super League"},  # SUI
    {'id': 210, 'name': "HNL"}
]

INIT_ELO = 1500
