"""
utils_archive.py
"""


import http.client
import json
import datetime
import utils as ut


KEY = "4a9e20eecbec58c517cb485f31552caf"

conn = http.client.HTTPSConnection("v3.football.api-sports.io")

headers = {
    'x-rapidapi-host': "v3.football.api-sports.io",
    'x-rapidapi-key': KEY
}


def get_seasons_with_odds_standings_lineups_players_events_statistics(data_leagues):
    fixtures_attributes = ['events', 'lineups', 'statistics_players', 'statistics_fixtures']
    # fixtures_attributes = []
    coverage_attributes = ['standings', 'players']

    # coverage_attributes = ['odds']

    def has_required_attributes(season_coverage, season_fixtures):
        fixtures_check = all(season_fixtures.get(attr, True) for attr in fixtures_attributes)
        coverage_check = all(season_coverage.get(attr, True) for attr in coverage_attributes)
        return fixtures_check and coverage_check

    filtered_seasons = []
    leagues = data_leagues['response']
    for league in leagues:
        seasons = league.get('seasons')
        for season in seasons:
            coverage = season.get('coverage', {})
            fixtures = coverage.get('fixtures', {})
            if has_required_attributes(coverage, fixtures):
                if int(season['year']) >= 2021:
                    filtered_seasons.append((league['country']['name'], league['league']['name'], season))

    return filtered_seasons


# Leagues
conn.request("GET", "/leagues", headers=headers)

res = conn.getresponse()
data = res.read()
data_leagues = json.loads(data)

filtered_seasons = get_seasons_with_odds_standings_lineups_players_events_statistics(data_leagues)

# Swiss Super League 2021
conn.request("GET", "/fixtures?season=2021&league=207", headers=headers)

res = conn.getresponse()
data = res.read()
data_swiss = json.loads(data)

for fixture in data_swiss['response']:
    # Statistics
    conn.request("GET", "/fixtures/statistics?fixture=" + str(fixture['fixture']['id']), headers=headers)

    res = conn.getresponse()
    data = res.read()
    data_statistics = json.loads(data)

    # Events
    conn.request("GET", "/fixtures/events?fixture=" + str(fixture['fixture']['id']), headers=headers)

    res = conn.getresponse()
    data = res.read()
    data_events = json.loads(data)

    # Lineups
    conn.request("GET", "/fixtures/lineups?fixture=" + str(fixture['fixture']['id']), headers=headers)

    res = conn.getresponse()
    data = res.read()
    data_lineups = json.loads(data)

    # Players
    conn.request("GET", "/fixtures/players?fixture=" + str(fixture['fixture']['id']), headers=headers)

    res = conn.getresponse()
    data = res.read()

    data_players = json.loads(data)

    # All rounds
    conn.request("GET", "/fixtures/rounds?league=207&season=2021", headers=headers)

    res = conn.getresponse()
    data = res.read()

    data_rounds = json.loads(data)

    # Round by comp season
    conn.request("GET", "/fixtures/rounds?league=207&season=2021&current=true", headers=headers)

    res = conn.getresponse()
    data = res.read()

    data_round_by_comp_season = json.loads(data)

# ------------------------------ Example of features ------------------------------

# Competition C, Season S
conn.request("GET", "/fixtures?season=2021&league=207", headers=headers)

res = conn.getresponse()
data = res.read()
season_fixtures = json.loads(data)

for fixture in season_fixtures['response']:
    # Match M
    fixture_id = int(fixture['fixture']['id'])  # unique match ID

    if fixture['fixture']['status']['short'] != "FT":
        raise Exception(f"Match {fixture_id} not finished")

    fixture_date = datetime.fromisoformat(fixture['fixture']['date'])
    fixture_hour = int(fixture_date.hour)  # FEATURE
    fixture_month = int(fixture_date.month)  # FEATURE

    competition_id = int(fixture['league']['id'])  # FEATURE
    competition_name = fixture['league']['name']
    country = fixture['league']['country']
    season_number = int(fixture['league']['season'])  # FEATURE

    # TODO: round_number
    # get_all_rounds_by_comp (/fixtures/rounds?league=207&season=2021)
    # get_round_by_comp_season (/fixtures/rounds?league=207&season=2021&current=true)
    # fixture['league']['round']

    home_team_id = int(fixture['teams']['home']['id'])  # FEATURE
    home_team_name = fixture['teams']['home']['name']

    winner_team_id = int(fixture['teams']['home']['winner']) if int(fixture['teams']['home']['winner']) is True else \
        int(fixture['teams']['away']['winner'])

    away_team_id = int(fixture['teams']['away']['id'])  # FEATURE
    away_team_name = fixture['teams']['away']['name']

    home_team_goals = fixture['goals']['home']
    away_team_goals = fixture['goals']['away']

    # TODO: Extra time/penalties handling?
