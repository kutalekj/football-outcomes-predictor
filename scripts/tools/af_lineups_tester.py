import json
import http.client
from datetime import timedelta, datetime
import settings


conn = http.client.HTTPSConnection(settings.HOST)
conn.request("GET", "/timezone", headers=settings.HEADERS)
res = conn.getresponse()
data = res.read()
data_tz = json.loads(data)

yesterday = datetime.now() - timedelta(days=1)
tomorrow = datetime.now() + timedelta(days=1)

# API call
request_string = "/fixtures?season=" + str(settings.LAST_SEASON) + "&league=" + str(218) + "&from=" + \
                 yesterday.strftime("%Y-%m-%d") + "&to=" + tomorrow.strftime("%Y-%m-%d") + "&timezone=Europe/Amsterdam"

conn = http.client.HTTPSConnection(settings.HOST)
conn.request("GET", request_string, headers=settings.HEADERS)
res = conn.getresponse()
data = res.read()
data_fixtures = json.loads(data)

# Lineups test
lineups_request_string = "/fixtures/lineups?fixture=" + str(1361671)
conn.request("GET", lineups_request_string, headers=settings.HEADERS)
res = conn.getresponse()
data = res.read()
data_lineups = json.loads(data)['response']
