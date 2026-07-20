from pathlib import Path

from football_outcomes.data import (
    footystats_client,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Response:
    def __init__(
        self,
        payload,
    ):
        self.payload = payload

    def json(self):
        return self.payload


def test_url_contains_key_and_parameters() -> None:
    client = footystats_client.FootyStatsClient(
        host="https://example.test/",
        api_key="secret key",
        http_get=lambda url: None,
    )

    url = client.build_url(
        "/league-season",
        {
            "season_id": 123,
            "include": "stats",
        },
    )

    assert url == ("https://example.test/" "league-season?" "key=secret+key&" "season_id=123&" "include=stats")


def test_get_data_returns_data_payload() -> None:
    requested_urls = []

    def fake_get(url):
        requested_urls.append(url)

        return Response({"data": {"id": 10}})

    client = footystats_client.FootyStatsClient(
        host="https://example.test",
        api_key="key",
        http_get=fake_get,
    )

    result = client.get_data(
        "match",
        {"match_id": 10},
    )

    assert result == {"id": 10}

    assert len(requested_urls) == 1
    assert requested_urls[0] == ("https://example.test/" "match?" "key=key&" "match_id=10")


def test_paginated_data_reads_all_pages() -> None:
    requested_urls = []

    def fake_get(url):
        requested_urls.append(url)

        if "page=2" in url:
            return Response(
                {
                    "data": [{"id": 3}],
                    "pager": {"max_page": 2},
                }
            )

        return Response(
            {
                "data": [
                    {"id": 1},
                    {"id": 2},
                ],
                "pager": {"max_page": 2},
            }
        )

    client = footystats_client.FootyStatsClient(
        host="https://example.test",
        api_key="key",
        http_get=fake_get,
    )

    rows = client.get_paginated_data(
        "league-players",
        {"season_id": 50},
    )

    assert rows == [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]

    assert len(requested_urls) == 2

    assert requested_urls[0] == ("https://example.test/" "league-players?" "key=key&" "season_id=50")

    assert requested_urls[1] == ("https://example.test/" "league-players?" "key=key&" "season_id=50&" "page=2")


def test_missing_pager_defaults_to_one_page() -> None:
    requested_urls = []

    def fake_get(url):
        requested_urls.append(url)

        return Response({"data": [{"id": 1}]})

    client = footystats_client.FootyStatsClient(
        host="https://example.test",
        api_key="key",
        http_get=fake_get,
    )

    rows = client.get_paginated_data("league-matches")

    assert rows == [{"id": 1}]
    assert len(requested_urls) == 1


def test_retrieval_module_has_no_direct_transport_logic() -> None:
    source_path = PROJECT_ROOT / "src" / "football_outcomes" / "data" / "fs_retrieve.py"
    source = source_path.read_text(encoding="utf-8")

    assert "requests.get" not in source
    assert "import requests" not in source
    assert "FS_HOST +" not in source
    assert "FS_KEY +" not in source

    assert "get_data(" in source
    assert "get_paginated_data(" in source
