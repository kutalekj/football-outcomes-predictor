from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
)
from typing import Any
from urllib.parse import urlencode

import requests

from football_outcomes.config import fs_settings as sett

JsonObject = dict[str, Any]
HttpGet = Callable[[str], Any]


class FootyStatsClient:
    """Small transport boundary for FootyStats HTTP calls."""

    def __init__(
        self,
        *,
        host: str,
        api_key: str,
        http_get: HttpGet = requests.get,
    ) -> None:
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.http_get = http_get

    def build_url(
        self,
        endpoint: str,
        params: (
            Mapping[
                str,
                object,
            ]
            | None
        ) = None,
    ) -> str:
        query: dict[
            str,
            object,
        ] = {"key": self.api_key}

        if params is not None:
            query.update(params)

        return f"{self.host}/" f"{endpoint.lstrip('/')}?" f"{urlencode(query)}"

    def get_json(
        self,
        endpoint: str,
        params: (
            Mapping[
                str,
                object,
            ]
            | None
        ) = None,
    ) -> JsonObject:
        response = self.http_get(
            self.build_url(
                endpoint,
                params,
            )
        )
        return response.json()

    def get_data(
        self,
        endpoint: str,
        params: (
            Mapping[
                str,
                object,
            ]
            | None
        ) = None,
    ) -> Any:
        return self.get_json(
            endpoint,
            params,
        )["data"]

    def get_paginated_data(
        self,
        endpoint: str,
        params: (
            Mapping[
                str,
                object,
            ]
            | None
        ) = None,
    ) -> list[dict[str, Any]]:
        base_params = dict(params or {})

        first_page = self.get_json(
            endpoint,
            base_params,
        )

        rows = list(first_page["data"])

        pager = first_page.get(
            "pager",
            {},
        )
        maximum_page = int(
            pager.get(
                "max_page",
                1,
            )
        )

        for page in range(
            2,
            maximum_page + 1,
        ):
            page_params = dict(base_params)
            page_params["page"] = page

            page_data = self.get_json(
                endpoint,
                page_params,
            )
            rows.extend(page_data["data"])

        return rows


def create_default_client() -> FootyStatsClient:
    return FootyStatsClient(
        host=sett.FS_HOST,
        api_key=sett.FS_KEY,
    )
