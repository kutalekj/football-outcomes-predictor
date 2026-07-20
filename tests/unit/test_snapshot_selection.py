from pathlib import Path

import pytest

from football_outcomes.application import (
    snapshot_selection,
)


def test_explicit_snapshot_path_wins(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "FOP_LOAD_SNAPSHOT_PATH",
        "environment.pkl",
    )

    result = snapshot_selection.resolve_snapshot_path("explicit.pkl")

    assert result == Path("explicit.pkl")


def test_environment_snapshot_path_is_used(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "FOP_LOAD_SNAPSHOT_PATH",
        "environment.pkl",
    )

    result = snapshot_selection.resolve_snapshot_path()

    assert result == Path("environment.pkl")


def test_missing_snapshot_path_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "FOP_LOAD_SNAPSHOT_PATH",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="snapshot path is required",
    ):
        (snapshot_selection.resolve_snapshot_path())
