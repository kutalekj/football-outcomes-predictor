from __future__ import annotations

import os
from pathlib import Path

SNAPSHOT_PATH_ENV = "FOP_LOAD_SNAPSHOT_PATH"


def resolve_snapshot_path(
    value: str | Path | None = None,
) -> Path:
    """
    Resolve an explicitly supplied snapshot path,
    falling back to the documented environment
    variable.
    """

    candidate = value

    if candidate is None:
        candidate = os.getenv(SNAPSHOT_PATH_ENV)

    if candidate is None:
        raise RuntimeError("A snapshot path is required. " "Pass it explicitly or set " f"{SNAPSHOT_PATH_ENV}.")

    candidate_text = str(candidate).strip()

    if not candidate_text:
        raise RuntimeError("The snapshot path must not be empty.")

    return Path(candidate_text).expanduser()
