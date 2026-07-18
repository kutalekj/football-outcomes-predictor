from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

Record = Mapping[str, Any]


def write_records_csv(
    path: Path,
    records: Sequence[Record],
) -> bool:
    """
    Write records using the sorted union of their keys.

    Returns False without creating a file when there
    are no records.
    """

    if not records:
        return False

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = sorted({key for record in records for key in record})

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records)

    return True


def write_json(
    path: Path,
    payload: Any,
) -> None:
    """Write a JSON artifact with stable indentation."""

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )
