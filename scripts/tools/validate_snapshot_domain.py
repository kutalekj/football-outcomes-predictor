from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from football_outcomes.application.snapshot_selection import (
    resolve_snapshot_path,
)
from football_outcomes.data.snapshots import (
    load_snapshot,
)
from football_outcomes.validation.domain import (
    validate_bundle_domain,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Validate primary IDs and domain " "relationships in a FootyStats " "snapshot.")
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help=("Snapshot path. When omitted, " "FOP_LOAD_SNAPSHOT_PATH is used."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Optional JSON output path."),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help=("Maximum examples retained for " "each finding code."),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    snapshot_path = resolve_snapshot_path(args.snapshot)
    bundle = load_snapshot(snapshot_path)

    report = validate_bundle_domain(
        bundle,
        max_examples_per_finding=(args.max_examples),
    )

    payload = {
        "validation": "domain_integrity",
        "snapshot_filename": (snapshot_path.name),
        **report.to_dict(),
    }

    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.output.write_text(
            rendered + "\n",
            encoding="utf-8",
        )

    print(rendered)

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
