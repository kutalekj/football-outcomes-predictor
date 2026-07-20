from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from football_outcomes.validation.coverage import (
    validate_coverage_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Classify findings in a snapshot " "coverage inventory."))
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    payload = json.loads(
        args.input.read_text(
            encoding="utf-8",
        )
    )

    summary = payload.get("summary")
    competition_seasons = payload.get("competition_seasons")

    if not isinstance(
        summary,
        dict,
    ):
        raise ValueError("Coverage input has no valid " "summary object.")

    if not isinstance(
        competition_seasons,
        list,
    ):
        raise ValueError("Coverage input has no valid " "competition_seasons array.")

    report = validate_coverage_summary(
        summary,
        competition_seasons,
        max_examples_per_finding=(args.max_examples),
    )

    output_payload = {
        "validation": ("sofifa_lineup_strength_coverage"),
        "coverage_source": (args.input.name),
        **report.to_dict(),
    }

    rendered = json.dumps(
        output_payload,
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
