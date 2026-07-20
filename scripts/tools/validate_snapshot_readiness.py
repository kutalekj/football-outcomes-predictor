from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from football_outcomes.application.snapshot_selection import (
    resolve_snapshot_path,
)
from football_outcomes.config import fs_settings as sett
from football_outcomes.data.snapshots import (
    load_snapshot,
)
from football_outcomes.validation.readiness import (
    FeatureReadinessConfig,
    validate_feature_readiness,
)
from football_outcomes.validation.selection import (
    SelectionValidationConfig,
    select_validation_matches,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Validate feature objects, model " "arrays and binary targets."))
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    snapshot_path = resolve_snapshot_path(args.snapshot)
    bundle = load_snapshot(snapshot_path)

    selection_config = SelectionValidationConfig(
        competitions=tuple(sett.COMPS_LEAGUE),
        first_season=(sett.FIRST_SEASON),
        last_season_exclusive=(sett.LAST_SEASON),
        excluded_competition_seasons=(frozenset(sett.EXCLUDED_COMP_SEASONS)),
        valid_round_ids_by_season=(sett.LEAGUE_VALID_ROUND_IDS_BY_SEASON),
    )

    selected = select_validation_matches(
        bundle.matches,
        selection_config,
    )

    readiness_config = FeatureReadinessConfig(
        competition_names=tuple(sett.COMPS_LEAGUE),
        chunk_size=args.chunk_size,
        max_goals_class=10,
        position_count=len(sett.FS_PLAYER_POSITION_TO_IDX),
    )

    report = validate_feature_readiness(
        selected,
        readiness_config,
        max_examples_per_finding=(args.max_examples),
    )

    payload = {
        "validation": ("feature_array_target_readiness"),
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
