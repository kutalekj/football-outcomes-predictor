from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from football_outcomes.application.snapshot_selection import (
    resolve_snapshot_path,
)
from football_outcomes.validation.reporting import (
    combine_validation_reports,
    render_validation_markdown,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "scripts" / "tools"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=("Run the complete deterministic " "Step 6 snapshot validation."))
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
    )
    return parser


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    source_path = str(PROJECT_ROOT / "src")
    existing = environment.get("PYTHONPATH")

    environment["PYTHONPATH"] = source_path if not existing else (source_path + os.pathsep + existing)

    return environment


def _run_tool(
    script_name: str,
    arguments: Sequence[str],
    *,
    accepted_exit_codes: set[int],
) -> None:
    command = [
        sys.executable,
        str(TOOLS_DIR / script_name),
        *arguments,
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode not in accepted_exit_codes:
        raise RuntimeError(
            f"{script_name} failed with "
            f"exit code "
            f"{result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def _load_json(
    path: Path,
) -> dict:
    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(f"{path} does not contain a " "JSON object.")

    return payload


def main(
    argv: Sequence[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    snapshot_path = resolve_snapshot_path(args.snapshot)

    args.json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.csv_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.markdown_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(prefix="fop-step-6-") as temporary_directory:
        temporary = Path(temporary_directory)

        domain_path = temporary / "domain.json"
        selection_path = temporary / "selection.json"
        readiness_path = temporary / "readiness.json"
        coverage_inventory_path = temporary / "coverage-inventory.json"
        coverage_csv_path = temporary / "coverage-by-scope.csv"
        coverage_findings_path = temporary / "coverage-findings.json"

        common_snapshot_arguments = [
            "--snapshot",
            str(snapshot_path),
        ]

        _run_tool(
            "validate_snapshot_domain.py",
            [
                *common_snapshot_arguments,
                "--output",
                str(domain_path),
                "--max-examples",
                str(args.max_examples),
            ],
            accepted_exit_codes={
                0,
                1,
            },
        )

        _run_tool(
            "validate_snapshot_selection.py",
            [
                *common_snapshot_arguments,
                "--output",
                str(selection_path),
                "--max-examples",
                str(args.max_examples),
            ],
            accepted_exit_codes={
                0,
                1,
            },
        )

        _run_tool(
            "validate_snapshot_readiness.py",
            [
                *common_snapshot_arguments,
                "--output",
                str(readiness_path),
                "--chunk-size",
                str(args.chunk_size),
                "--max-examples",
                str(args.max_examples),
            ],
            accepted_exit_codes={
                0,
                1,
            },
        )

        _run_tool(
            "inspect_snapshot_coverage.py",
            [
                *common_snapshot_arguments,
                "--json-output",
                str(coverage_inventory_path),
                "--csv-output",
                str(coverage_csv_path),
            ],
            accepted_exit_codes={0},
        )

        _run_tool(
            "validate_snapshot_coverage.py",
            [
                "--input",
                str(coverage_inventory_path),
                "--output",
                str(coverage_findings_path),
                "--max-examples",
                str(args.max_examples),
            ],
            accepted_exit_codes={
                0,
                1,
            },
        )

        domain = _load_json(domain_path)
        selection = _load_json(selection_path)
        readiness = _load_json(readiness_path)
        coverage_inventory = _load_json(coverage_inventory_path)
        coverage = _load_json(coverage_findings_path)

        final_report = combine_validation_reports(
            snapshot_filename=(snapshot_path.name),
            snapshot_sha256=(sha256_file(snapshot_path)),
            snapshot_size_bytes=(snapshot_path.stat().st_size),
            reports={
                "domain": domain,
                "selection": selection,
                "readiness": readiness,
                "coverage": coverage,
            },
            coverage_inventory={
                "lineup_positions": (
                    coverage_inventory.get(
                        "lineup_positions",
                        {},
                    )
                ),
                "cache_reasons": (
                    coverage_inventory.get(
                        "cache_reasons",
                        {},
                    )
                ),
            },
        )

        args.json_output.write_text(
            json.dumps(
                final_report,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        args.markdown_output.write_text(
            render_validation_markdown(final_report),
            encoding="utf-8",
        )

        shutil.copyfile(
            coverage_csv_path,
            args.csv_output,
        )

    console_summary = {
        "overall_ok": final_report["overall_ok"],
        "critical_issue_count": (final_report["critical_issue_count"]),
        "warning_count": final_report["warning_count"],
        "json_output": str(args.json_output),
        "csv_output": str(args.csv_output),
        "markdown_output": str(args.markdown_output),
    }

    print(
        json.dumps(
            console_summary,
            indent=2,
            sort_keys=True,
        )
    )

    return 0 if final_report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
