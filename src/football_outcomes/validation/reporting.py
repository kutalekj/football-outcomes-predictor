from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

COMPONENT_ORDER = (
    "domain",
    "selection",
    "readiness",
    "coverage",
)


def sha256_file(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest().upper()


def combine_validation_reports(
    *,
    snapshot_filename: str,
    snapshot_sha256: str,
    snapshot_size_bytes: int,
    reports: Mapping[
        str,
        Mapping[str, Any],
    ],
    coverage_inventory: (
        Mapping[
            str,
            Any,
        ]
        | None
    ) = None,
) -> dict[str, Any]:
    missing_components = [name for name in COMPONENT_ORDER if name not in reports]

    if missing_components:
        raise ValueError("Missing validation components: " + ", ".join(missing_components))

    components = {name: dict(reports[name]) for name in COMPONENT_ORDER}

    critical_issue_count = sum(
        int(
            components[name].get(
                "critical_issue_count",
                0,
            )
        )
        for name in COMPONENT_ORDER
    )
    warning_count = sum(
        int(
            components[name].get(
                "warning_count",
                0,
            )
        )
        for name in COMPONENT_ORDER
    )

    overall_ok = all(
        bool(
            components[name].get(
                "ok",
                False,
            )
        )
        for name in COMPONENT_ORDER
    )

    result: dict[str, Any] = {
        "validation": ("step_6_full_dataset"),
        "overall_ok": overall_ok,
        "critical_issue_count": (critical_issue_count),
        "warning_count": warning_count,
        "snapshot": {
            "filename": snapshot_filename,
            "sha256": snapshot_sha256,
            "size_bytes": (snapshot_size_bytes),
        },
        "components": components,
    }

    if coverage_inventory is not None:
        result["coverage_inventory"] = dict(coverage_inventory)

    return result


def _format_percent(
    value: object,
) -> str:
    if not isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        return "n/a"

    if isinstance(value, bool):
        return "n/a"

    return f"{100.0 * float(value):.2f}%"


def _component_status(
    component: Mapping[str, Any],
) -> str:
    return "PASS" if component.get("ok") is True else "FAIL"


def render_validation_markdown(
    report: Mapping[str, Any],
) -> str:
    snapshot = report["snapshot"]
    components = report["components"]

    lines = [
        "# Step 6 full-dataset validation report",
        "",
        "## Result",
        "",
        ("**PASS**" if report["overall_ok"] else "**FAIL**"),
        "",
        ("- Critical observations: " f"`{report['critical_issue_count']}`"),
        ("- Warning observations: " f"`{report['warning_count']}`"),
        "",
        "Warnings quantify affected data " "entities or cells and are not a " "count of distinct defect categories.",
        "",
        "## Frozen snapshot",
        "",
        ("- Filename: " f"`{snapshot['filename']}`"),
        ("- SHA-256: " f"`{snapshot['sha256']}`"),
        ("- Size in bytes: " f"`{snapshot['size_bytes']}`"),
        "",
        "## Validation components",
        "",
        ("| Component | Status | " "Critical | Warnings |"),
        "|---|---:|---:|---:|",
    ]

    for name in COMPONENT_ORDER:
        component = components[name]

        lines.append(
            "| "
            f"{name.title()} | "
            f"{_component_status(component)} | "
            f"{component.get('critical_issue_count', 0)} | "
            f"{component.get('warning_count', 0)} |"
        )

    selection_metrics = components["selection"].get(
        "metrics",
        {},
    )
    readiness_metrics = components["readiness"].get(
        "metrics",
        {},
    )
    coverage_metrics = components["coverage"].get(
        "metrics",
        {},
    )

    lines.extend(
        [
            "",
            "## Model-development scope",
            "",
            ("- Selected matches: " f"`{selection_metrics.get('selected_matches', 'n/a')}`"),
            ("- Competitions: " f"`{selection_metrics.get('selected_competitions', 'n/a')}`"),
            ("- Competition-seasons: " f"`{selection_metrics.get('selected_competition_seasons', 'n/a')}`"),
            ("- Constructed rounds: " f"`{selection_metrics.get('constructed_rounds', 'n/a')}`"),
            ("- Array-ready matches: " f"`{readiness_metrics.get('processed_array_matches', 'n/a')}`"),
            ("- Under 2.5 targets: " f"`{readiness_metrics.get('binary_under_25_count', 'n/a')}`"),
            ("- Over 2.5 targets: " f"`{readiness_metrics.get('binary_over_25_count', 'n/a')}`"),
            "",
            "## Coverage summary",
            "",
            ("- Selected-team mapping: " f"`{_format_percent(coverage_metrics.get('selected_team_mapping_rate'))}`"),
            (
                "- Unique-player matching: "
                f"`{_format_percent(coverage_metrics.get('unique_player_cache_match_rate'))}`"
            ),
            (
                "- Lineup-reference matching: "
                f"`{_format_percent(coverage_metrics.get('lineup_reference_cache_match_rate'))}`"
            ),
            (
                "- Observed strength cells: "
                f"`{_format_percent(coverage_metrics.get('strength_cell_observation_rate'))}`"
            ),
            "",
            "## Finding categories",
            "",
        ]
    )

    finding_found = False

    for name in COMPONENT_ORDER:
        findings = components[name].get(
            "findings",
            [],
        )

        if not findings:
            continue

        finding_found = True
        lines.append(f"### {name.title()}")
        lines.append("")

        for finding in findings:
            lines.append(
                "- "
                f"`{finding.get('code')}`: "
                f"{finding.get('count', 0)} "
                f"({finding.get('severity', 'unknown')})"
            )

        lines.append("")

    if not finding_found:
        lines.append("No findings were reported.")
        lines.append("")

    lines.extend(
        [
            "## Acceptance decision",
            "",
            (
                "Step 6 is accepted because " "all critical validation " "components passed."
                if report["overall_ok"]
                else ("Step 6 is not accepted " "because one or more " "critical validation " "components failed.")
            ),
            "",
            (
                "The documented quality gaps "
                "remain inputs to Step 7 "
                "imputation and are not "
                "silently repaired in the "
                "snapshot."
            ),
            "",
        ]
    )

    return "\n".join(lines)
