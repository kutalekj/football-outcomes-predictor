from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def choose_audit_fold_indices(
    *,
    round_count: int,
    window_rounds: int,
    audit_fold_count: int,
) -> tuple[int, ...]:
    if type(round_count) is not int or round_count < 0:
        raise ValueError("round_count must be a non-negative integer.")

    if type(window_rounds) is not int or window_rounds <= 0:
        raise ValueError("window_rounds must be a positive integer.")

    if type(audit_fold_count) is not int or audit_fold_count <= 0:
        raise ValueError("audit_fold_count must be a positive integer.")

    eligible_count = round_count - window_rounds

    if eligible_count <= 0:
        return ()

    selected_count = min(
        audit_fold_count,
        eligible_count,
    )
    first_index = window_rounds
    last_index = round_count - 1

    if selected_count == 1:
        return (last_index,)

    span = eligible_count - 1
    denominator = selected_count - 1

    return tuple(first_index + (position * span) // denominator for position in range(selected_count))


def _finding(
    code: str,
    count: int,
    message: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "count": int(count),
        "message": message,
    }


def build_step7_validation_report(
    *,
    snapshot: Mapping[str, Any],
    config: Mapping[str, Any],
    scope: Mapping[str, Any],
    legacy: Mapping[str, Any],
    past_only: Mapping[str, Any],
    comparison: Mapping[str, Any],
    rolling_audit: Mapping[str, Any],
) -> dict[str, Any]:
    critical_findings: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    total_cells = int(scope.get("total_strength_cells", 0))

    legacy_classified = int(legacy.get("observed_cells", 0)) + int(legacy.get("missing_cells", 0))

    if legacy_classified != total_cells:
        critical_findings.append(
            _finding(
                "legacy_cell_conservation_failure",
                abs(total_cells - legacy_classified),
                ("Legacy observed and missing cells do not " "sum to the selected-scope total."),
            )
        )

    past_classified = int(past_only.get("observed_cells", 0)) + int(past_only.get("unresolved_cells", 0))

    if past_classified != total_cells:
        critical_findings.append(
            _finding(
                "past_only_cell_conservation_failure",
                abs(total_cells - past_classified),
                ("Past-only observed and unresolved cells do " "not sum to the selected-scope total."),
            )
        )

    comparison_total = sum(
        int(comparison.get(key, 0))
        for key in (
            "both_observed_equal_cells",
            "both_observed_changed_cells",
            "legacy_only_observed_cells",
            "past_only_only_observed_cells",
            "both_missing_cells",
        )
    )

    if comparison_total != total_cells:
        critical_findings.append(
            _finding(
                "comparison_cell_conservation_failure",
                abs(total_cells - comparison_total),
                ("Legacy-versus-past-only comparison classes " "do not sum to the selected-scope total."),
            )
        )

    audited_classified = (
        int(
            rolling_audit.get(
                "observed_provenance_cells",
                0,
            )
        )
        + int(
            rolling_audit.get(
                "imputed_provenance_cells",
                0,
            )
        )
        + int(
            rolling_audit.get(
                "unresolved_provenance_cells",
                0,
            )
        )
    )
    audited_total = int(
        rolling_audit.get(
            "validation_cells",
            0,
        )
    )

    if audited_classified != audited_total:
        critical_findings.append(
            _finding(
                "audited_cell_conservation_failure",
                abs(audited_total - audited_classified),
                (
                    "Audited observed, imputed and unresolved "
                    "provenance cells do not sum to the audited "
                    "validation-cell total."
                ),
            )
        )

    critical_checks = (
        (
            "invalid_legacy_matrices",
            int(legacy.get("invalid_matrices", 0)),
            "Persisted legacy matrices have invalid shapes.",
        ),
        (
            "future_source_cells",
            int(past_only.get("future_source_cells", 0)),
            "Past-only reconstruction used future-dated cells.",
        ),
        (
            "source_cells_beyond_max_age",
            int(
                past_only.get(
                    "source_cells_beyond_max_age",
                    0,
                )
            ),
            ("Past-only reconstruction used cells older than " "the configured temporal window."),
        ),
        (
            "invalid_past_only_values",
            int(
                past_only.get(
                    "invalid_observed_values",
                    0,
                )
            ),
            ("Past-only reconstruction produced invalid " "observed skill values."),
        ),
        (
            "invalid_past_only_matrices",
            int(
                past_only.get(
                    "invalid_matrices",
                    0,
                )
            ),
            ("Past-only reconstruction produced invalid " "matrix shapes."),
        ),
        (
            "invalid_past_only_provenance",
            int(
                past_only.get(
                    "invalid_provenance_cells",
                    0,
                )
            ),
            ("Past-only reconstruction produced unsupported " "provenance codes."),
        ),
        (
            "past_only_age_marker_mismatches",
            int(
                past_only.get(
                    "unresolved_age_mismatches",
                    0,
                )
            ),
            ("Unresolved past-only cells do not retain source " "age -1."),
        ),
        (
            "audited_unresolved_cells",
            int(
                rolling_audit.get(
                    "unresolved_provenance_cells",
                    0,
                )
            ),
            ("Fold-local completion left unresolved cells in " "audited validation folds."),
        ),
        (
            "audited_mask_mismatches",
            int(
                rolling_audit.get(
                    "observed_mask_mismatches",
                    0,
                )
            ),
            ("Audited observed masks disagree with temporal " "provenance."),
        ),
        (
            "audited_array_contract_mismatches",
            int(
                rolling_audit.get(
                    "nonstructured_array_mismatches",
                    0,
                )
            ),
            ("Fold-local integration changed non-strength " "validation arrays."),
        ),
        (
            "audited_invalid_strength_values",
            int(
                rolling_audit.get(
                    "invalid_strength_values",
                    0,
                )
            ),
            ("Fold-local arrays contain invalid normalized " "strength values."),
        ),
        (
            "audited_invalid_masks",
            int(
                rolling_audit.get(
                    "invalid_masks",
                    0,
                )
            ),
            "Fold-local arrays contain non-binary masks.",
        ),
        (
            "audited_provenance_conservation_failure",
            int(
                rolling_audit.get(
                    "provenance_conservation_failures",
                    0,
                )
            ),
            ("Audited provenance counts do not match audited " "validation-cell totals."),
        ),
    )

    for code, count, message in critical_checks:
        if count > 0:
            critical_findings.append(
                _finding(
                    code,
                    count,
                    message,
                )
            )

    warning_checks = (
        (
            "legacy_missing_cells",
            int(legacy.get("missing_cells", 0)),
            "Persisted legacy strength cells are missing.",
        ),
        (
            "past_only_unresolved_cells",
            int(past_only.get("unresolved_cells", 0)),
            ("Strict past-only reconstruction cannot resolve " "all selected-scope cells without imputation."),
        ),
        (
            "legacy_only_observed_cells",
            int(
                comparison.get(
                    "legacy_only_observed_cells",
                    0,
                )
            ),
            ("Legacy matrices contain observations that are " "not available under the strict temporal policy."),
        ),
        (
            "past_only_only_observed_cells",
            int(
                comparison.get(
                    "past_only_only_observed_cells",
                    0,
                )
            ),
            ("Past-only reconstruction recovers observations " "missing from persisted matrices."),
        ),
        (
            "changed_observed_cells",
            int(
                comparison.get(
                    "both_observed_changed_cells",
                    0,
                )
            ),
            ("Legacy and strict past-only paths select different " "values for cells observed by both paths."),
        ),
        (
            "audited_neutral_fallback_cells",
            int(
                rolling_audit.get(
                    "neutral_fallback_cells",
                    0,
                )
            ),
            ("Audited validation folds require the fixed neutral " "fallback after grouped and global medians."),
        ),
    )

    for code, count, message in warning_checks:
        if count > 0:
            warnings.append(
                _finding(
                    code,
                    count,
                    message,
                )
            )

    critical_issue_count = sum(int(finding["count"]) for finding in critical_findings)
    warning_count = sum(int(finding["count"]) for finding in warnings)

    return {
        "validation": ("step_7_leakage_safe_sofifa_imputation"),
        "overall_ok": critical_issue_count == 0,
        "critical_issue_count": critical_issue_count,
        "warning_count": warning_count,
        "critical_findings": critical_findings,
        "warnings": warnings,
        "snapshot": dict(snapshot),
        "config": dict(config),
        "scope": dict(scope),
        "legacy": dict(legacy),
        "past_only": dict(past_only),
        "comparison": dict(comparison),
        "rolling_audit": dict(rolling_audit),
    }


def _format_percent(value: object) -> str:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        return "n/a"

    return f"{100.0 * float(value):.2f}%"


def render_step7_validation_markdown(
    report: Mapping[str, Any],
) -> str:
    snapshot = report["snapshot"]
    config = report["config"]
    scope = report["scope"]
    legacy = report["legacy"]
    past_only = report["past_only"]
    comparison = report["comparison"]
    rolling = report["rolling_audit"]

    lines = [
        "# Step 7 leakage-safe SoFIFA imputation report",
        "",
        "## Result",
        "",
        "**PASS**" if report["overall_ok"] else "**FAIL**",
        "",
        ("- Critical observations: " f"`{report['critical_issue_count']}`"),
        ("- Warning observations: " f"`{report['warning_count']}`"),
        "",
        "## Frozen snapshot",
        "",
        f"- Filename: `{snapshot['filename']}`",
        f"- SHA-256: `{snapshot['sha256']}`",
        f"- Size in bytes: `{snapshot['size_bytes']}`",
        "",
        "## Validation scope",
        "",
        f"- Selected matches: `{scope['selected_matches']}`",
        f"- Array-ready matches: `{scope['array_ready_matches']}`",
        f"- Constructed rounds: `{scope['round_count']}`",
        ("- Strength cells: " f"`{scope['total_strength_cells']}`"),
        "",
        "## Temporal and imputation policy",
        "",
        f"- Rolling window: `{config['window_rounds']}` rounds",
        ("- Audited folds: " f"`{rolling['audit_fold_count']}`"),
        ("- Maximum SoFIFA source age: " f"`{config['max_age_days']}` days"),
        ("- Maximum snapshots scanned per player: " f"`{config['max_snapshots']}`"),
        ("- Minimum grouped-median support: " f"`{config['minimum_group_support']}`"),
        ("- Neutral fallback: " f"`{config['neutral_value']}`"),
        "",
        "## Full selected-scope coverage",
        "",
        ("| Representation | Observed cells | Missing or " "unresolved cells | Observation rate |"),
        "|---|---:|---:|---:|",
        (
            "| Persisted legacy | "
            f"{legacy['observed_cells']} | "
            f"{legacy['missing_cells']} | "
            f"{_format_percent(legacy['observation_rate'])} |"
        ),
        (
            "| Strict past-only | "
            f"{past_only['observed_cells']} | "
            f"{past_only['unresolved_cells']} | "
            f"{_format_percent(past_only['observation_rate'])} |"
        ),
        "",
        "## Legacy versus strict past-only",
        "",
        ("- Observed and equal: " f"`{comparison['both_observed_equal_cells']}`"),
        ("- Observed but changed: " f"`{comparison['both_observed_changed_cells']}`"),
        ("- Legacy-only observations: " f"`{comparison['legacy_only_observed_cells']}`"),
        ("- Past-only-only observations: " f"`{comparison['past_only_only_observed_cells']}`"),
        ("- Missing in both paths: " f"`{comparison['both_missing_cells']}`"),
        "",
        "## Strict temporal evidence",
        "",
        ("- Nearest-past cells: " f"`{past_only['nearest_past_cells']}`"),
        ("- Older-past completion cells: " f"`{past_only['older_past_cells']}`"),
        ("- Maximum observed source age: " f"`{past_only['maximum_source_age_days']}` days"),
        ("- Future-source cells: " f"`{past_only['future_source_cells']}`"),
        "",
        "## Fold-local production-path audit",
        "",
        ("- Audited validation matches: " f"`{rolling['validation_matches']}`"),
        ("- Audited validation cells: " f"`{rolling['validation_cells']}`"),
        ("- Genuine past-only cells: " f"`{rolling['observed_provenance_cells']}`"),
        ("- Statistically imputed cells: " f"`{rolling['imputed_provenance_cells']}`"),
        ("- Cells left unresolved: " f"`{rolling['unresolved_provenance_cells']}`"),
        ("- Non-strength array mismatches: " f"`{rolling['nonstructured_array_mismatches']}`"),
        "",
        "### Audited provenance",
        "",
    ]

    provenance = rolling.get(
        "validation_provenance_counts",
        {},
    )

    for name, count in sorted(provenance.items()):
        lines.append(f"- `{name}`: `{count}`")

    lines.extend(
        [
            "",
            "## Findings",
            "",
        ]
    )

    findings = [
        *report.get("critical_findings", []),
        *report.get("warnings", []),
    ]

    if not findings:
        lines.append("No findings were reported.")
    else:
        for finding in findings:
            lines.append("- " f"`{finding['code']}`: " f"{finding['count']} — " f"{finding['message']}")

    lines.extend(
        [
            "",
            "## Acceptance decision",
            "",
            (
                "Step 7 is accepted because strict past-only "
                "reconstruction used no future information and all "
                "audited fold-local arrays satisfied the production "
                "input contract."
                if report["overall_ok"]
                else (
                    "Step 7 is not accepted because one or more "
                    "temporal or fold-local integration invariants "
                    "failed."
                )
            ),
            "",
            (
                "The persisted snapshot remains unchanged. The new "
                "path is transient, explicitly configurable and "
                "disabled by default."
            ),
            "",
        ]
    )

    return "\n".join(lines)
