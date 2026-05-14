from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from football_outcomes.config import fs_settings as sett

matplotlib.use("Agg")

OUT_DIR = Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "missing_skill_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "zero_mask": Path(sett.DATA_DIR) / "tensorboard_logs" / "missing_skill_zero_mask_selected_mlp_binary_u25",
    "position_mean": Path(sett.DATA_DIR) / "tensorboard_logs" / "missing_skill_position_mean_selected_mlp_binary_u25",
}

LABELS = {
    "zero_mask": "Zero + mask",
    "position_mean": "Position-mean imputation + mask",
}

VARIANT_COLORS = {
    "zero_mask": "#6f8fbf",
    "position_mean": "#8a7fa6",
}

BAR_EDGE_COLOR = "#000000"
BAR_EDGE_WIDTH = 0.6


def read_summary(path: Path) -> dict:
    with (path / "summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def read_round_metrics(path: Path) -> list[dict]:
    with (path / "round_metrics.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def moving_average(values: list[float], window: int = 17) -> list[float]:
    half = window // 2
    out = []

    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        vals = [v for v in values[lo:hi] if not np.isnan(v)]
        out.append(float(np.mean(vals)) if vals else float("nan"))

    return out


def main() -> None:
    summaries = {variant: read_summary(path) for variant, path in VARIANTS.items()}
    rounds = {variant: read_round_metrics(path) for variant, path in VARIANTS.items()}

    with (OUT_DIR / "missing_skill_sensitivity_global_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "auc", "accuracy", "brier"],
        )
        writer.writeheader()

        for variant, summary in summaries.items():
            writer.writerow(
                {
                    "variant": variant,
                    "auc": summary.get("pooled_auc"),
                    "accuracy": summary.get("pooled_accuracy"),
                    "brier": summary.get("pooled_brier"),
                }
            )

    # Figure: global metrics + round AUC curves
    fig, (ax_bar, ax_line) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10.8, 8.8),
        gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.35},
    )

    metric_names = ["pooled_auc", "pooled_accuracy", "pooled_brier"]
    metric_labels = ["AUC", "Accuracy", "Brier score"]
    x = np.arange(len(metric_names))
    width = 0.34

    vals_zero = [summaries["zero_mask"][metric] for metric in metric_names]
    vals_imp = [summaries["position_mean"][metric] for metric in metric_names]

    ax_bar.bar(
        x - width / 2,
        vals_zero,
        width=width,
        label=LABELS["zero_mask"],
        color=VARIANT_COLORS["zero_mask"],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )
    ax_bar.bar(
        x + width / 2,
        vals_imp,
        width=width,
        label=LABELS["position_mean"],
        color=VARIANT_COLORS["position_mean"],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )

    ax_bar.set_title("Missing-skill handling: pooled metrics", fontsize=16)
    ax_bar.set_xlabel("Metric", fontsize=13)
    ax_bar.set_ylabel("Metric value", fontsize=13)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels)
    ax_bar.set_ylim(0.0, max(max(vals_zero), max(vals_imp)) + 0.08)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.5)
    ax_bar.legend()
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    for offset, vals in [(-width / 2, vals_zero), (width / 2, vals_imp)]:
        for i, value in enumerate(vals):
            ax_bar.text(
                i + offset,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )

    for variant, rows in rounds.items():
        xs = [int(float(r["round_idx"])) for r in rows]
        ys = [float(r["val_auc"]) for r in rows]
        ys_smooth = moving_average(ys, window=17)

        ax_line.plot(
            xs,
            ys,
            color=VARIANT_COLORS[variant],
            alpha=0.12,
            linewidth=0.7,
        )
        ax_line.plot(
            xs,
            ys_smooth,
            color=VARIANT_COLORS[variant],
            linewidth=2.0,
            label=LABELS[variant],
        )

    ax_line.axhline(0.5, linestyle="--", color="#555555", linewidth=1.0, alpha=0.75)
    ax_line.set_title("Round-level validation AUC", fontsize=16)
    ax_line.set_xlabel("Rolling validation round", fontsize=13)
    ax_line.set_ylabel("Validation AUC", fontsize=13)
    ax_line.set_ylim(0.45, 0.65)
    ax_line.grid(axis="y", linestyle=":", alpha=0.5)
    ax_line.legend()
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)

    fig.savefig(
        OUT_DIR / "missing_skill_sensitivity_global_and_round_auc.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / "missing_skill_sensitivity_global_and_round_auc.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
