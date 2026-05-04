from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from football_outcomes.config import fs_settings as sett

matplotlib.use("Agg")

IN_ROOT = Path(sett.DATA_DIR) / "comparison" / "dataset_sensitivity"
OUT_DIR = Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "dataset_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "clean": Path(sett.DATA_DIR) / "tensorboard_logs" / "dataset_sensitivity_clean_selected_mlp_binary_u25",
    "less_restricted": Path(sett.DATA_DIR)
    / "tensorboard_logs"
    / "dataset_sensitivity_less_restricted_selected_mlp_binary_u25",
}

LABELS = {
    "clean": "Cleaned dataset",
    "less_restricted": "Less-restricted variant",
}

VARIANT_COLORS = {
    "clean": "#6f8fbf",
    "less_restricted": "#8a7fa6",
}

BAR_EDGE_COLOR = "#000000"
BAR_EDGE_WIDTH = 0.6


def read_oos(path: Path) -> list[dict]:
    with (path / "oos_predictions.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_round_metrics(path: Path) -> list[dict]:
    with (path / "round_metrics.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_auc(y_true, y_prob) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def metrics_from_rows(rows: list[dict]) -> dict:
    y_true = np.asarray([int(float(r["y_true"])) for r in rows], dtype=np.int32)
    y_prob = np.asarray([float(r["y_prob_under25"]) for r in rows], dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(np.int32)

    return {
        "auc": safe_auc(y_true.tolist(), y_prob.tolist()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "num_predictions": int(len(rows)),
    }


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
    rows_by_variant = {variant: read_oos(path) for variant, path in VARIANTS.items()}
    round_by_variant = {variant: read_round_metrics(path) for variant, path in VARIANTS.items()}

    global_metrics = {variant: metrics_from_rows(rows) for variant, rows in rows_by_variant.items()}

    with (OUT_DIR / "dataset_sensitivity_global_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(global_metrics, f, indent=2)

    with (OUT_DIR / "dataset_sensitivity_global_metrics.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "auc", "accuracy", "brier", "num_predictions"],
        )
        writer.writeheader()
        for variant, metrics in global_metrics.items():
            writer.writerow({"variant": variant, **metrics})

    # ------------------------------------------------------------
    # Figure 1: global metrics + round AUC curves
    # ------------------------------------------------------------
    fig, (ax_bar, ax_line) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10.8, 8.8),
        gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.35},
    )

    metric_names = ["auc", "accuracy", "brier"]
    metric_labels = ["AUC", "Accuracy", "Brier score"]
    x = np.arange(len(metric_names))
    width = 0.34

    clean_vals = [global_metrics["clean"][m] for m in metric_names]
    less_vals = [global_metrics["less_restricted"][m] for m in metric_names]

    ax_bar.bar(
        x - width / 2,
        clean_vals,
        width=width,
        label=LABELS["clean"],
        color=VARIANT_COLORS["clean"],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )
    ax_bar.bar(
        x + width / 2,
        less_vals,
        width=width,
        label=LABELS["less_restricted"],
        color=VARIANT_COLORS["less_restricted"],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )

    ax_bar.set_title("Dataset sensitivity: pooled metrics", fontsize=16)
    ax_bar.set_xlabel("Metric", fontsize=13)
    ax_bar.set_ylabel("Metric value", fontsize=13)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels)
    ax_bar.set_ylim(0.0, max(max(clean_vals), max(less_vals)) + 0.08)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.5)
    ax_bar.legend()
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    for offset, vals in [(-width / 2, clean_vals), (width / 2, less_vals)]:
        for i, v in enumerate(vals):
            ax_bar.text(
                i + offset,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                fontsize=9,
            )

    for variant, round_rows in round_by_variant.items():
        xs = [int(float(r["round_idx"])) for r in round_rows]
        ys = [float(r["val_auc"]) for r in round_rows]
        ys_smooth = moving_average(ys, window=25)

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
        OUT_DIR / "dataset_sensitivity_global_and_round_auc.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / "dataset_sensitivity_global_and_round_auc.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Figure 2: per-competition AUC delta
    # ------------------------------------------------------------
    def per_comp_auc(rows: list[dict]) -> dict[str, dict]:
        out = {}
        comps = sorted({r["competition"] for r in rows})
        for comp in comps:
            rr = [r for r in rows if r["competition"] == comp]
            y_true = [int(float(r["y_true"])) for r in rr]
            y_prob = [float(r["y_prob_under25"]) for r in rr]
            out[comp] = {
                "auc": safe_auc(y_true, y_prob),
                "matches": len(rr),
            }
        return out

    clean_pc = per_comp_auc(rows_by_variant["clean"])
    less_pc = per_comp_auc(rows_by_variant["less_restricted"])

    delta_rows = []
    for comp in sorted(set(clean_pc) | set(less_pc)):
        clean_auc = clean_pc.get(comp, {}).get("auc", float("nan"))
        less_auc = less_pc.get(comp, {}).get("auc", float("nan"))

        delta_rows.append(
            {
                "competition": comp,
                "clean_auc": clean_auc,
                "less_restricted_auc": less_auc,
                "delta_auc": (
                    less_auc - clean_auc if not np.isnan(clean_auc) and not np.isnan(less_auc) else float("nan")
                ),
                "clean_matches": clean_pc.get(comp, {}).get("matches", 0),
                "less_restricted_matches": less_pc.get(comp, {}).get("matches", 0),
            }
        )

    delta_rows.sort(key=lambda r: np.nan_to_num(r["delta_auc"], nan=-999.0))

    with (OUT_DIR / "dataset_sensitivity_per_competition_delta_auc.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(delta_rows)

    comps = [r["competition"] for r in delta_rows if not np.isnan(r["delta_auc"])]
    vals = [r["delta_auc"] for r in delta_rows if not np.isnan(r["delta_auc"])]

    fig, ax = plt.subplots(figsize=(14.5, 5.8))
    ax.bar(
        range(len(comps)),
        vals,
        color=[sett.COMPS_LEAGUE_COLORS.get(c, "#bdbdbd") for c in comps],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )
    ax.axhline(0.0, linestyle="--", color="#555555", linewidth=1.0)

    ax.set_title("Dataset sensitivity: per-competition AUC change", fontsize=16)
    ax.set_ylabel("AUC less-restricted - AUC cleaned", fontsize=13)
    ax.set_xlabel("Competition", fontsize=13)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, rotation=55, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        OUT_DIR / "dataset_sensitivity_per_competition_delta_auc.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / "dataset_sensitivity_per_competition_delta_auc.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
