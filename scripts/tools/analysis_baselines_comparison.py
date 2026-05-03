from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from football_outcomes.config import fs_settings as sett

matplotlib.use("Agg")

TITLE_SIZE = 15
LABEL_SIZE = 12.5
TICK_SIZE = 10.5
LEGEND_SIZE = 10.5
GRID_COLOR = "#d9d9d9"

COLORS = {
    "majority": "#8c8c8c",
    "logreg": "#6f8fbf",
    "rf": "#5f9e6e",
    "ridge": "#6f8fbf",
    "multinomial_logreg": "#6f8fbf",
    "selected_mlp": "#b85c5c",
}

LINESTYLES = {
    "majority": ":",
    "logreg": "--",
    "rf": "-.",
    "ridge": "--",
    "multinomial_logreg": "--",
    "selected_mlp": "-",
}

Y_LIMITS = {
    "binary_u25": (0.45, 0.70),
    "goals_reg": (1.05, 2.05),
    "goals_dist": (1.15, 2.15),
}

RUNS = [
    {
        "mode": "binary_u25",
        "model_name": "majority",
        "label": "Majority",
        "path": Path(sett.DATA_DIR) / "baseline_logs" / "baseline_binary_majority" / "round_metrics.csv",
    },
    {
        "mode": "binary_u25",
        "model_name": "logreg",
        "label": "Logistic regression",
        "path": Path(sett.DATA_DIR) / "baseline_logs" / "baseline_binary_logreg" / "round_metrics.csv",
    },
    {
        "mode": "binary_u25",
        "model_name": "rf",
        "label": "Random forest",
        "path": Path(sett.DATA_DIR) / "baseline_logs" / "baseline_binary_rf" / "round_metrics.csv",
    },
    {
        "mode": "binary_u25",
        "model_name": "selected_mlp",
        "label": "Selected MLP",
        "path": Path(sett.DATA_DIR) / "tensorboard_logs" / "selected_mlp_binary_u25" / "round_metrics.csv",
    },
    {
        "mode": "goals_reg",
        "model_name": "majority",
        "label": "Mean goals",
        "path": Path(sett.DATA_DIR) / "baseline_logs" / "baseline_reg_mean_goals" / "round_metrics.csv",
    },
    {
        "mode": "goals_reg",
        "model_name": "ridge",
        "label": "Ridge regression",
        "path": Path(sett.DATA_DIR) / "baseline_logs" / "baseline_reg_ridge" / "round_metrics.csv",
    },
    {
        "mode": "goals_reg",
        "model_name": "selected_mlp",
        "label": "Selected MLP",
        "path": Path(sett.DATA_DIR) / "tensorboard_logs" / "selected_mlp_goals_reg" / "round_metrics.csv",
    },
    {
        "mode": "goals_dist",
        "model_name": "majority",
        "label": "Majority class",
        "path": Path(sett.DATA_DIR) / "baseline_logs" / "baseline_multiclass_majority" / "round_metrics.csv",
    },
    {
        "mode": "goals_dist",
        "model_name": "multinomial_logreg",
        "label": "Multinomial logreg",
        "path": Path(sett.DATA_DIR) / "baseline_logs" / "baseline_multiclass_logreg" / "round_metrics.csv",
    },
    {
        "mode": "goals_dist",
        "model_name": "selected_mlp",
        "label": "Selected MLP",
        "path": Path(sett.DATA_DIR) / "tensorboard_logs" / "selected_mlp_goals_dist" / "round_metrics.csv",
    },
]

METRIC_BY_MODE = {
    "binary_u25": {
        "aliases": ["val_auc"],
        "title": "Binary Under/Over 2.5 prediction",
        "ylabel": "Validation AUC per round",
        "higher_is_better": True,
    },
    "goals_reg": {
        "aliases": ["val_mae"],
        "title": "Total-goals regression",
        "ylabel": "Validation MAE per round",
        "higher_is_better": False,
    },
    "goals_dist": {
        "aliases": ["val_expected_goals_mae", "expected_goals_mae"],
        "title": "Multiclass goal-count prediction",
        "ylabel": "Expected-goals MAE per round",
        "higher_is_better": False,
    },
}


def apply_axis_style(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_round_metric(path: Path, aliases: list[str]) -> tuple[list[int], list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing round metrics file: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows in file: {path}")

    metric_col = None
    for alias in aliases:
        if alias in rows[0]:
            metric_col = alias
            break

    if metric_col is None:
        available = ", ".join(rows[0].keys())
        raise KeyError(f"None of metric aliases {aliases} found in {path}. Available columns: {available}")

    x = [int(float(r["round_idx"])) for r in rows if r.get(metric_col) not in ("", None)]
    y = [float(r[metric_col]) for r in rows if r.get(metric_col) not in ("", None)]
    return x, y


def moving_average(values: list[float], window: int = 9) -> list[float]:
    if window <= 1 or len(values) < window:
        return values

    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def plot_mode(ax, mode: str, runs: list[dict]) -> None:
    spec = METRIC_BY_MODE[mode]
    aliases = spec["aliases"]

    for run in runs:
        x, y = read_round_metric(run["path"], aliases)
        y_smooth = moving_average(y, window=25)

        color = COLORS.get(run["model_name"], "#888888")
        linestyle = LINESTYLES.get(run["model_name"], "-")

        ax.plot(x, y, color=color, linewidth=0.55, alpha=0.25)
        ax.plot(
            x,
            y_smooth,
            color=color,
            linestyle=linestyle,
            linewidth=2.0 if run["model_name"] == "selected_mlp" else 1.6,
            label=run["label"],
        )

    ax.set_title(spec["title"], fontsize=TITLE_SIZE, pad=10)
    ax.set_ylabel(spec["ylabel"], fontsize=LABEL_SIZE)
    ax.set_xlabel("Rolling validation round", fontsize=LABEL_SIZE)

    # Manual y-axis scaling per subplot/mode.
    y_limits = Y_LIMITS.get(mode)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.legend(frameon=True, fontsize=LEGEND_SIZE, ncol=2)
    apply_axis_style(ax)


def main() -> None:
    out_dir = Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "baseline_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(11.2, 11.5),
        gridspec_kw={"hspace": 0.68},
        constrained_layout=False,
    )

    for ax, mode in zip(axes, ["binary_u25", "goals_reg", "goals_dist"]):
        mode_runs = [r for r in RUNS if r["mode"] == mode]
        plot_mode(ax, mode, mode_runs)

    fig.suptitle(
        "Round-level comparison of selected MLP model with simple baselines",
        fontsize=16,
        y=0.995,
    )

    png_path = out_dir / "baseline_comparison_round_curves.png"
    pdf_path = out_dir / "baseline_comparison_round_curves.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG to: {png_path}")
    print(f"Saved PDF to: {pdf_path}")


if __name__ == "__main__":
    main()
