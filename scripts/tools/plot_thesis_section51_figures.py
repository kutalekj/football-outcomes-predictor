from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from football_outcomes.config import fs_settings as sett

matplotlib.use("Agg")

TB_ROOT = Path(sett.DATA_DIR) / "tensorboard_logs"
OUT_DIR = Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "section_51_final_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 300

SMOOTH_DIAG_WINDOW = 13
SMOOTH_EPOCH_WINDOW = 91

TITLE_FONTSIZE = 20
SUBTITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 11

TRAIN_LINE_WIDTH = 1.7
VAL_LINE_WIDTH = 2.5
DIAG_LINE_WIDTH = 2.4
TRAIN_ALPHA = 0.68
VAL_ALPHA = 1.0
GRID_ALPHA = 0.35

ABLATION_COLORS = {
    "full": "#1f77b4",
    "no_strength": "#ff7f0e",
    "no_positions": "#2ca02c",
}

ARCH_COLORS = {
    "v1": "#1f77b4",
    "v2_lite": "#9467bd",
    "v2": "#d62728",
}

PRETRAIN_COLORS = {
    "scratch": "#1f77b4",
    "pretrained": "#17becf",
    "freeze3": "#bcbd22",
    "freeze25": "#7f7f7f",
}

TRAIN_STYLE = "-"
VAL_STYLE = "-"

FIG51_RUNS = {
    "Full": ("thesis_fig51_diag_full", "full"),
    "No strength": ("thesis_fig51_diag_no_strength", "no_strength"),
    "No positions": ("thesis_fig51_diag_no_positions", "no_positions"),
}
FIG51_PROBE_LAYERS = {
    "Competition Embedding": "competition_embedding_flat",
    "Home Team Embedding": "home_embedding_flat",
    "Home Strength Embedding": "home_strength_embedding",
}
FIG51_YLIM = {
    "Competition Embedding": None,
    "Home Team Embedding": None,
    "Home Strength Embedding": None,
}

FIG52_RUNS = {
    "Full": ("thesis_fig52_epoch_v1_full", "full"),
    "No strength": ("thesis_fig52_epoch_v1_no_strength", "no_strength"),
    "No positions": ("thesis_fig52_epoch_v1_no_positions", "no_positions"),
}
FIG52_ACC_YLIM = (0.49, 0.625)
FIG52_LOSS_YLIM = (0.650, 0.705)

FIG54_RUNS = {
    "v1": ("thesis_fig54_epoch_v1_full", "v1"),
    "v2-lite": ("thesis_fig54_epoch_v2_lite_full", "v2_lite"),
    "v2": ("thesis_fig54_epoch_v2_full_approx", "v2"),
}
FIG54_ACC_YLIM = (0.47, 0.765)
FIG54_LOSS_YLIM = (0.50, 1.05)
FIG54_PLOT_LOSS = True

FIG55_RUNS = {
    "v1": ("thesis_fig55_diag_v1_full", "v1"),
    "v2-lite": ("thesis_fig55_diag_v2_lite_full", "v2_lite"),
}
FIG55_DRIFT_PANELS = {
    "Competition Embedding Drift": "competition_embedding",
    "Strength-Branch Drift": "strength_projection",
}
FIG55_PROBE_PANELS = {
    "Competition Representation": {
        "v1": "competition_embedding_flat",
        "v2-lite": "competition_embedding_flat",
    },
    "Home Team Representation": {
        "v1": "home_embedding_flat",
        "v2-lite": "home_embedding_flat",
    },
    "Structured-Branch Representation": {
        "v1": "home_strength_embedding",
        "v2-lite": "team_branch_proj",
    },
}

FIG56_RUNS = {
    "Scratch": ("thesis_fig56_epoch_v1_full_scratch_lr8e5", "scratch"),
    "Pretrained init": ("thesis_fig56_epoch_v1_full_pretrained_init_lr8e5", "pretrained"),
    "Freeze 3": ("thesis_fig56_epoch_v1_full_pretrained_init_freeze3_lr8e5", "freeze3"),
    "Freeze 25": ("thesis_fig56_epoch_v1_full_pretrained_init_freeze25_lr8e5", "freeze25"),
}
FIG56_ACC_YLIM = (0.50, 0.615)
FIG56_LOSS_YLIM = (0.660, 0.700)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) <= 2:
        return values

    out = np.empty_like(values, dtype=float)
    half = window // 2

    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        chunk = values[lo:hi]
        chunk = chunk[np.isfinite(chunk)]
        out[i] = np.mean(chunk) if len(chunk) else np.nan

    return out


def read_epoch_metrics(run_name: str) -> pd.DataFrame:
    path = TB_ROOT / run_name / "epoch_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing epoch metrics: {path}")
    return pd.read_csv(path)


def read_diagnostics(run_name: str) -> pd.DataFrame:
    path = TB_ROOT / run_name / "diagnostics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing diagnostics: {path}")
    return pd.read_csv(path)


def first_existing_layer(df: pd.DataFrame, candidates: list[str]) -> str | None:
    layers = set(df["layer"].dropna().astype(str))
    for candidate in candidates:
        if candidate in layers:
            return candidate
    return None


def save_fig(fig: plt.Figure, stem: str) -> None:
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"

    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {pdf}")
    print(f"[saved] {png}")


def style_axis(ax):
    ax.grid(axis="y", linestyle=":", alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def set_labels(ax, xlabel=None, ylabel=None):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)


def plot_fig51():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True)
    fig.suptitle("Branch Probe Diagnostics in the Initial v1 Ablation", fontsize=TITLE_FONTSIZE)

    for ax, (title, layer_name) in zip(axes, FIG51_PROBE_LAYERS.items()):
        for label, (run_name, color_key) in FIG51_RUNS.items():
            df = read_diagnostics(run_name)
            sub = df[(df["metric_family"] == "probe_meanabs") & (df["layer"] == layer_name)].copy()

            if sub.empty:
                continue

            x = sub["diag_step"].to_numpy()
            y = moving_average(sub["value"].to_numpy(), SMOOTH_DIAG_WINDOW)

            ax.plot(
                x,
                y,
                label=label,
                color=ABLATION_COLORS[color_key],
                linewidth=DIAG_LINE_WIDTH,
            )

        ax.set_title(title, fontsize=SUBTITLE_FONTSIZE)
        set_labels(ax, "Diagnostic step", "Mean absolute activation")
        if FIG51_YLIM.get(title) is not None:
            ax.set_ylim(*FIG51_YLIM[title])
        style_axis(ax)

    axes[0].legend(loc="best", fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    save_fig(fig, "fig_5_1_exp_diag_initial")


def plot_epoch_accuracy_loss(
    runs: dict[str, tuple[str, str]],
    title: str,
    stem: str,
    colors: dict[str, str],
    acc_ylim=None,
    loss_ylim=None,
    plot_loss: bool = True,
):
    nrows = 2 if plot_loss else 1
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(13.0, 7.2 if plot_loss else 4.8),
        sharex=True,
    )

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax_acc = axes[0]
    ax_loss = axes[1] if plot_loss else None

    for label, (run_name, color_key) in runs.items():
        df = read_epoch_metrics(run_name)
        x = df["global_epoch_step"].to_numpy()
        color = colors[color_key]

        train_acc = moving_average(df["accuracy"].to_numpy(), SMOOTH_EPOCH_WINDOW)
        val_acc = moving_average(df["val_accuracy"].to_numpy(), SMOOTH_EPOCH_WINDOW)

        ax_acc.plot(
            x,
            train_acc,
            linestyle=TRAIN_STYLE,
            color=color,
            alpha=TRAIN_ALPHA,
            linewidth=TRAIN_LINE_WIDTH,
            label=f"{label} train",
        )
        ax_acc.plot(
            x,
            val_acc,
            linestyle=VAL_STYLE,
            color=color,
            alpha=VAL_ALPHA,
            linewidth=VAL_LINE_WIDTH,
            label=f"{label} validation",
        )

        if plot_loss:
            train_loss = moving_average(df["loss"].to_numpy(), SMOOTH_EPOCH_WINDOW)
            val_loss = moving_average(df["val_loss"].to_numpy(), SMOOTH_EPOCH_WINDOW)

            ax_loss.plot(
                x,
                train_loss,
                linestyle=TRAIN_STYLE,
                color=color,
                alpha=TRAIN_ALPHA,
                linewidth=TRAIN_LINE_WIDTH,
                label=f"{label} train",
            )
            ax_loss.plot(
                x,
                val_loss,
                linestyle=VAL_STYLE,
                color=color,
                alpha=VAL_ALPHA,
                linewidth=VAL_LINE_WIDTH,
                label=f"{label} validation",
            )

    ax_acc.set_title("Accuracy", fontsize=SUBTITLE_FONTSIZE)
    set_labels(ax_acc, ylabel="Accuracy")
    if acc_ylim is not None:
        ax_acc.set_ylim(*acc_ylim)
    style_axis(ax_acc)
    ax_acc.legend(ncol=2, fontsize=LEGEND_FONTSIZE)

    if plot_loss:
        ax_loss.set_title("Loss", fontsize=SUBTITLE_FONTSIZE)
        set_labels(ax_loss, "Global epoch step", "Loss")
        if loss_ylim is not None:
            ax_loss.set_ylim(*loss_ylim)
        style_axis(ax_loss)
        ax_loss.legend(ncol=2, fontsize=LEGEND_FONTSIZE)
    else:
        set_labels(ax_acc, "Global epoch step")

    fig.suptitle(title, fontsize=TITLE_FONTSIZE)
    fig.tight_layout()
    save_fig(fig, stem)


def plot_fig52():
    plot_epoch_accuracy_loss(
        runs=FIG52_RUNS,
        title="Initial v1 Ablation: Epoch Accuracy and Loss",
        stem="fig_5_2_exp_initial_ablation_curves",
        colors=ABLATION_COLORS,
        acc_ylim=FIG52_ACC_YLIM,
        loss_ylim=FIG52_LOSS_YLIM,
        plot_loss=True,
    )


def plot_fig54():
    plot_epoch_accuracy_loss(
        runs=FIG54_RUNS,
        title="v1, v2-lite and v2 Comparison: Epoch Metrics",
        stem="fig_5_4_exp_v1_v2_curves",
        colors=ARCH_COLORS,
        acc_ylim=FIG54_ACC_YLIM,
        loss_ylim=FIG54_LOSS_YLIM,
        plot_loss=FIG54_PLOT_LOSS,
    )


def plot_fig56():
    plot_epoch_accuracy_loss(
        runs=FIG56_RUNS,
        title="v1 Pretraining and Fine-Tuning Comparison",
        stem="fig_5_6_exp_pretrain_finetune_v1",
        colors=PRETRAIN_COLORS,
        acc_ylim=FIG56_ACC_YLIM,
        loss_ylim=FIG56_LOSS_YLIM,
        plot_loss=True,
    )


def get_drift_layer(df: pd.DataFrame, preferred: str) -> str | None:
    candidates = [preferred]

    if preferred == "competition_embedding":
        candidates += ["competition_embedding", "competition_embedding_flat"]

    if preferred == "strength_projection":
        candidates += [
            "strength_projection",
            "team_branch_proj",
            "home_strength_embedding",
            "strength_dense_2",
        ]

    return first_existing_layer(df, candidates)


def get_probe_layer(df: pd.DataFrame, preferred: str) -> str | None:
    candidates = [preferred]

    if preferred == "team_branch_proj":
        candidates += ["team_branch_proj", "home_team_repr", "home_strength_embedding"]

    if preferred == "home_strength_embedding":
        candidates += ["home_strength_embedding", "team_branch_proj", "home_team_repr"]

    return first_existing_layer(df, candidates)


def plot_fig55():
    fig = plt.figure(figsize=(14.8, 8.8))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0])

    drift_axes = [
        fig.add_subplot(gs[0, 0:3]),
        fig.add_subplot(gs[0, 3:6]),
    ]

    probe_axes = [
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
    ]

    fig.suptitle("v1 vs. v2-lite Branch Diagnostics", fontsize=TITLE_FONTSIZE)

    for ax, (panel_title, layer_name) in zip(drift_axes, FIG55_DRIFT_PANELS.items()):
        for model_label, (run_name, color_key) in FIG55_RUNS.items():
            df = read_diagnostics(run_name)
            drift_df = df[df["metric_family"] == "drift"].copy()

            actual_layer = get_drift_layer(drift_df, layer_name)
            if actual_layer is None:
                continue

            sub = drift_df[drift_df["layer"] == actual_layer].copy()
            x = sub["diag_step"].to_numpy()
            y = moving_average(sub["value"].to_numpy(), SMOOTH_DIAG_WINDOW)

            ax.plot(
                x,
                y,
                color=ARCH_COLORS[color_key],
                linewidth=DIAG_LINE_WIDTH,
                label=model_label,
            )

        ax.set_title(panel_title, fontsize=SUBTITLE_FONTSIZE)
        set_labels(ax, "Diagnostic step", "L2 distance")
        style_axis(ax)
        ax.legend(fontsize=LEGEND_FONTSIZE)

    for ax, (panel_title, per_model_layers) in zip(probe_axes, FIG55_PROBE_PANELS.items()):
        for model_label, (run_name, color_key) in FIG55_RUNS.items():
            df = read_diagnostics(run_name)
            probe_df = df[df["metric_family"] == "probe_meanabs"].copy()

            preferred = per_model_layers[model_label]
            actual_layer = get_probe_layer(probe_df, preferred)
            if actual_layer is None:
                continue

            sub = probe_df[probe_df["layer"] == actual_layer].copy()
            x = sub["diag_step"].to_numpy()
            y = moving_average(sub["value"].to_numpy(), SMOOTH_DIAG_WINDOW)

            ax.plot(
                x,
                y,
                color=ARCH_COLORS[color_key],
                linewidth=DIAG_LINE_WIDTH,
                label=model_label,
            )

        ax.set_title(panel_title, fontsize=SUBTITLE_FONTSIZE)
        set_labels(ax, "Diagnostic step", "Mean absolute activation")
        style_axis(ax)
        ax.legend(fontsize=LEGEND_FONTSIZE)

    fig.tight_layout()
    save_fig(fig, "fig_5_5_exp_v2_lite_diagnostics")


def main():
    plot_fig51()
    plot_fig52()
    plot_fig54()
    plot_fig55()
    plot_fig56()
    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
