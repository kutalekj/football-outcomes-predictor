from __future__ import annotations

import statistics
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import try_load_snapshot
from football_outcomes.data.fs_retrieve import fill_globals_with_cache
from football_outcomes.training.fs_training_utils import distribute_matches_into_rounds
from football_outcomes.utils import fs_common as utils
from football_outcomes.utils import fs_feature_utils as fu

matplotlib.use("Agg")


# ---------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------
BAR_COLOR = "#6f8fbf"
EDGE_COLOR = "#334e68"
MEAN_COLOR = "#2f5f73"
MEDIAN_COLOR = "#7a7a7a"
GRID_COLOR = "#d9d9d9"

TITLE_SIZE = 15
LABEL_SIZE = 13
TICK_SIZE = 10.5
LEGEND_SIZE = 10.5


def apply_axis_style(ax) -> None:
    """Apply a consistent thesis-friendly visual style to one axis."""
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def set_tight_y_limits(ax, values: list[int], padding_ratio: float = 0.08) -> None:
    """Set a compact but safe y-axis range for count plots."""
    max_value = max(values)
    padding = max(5, max_value * padding_ratio)
    ax.set_ylim(0, max_value + padding)


def main() -> None:
    cache = try_load_snapshot()
    if cache is None:
        raise RuntimeError("No cached snapshot available.")

    fill_globals_with_cache(cache, update_leagues_list=False)
    g = Global.get_instance()

    all_matches_sorted = sorted(g.all_matches, key=fu.match_sort_key)
    league_matches_sorted = utils.filter_clean_league_matches(all_matches_sorted)
    league_matches_sorted = utils.filter_valid_round_matches(league_matches_sorted)
    league_matches_sorted = [
        m
        for m in league_matches_sorted
        if getattr(m, "season", None) is not None and sett.FIRST_SEASON <= m.season < sett.LAST_SEASON
    ]

    rounds = distribute_matches_into_rounds(league_matches_sorted)
    round_sizes = [len(r) for r in rounds]

    num_rounds = len(rounds)
    min_size = min(round_sizes)
    max_size = max(round_sizes)
    mean_size = statistics.mean(round_sizes)
    median_size = statistics.median(round_sizes)

    print(f"Num rounds: {num_rounds}")
    print(f"Min round size: {min_size}")
    print(f"Max round size: {max_size}")
    print(f"Mean round size: {mean_size:.2f}")
    print(f"Median round size: {median_size:.2f}")

    out_dir = Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "thesis_round_sizes_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Combined figure: histogram + chronological sizes
    # --------------------------------------------------
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10.5, 8.4),
        gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.5},
        constrained_layout=False,
    )

    # --- Top: histogram
    ax = axes[0]
    ax.hist(
        round_sizes,
        bins=20,
        color=BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        alpha=0.88,
    )
    ax.set_title("Distribution of training-round sizes", fontsize=TITLE_SIZE, pad=10)
    ax.set_xlabel("Number of matches in a training round", fontsize=LABEL_SIZE)
    ax.set_ylabel("Frequency", fontsize=LABEL_SIZE)
    ax.axvline(
        mean_size,
        color=MEAN_COLOR,
        linestyle="--",
        linewidth=1.6,
        label=f"Mean = {mean_size:.2f}",
    )
    ax.axvline(
        median_size,
        color=MEDIAN_COLOR,
        linestyle=":",
        linewidth=1.8,
        label=f"Median = {median_size:.2f}",
    )
    ax.legend(frameon=True, fontsize=LEGEND_SIZE)
    apply_axis_style(ax)

    # --- Bottom: chronological round sizes
    ax = axes[1]
    round_indices = list(range(1, num_rounds + 1))
    ax.bar(
        round_indices,
        round_sizes,
        width=0.88,
        color=BAR_COLOR,
        edgecolor=BAR_COLOR,
        linewidth=0.25,
        alpha=0.78,
    )
    ax.set_title("Training-round sizes in chronological order", fontsize=TITLE_SIZE, pad=10)
    ax.set_xlabel("Training round index", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of matches", fontsize=LABEL_SIZE)
    ax.axhline(
        mean_size,
        color=MEAN_COLOR,
        linestyle="--",
        linewidth=1.6,
        label=f"Mean = {mean_size:.2f}",
    )
    ax.axhline(
        median_size,
        color=MEDIAN_COLOR,
        linestyle=":",
        linewidth=1.8,
        label=f"Median = {median_size:.2f}",
    )
    ax.legend(frameon=True, fontsize=LEGEND_SIZE)
    apply_axis_style(ax)
    set_tight_y_limits(ax, round_sizes)

    combined_png_path = out_dir / "round_size_analysis_combined.png"
    combined_pdf_path = out_dir / "round_size_analysis_combined.pdf"
    fig.savefig(combined_png_path, dpi=300, bbox_inches="tight")
    fig.savefig(combined_pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined PNG figure to: {combined_png_path}")
    print(f"Saved combined PDF figure to: {combined_pdf_path}")

    # --------------------------------------------------
    # Optional: separate chronological-only plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.bar(
        round_indices,
        round_sizes,
        width=0.88,
        color=BAR_COLOR,
        edgecolor=BAR_COLOR,
        linewidth=0.25,
        alpha=0.78,
    )
    ax.set_title("Training-round sizes in chronological order", fontsize=TITLE_SIZE, pad=10)
    ax.set_xlabel("Training round index", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of matches", fontsize=LABEL_SIZE)
    ax.axhline(
        mean_size,
        color=MEAN_COLOR,
        linestyle="--",
        linewidth=1.6,
        label=f"Mean = {mean_size:.2f}",
    )
    ax.axhline(
        median_size,
        color=MEDIAN_COLOR,
        linestyle=":",
        linewidth=1.8,
        label=f"Median = {median_size:.2f}",
    )
    ax.legend(frameon=True, fontsize=LEGEND_SIZE)
    apply_axis_style(ax)
    set_tight_y_limits(ax, round_sizes)

    chrono_png_path = out_dir / "round_size_chronological.png"
    chrono_pdf_path = out_dir / "round_size_chronological.pdf"
    fig.savefig(chrono_png_path, dpi=300, bbox_inches="tight")
    fig.savefig(chrono_pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved chronological PNG figure to: {chrono_png_path}")
    print(f"Saved chronological PDF figure to: {chrono_pdf_path}")


if __name__ == "__main__":
    main()
