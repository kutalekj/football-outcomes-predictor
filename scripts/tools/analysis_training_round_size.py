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
        figsize=(10, 9),
        gridspec_kw={"height_ratios": [1, 1.2]},
        constrained_layout=True,
    )

    # --- Top: histogram
    ax = axes[0]
    ax.hist(round_sizes, bins=20, edgecolor="black")
    ax.set_title("Distribution of matches per training round")
    ax.set_xlabel("Number of matches in training round")
    ax.set_ylabel("Frequency")
    ax.axvline(mean_size, linestyle="--", label=f"Mean = {mean_size:.2f}")
    ax.axvline(median_size, linestyle=":", label=f"Median = {median_size:.2f}")
    ax.legend()

    # --- Bottom: chronological round sizes
    ax = axes[1]
    round_indices = list(range(1, num_rounds + 1))
    ax.bar(round_indices, round_sizes, width=0.9)
    ax.set_title("Training round sizes in chronological order")
    ax.set_xlabel("Training round index")
    ax.set_ylabel("Number of matches")
    ax.axhline(mean_size, linestyle="--", label=f"Mean = {mean_size:.2f}")
    ax.axhline(median_size, linestyle=":", label=f"Median = {median_size:.2f}")
    ax.legend()

    combined_path = out_dir / "round_size_analysis_combined.png"
    fig.savefig(combined_path, dpi=200)
    plt.close(fig)

    print(f"Saved combined figure to: {combined_path}")

    # --------------------------------------------------
    # Optional: separate chronological-only plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar(round_indices, round_sizes, width=0.9)
    ax.set_title("Training round sizes in chronological order")
    ax.set_xlabel("Training round index")
    ax.set_ylabel("Number of matches")
    ax.axhline(mean_size, linestyle="--", label=f"Mean = {mean_size:.2f}")
    ax.axhline(median_size, linestyle=":", label=f"Median = {median_size:.2f}")
    ax.legend()

    chrono_path = out_dir / "round_size_chronological.png"
    fig.savefig(chrono_path, dpi=200)
    plt.close(fig)

    print(f"Saved chronological figure to: {chrono_path}")


if __name__ == "__main__":
    main()
