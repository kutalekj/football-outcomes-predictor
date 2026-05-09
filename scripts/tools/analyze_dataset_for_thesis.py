from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import PowerNorm

import football_outcomes.utils.fs_common as utils
from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import load_snapshot
from football_outcomes.data.fs_models import FSMatch
from football_outcomes.data.fs_retrieve import fill_globals_with_cache

matplotlib.use("Agg")

# Raw match-stat fields where literal zero may also mean “missing / unavailable” in provider data.
ZERO_AS_MISSING_STATS = {
    "home_total_shots",
    "away_total_shots",
    "home_fouls",
    "away_fouls",
    "home_possession",
    "away_possession",
    "home_attacks",
    "away_attacks",
    "home_dangerous_attacks",
    "away_dangerous_attacks",
    "home_prematch_xg",
    "away_prematch_xg",
}

# Shared percentage-plot configuration.
# Keep the semantic range at 0–100, but use nonlinear normalization so sparse percentages remain distinguishable.
PCT_MIN = 0.0
PCT_MAX = 100.0
HEATMAP_GAMMA = 0.40
PCT_CMAP = "inferno"
PCT_BAR_YMAX = 100.0
PCT_CBAR_TICKS = [0, 2, 5, 10, 15, 20, 30, 50, 100]

THESIS_PAIR_GRID_COLOR = "#3a3a3a"
THESIS_PAIR_GRID_WIDTH = 0.14
THESIS_PAIR_MASK_COLOR = "#504d4d"

THESIS_PAIR_TITLE_FONTSIZE = 32
THESIS_PAIR_AXIS_LABEL_FONTSIZE = 24
THESIS_PAIR_CBAR_LABEL_FONTSIZE = 24
THESIS_PAIR_CBAR_TICK_FONTSIZE = 20

THESIS_PAIR_SUPXLABEL_Y = 0.065
THESIS_PAIR_SUPYLABEL_X = 0.1

BAR_TITLE_SIZE = 24
BAR_LABEL_SIZE = 20
BAR_TICK_SIZE = 12
BAR_ANNOTATION_SIZE = 12
BAR_GRID_COLOR = "#d9d9d9"


def _get_comp_colors() -> Dict[str, str]:
    cfg_colors = getattr(sett, "COMPS_LEAGUE_COLORS", None)
    if not isinstance(cfg_colors, dict):
        raise ValueError("No colors definition for league competitions was found.")
    # allow typo correction in config
    fixed = {}
    for comp, color in cfg_colors.items():
        if color == "bluevioldet":
            color = "blueviolet"
        fixed[comp] = color
    return fixed


def load_data_into_globals() -> None:
    bundle = load_snapshot()
    fill_globals_with_cache(bundle, update_leagues_list=False)
    g = Global.get_instance()

    print(
        f"Loaded {len(g.all_matches)} total matches in globals; "
        f"{len(sett.COMPS_LEAGUE)} league competitions configured for "
        f"seasons {sett.FIRST_SEASON}..{sett.LAST_SEASON - 1}."
    )


def get_league_matches(*, apply_clean_filter: bool) -> List[FSMatch]:
    g = Global.get_instance()

    if apply_clean_filter:
        league_matches = utils.filter_clean_league_matches(g.all_matches)
    else:
        league_matches = [m for m in g.all_matches if getattr(m, "comp_name", None) in sett.COMPS_LEAGUE]

    league_matches = [
        m
        for m in league_matches
        if getattr(m, "season", None) is not None and sett.FIRST_SEASON <= m.season < sett.LAST_SEASON
    ]
    league_matches.sort(key=lambda m: ((getattr(m, "datetime", None) or 0), getattr(m, "hour_utc", -1), m.id))
    return league_matches


def get_allowed_stat_keys_for_mode(stat_keys: List[str], *, ignore_ignored_stats: bool) -> List[str]:
    if ignore_ignored_stats:
        return utils.get_allowed_match_stat_keys(stat_keys)
    return list(stat_keys)


def build_missingness_pivot(
    df: pd.DataFrame,
    *,
    row_cols: List[str],
    col_name: str,
    value_field: str = "missing_pct",
) -> pd.DataFrame:
    pivot = df.pivot_table(index=row_cols, columns=col_name, values=value_field, aggfunc="mean")
    pivot = pivot.sort_index()

    if pivot.empty:
        return pivot

    row_labels = [" | ".join(map(str, idx if isinstance(idx, tuple) else (idx,))) for idx in pivot.index]
    pivot = pivot.copy()
    pivot.index = row_labels
    return pivot


def _reindex_for_comparable_heatmaps(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    union_rows: bool = True,
    union_cols: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reindex two pivots to the same row/column space.
    Missing rows/cols become NaN and can be rendered as gray.
    """
    if union_rows:
        all_rows = sorted(set(left.index).union(set(right.index)))
    else:
        all_rows = list(left.index)

    if union_cols:
        all_cols = list(dict.fromkeys(list(left.columns) + list(right.columns)))
    else:
        all_cols = list(left.columns)

    left2 = left.reindex(index=all_rows, columns=all_cols)
    right2 = right.reindex(index=all_rows, columns=all_cols)
    return left2, right2


def _plot_pct_heatmap_thesis(
    pivot_df: pd.DataFrame,
    ax: Axes,
    *,
    cmap: str = PCT_CMAP,
    show_cbar: bool = False,
    cbar_label: str = "Missing values (%)",
    hide_ticks: bool = True,
    mask_color: str = THESIS_PAIR_MASK_COLOR,
) -> None:
    """
    Minimal thesis-ready heatmap:
      - no title
      - optional colorbar
      - optional hidden ticks
      - NaN cells shown in a darker gray mask
    """
    data = pivot_df.astype(float)
    norm = PowerNorm(gamma=HEATMAP_GAMMA, vmin=PCT_MIN, vmax=PCT_MAX)

    cmap_obj = matplotlib.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(mask_color)

    hm = sns.heatmap(
        data,
        ax=ax,
        cmap=cmap_obj,
        vmin=PCT_MIN,
        vmax=PCT_MAX,
        norm=norm,
        linewidths=THESIS_PAIR_GRID_WIDTH,
        linecolor=THESIS_PAIR_GRID_COLOR,
        cbar=show_cbar,
        cbar_kws=(
            {
                "label": cbar_label,
                "ticks": PCT_CBAR_TICKS,
            }
            if show_cbar
            else None
        ),
    )

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=9,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=9,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_cbar:
        cbar = hm.collections[0].colorbar
        cbar.set_ticks(PCT_CBAR_TICKS)
        cbar.set_ticklabels([f"{t:g}" for t in PCT_CBAR_TICKS])
        cbar.ax.tick_params(labelsize=THESIS_PAIR_CBAR_TICK_FONTSIZE)
        cbar.set_label(cbar_label, fontsize=THESIS_PAIR_CBAR_LABEL_FONTSIZE)


def plot_thesis_pair_heatmaps(
    left_pivot: pd.DataFrame,
    right_pivot: pd.DataFrame,
    *,
    out_dir: Path,
    stem: str,
    cbar_label: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    align_shapes: bool = True,
    mask_missing_shape: bool = True,
    hide_ticks: bool = True,
    show_cbar: bool = True,
    left_caption: Optional[str] = None,
    right_caption: Optional[str] = None,
) -> None:
    """
    Two aligned heatmaps side by side for thesis use.
    """
    if left_pivot.empty or right_pivot.empty:
        print(f"Skipping thesis pair heatmap {stem}: one side is empty.")
        return

    if align_shapes:
        left_plot, right_plot = _reindex_for_comparable_heatmaps(left_pivot, right_pivot)
    else:
        left_plot, right_plot = left_pivot.copy(), right_pivot.copy()

    if not mask_missing_shape:
        left_plot = left_plot.dropna(axis=0, how="all").dropna(axis=1, how="all")
        right_plot = right_plot.dropna(axis=0, how="all").dropna(axis=1, how="all")

    n_rows = max(len(left_plot.index), len(right_plot.index))
    n_cols = max(len(left_plot.columns), len(right_plot.columns))

    # Thesis-only heatmaps are intentionally stretched horizontally and compressed vertically.
    # The goal is to show missingness patterns/trends on portrait-oriented thesis pages,
    # not to make every individual cell label-readable.
    fig_w = max(15.5, 0.46 * n_cols * 2 + (2.6 if show_cbar else 0.0))
    fig_h = max(4.2, 0.105 * n_rows + 1.25)

    if show_cbar:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(fig_w, fig_h),
            gridspec_kw={"width_ratios": [1, 1, 0.055], "wspace": 0.025},
        )
        ax_left, ax_right, cax = axes
    else:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(fig_w, fig_h),
            gridspec_kw={"wspace": 0.025},
        )
        ax_left, ax_right = axes
        cax = None

    _plot_pct_heatmap_thesis(
        left_plot,
        ax_left,
        show_cbar=False,
        cbar_label=cbar_label,
        hide_ticks=hide_ticks,
    )
    _plot_pct_heatmap_thesis(
        right_plot,
        ax_right,
        show_cbar=False,
        cbar_label=cbar_label,
        hide_ticks=hide_ticks,
    )

    if left_caption:
        ax_left.set_title(left_caption, fontsize=THESIS_PAIR_TITLE_FONTSIZE, pad=8)
    if right_caption:
        ax_right.set_title(right_caption, fontsize=THESIS_PAIR_TITLE_FONTSIZE, pad=8)

    if show_cbar:
        norm = PowerNorm(gamma=HEATMAP_GAMMA, vmin=PCT_MIN, vmax=PCT_MAX)
        cmap_obj = matplotlib.cm.get_cmap(PCT_CMAP).copy()
        cmap_obj.set_bad(THESIS_PAIR_MASK_COLOR)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label, fontsize=THESIS_PAIR_CBAR_LABEL_FONTSIZE)
        cbar.set_ticks(PCT_CBAR_TICKS)
        cbar.set_ticklabels([f"{t:g}" for t in PCT_CBAR_TICKS])
        cbar.ax.tick_params(labelsize=THESIS_PAIR_CBAR_TICK_FONTSIZE)

    if x_label is not None:
        fig.supxlabel(
            x_label,
            fontsize=THESIS_PAIR_AXIS_LABEL_FONTSIZE,
            y=THESIS_PAIR_SUPXLABEL_Y,
        )

    if y_label is not None:
        fig.supylabel(
            y_label,
            fontsize=THESIS_PAIR_AXIS_LABEL_FONTSIZE,
            x=THESIS_PAIR_SUPYLABEL_X,
        )

    fig.tight_layout(rect=[0.06, 0.08, 0.985, 0.98])

    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_thesis_match_stats_before_after(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    out_dir: Path,
    stem: str = "thesis_match_stats_missingness_before_after",
    value_field: str = "missing_pct",
) -> None:
    left_pivot = build_missingness_pivot(
        df_before,
        row_cols=["competition", "season"],
        col_name="stat",
        value_field=value_field,
    )
    right_pivot = build_missingness_pivot(
        df_after,
        row_cols=["competition", "season"],
        col_name="stat",
        value_field=value_field,
    )

    plot_thesis_pair_heatmaps(
        left_pivot,
        right_pivot,
        out_dir=out_dir,
        stem=stem,
        cbar_label="Missing values (%)",
        x_label="Match statistics",
        y_label="Competition - season",
        align_shapes=True,
        mask_missing_shape=True,
        hide_ticks=True,
        show_cbar=True,
        left_caption="Before Cleaning",
        right_caption="After Cleaning",
    )


def plot_thesis_sofifa_skill_raw_vs_persistent(
    df_raw: pd.DataFrame,
    df_persistent: pd.DataFrame,
    *,
    out_dir: Path,
    stem: str = "thesis_sofifa_skill_missingness_raw_vs_persistent",
    value_field: str = "missing_pct",
) -> None:
    left_pivot = build_missingness_pivot(
        df_raw,
        row_cols=["competition", "season"],
        col_name="skill",
        value_field=value_field,
    )
    right_pivot = build_missingness_pivot(
        df_persistent,
        row_cols=["competition", "season"],
        col_name="skill",
        value_field=value_field,
    )

    plot_thesis_pair_heatmaps(
        left_pivot,
        right_pivot,
        out_dir=out_dir,
        stem=stem,
        cbar_label="Missing values (%)",
        x_label="Player skills",
        y_label="Competition - season",
        align_shapes=True,
        mask_missing_shape=False,
        hide_ticks=True,
        show_cbar=True,
        left_caption="Missing in Raw Snapshots",
        right_caption="Persistently Missing",
    )


def plot_match_counts_per_comp(league_matches: List[FSMatch], out_dir: Path) -> pd.DataFrame:
    counts = Counter(m.comp_name for m in league_matches)
    rows = [(comp, counts.get(comp, 0)) for comp in sett.COMPS_LEAGUE]
    df = pd.DataFrame(rows, columns=["competition", "n_matches"])
    df = df.sort_values("n_matches", ascending=False).reset_index(drop=True)

    colors = _get_comp_colors()
    bar_colors = [colors.get(comp, "lightgray") for comp in df["competition"]]

    fig, ax = plt.subplots(figsize=(15.5, 8.8))
    x = np.arange(len(df))
    bars = ax.bar(x, df["n_matches"], color=bar_colors, edgecolor="black", linewidth=0.7)

    ax.set_title(
        f"Number of League Matches per Competition ({sett.FIRST_SEASON}/{sett.FIRST_SEASON + 1}–"
        f"{sett.LAST_SEASON - 1}/{sett.LAST_SEASON})",
        fontsize=BAR_TITLE_SIZE,
        pad=12,
    )
    ax.set_xlabel("Competition", fontsize=BAR_LABEL_SIZE)
    ax.set_ylabel("Number of matches", fontsize=BAR_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(df["competition"], rotation=45, ha="right", fontsize=BAR_TICK_SIZE)
    ax.tick_params(axis="y", labelsize=BAR_TICK_SIZE)
    ax.grid(axis="y", linestyle="--", color=BAR_GRID_COLOR, alpha=0.8)
    ax.set_axisbelow(True)

    y_max = max(df["n_matches"].max(), 1)
    ax.set_ylim(0, y_max * 1.14)
    for bar, n in zip(bars, df["n_matches"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.012,
            f"{int(n)}",
            ha="center",
            va="bottom",
            fontsize=BAR_ANNOTATION_SIZE,
            rotation=0,
        )

    fig.tight_layout()
    png_path = out_dir / "matches_per_league_competition.png"
    pdf_path = out_dir / "matches_per_league_competition.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = out_dir / "matches_per_league_competition.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {csv_path}")
    return df


def _is_missing_match_stat(stat: str, value, *, count_zero_as_missing: bool = True) -> bool:
    if value is None or value == -1:
        return True
    if count_zero_as_missing and stat in ZERO_AS_MISSING_STATS and value == 0:
        return True
    return False


def _is_missing_skill_value(value) -> bool:
    if value is None:
        return True
    try:
        if float(value) == -1.0:
            return True
    except Exception:
        pass
    return isinstance(value, float) and math.isnan(value)


def build_match_stats_missingness(
    league_matches: List[FSMatch],
    *,
    count_zero_as_missing: bool = True,
    ignore_ignored_stats: bool = True,
) -> pd.DataFrame:
    if not league_matches:
        return pd.DataFrame(
            columns=[
                "competition",
                "season",
                "stat",
                "n_missing",
                "n_zero",
                "n_total",
                "missing_pct",
                "zero_pct",
                "missing_or_zero_pct",
            ]
        )

    stat_keys = get_allowed_stat_keys_for_mode(
        list(league_matches[0].stats.keys()),
        ignore_ignored_stats=ignore_ignored_stats,
    )

    totals: Dict[Tuple[str, int, str], int] = defaultdict(int)
    missing: Dict[Tuple[str, int, str], int] = defaultdict(int)
    zeros: Dict[Tuple[str, int, str], int] = defaultdict(int)

    for m in league_matches:
        comp = m.comp_name
        season = m.season
        for stat in stat_keys:
            key = (comp, season, stat)
            totals[key] += 1
            v = m.stats.get(stat, -1)

            if v == 0 and stat in ZERO_AS_MISSING_STATS:
                zeros[key] += 1

            if _is_missing_match_stat(stat, v, count_zero_as_missing=count_zero_as_missing):
                missing[key] += 1

    rows = []
    for key, n_total in totals.items():
        n_missing = missing.get(key, 0)
        n_zero = zeros.get(key, 0)

        missing_pct = 100.0 * n_missing / n_total if n_total else math.nan
        zero_pct = 100.0 * n_zero / n_total if n_total else math.nan

        rows.append(
            {
                "competition": key[0],
                "season": key[1],
                "stat": key[2],
                "n_missing": n_missing,
                "n_zero": n_zero,
                "n_total": n_total,
                "missing_pct": missing_pct,
                "zero_pct": zero_pct,
                "missing_or_zero_pct": missing_pct,
            }
        )

    return pd.DataFrame(rows).sort_values(["competition", "season", "stat"]).reset_index(drop=True)


def _snapshot_date_to_season_year(snapshot_date) -> int:
    # Dates from July onward belong to the season YYYY/YYYY+1.
    if snapshot_date.month >= 7:
        return snapshot_date.year
    return snapshot_date.year - 1


def _iter_sofifa_records():
    g = Global.get_instance()

    # Invert mapping: SOFIFA league_id -> FS league name
    sofifa_league_id_to_fs_name = {
        int(sofifa_id): fs_name for fs_name, sofifa_id in sett.FS_LEAGUE_TO_SOFIFA_LEAGUE_ID.items()
    }

    # Keep full snapshot timeline available for neighboring-snapshot logic.
    snapshots_sorted = sorted(g.sofifa_snapshots, key=lambda x: x[0])

    for snap_idx, (snapshot_date, players_by_id) in enumerate(snapshots_sorted):
        season = _snapshot_date_to_season_year(snapshot_date)

        # Report only thesis seasons: 2021/2022 .. 2024/2025
        if not (sett.FIRST_SEASON <= season < sett.LAST_SEASON):
            continue

        for sofifa_id, rec in players_by_id.items():
            if not isinstance(rec, dict):
                continue

            league_id = rec.get("club_league_id")
            try:
                league_id = int(league_id) if league_id is not None else None
            except (TypeError, ValueError):
                league_id = None

            if league_id is None:
                continue

            league_name = sofifa_league_id_to_fs_name.get(league_id)
            if league_name is None:
                continue

            skills = rec.get("skills")
            if not isinstance(skills, list):
                continue

            yield {
                "snap_idx": snap_idx,
                "snapshot_date": snapshot_date,
                "season": season,
                "league_name": league_name,
                "sofifa_id": int(sofifa_id),
                "skills": skills,
            }


def build_player_skill_missingness_from_raw_sofifa() -> Tuple[pd.DataFrame, pd.DataFrame]:
    skill_names = list(sett.PLAYER_SKILLS)
    totals: Dict[Tuple[str, int, str], int] = defaultdict(int)
    missing: Dict[Tuple[str, int, str], int] = defaultdict(int)
    player_row_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    snapshot_sets: Dict[Tuple[str, int], set] = defaultdict(set)

    total_records = 0
    total_missing_cells = 0

    for rec in _iter_sofifa_records():
        snapshot_date = rec["snapshot_date"]
        season = rec["season"]
        league_name = rec["league_name"]
        skills = rec["skills"]

        player_row_counts[(league_name, season)] += 1
        snapshot_sets[(league_name, season)].add(snapshot_date)
        total_records += 1

        for idx, skill in enumerate(skill_names):
            key = (league_name, season, skill)
            totals[key] += 1

            v = skills[idx] if idx < len(skills) else -1.0
            if _is_missing_skill_value(v):
                missing[key] += 1
                total_missing_cells += 1

    rows = []
    for key, n_total in totals.items():
        n_missing = missing.get(key, 0)
        rows.append(
            {
                "competition": key[0],
                "season": key[1],
                "skill": key[2],
                "n_missing": n_missing,
                "n_total": n_total,
                "missing_pct": 100.0 * n_missing / n_total if n_total else math.nan,
            }
        )

    if rows:
        df = pd.DataFrame(rows).sort_values(["competition", "season", "skill"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["competition", "season", "skill", "n_missing", "n_total", "missing_pct"])

    print(f"Loaded raw SOFIFA records for missingness analysis: {total_records}")
    if total_records > 0:
        total_cells = total_records * len(skill_names)
        print(
            f"Raw SOFIFA missing skill cells: {total_missing_cells}/{total_cells} "
            f"({total_missing_cells / total_cells:.2%})"
        )

    snapshot_rows = []
    for comp, season in sorted(player_row_counts):
        snapshot_rows.append(
            {
                "competition": comp,
                "season": season,
                "n_player_snapshot_rows": player_row_counts[(comp, season)],
                "n_unique_snapshots": len(snapshot_sets[(comp, season)]),
            }
        )
    snapshot_df = pd.DataFrame(snapshot_rows)
    return df, snapshot_df


def build_player_skill_missingness_from_raw_sofifa_persistent() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A skill cell is counted as persistently missing only if:
    - it is missing in the current snapshot row, and
    - neither the previous nor next global SOFIFA snapshot provides a valid value for that player+skill.
    """
    g = Global.get_instance()
    skill_names = list(sett.PLAYER_SKILLS)
    snapshots_sorted = sorted(g.sofifa_snapshots, key=lambda x: x[0])

    totals: Dict[Tuple[str, int, str], int] = defaultdict(int)
    persistent_missing: Dict[Tuple[str, int, str], int] = defaultdict(int)
    player_row_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    snapshot_sets: Dict[Tuple[str, int], set] = defaultdict(set)

    # reuse filtered reporting rows, but neighbors can come from full timeline
    iter_rows = list(_iter_sofifa_records())
    total_records = 0
    total_persistent_cells = 0

    for rec in iter_rows:
        snap_idx = rec["snap_idx"]
        snapshot_date = rec["snapshot_date"]
        season = rec["season"]
        league_name = rec["league_name"]
        sofifa_id = rec["sofifa_id"]
        skills = rec["skills"]

        player_row_counts[(league_name, season)] += 1
        snapshot_sets[(league_name, season)].add(snapshot_date)
        total_records += 1

        prev_players = snapshots_sorted[snap_idx - 1][1] if snap_idx - 1 >= 0 else None
        next_players = snapshots_sorted[snap_idx + 1][1] if snap_idx + 1 < len(snapshots_sorted) else None

        prev_rec = prev_players.get(sofifa_id) if isinstance(prev_players, dict) else None
        next_rec = next_players.get(sofifa_id) if isinstance(next_players, dict) else None

        prev_skills = prev_rec.get("skills") if isinstance(prev_rec, dict) else None
        next_skills = next_rec.get("skills") if isinstance(next_rec, dict) else None

        for idx, skill in enumerate(skill_names):
            key = (league_name, season, skill)
            totals[key] += 1

            curr_v = skills[idx] if idx < len(skills) else -1.0
            if not _is_missing_skill_value(curr_v):
                continue

            prev_v = prev_skills[idx] if isinstance(prev_skills, list) and idx < len(prev_skills) else -1.0
            next_v = next_skills[idx] if isinstance(next_skills, list) and idx < len(next_skills) else -1.0

            if _is_missing_skill_value(prev_v) and _is_missing_skill_value(next_v):
                persistent_missing[key] += 1
                total_persistent_cells += 1

    rows = []
    for key, n_total in totals.items():
        n_missing = persistent_missing.get(key, 0)
        rows.append(
            {
                "competition": key[0],
                "season": key[1],
                "skill": key[2],
                "n_missing": n_missing,
                "n_total": n_total,
                "missing_pct": 100.0 * n_missing / n_total if n_total else math.nan,
            }
        )

    if rows:
        df = pd.DataFrame(rows).sort_values(["competition", "season", "skill"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["competition", "season", "skill", "n_missing", "n_total", "missing_pct"])

    print(f"Loaded raw SOFIFA records for persistent-missingness analysis: {total_records}")
    if total_records > 0:
        total_cells = total_records * len(skill_names)
        print(
            f"Persistent raw SOFIFA missing skill cells: {total_persistent_cells}/{total_cells} "
            f"({total_persistent_cells / total_cells:.2%})"
        )

    snapshot_rows = []
    for comp, season in sorted(player_row_counts):
        snapshot_rows.append(
            {
                "competition": comp,
                "season": season,
                "n_player_snapshot_rows": player_row_counts[(comp, season)],
                "n_unique_snapshots": len(snapshot_sets[(comp, season)]),
            }
        )
    snapshot_df = pd.DataFrame(snapshot_rows)
    return df, snapshot_df


def save_missingness_tables(df: pd.DataFrame, out_dir: Path, stem: str, value_col: str) -> None:
    csv_path = out_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    grouped = (
        df.groupby(value_col, as_index=False)
        .agg(n_missing=("n_missing", "sum"), n_total=("n_total", "sum"))
        .assign(missing_pct=lambda x: 100.0 * x["n_missing"] / x["n_total"])
        .sort_values("missing_pct", ascending=False)
    )

    if "n_zero" in df.columns:
        grouped_zero = (
            df.groupby(value_col, as_index=False)
            .agg(n_zero=("n_zero", "sum"), n_total=("n_total", "sum"))
            .assign(zero_pct=lambda x: 100.0 * x["n_zero"] / x["n_total"])
        )
        grouped = grouped.merge(grouped_zero[[value_col, "n_zero", "zero_pct"]], on=value_col, how="left")
        grouped["missing_or_zero_pct"] = grouped["missing_pct"]

    feature_summary_path = out_dir / f"{stem}_feature_summary.csv"
    grouped.to_csv(feature_summary_path, index=False)
    print(f"Saved: {feature_summary_path}")

    grouped_cs = (
        df.groupby(["competition", "season"], as_index=False)
        .agg(n_missing=("n_missing", "sum"), n_total=("n_total", "sum"))
        .assign(missing_pct=lambda x: 100.0 * x["n_missing"] / x["n_total"])
        .sort_values(["competition", "season"])
    )
    if "n_zero" in df.columns:
        grouped_cs_zero = (
            df.groupby(["competition", "season"], as_index=False)
            .agg(n_zero=("n_zero", "sum"), n_total=("n_total", "sum"))
            .assign(zero_pct=lambda x: 100.0 * x["n_zero"] / x["n_total"])
        )
        grouped_cs = grouped_cs.merge(
            grouped_cs_zero[["competition", "season", "n_zero", "zero_pct"]],
            on=["competition", "season"],
            how="left",
        )
        grouped_cs["missing_or_zero_pct"] = grouped_cs["missing_pct"]

    comp_season_summary_path = out_dir / f"{stem}_comp_season_summary.csv"
    grouped_cs.to_csv(comp_season_summary_path, index=False)
    print(f"Saved: {comp_season_summary_path}")


def plot_pct_heatmap(
    pivot_df: pd.DataFrame,
    ax: Axes,
    *,
    title: str,
    cbar_label: str,
    cmap: str = PCT_CMAP,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    x_rotation: float = 45.0,
    y_rotation: float = 0.0,
) -> None:
    data = pivot_df.astype(float)
    norm = PowerNorm(gamma=HEATMAP_GAMMA, vmin=PCT_MIN, vmax=PCT_MAX)

    hm = sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        vmin=PCT_MIN,
        vmax=PCT_MAX,
        norm=norm,
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        cbar_kws={
            "label": cbar_label,
            "ticks": PCT_CBAR_TICKS,
        },
    )

    ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    ax.tick_params(axis="x", rotation=x_rotation)
    ax.tick_params(axis="y", rotation=y_rotation)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=9,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=10,
    )

    # Make colorbar labels explicit and easier to read
    cbar = hm.collections[0].colorbar
    cbar.set_ticks(PCT_CBAR_TICKS)
    cbar.set_ticklabels([f"{t:g}" for t in PCT_CBAR_TICKS])


def plot_missingness_heatmap(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    row_cols: List[str],
    col_name: str,
    stem: str,
    title: str,
    value_field: str = "missing_pct",
    cbar_label: str = "Missing values [%]",
) -> None:
    pivot = df.pivot_table(index=row_cols, columns=col_name, values=value_field, aggfunc="mean")
    pivot = pivot.sort_index()

    if pivot.empty:
        print(f"Skipping empty heatmap: {stem}")
        return

    row_labels = [" | ".join(map(str, idx if isinstance(idx, tuple) else (idx,))) for idx in pivot.index]
    pivot = pivot.copy()
    pivot.index = row_labels

    fig_w = max(12, 0.35 * len(pivot.columns) + 6)
    fig_h = max(8, 0.28 * len(pivot.index) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_pct_heatmap(
        pivot,
        ax,
        title=title,
        cbar_label=cbar_label,
        x_label=col_name.replace("_", " ").title(),
        y_label=" / ".join(x.replace("_", " ") for x in row_cols).title(),
    )

    fig.tight_layout()
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def build_match_stats_team_month_missingness(
    league_matches: List[FSMatch],
    *,
    count_zero_as_missing: bool = True,
    ignore_ignored_stats: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Localize missingness by (competition, season, team, month, stat).
    """
    if not league_matches:
        empty_cols_detail = [
            "competition",
            "season",
            "team_id",
            "team_name",
            "month",
            "stat",
            "n_missing",
            "n_total",
            "n_matches",
            "missing_pct",
        ]
        empty_cols_overall = [
            "competition",
            "season",
            "team_id",
            "team_name",
            "month",
            "n_missing",
            "n_total",
            "n_matches",
            "missing_pct",
        ]
        empty_cols_summary = [
            "competition",
            "season",
            "n_team_month_slices",
            "mean_slice_missing_pct",
            "median_slice_missing_pct",
            "worst_slice_missing_pct",
            "worst_team_name",
            "worst_month",
            "worst_n_matches",
        ]
        return (
            pd.DataFrame(columns=empty_cols_detail),
            pd.DataFrame(columns=empty_cols_overall),
            pd.DataFrame(columns=empty_cols_summary),
        )

    stat_keys = get_allowed_stat_keys_for_mode(
        list(league_matches[0].stats.keys()),
        ignore_ignored_stats=ignore_ignored_stats,
    )

    totals: Dict[Tuple[str, int, int, str, int, str], int] = defaultdict(int)
    missing: Dict[Tuple[str, int, int, str, int, str], int] = defaultdict(int)
    match_ids: Dict[Tuple[str, int, int, str, int], set] = defaultdict(set)

    for m in league_matches:
        month = getattr(m, "month", None)
        if month is None and getattr(m, "datetime", None) is not None:
            month = m.datetime.month
        if month is None:
            continue

        team_pairs = []
        if getattr(m, "home_team", None) is not None:
            team_pairs.append((m.home_team.id, m.home_team.name))
        if getattr(m, "away_team", None) is not None:
            team_pairs.append((m.away_team.id, m.away_team.name))

        for team_id, team_name in team_pairs:
            overall_key = (m.comp_name, m.season, team_id, team_name, int(month))
            match_ids[overall_key].add(m.id)

            for stat in stat_keys:
                key = (m.comp_name, m.season, team_id, team_name, int(month), stat)
                totals[key] += 1
                v = m.stats.get(stat, -1)
                if _is_missing_match_stat(stat, v, count_zero_as_missing=count_zero_as_missing):
                    missing[key] += 1

    detail_rows = []
    for key, n_total in totals.items():
        comp, season, team_id, team_name, month, stat = key
        n_missing = missing.get(key, 0)
        overall_key = (comp, season, team_id, team_name, month)
        detail_rows.append(
            {
                "competition": comp,
                "season": season,
                "team_id": team_id,
                "team_name": team_name,
                "month": month,
                "stat": stat,
                "n_missing": n_missing,
                "n_total": n_total,
                "n_matches": len(match_ids[overall_key]),
                "missing_pct": 100.0 * n_missing / n_total if n_total else math.nan,
            }
        )

    detail_df = (
        pd.DataFrame(detail_rows)
        .sort_values(["competition", "season", "team_name", "month", "stat"])
        .reset_index(drop=True)
    )

    overall_df = (
        detail_df.groupby(["competition", "season", "team_id", "team_name", "month"], as_index=False)
        .agg(n_missing=("n_missing", "sum"), n_total=("n_total", "sum"), n_matches=("n_matches", "max"))
        .assign(missing_pct=lambda x: 100.0 * x["n_missing"] / x["n_total"])
        .sort_values(["competition", "season", "missing_pct"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    summary_rows = []
    for (comp, season), gdf in overall_df.groupby(["competition", "season"], sort=True):
        worst = gdf.sort_values(["missing_pct", "n_matches"], ascending=[False, False]).iloc[0]
        summary_rows.append(
            {
                "competition": comp,
                "season": season,
                "n_team_month_slices": len(gdf),
                "mean_slice_missing_pct": gdf["missing_pct"].mean(),
                "median_slice_missing_pct": gdf["missing_pct"].median(),
                "worst_slice_missing_pct": worst["missing_pct"],
                "worst_team_name": worst["team_name"],
                "worst_month": int(worst["month"]),
                "worst_n_matches": int(worst["n_matches"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["competition", "season"]).reset_index(drop=True)
    return detail_df, overall_df, summary_df


def _competition_from_label(label: str) -> str:
    # Example label:
    # "Hatay Spor Kulübü | Turkey Süper Lig 2022 | m04"

    middle = label.split(" | ")[1]  # "Turkey Süper Lig 2022"
    competition = middle.rsplit(" ", 1)[0]  # remove trailing season
    return competition


def plot_top_team_month_slices(overall_df: pd.DataFrame, out_dir: Path, top_n: int = 30) -> None:
    if overall_df.empty:
        print("Skipping team-month bar chart because the dataframe is empty.")
        return

    # Prefer slices with enough matches so one-match artifacts do not dominate.
    filtered = overall_df[overall_df["n_matches"] >= 4].copy()
    if filtered.empty:
        filtered = overall_df.copy()

    top = filtered.sort_values(["missing_pct", "n_matches"], ascending=[False, False]).head(top_n).copy()
    top["label"] = top.apply(
        lambda r: f"{r['team_name']} | {r['competition']} {int(r['season'])} | m{int(r['month']):02d}",
        axis=1,
    )

    bar_colors = [sett.COMPS_LEAGUE_COLORS.get(_competition_from_label(lbl), "steelblue") for lbl in top["label"]]

    fig_h = max(8, 0.34 * len(top) + 2.5)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    y = np.arange(len(top))
    bars = ax.barh(y, top["missing_pct"], color=bar_colors, edgecolor="black", linewidth=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(top["label"])
    ax.invert_yaxis()
    ax.set_xlim(PCT_MIN, PCT_BAR_YMAX)
    ax.set_xlabel("Missing values (%)")
    ax.set_ylabel("Team / competition-season / month")
    ax.set_title("Most problematic team-month slices for match-stat missingness")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, pct in zip(bars, top["missing_pct"]):
        ax.text(
            min(bar.get_width() + 1.0, PCT_BAR_YMAX - 1.5),
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
        )

    fig.tight_layout()
    png_path = out_dir / "match_stats_missingness_top_team_month_slices.png"
    pdf_path = out_dir / "match_stats_missingness_top_team_month_slices.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def _match_sort_key_for_analysis(m) -> Tuple:
    dt = getattr(m, "datetime", None)
    hr = getattr(m, "hour_utc", None)
    hr = int(hr) if isinstance(hr, int) else -1
    return dt, hr, m.id


def _is_missing_timeline_value(v, *, count_zero_as_missing: bool) -> bool:
    if v is None or v == -1:
        return True
    if count_zero_as_missing and v == 0:
        return True
    return False


def plot_stat_missingness_timeline(
    stat_key: str,
    *,
    apply_clean_filter: bool,
    count_zero_as_missing: bool = False,
    output_path: str | Path | None = None,
    max_matches_cap: int | None = None,
    title_prefix: str = "Missingness timeline",
) -> None:
    g = Global.get_instance()

    if apply_clean_filter:
        league_matches = utils.filter_clean_league_matches(g.all_matches)
    else:
        league_matches = [m for m in g.all_matches if getattr(m, "comp_name", None) in sett.COMPS_LEAGUE]

    league_matches = [
        m
        for m in league_matches
        if getattr(m, "season", None) is not None
        and sett.FIRST_SEASON <= m.season < sett.LAST_SEASON
        and getattr(m, "datetime", None) is not None
    ]

    if not league_matches:
        print(f"[analysis] No league matches found for timeline plot of {stat_key}.")
        return

    sample_stats = getattr(league_matches[0], "stats", {}) or {}
    if stat_key not in sample_stats:
        raise ValueError(f"stat_key '{stat_key}' not found in match.stats")

    grouped: dict[str, List[FSMatch]] = {}
    for m in league_matches:
        row_label = f"{m.comp_name} | {m.season}"
        grouped.setdefault(row_label, []).append(m)

    def _row_sort_key(label: str):
        comp_name, season_str = label.rsplit(" | ", 1)
        try:
            season = int(season_str)
        except Exception:
            season = 999999
        return comp_name, season

    row_labels = sorted(grouped.keys(), key=_row_sort_key)

    rows_as_binary: List[List[float]] = []
    max_len = 0

    for row_label in row_labels:
        ms = sorted(grouped[row_label], key=_match_sort_key_for_analysis)

        row_vals = []
        for m in ms:
            v = (getattr(m, "stats", {}) or {}).get(stat_key, -1)
            row_vals.append(1.0 if _is_missing_timeline_value(v, count_zero_as_missing=count_zero_as_missing) else 0.0)

        rows_as_binary.append(row_vals)
        max_len = max(max_len, len(row_vals))

    if max_len == 0:
        print(f"[analysis] No matches found for {stat_key}; skipping timeline plot.")
        return

    if max_matches_cap is not None:
        max_len = min(max_len, max_matches_cap)

    arr = np.full((len(rows_as_binary), max_len), np.nan, dtype=np.float32)
    for i, row_vals in enumerate(rows_as_binary):
        clipped = row_vals[:max_len]
        arr[i, : len(clipped)] = clipped

    fig_h = max(10, 0.28 * len(row_labels))
    fig_w = max(16, max_len / 10)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="magma")

    missing_note = "counting 0 as missing" if count_zero_as_missing else "counting only -1/None as missing"
    pretty_name = stat_key.replace("_", " ")
    plt.title(f"{title_prefix} for {pretty_name} by competition and season\n({missing_note})")
    plt.xlabel("Match order within competition season")
    plt.ylabel("Competition / Season")

    plt.yticks(np.arange(len(row_labels)), row_labels, fontsize=8)

    if max_len <= 40:
        xticks = np.arange(max_len)
    else:
        step = max(1, math.ceil(max_len / 20))
        xticks = np.arange(0, max_len, step)
    plt.xticks(xticks, xticks + 1)

    cbar = plt.colorbar(im)
    cbar.set_label("Missing value (1=yes, 0=no)")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["present", "missing"])

    plt.tight_layout()

    if output_path is None:
        suffix = "zero_missing" if count_zero_as_missing else "strict_missing"
        output_path = f"{stat_key}_missingness_timeline_{suffix}.png"

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[analysis] Saved stat timeline plot: {output_path}")


def _competition_season_first_match_dates(league_matches: List[FSMatch]) -> Dict[Tuple[str, int], datetime]:
    out: Dict[Tuple[str, int], datetime] = {}
    for m in league_matches:
        comp = getattr(m, "comp_name", None)
        season = getattr(m, "season", None)
        dt = getattr(m, "datetime", None)
        if comp is None or season is None or dt is None:
            continue

        key = (comp, int(season))
        prev = out.get(key)
        if prev is None or dt < prev:
            out[key] = dt
    return out


def build_fs_vs_sofifa_team_counts_per_comp_season(league_matches: List[FSMatch]) -> pd.DataFrame:
    """
    Compare:
      - FootyStats distinct teams appearing in matches of a competition season
      - SOFIFA distinct clubs in the mapped SOFIFA league, using the snapshot
        nearest to the first match date of that competition season

    This is a source-coverage comparison, not a matching-quality plot.
    """
    g = Global.get_instance()

    # ---- FootyStats counts ----
    fs_team_ids_by_cs: Dict[Tuple[str, int], set[int]] = defaultdict(set)
    for m in league_matches:
        comp_name = getattr(m, "comp_name", None)
        season = getattr(m, "season", None)
        if comp_name is None or season is None:
            continue

        key = (comp_name, int(season))
        if getattr(m, "home_team", None) is not None:
            fs_team_ids_by_cs[key].add(int(m.home_team.id))
        if getattr(m, "away_team", None) is not None:
            fs_team_ids_by_cs[key].add(int(m.away_team.id))

    first_match_dates = _competition_season_first_match_dates(league_matches)

    # ---- SOFIFA counts via nearest snapshot to season start ----
    snapshots = getattr(g, "sofifa_snapshots", []) or []

    rows = []
    for (comp_name, season), fs_team_ids in sorted(fs_team_ids_by_cs.items(), key=lambda x: (x[0][0], x[0][1])):
        sofifa_league_id = sett.FS_LEAGUE_TO_SOFIFA_LEAGUE_ID.get(comp_name)
        first_dt = first_match_dates.get((comp_name, season))

        sofifa_team_ids: set[int] = set()
        chosen_snapshot_date = None
        snapshot_gap_days = None

        if sofifa_league_id is not None and first_dt is not None and snapshots:
            # choose snapshot nearest to season start
            best_snap = min(
                snapshots,
                key=lambda x: abs((x[0] - first_dt.date()).days),
            )
            chosen_snapshot_date, snap_players = best_snap
            snapshot_gap_days = abs((chosen_snapshot_date - first_dt.date()).days)

            for _, rec in snap_players.items():
                if not isinstance(rec, dict):
                    continue

                league_id = rec.get("club_league_id")
                club_id = rec.get("club_id")

                try:
                    league_id = int(league_id) if league_id not in (None, "") else None
                except Exception:
                    league_id = None

                try:
                    club_id = int(club_id) if club_id not in (None, "") else None
                except Exception:
                    club_id = None

                if league_id == int(sofifa_league_id) and club_id is not None:
                    sofifa_team_ids.add(club_id)

        rows.append(
            {
                "competition": comp_name,
                "season": int(season),
                "fs_n_teams": len(fs_team_ids),
                "sofifa_n_teams": len(sofifa_team_ids) if sofifa_league_id is not None else np.nan,
                "team_gap": (len(fs_team_ids) - len(sofifa_team_ids)) if sofifa_league_id is not None else np.nan,
                "sofifa_league_id": sofifa_league_id,
                "season_start_date": first_dt.date().isoformat() if first_dt is not None else None,
                "sofifa_snapshot_date": chosen_snapshot_date.isoformat() if chosen_snapshot_date is not None else None,
                "snapshot_gap_days": snapshot_gap_days,
            }
        )

    df = pd.DataFrame(rows).sort_values(["competition", "season"]).reset_index(drop=True)
    return df


def plot_fs_vs_sofifa_team_counts(team_counts_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Plot only SOFIFA undercoverage relative to FootyStats:
      deficit = max(0, fs_n_teams - sofifa_n_teams)

    This intentionally suppresses the visually noisy cases where SOFIFA has
    more clubs than FootyStats teams, because those are usually overcoverage /
    league-bucket breadth rather than harmful missingness.

    Output:
      - one bar per competition season
      - bars shown only when deficit > 0
      - labels show "FS X vs SOFIFA Y"
    """
    if team_counts_df.empty:
        print("Skipping FS vs SOFIFA deficit plot because dataframe is empty.")
        return

    df = team_counts_df.copy()
    df["label"] = df.apply(lambda r: f"{r['competition']} | {int(r['season'])}", axis=1)

    # Keep stable order
    df = df.sort_values(["competition", "season"]).reset_index(drop=True)

    # Deficit = only the problematic direction
    df["team_deficit"] = (df["fs_n_teams"] - df["sofifa_n_teams"]).clip(lower=0)

    # Split into all rows vs highlighted rows
    df_bad = df[df["team_deficit"] > 0].copy()

    if df_bad.empty:
        print("No competition seasons with SOFIFA undercoverage were found.")
        return

    fig_h = max(5, 0.55 * len(df_bad) + 2.5)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))

    y = np.arange(len(df_bad))
    bars = ax.barh(
        y,
        df_bad["team_deficit"],
        color="crimson",
        edgecolor="black",
        linewidth=0.7,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df_bad["label"], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Missing SOFIFA clubs relative to FootyStats team count")
    ax.set_ylabel("Competition / Season")
    ax.set_title("SOFIFA team undercoverage by competition season")

    xmax = max(2, int(df_bad["team_deficit"].max()) + 2)
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, (_, row) in zip(bars, df_bad.iterrows()):
        fs_n = int(row["fs_n_teams"])
        sf_n = int(row["sofifa_n_teams"])
        deficit = int(row["team_deficit"])

        txt = f"{deficit}  (FS {fs_n} vs SOFIFA {sf_n})"
        ax.text(
            min(bar.get_width() + 0.12, xmax - 0.1),
            bar.get_y() + bar.get_height() / 2,
            txt,
            va="center",
            ha="left",
            fontsize=9,
            color="black",
        )

    fig.tight_layout()

    png_path = out_dir / "fs_vs_sofifa_team_undercoverage_per_comp_season.png"
    pdf_path = out_dir / "fs_vs_sofifa_team_undercoverage_per_comp_season.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def run_analysis_suite(
    *,
    out_dir: Path,
    apply_clean_filter: bool,
    ignore_ignored_stats: bool,
    include_offsides_timelines: bool,
) -> Dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    league_matches = get_league_matches(apply_clean_filter=apply_clean_filter)

    # 1) Competition distribution
    plot_match_counts_per_comp(league_matches, out_dir)

    # 1b) FS vs SOFIFA source coverage
    df_fs_vs_sofifa = build_fs_vs_sofifa_team_counts_per_comp_season(league_matches)
    path = out_dir / "fs_vs_sofifa_team_counts_per_comp_season.csv"
    df_fs_vs_sofifa.to_csv(path, index=False)
    print(f"Saved: {path}")
    plot_fs_vs_sofifa_team_counts(df_fs_vs_sofifa, out_dir)

    # 2a) Match-stat missingness
    df_match = build_match_stats_missingness(
        league_matches,
        count_zero_as_missing=True,
        ignore_ignored_stats=ignore_ignored_stats,
    )
    save_missingness_tables(df_match, out_dir, "match_stats_missingness", "stat")
    plot_missingness_heatmap(
        df_match,
        out_dir=out_dir,
        row_cols=["competition", "season"],
        col_name="stat",
        stem="match_stats_missingness_heatmap_comp_season",
        title="Missing match statistics by competition and season\n(zeros counted as missing for selected stats)",
        value_field="missing_pct",
        cbar_label="Missing or zero-as-missing values (%)",
    )

    df_match_strict = build_match_stats_missingness(
        league_matches,
        count_zero_as_missing=False,
        ignore_ignored_stats=ignore_ignored_stats,
    )
    save_missingness_tables(df_match_strict, out_dir, "match_stats_missingness_strict", "stat")
    plot_missingness_heatmap(
        df_match_strict,
        out_dir=out_dir,
        row_cols=["competition", "season"],
        col_name="stat",
        stem="match_stats_missingness_strict_heatmap_comp_season",
        title="Missing match statistics by competition and season\n(only -1 / None counted as missing)",
        value_field="missing_pct",
        cbar_label="Missing values (%)",
    )

    df_match_zero_subset = df_match[df_match["stat"].isin(ZERO_AS_MISSING_STATS)].copy()
    if not df_match_zero_subset.empty:
        plot_missingness_heatmap(
            df_match_zero_subset,
            out_dir=out_dir,
            row_cols=["competition", "season"],
            col_name="stat",
            stem="match_stats_zero_values_heatmap_comp_season",
            title="Zero values in selected match statistics by competition and season",
            value_field="zero_pct",
            cbar_label="Zero values (%)",
        )

    df_team_month_detail, df_team_month_overall, df_team_month_summary = build_match_stats_team_month_missingness(
        league_matches,
        count_zero_as_missing=True,
        ignore_ignored_stats=ignore_ignored_stats,
    )
    if not df_team_month_detail.empty:
        path = out_dir / "match_stats_missingness_team_month_detail.csv"
        df_team_month_detail.to_csv(path, index=False)
        print(f"Saved: {path}")
    if not df_team_month_overall.empty:
        path = out_dir / "match_stats_missingness_team_month_overall.csv"
        df_team_month_overall.to_csv(path, index=False)
        print(f"Saved: {path}")
        plot_top_team_month_slices(df_team_month_overall, out_dir)
    if not df_team_month_summary.empty:
        path = out_dir / "match_stats_missingness_team_month_comp_season_summary.csv"
        df_team_month_summary.to_csv(path, index=False)
        print(f"Saved: {path}")

    # 2b) SOFIFA skill missingness
    df_skill, df_skill_rows = build_player_skill_missingness_from_raw_sofifa()
    save_missingness_tables(df_skill, out_dir, "player_skill_missingness_raw_sofifa", "skill")
    if not df_skill.empty:
        plot_missingness_heatmap(
            df_skill,
            out_dir=out_dir,
            row_cols=["competition", "season"],
            col_name="skill",
            stem="player_skill_missingness_raw_sofifa_heatmap_comp_season",
            title="Missing raw SOFIFA player-skill values by competition and season",
            cbar_label="Missing values (%)",
        )
    if not df_skill_rows.empty:
        skill_rows_path = out_dir / "player_skill_missingness_raw_sofifa_snapshot_rows.csv"
        df_skill_rows.to_csv(skill_rows_path, index=False)
        print(f"Saved: {skill_rows_path}")

    df_skill_persistent, df_skill_persistent_rows = build_player_skill_missingness_from_raw_sofifa_persistent()
    save_missingness_tables(
        df_skill_persistent,
        out_dir,
        "player_skill_missingness_raw_sofifa_persistent",
        "skill",
    )
    if not df_skill_persistent.empty:
        plot_missingness_heatmap(
            df_skill_persistent,
            out_dir=out_dir,
            row_cols=["competition", "season"],
            col_name="skill",
            stem="player_skill_missingness_raw_sofifa_persistent_heatmap_comp_season",
            title="Persistently missing raw SOFIFA player-skill values by competition and season",
            cbar_label="Persistently missing values (%)",
        )
    if not df_skill_persistent_rows.empty:
        skill_rows_path = out_dir / "player_skill_missingness_raw_sofifa_persistent_snapshot_rows.csv"
        df_skill_persistent_rows.to_csv(skill_rows_path, index=False)
        print(f"Saved: {skill_rows_path}")

    if (not df_skill.empty) and (not df_skill_persistent.empty):
        plot_thesis_sofifa_skill_raw_vs_persistent(
            df_skill,
            df_skill_persistent,
            out_dir=out_dir,
        )

    # 3) Pre-match xG timelines
    plot_stat_missingness_timeline(
        "home_prematch_xg",
        apply_clean_filter=apply_clean_filter,
        count_zero_as_missing=True,
        output_path=out_dir / "home_prematch_xg_missingness_timeline_zero_missing.png",
        title_prefix="Missingness timeline",
    )
    plot_stat_missingness_timeline(
        "away_prematch_xg",
        apply_clean_filter=apply_clean_filter,
        count_zero_as_missing=True,
        output_path=out_dir / "away_prematch_xg_missingness_timeline_zero_missing.png",
        title_prefix="Missingness timeline",
    )

    # 4) Offsides timelines only when desired
    if include_offsides_timelines:
        plot_stat_missingness_timeline(
            "home_offsides",
            apply_clean_filter=apply_clean_filter,
            count_zero_as_missing=False,
            output_path=out_dir / "home_offsides_missingness_timeline_strict_missing.png",
            title_prefix="Missingness timeline",
        )
        plot_stat_missingness_timeline(
            "away_offsides",
            apply_clean_filter=apply_clean_filter,
            count_zero_as_missing=False,
            output_path=out_dir / "away_offsides_missingness_timeline_strict_missing.png",
            title_prefix="Missingness timeline",
        )

    print(f"All outputs saved into: {out_dir}")
    return {
        "df_match": df_match,
        "df_match_strict": df_match_strict,
        "df_skill": df_skill,
        "df_skill_persistent": df_skill_persistent,
        "df_fs_vs_sofifa": df_fs_vs_sofifa,
    }


def main() -> None:
    base_out_dir = Path(sett.PROJECT_ROOT) / "docs/experiments/thesis_data_overview"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw snapshot data once
    load_data_into_globals()

    before_out_dir = base_out_dir / "before_comp_season_filtering"
    after_out_dir = base_out_dir / "after_comp_season_filtering"

    # BEFORE: all league competition seasons, all raw match stats including offsides
    before_results = run_analysis_suite(
        out_dir=before_out_dir,
        apply_clean_filter=False,
        ignore_ignored_stats=False,
        include_offsides_timelines=True,
    )

    # AFTER: excluded competition seasons removed, ignored stats removed from stat-based summaries
    after_results = run_analysis_suite(
        out_dir=after_out_dir,
        apply_clean_filter=True,
        ignore_ignored_stats=True,
        include_offsides_timelines=False,
    )

    # Thesis-ready before vs after match-stat comparison
    if (
        before_results["df_match_strict"] is not None
        and not before_results["df_match_strict"].empty
        and after_results["df_match_strict"] is not None
        and not after_results["df_match_strict"].empty
    ):
        plot_thesis_match_stats_before_after(
            before_results["df_match_strict"],
            after_results["df_match_strict"],
            out_dir=base_out_dir,
        )

    print(f"All thesis overview outputs saved under: {base_out_dir}")


if __name__ == "__main__":
    main()
