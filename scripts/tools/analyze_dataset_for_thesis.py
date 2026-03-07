from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import load_snapshot
from football_outcomes.data.fs_models import FSMatch
from football_outcomes.data.fs_retrieve import fill_globals_with_cache

matplotlib.use("Agg")

# Raw match-stat fields where literal zero may also mean “missing / unavailable” in provider data.
ZERO_AS_MISSING_STATS = {
    "home_attacks",
    "away_attacks",
    "home_dangerous_attacks",
    "away_dangerous_attacks",
    "home_prematch_xg",
    "away_prematch_xg",
}


def _get_comp_colors() -> Dict[str, str]:
    cfg_colors = getattr(sett, "COMPS_LEAGUE_COLORS", None)
    if not isinstance(cfg_colors, dict):
        raise ValueError("No colors definition for league competitions was found.")
    return {comp: color for comp, color in cfg_colors.items()}


def load_data_into_globals() -> List[FSMatch]:
    bundle = load_snapshot()
    fill_globals_with_cache(bundle, update_leagues_list=False)
    g = Global.get_instance()

    league_matches = [m for m in g.all_matches if getattr(m, "comp_name", None) in sett.COMPS_LEAGUE]
    league_matches.sort(key=lambda m: ((getattr(m, "datetime", None) or 0), getattr(m, "hour_utc", -1), m.id))

    print(f"Loaded {len(league_matches)} league matches across {len(sett.COMPS_LEAGUE)} league competitions.")
    return league_matches


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

    ax.set_title("Number of league matches per competition (all seasons)")
    ax.set_xlabel("Competition")
    ax.set_ylabel("Number of matches")
    ax.set_xticks(x)
    ax.set_xticklabels(df["competition"], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
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
            fontsize=8,
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


def _is_missing_match_stat(stat: str, value) -> bool:
    if value is None or value == -1:
        return True
    if stat in ZERO_AS_MISSING_STATS and value == 0:
        return True
    return False


def build_match_stats_missingness(league_matches: List[FSMatch]) -> pd.DataFrame:
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

    stat_keys = list(league_matches[0].stats.keys())
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
            if _is_missing_match_stat(stat, v):
                missing[key] += 1

    rows = []
    for key, n_total in totals.items():
        n_missing = missing.get(key, 0)
        n_zero = zeros.get(key, 0)
        rows.append(
            {
                "competition": key[0],
                "season": key[1],
                "stat": key[2],
                "n_missing": n_missing,
                "n_zero": n_zero,
                "n_total": n_total,
                "missing_pct": 100.0 * n_missing / n_total if n_total else math.nan,
                "zero_pct": 100.0 * n_zero / n_total if n_total else math.nan,
                "missing_or_zero_pct": 100.0 * n_missing / n_total if n_total else math.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(["competition", "season", "stat"]).reset_index(drop=True)


def _snapshot_date_to_season_year(snapshot_date) -> int:
    # Mirrors European season convention in your project: dates from Aug onward belong to season YYYY/YY+1.
    if snapshot_date.month >= 7:
        return snapshot_date.year
    return snapshot_date.year - 1


def _iter_sofifa_records():
    g = Global.get_instance()

    # Invert mapping: SOFIFA league_id -> FS league name
    sofifa_league_id_to_fs_name = {
        int(sofifa_id): fs_name for fs_name, sofifa_id in sett.FS_LEAGUE_TO_SOFIFA_LEAGUE_ID.items()
    }

    for snapshot_date, players_by_id in g.sofifa_snapshots:
        season = _snapshot_date_to_season_year(snapshot_date)

        # Keep only thesis seasons: 2021/2022 .. 2024/2025
        if not (sett.FIRST_SEASON <= season < sett.LAST_SEASON):
            continue

        for _sofifa_id, rec in players_by_id.items():
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

            yield snapshot_date, season, league_name, skills


def build_player_skill_missingness_from_raw_sofifa() -> Tuple[pd.DataFrame, pd.DataFrame]:
    skill_names = list(sett.PLAYER_SKILLS)
    totals: Dict[Tuple[str, int, str], int] = defaultdict(int)
    missing: Dict[Tuple[str, int, str], int] = defaultdict(int)
    player_row_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    snapshot_sets: Dict[Tuple[str, int], set] = defaultdict(set)

    total_records = 0
    total_missing_cells = 0

    for snapshot_date, season, league_name, skills in _iter_sofifa_records():
        player_row_counts[(league_name, season)] += 1
        snapshot_sets[(league_name, season)].add(snapshot_date)
        total_records += 1

        for idx, skill in enumerate(skill_names):
            key = (league_name, season, skill)
            totals[key] += 1

            v = skills[idx] if idx < len(skills) else -1.0
            if v is None or v == -1.0 or (isinstance(v, float) and math.isnan(v)):
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

    row_labels = [" | ".join(map(str, idx if isinstance(idx, tuple) else (idx,))) for idx in pivot.index]
    col_labels = list(pivot.columns)
    arr = pivot.to_numpy(dtype=float)

    fig_w = max(12, 0.35 * len(col_labels) + 6)
    fig_h = max(8, 0.28 * len(row_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, aspect="auto", interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel(col_name.replace("_", " ").title())
    ax.set_ylabel(" / ".join(x.replace("_", " ") for x in row_cols).title())
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    out_dir = Path(sett.PROJECT_ROOT) / "docs/experiments/thesis_data_overview"
    out_dir.mkdir(parents=True, exist_ok=True)

    league_matches = load_data_into_globals()

    # 1) Thesis-ready competition distribution figure
    plot_match_counts_per_comp(league_matches, out_dir)

    # 2a) Detailed raw match-stat missingness (+ selected zero values treated as missing)
    df_match = build_match_stats_missingness(league_matches)
    save_missingness_tables(df_match, out_dir, "match_stats_missingness", "stat")
    plot_missingness_heatmap(
        df_match,
        out_dir=out_dir,
        row_cols=["competition", "season"],
        col_name="stat",
        stem="match_stats_missingness_heatmap_comp_season",
        title="Missing match statistics by competition and season",
        value_field="missing_pct",
        cbar_label="Missing or zero-as-missing values [%]",
    )

    # Useful extra heatmap: literal zeros only for the selected ambiguous stats.
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
            cbar_label="Zero values [%]",
        )

    # 2b) Detailed player-skill missingness from raw SOFIFA CSV snapshots.
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
        )
    if not df_skill_rows.empty:
        skill_rows_path = out_dir / "player_skill_missingness_raw_sofifa_snapshot_rows.csv"
        df_skill_rows.to_csv(skill_rows_path, index=False)
        print(f"Saved: {skill_rows_path}")

    print(f"All outputs saved into: {out_dir}")


if __name__ == "__main__":
    main()
