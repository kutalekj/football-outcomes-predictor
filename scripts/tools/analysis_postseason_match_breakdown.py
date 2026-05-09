from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import football_outcomes.utils.fs_common as utils
from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import load_snapshot
from football_outcomes.data.fs_models import FSMatch
from football_outcomes.data.fs_retrieve import fill_globals_with_cache

matplotlib.use("Agg")

APPLY_CLEAN_FILTER = True
ONLY_AFFECTED_SEASONS_FOR_ROUND_PLOTS = True

TITLE_SIZE = 24
LABEL_SIZE = 20
TICK_SIZE = 12
ANNOTATION_SIZE = 9.5
LEGEND_SIZE = 12
GRID_COLOR = "#d9d9d9"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class TeeLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        _ensure_dir(log_path.parent)
        self._fh = log_path.open("w", encoding="utf-8")

    def log(self, msg: str = "") -> None:
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


@dataclass
class SeasonBreakdown:
    comp_name: str
    season: int
    comp_season_id: int
    n_teams: int
    total_matches: int
    regular_matches: int
    non_regular_matches: int
    non_regular_kept_matches: int
    non_regular_filtered_matches: int
    non_regular_share: float
    distinct_round_ids: int
    regular_round_ids: int
    non_regular_round_ids: int
    kept_round_ids: int
    filtered_round_ids: int
    min_team_matches: int
    max_team_matches: int
    median_team_matches: float
    first_match_date: Optional[str]
    last_match_date: Optional[str]
    notes: str


@dataclass
class RoundTypeBreakdown:
    comp_name: str
    season: int
    comp_season_id: int
    round_id: int
    match_count: int
    distinct_teams: int
    regular_matches: int
    non_regular_matches: int
    kept_matches: int
    filtered_matches: int
    whitelist_status: str
    inferred_round_type: str
    first_match_date: Optional[str]
    last_match_date: Optional[str]


def load_data_into_globals() -> None:
    bundle = load_snapshot()
    fill_globals_with_cache(bundle, update_leagues_list=False)
    utils.link_matches_to_comp_seasons()
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


def _fmt_dt(dt) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _group_matches_by_comp_and_season(matches: List[FSMatch]) -> Dict[Tuple[str, int, int], List[FSMatch]]:
    out: Dict[Tuple[str, int, int], List[FSMatch]] = {}
    for m in matches:
        comp_name = getattr(m, "comp_name", None)
        season = getattr(m, "season", None)
        comp_season_id = getattr(m, "comp_season_id", None)
        if comp_name is None or season is None or comp_season_id is None:
            continue
        out.setdefault((comp_name, int(season), int(comp_season_id)), []).append(m)
    return out


def _team_match_counts(matches: List[FSMatch]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for m in matches:
        if m.home_team is not None:
            counts[m.home_team.id] = counts.get(m.home_team.id, 0) + 1
        if m.away_team is not None:
            counts[m.away_team.id] = counts.get(m.away_team.id, 0) + 1
    return counts


def _valid_round_ids(comp_name: str, season: int) -> set[int]:
    mapping = getattr(sett, "LEAGUE_VALID_ROUND_IDS_BY_SEASON", {})
    vals = mapping.get((comp_name, season), set())
    return {int(x) for x in vals}


def build_season_breakdown(league_matches: List[FSMatch]) -> List[SeasonBreakdown]:
    rows: List[SeasonBreakdown] = []
    grouped = _group_matches_by_comp_and_season(league_matches)

    for (comp_name, season, comp_season_id), matches in sorted(
        grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])
    ):
        team_counts = _team_match_counts(matches)
        total_matches = len(matches)
        regular_matches = sum(1 for m in matches if bool(getattr(m, "regular_season", False)))
        non_regular_matches = total_matches - regular_matches

        valid_round_ids = _valid_round_ids(comp_name, season)
        non_regular_kept_matches = sum(
            1
            for m in matches
            if (not bool(getattr(m, "regular_season", False))) and getattr(m, "round_id", None) in valid_round_ids
        )
        non_regular_filtered_matches = sum(
            1
            for m in matches
            if (not bool(getattr(m, "regular_season", False))) and getattr(m, "round_id", None) not in valid_round_ids
        )
        non_regular_share = float(non_regular_matches / total_matches) if total_matches > 0 else 0.0

        round_ids = {getattr(m, "round_id", None) for m in matches if getattr(m, "round_id", None) is not None}
        regular_round_ids = {
            getattr(m, "round_id", None)
            for m in matches
            if bool(getattr(m, "regular_season", False)) and getattr(m, "round_id", None) is not None
        }
        non_regular_round_ids = {
            getattr(m, "round_id", None)
            for m in matches
            if (not bool(getattr(m, "regular_season", False))) and getattr(m, "round_id", None) is not None
        }
        kept_round_ids = {
            getattr(m, "round_id", None) for m in matches if getattr(m, "round_id", None) in valid_round_ids
        }
        filtered_round_ids = round_ids - kept_round_ids

        dts = [getattr(m, "datetime", None) for m in matches if getattr(m, "datetime", None) is not None]
        first_dt = min(dts) if dts else None
        last_dt = max(dts) if dts else None

        notes = (
            "Exact regular/non-regular split taken from persisted FSMatch.regular_season flag. "
            "Kept/filtered non-regular matches derived from LEAGUE_VALID_ROUND_IDS_BY_SEASON."
        )

        rows.append(
            SeasonBreakdown(
                comp_name=comp_name,
                season=season,
                comp_season_id=comp_season_id,
                n_teams=len(team_counts),
                total_matches=total_matches,
                regular_matches=regular_matches,
                non_regular_matches=non_regular_matches,
                non_regular_kept_matches=non_regular_kept_matches,
                non_regular_filtered_matches=non_regular_filtered_matches,
                non_regular_share=non_regular_share,
                distinct_round_ids=len(round_ids),
                regular_round_ids=len(regular_round_ids),
                non_regular_round_ids=len(non_regular_round_ids),
                kept_round_ids=len(kept_round_ids),
                filtered_round_ids=len(filtered_round_ids),
                min_team_matches=min(team_counts.values()) if team_counts else 0,
                max_team_matches=max(team_counts.values()) if team_counts else 0,
                median_team_matches=float(median(team_counts.values())) if team_counts else 0.0,
                first_match_date=_fmt_dt(first_dt),
                last_match_date=_fmt_dt(last_dt),
                notes=notes,
            )
        )

    return rows


def build_round_type_breakdown(league_matches: List[FSMatch], *, only_affected: bool) -> List[RoundTypeBreakdown]:
    rows: List[RoundTypeBreakdown] = []
    grouped = _group_matches_by_comp_and_season(league_matches)

    for (comp_name, season, comp_season_id), matches in sorted(
        grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])
    ):
        has_non_regular = any(not bool(getattr(m, "regular_season", False)) for m in matches)
        if only_affected and not has_non_regular:
            continue

        valid_round_ids = _valid_round_ids(comp_name, season)

        by_round: Dict[int, List[FSMatch]] = defaultdict(list)
        for m in matches:
            round_id = getattr(m, "round_id", None)
            if round_id is None:
                continue
            by_round[int(round_id)].append(m)

        for round_id, round_matches in sorted(by_round.items(), key=lambda x: x[0]):
            regular_count = sum(1 for m in round_matches if bool(getattr(m, "regular_season", False)))
            non_regular_count = len(round_matches) - regular_count
            kept_count = sum(1 for m in round_matches if getattr(m, "round_id", None) in valid_round_ids)
            filtered_count = len(round_matches) - kept_count

            if regular_count > 0 and non_regular_count == 0:
                inferred = "regular-season round type"
            elif non_regular_count > 0 and regular_count == 0:
                inferred = "non-regular round type"
            else:
                inferred = "mixed / check manually"

            status = "kept" if round_id in valid_round_ids else "filtered"

            teams = set()
            for m in round_matches:
                if m.home_team is not None:
                    teams.add(m.home_team.id)
                if m.away_team is not None:
                    teams.add(m.away_team.id)

            dts = [getattr(m, "datetime", None) for m in round_matches if getattr(m, "datetime", None) is not None]
            first_dt = min(dts) if dts else None
            last_dt = max(dts) if dts else None

            rows.append(
                RoundTypeBreakdown(
                    comp_name=comp_name,
                    season=season,
                    comp_season_id=comp_season_id,
                    round_id=round_id,
                    match_count=len(round_matches),
                    distinct_teams=len(teams),
                    regular_matches=regular_count,
                    non_regular_matches=non_regular_count,
                    kept_matches=kept_count,
                    filtered_matches=filtered_count,
                    whitelist_status=status,
                    inferred_round_type=inferred,
                    first_match_date=_fmt_dt(first_dt),
                    last_match_date=_fmt_dt(last_dt),
                )
            )

    return rows


def _season_rows_to_df(rows: List[SeasonBreakdown]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows])


def _round_rows_to_df(rows: List[RoundTypeBreakdown]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows])


def _aggregate_competition_totals(df_seasons: pd.DataFrame) -> pd.DataFrame:
    g = (
        df_seasons.groupby("comp_name", as_index=False)[
            [
                "total_matches",
                "regular_matches",
                "non_regular_matches",
                "non_regular_kept_matches",
                "non_regular_filtered_matches",
            ]
        ]
        .sum()
        .copy()
    )
    g["non_regular_share"] = g["non_regular_matches"] / g["total_matches"]
    g = g.sort_values(["total_matches", "comp_name"], ascending=[False, True]).reset_index(drop=True)
    return g


def _get_comp_color(comp_name: str):
    color_map = getattr(sett, "COMPS_LEAGUE_COLORS", {})
    return color_map.get(comp_name, None)


def _make_stacked_bar(df_comp: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    if df_comp.empty:
        return

    fig, ax = plt.subplots(figsize=(18, 8))

    x = list(range(len(df_comp)))
    regular = df_comp["regular_matches"].tolist()
    non_regular = df_comp["non_regular_matches"].tolist()
    labels = df_comp["comp_name"].astype(str).tolist()

    ax.bar(x, regular, label="Regular-season matches")
    ax.bar(x, non_regular, bottom=regular, label="Non-regular matches")

    ymax = max(df_comp["total_matches"]) if len(df_comp) else 0
    for i, (r, p) in enumerate(zip(regular, non_regular)):
        total = r + p
        ax.text(i, total + ymax * 0.01, str(int(total)), ha="center", va="bottom", fontsize=ANNOTATION_SIZE)
        if p > 0:
            ax.text(i, r + p / 2.0, str(int(p)), ha="center", va="center", fontsize=8)

    ax.set_title("League matches per competition (2021/2022–2024/2025): regular season vs non-regular")
    ax.set_xlabel("Competition", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of matches", fontsize=LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.grid(axis="y", linestyle="--", color=GRID_COLOR, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=LEGEND_SIZE)
    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _make_non_regular_share_plot(df_comp: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    if df_comp.empty:
        return

    df = df_comp.copy().sort_values("non_regular_share", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(15, 7))
    x = list(range(len(df)))
    shares = (100.0 * df["non_regular_share"]).tolist()
    labels = df["comp_name"].astype(str).tolist()

    ax.bar(x, shares)
    for i, val in enumerate(shares):
        ax.text(i, val + 0.6, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_title("Share of non-regular league matches by competition")
    ax.set_xlabel("Competition")
    ax.set_ylabel("Non-regular share [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _make_round_type_plot(df_rounds: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    if df_rounds.empty:
        return

    affected_keys = (
        df_rounds[["comp_name", "season", "comp_season_id"]]
        .drop_duplicates()
        .sort_values(["comp_name", "season", "comp_season_id"])
        .values.tolist()
    )

    n_panels = len(affected_keys)
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, max(4.5 * nrows, 5.5)), squeeze=False)
    axes_flat = axes.flatten()

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    for ax, (comp_name, season, comp_season_id) in zip(axes_flat, affected_keys):
        sub = df_rounds[
            (df_rounds["comp_name"] == comp_name)
            & (df_rounds["season"] == season)
            & (df_rounds["comp_season_id"] == comp_season_id)
        ].copy()
        sub = sub.sort_values(["regular_matches", "non_regular_matches", "round_id"], ascending=[False, False, True])

        labels = [str(int(x)) for x in sub["round_id"].tolist()]
        reg = sub["regular_matches"].tolist()
        nonreg = sub["non_regular_matches"].tolist()
        x = list(range(len(sub)))

        ax.bar(x, reg, label="Regular-season matches")
        ax.bar(x, nonreg, bottom=reg, label="Non-regular matches")

        max_count = max(sub["match_count"]) if len(sub) else 0
        for i, total in enumerate((sub["match_count"]).tolist()):
            ax.text(i, total + max_count * 0.03, str(int(total)), ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{comp_name} {int(season)}/{int(season) + 1}")
        ax.set_xlabel("round_id")
        ax.set_ylabel("Matches")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Round-type counts by competition season with non-regular matches", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _make_filtering_outcome_stacked_plot(df_comp: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    if df_comp.empty:
        return

    fig, ax = plt.subplots(figsize=(18, 8))

    x = list(range(len(df_comp)))
    labels = df_comp["comp_name"].astype(str).tolist()
    regular = df_comp["regular_matches"].astype(int).tolist()
    nonreg_kept = df_comp["non_regular_kept_matches"].astype(int).tolist()
    nonreg_filtered = df_comp["non_regular_filtered_matches"].astype(int).tolist()
    totals = df_comp["total_matches"].astype(int).tolist()

    first_regular = True
    first_kept = True
    first_filtered = True

    for i, comp_name in enumerate(labels):
        base_color = _get_comp_color(comp_name)

        ax.bar(
            i,
            regular[i],
            color=base_color,
            edgecolor="black",
            linewidth=0.6,
            label="Regular season matches kept" if first_regular else None,
        )
        first_regular = False

        ax.bar(
            i,
            nonreg_kept[i],
            bottom=regular[i],
            color=base_color,
            edgecolor="black",
            linewidth=0.6,
            hatch="///",
            label="Non-regular season matches kept" if first_kept else None,
        )
        first_kept = False

        ax.bar(
            i,
            nonreg_filtered[i],
            bottom=regular[i] + nonreg_kept[i],
            color="black",
            edgecolor="black",
            linewidth=0.6,
            label="Non-regular season matches filtered out" if first_filtered else None,
        )
        first_filtered = False

    ymax = max(totals) if totals else 0
    for i, total in enumerate(totals):
        ax.text(i, total + ymax * 0.01, str(int(total)), ha="center", va="bottom", fontsize=ANNOTATION_SIZE)

        filtered = nonreg_filtered[i]
        if total > 0 and filtered > 0:
            pct = 100.0 * float(filtered) / float(total)
            ax.text(
                i,
                total + ymax * 0.045,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color="black",
            )

    ax.set_title(
        "League Matches per Competition: " "Regular-Season Matches and Retained/Filtered Non-Regular Matches",
        fontsize=TITLE_SIZE,
        pad=12,
    )
    ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)
    ax.set_xlabel("Competition", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of matches", fontsize=LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.grid(axis="y", linestyle="--", color=GRID_COLOR, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=LEGEND_SIZE)
    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _write_report(log: TeeLogger, df_seasons: pd.DataFrame, df_comp: pd.DataFrame) -> None:
    log.log("=" * 96)
    log.log("REGULAR VS NON-REGULAR LEAGUE MATCH ANALYSIS")
    log.log("=" * 96)
    log.log("Goal: use persisted FSMatch.regular_season to quantify regular vs non-regular matches exactly.")
    log.log("Additional goal: quantify which non-regular matches are retained by the round whitelist and which")
    log.log("are filtered out as playoff/relegation/promotion noise.")
    log.log()

    log.log("Per-season breakdown:")
    for _, row in df_seasons.iterrows():
        log.log(
            f"[{row['comp_name']}, {int(row['season'])}] total={int(row['total_matches'])}, "
            f"regular={int(row['regular_matches'])}, non_regular={int(row['non_regular_matches'])}, "
            f"non_regular_kept={int(row['non_regular_kept_matches'])}, "
            f"non_regular_filtered={int(row['non_regular_filtered_matches'])}"
        )
    log.log()

    log.log("Aggregated competition totals:")
    for _, row in df_comp.iterrows():
        log.log(
            f"[{row['comp_name']}] total={int(row['total_matches'])}, "
            f"regular={int(row['regular_matches'])}, "
            f"non_regular_kept={int(row['non_regular_kept_matches'])}, "
            f"non_regular_filtered={int(row['non_regular_filtered_matches'])}"
        )
    log.log()


def run_analysis(out_dir: Path, *, apply_clean_filter: bool = True) -> None:
    _ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    load_data_into_globals()
    league_matches = get_league_matches(apply_clean_filter=apply_clean_filter)

    season_rows = build_season_breakdown(league_matches)
    round_rows_all = build_round_type_breakdown(league_matches, only_affected=False)
    round_rows_for_plot = build_round_type_breakdown(
        league_matches, only_affected=ONLY_AFFECTED_SEASONS_FOR_ROUND_PLOTS
    )

    df_seasons = (
        _season_rows_to_df(season_rows).sort_values(["comp_name", "season", "comp_season_id"]).reset_index(drop=True)
    )
    df_comp = _aggregate_competition_totals(df_seasons)
    df_rounds_all = (
        _round_rows_to_df(round_rows_all)
        .sort_values(["comp_name", "season", "comp_season_id", "round_id"])
        .reset_index(drop=True)
    )
    df_rounds_plot = (
        _round_rows_to_df(round_rows_for_plot)
        .sort_values(["comp_name", "season", "comp_season_id", "round_id"])
        .reset_index(drop=True)
    )

    seasons_csv = out_dir / f"postseason_breakdown_by_season_{ts}.csv"
    comp_csv = out_dir / f"postseason_breakdown_by_competition_{ts}.csv"
    rounds_csv = out_dir / f"round_type_breakdown_by_season_{ts}.csv"

    df_seasons.to_csv(seasons_csv, index=False, encoding="utf-8")
    df_comp.to_csv(comp_csv, index=False, encoding="utf-8")
    df_rounds_all.to_csv(rounds_csv, index=False, encoding="utf-8")

    stacked_png = out_dir / f"postseason_regular_vs_nonregular_stacked_{ts}.png"
    stacked_pdf = out_dir / f"postseason_regular_vs_nonregular_stacked_{ts}.pdf"
    share_png = out_dir / f"postseason_nonregular_share_{ts}.png"
    share_pdf = out_dir / f"postseason_nonregular_share_{ts}.pdf"
    rounds_png = out_dir / f"round_type_breakdown_{ts}.png"
    rounds_pdf = out_dir / f"round_type_breakdown_{ts}.pdf"
    outcome_png = out_dir / f"postseason_filtering_outcome_stacked_{ts}.png"
    outcome_pdf = out_dir / f"postseason_filtering_outcome_stacked_{ts}.pdf"

    _make_stacked_bar(df_comp, stacked_png, stacked_pdf)
    _make_non_regular_share_plot(df_comp, share_png, share_pdf)
    _make_round_type_plot(df_rounds_plot, rounds_png, rounds_pdf)
    _make_filtering_outcome_stacked_plot(df_comp, outcome_png, outcome_pdf)

    log_path = out_dir / f"postseason_analysis_report_{ts}.txt"
    logger = TeeLogger(log_path)
    try:
        logger.log(f"Saved CSV: {seasons_csv}")
        logger.log(f"Saved CSV: {comp_csv}")
        logger.log(f"Saved CSV: {rounds_csv}")
        logger.log()
        logger.log(f"Saved plot: {stacked_png}")
        logger.log(f"Saved plot: {stacked_pdf}")
        logger.log(f"Saved plot: {share_png}")
        logger.log(f"Saved plot: {share_pdf}")
        logger.log(f"Saved plot: {rounds_png}")
        logger.log(f"Saved plot: {rounds_pdf}")
        logger.log(f"Saved plot: {outcome_png}")
        logger.log(f"Saved plot: {outcome_pdf}")
        logger.log()
        _write_report(logger, df_seasons, df_comp)
    finally:
        logger.close()

    print(f"Saved log: {log_path}")


def main() -> None:
    out_dir = Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "thesis_postseason_match_breakdown"
    run_analysis(out_dir=out_dir, apply_clean_filter=APPLY_CLEAN_FILTER)


if __name__ == "__main__":
    main()
