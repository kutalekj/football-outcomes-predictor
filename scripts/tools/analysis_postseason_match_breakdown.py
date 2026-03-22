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
    non_regular_share: float
    distinct_round_ids: int
    regular_round_ids: int
    non_regular_round_ids: int
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
    inferred_round_type: str
    first_match_date: Optional[str]
    last_match_date: Optional[str]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------


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


def _fmt_dt(dt) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


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
            if not bool(getattr(m, "regular_season", False)) and getattr(m, "round_id", None) is not None
        }

        dts = [getattr(m, "datetime", None) for m in matches if getattr(m, "datetime", None) is not None]
        first_dt = min(dts) if dts else None
        last_dt = max(dts) if dts else None

        notes = (
            "Exact regular/non-regular split taken from persisted FSMatch.regular_season flag. "
            "Round-type counts are grouped by round_id."
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
                non_regular_share=non_regular_share,
                distinct_round_ids=len(round_ids),
                regular_round_ids=len(regular_round_ids),
                non_regular_round_ids=len(non_regular_round_ids),
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

        by_round: Dict[int, List[FSMatch]] = defaultdict(list)
        for m in matches:
            round_id = getattr(m, "round_id", None)
            if round_id is None:
                continue
            by_round[int(round_id)].append(m)

        for round_id, round_matches in sorted(by_round.items(), key=lambda x: x[0]):
            regular_count = sum(1 for m in round_matches if bool(getattr(m, "regular_season", False)))
            non_regular_count = len(round_matches) - regular_count

            if regular_count > 0 and non_regular_count == 0:
                inferred = "regular-season round type"
            elif non_regular_count > 0 and regular_count == 0:
                inferred = "non-regular round type"
            else:
                inferred = "mixed / check manually"

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
                    inferred_round_type=inferred,
                    first_match_date=_fmt_dt(first_dt),
                    last_match_date=_fmt_dt(last_dt),
                )
            )

    return rows


# -----------------------------------------------------------------------------
# DataFrames / plots
# -----------------------------------------------------------------------------


def _season_rows_to_df(rows: List[SeasonBreakdown]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows])


def _round_rows_to_df(rows: List[RoundTypeBreakdown]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows])


def _aggregate_competition_totals(df_seasons: pd.DataFrame) -> pd.DataFrame:
    g = (
        df_seasons.groupby("comp_name", as_index=False)[["total_matches", "regular_matches", "non_regular_matches"]]
        .sum()
        .copy()
    )
    g["non_regular_share"] = g["non_regular_matches"] / g["total_matches"]

    comp_order = [name for name in sett.COMPS_LEAGUE if name in set(g["comp_name"])]
    g["comp_name"] = pd.Categorical(g["comp_name"], categories=comp_order, ordered=True)
    g = g.sort_values(["comp_name"]).reset_index(drop=True)
    return g


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
        ax.text(i, total + ymax * 0.01, str(int(total)), ha="center", va="bottom", fontsize=9)
        if p > 0:
            ax.text(i, r + p / 2.0, str(int(p)), ha="center", va="center", fontsize=8)

    ax.set_title("League matches per competition (2021/2022–2024/2025): regular season vs non-regular")
    ax.set_xlabel("Competition")
    ax.set_ylabel("Number of matches")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
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
    ax.set_xticklabels(labels, rotation=45, ha="right")
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

        for i, total in enumerate((sub["match_count"]).tolist()):
            ax.text(i, total + max(sub["match_count"]) * 0.03, str(int(total)), ha="center", va="bottom", fontsize=8)

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


# -----------------------------------------------------------------------------
# Reporting / run
# -----------------------------------------------------------------------------


def _write_report(log: TeeLogger, df_seasons: pd.DataFrame, df_comp: pd.DataFrame, df_rounds: pd.DataFrame) -> None:
    log.log("=" * 96)
    log.log("REGULAR VS NON-REGULAR LEAGUE MATCH ANALYSIS")
    log.log("=" * 96)
    log.log("Goal: use persisted FSMatch.regular_season to quantify regular vs non-regular matches exactly.")
    log.log("Additional goal: summarize round_id counts for affected competition seasons to distinguish split")
    log.log("formats from seasons where only a small playoff block appears at the end.")
    log.log()

    log.log("Per-season breakdown:")
    for _, row in df_seasons.iterrows():
        log.log(
            f"[{row['comp_name']}, {int(row['season'])}] total={int(row['total_matches'])}, "
            f"regular={int(row['regular_matches'])}, non_regular={int(row['non_regular_matches'])} "
            f"({100.0 * float(row['non_regular_share']):.2f}%), teams={int(row['n_teams'])}, "
            f"round_ids={int(row['distinct_round_ids'])}, regular_round_ids={int(row['regular_round_ids'])}, "
            f"non_regular_round_ids={int(row['non_regular_round_ids'])}, team_matches=min/med/max="
            f"{int(row['min_team_matches'])}/{float(row['median_team_matches']):.1f}/{int(row['max_team_matches'])}"
        )
        log.log(f"  notes: {row['notes']}")
        log.log()

    log.log("Aggregated competition totals:")
    for _, row in df_comp.iterrows():
        log.log(
            f"{row['comp_name']}: total={int(row['total_matches'])}, regular={int(row['regular_matches'])}, "
            f"non_regular={int(row['non_regular_matches'])} ({100.0 * float(row['non_regular_share']):.2f}%)"
        )
    log.log()

    if not df_rounds.empty:
        log.log("Round-type counts for affected competition seasons:")
        for _, row in df_rounds.sort_values(["comp_name", "season", "round_id"]).iterrows():
            log.log(
                f"[{row['comp_name']}, {int(row['season'])}, round_id={int(row['round_id'])}] "
                f"matches={int(row['match_count'])}, teams={int(row['distinct_teams'])}, "
                f"regular={int(row['regular_matches'])}, non_regular={int(row['non_regular_matches'])}, "
                f"type={row['inferred_round_type']}"
            )


def run_analysis(*, out_dir: Path, apply_clean_filter: bool) -> None:
    load_data_into_globals()
    league_matches = get_league_matches(apply_clean_filter=apply_clean_filter)

    if not league_matches:
        print("No league matches available for regular/non-regular analysis.")
        return

    if not any(hasattr(m, "regular_season") for m in league_matches):
        raise RuntimeError(
            "Loaded snapshot matches do not contain regular_season. Rebuild / reload the updated snapshot first."
        )

    _ensure_dir(out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"analysis_postseason_match_breakdown_{timestamp}.log"
    logger = TeeLogger(log_path)

    try:
        logger.log(f"Output directory: {out_dir}")
        logger.log(f"League matches analyzed: {len(league_matches)}")
        logger.log(f"Apply clean filter: {apply_clean_filter}")
        logger.log(f"Season range: {sett.FIRST_SEASON}..{sett.LAST_SEASON - 1}")
        logger.log()

        season_rows = build_season_breakdown(league_matches)
        round_rows = build_round_type_breakdown(
            league_matches,
            only_affected=ONLY_AFFECTED_SEASONS_FOR_ROUND_PLOTS,
        )

        df_seasons = _season_rows_to_df(season_rows)
        df_comp = _aggregate_competition_totals(df_seasons)
        df_rounds = _round_rows_to_df(round_rows)

        season_csv = out_dir / f"postseason_breakdown_by_season_{timestamp}.csv"
        comp_csv = out_dir / f"postseason_breakdown_by_competition_{timestamp}.csv"
        rounds_csv = out_dir / f"round_type_breakdown_by_season_{timestamp}.csv"
        df_seasons.to_csv(season_csv, index=False)
        df_comp.to_csv(comp_csv, index=False)
        df_rounds.to_csv(rounds_csv, index=False)

        logger.log(f"Saved table: {season_csv}")
        logger.log(f"Saved table: {comp_csv}")
        logger.log(f"Saved table: {rounds_csv}")
        logger.log()

        stacked_png = out_dir / f"postseason_breakdown_stacked_{timestamp}.png"
        stacked_pdf = out_dir / f"postseason_breakdown_stacked_{timestamp}.pdf"
        share_png = out_dir / f"postseason_breakdown_share_{timestamp}.png"
        share_pdf = out_dir / f"postseason_breakdown_share_{timestamp}.pdf"
        rounds_png = out_dir / f"round_type_breakdown_{timestamp}.png"
        rounds_pdf = out_dir / f"round_type_breakdown_{timestamp}.pdf"

        _make_stacked_bar(df_comp, stacked_png, stacked_pdf)
        _make_non_regular_share_plot(df_comp, share_png, share_pdf)
        _make_round_type_plot(df_rounds, rounds_png, rounds_pdf)

        logger.log(f"Saved plot: {stacked_png}")
        logger.log(f"Saved plot: {stacked_pdf}")
        logger.log(f"Saved plot: {share_png}")
        logger.log(f"Saved plot: {share_pdf}")
        if not df_rounds.empty:
            logger.log(f"Saved plot: {rounds_png}")
            logger.log(f"Saved plot: {rounds_pdf}")
        logger.log()

        _write_report(logger, df_seasons, df_comp, df_rounds)
    finally:
        logger.close()

    print(f"Saved log: {log_path}")


def main() -> None:
    out_dir = Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "thesis_postseason_match_breakdown"
    run_analysis(out_dir=out_dir, apply_clean_filter=APPLY_CLEAN_FILTER)


if __name__ == "__main__":
    main()
