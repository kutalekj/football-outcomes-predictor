from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import football_outcomes.utils.fs_common as utils
from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import load_snapshot
from football_outcomes.data.fs_models import FSMatch
from football_outcomes.data.fs_retrieve import fill_globals_with_cache
from football_outcomes.utils.fs_feature_utils import build_team_match_index, get_n_previous_matches, is_within_days

matplotlib.use("Agg")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# One clean thesis-friendly rule for all clipping constants.
CLIP_PERCENTILE = 99.0

# Keep this aligned with your missingness analysis style.
APPLY_CLEAN_FILTER = True

TEAM_LEVEL_STAT_SPECS: List[Tuple[str, Optional[str], Tuple[str, str]]] = [
    ("goals scored", "GOALS_NORM_COEFFICIENT", ("home_goals", "away_goals")),
    ("shots on target", "SOG_NORM_COEFFICIENT", ("home_shots_on_target", "away_shots_on_target")),
    ("shots off target", "SHOTS_OFF_G_NORM_COEFFICIENT", ("home_shots_off_target", "away_shots_off_target")),
    ("total shots", "TOTAL_SHOTS_NORM_COEFFICIENT", ("home_total_shots", "away_total_shots")),
    ("corner kicks", "CORNER_KICKS_NORM_COEFFICIENT", ("home_corners", "away_corners")),
    ("fouls committed", "FOULS_NORM_COEFFICIENT", ("home_fouls", "away_fouls")),
    ("ball possession", None, ("home_possession", "away_possession")),
    ("attacks", "ATTACKS_NORM_COEFFICIENT", ("home_attacks", "away_attacks")),
    ("dangerous attacks", "DANG_ATTACKS_NORM_COEFFICIENT", ("home_dangerous_attacks", "away_dangerous_attacks")),
    ("in-match xG", "TEAM_XG_NORM_COEFFICIENT", ("home_xg", "away_xg")),
    ("pre-match xG", "TEAM_PRE_MATCH_XG_NORM_COEFFICIENT", ("home_prematch_xg", "away_prematch_xg")),
]

TOTAL_LEVEL_STAT_SPECS: List[Tuple[str, str, Tuple[str, str]]] = [
    ("total in-match xG", "TOTAL_XG_NORM_COEFFICIENT", ("home_xg", "away_xg")),
    ("total pre-match xG", "TOTAL_PRE_MATCH_XG_NORM_COEFFICIENT", ("home_prematch_xg", "away_prematch_xg")),
]

NAME_TO_SETTING = {
    "goals scored": "GOALS_NORM_COEFFICIENT",
    "shots on target": "SOG_NORM_COEFFICIENT",
    "shots off target": "SHOTS_OFF_G_NORM_COEFFICIENT",
    "total shots": "TOTAL_SHOTS_NORM_COEFFICIENT",
    "corner kicks": "CORNER_KICKS_NORM_COEFFICIENT",
    "fouls committed": "FOULS_NORM_COEFFICIENT",
    "attacks": "ATTACKS_NORM_COEFFICIENT",
    "dangerous attacks": "DANG_ATTACKS_NORM_COEFFICIENT",
    "in-match xG": "TEAM_XG_NORM_COEFFICIENT",
    "total in-match xG": "TOTAL_XG_NORM_COEFFICIENT",
    "pre-match xG": "TEAM_PRE_MATCH_XG_NORM_COEFFICIENT",
    "total pre-match xG": "TOTAL_PRE_MATCH_XG_NORM_COEFFICIENT",
    "match load per day": "MATCH_LOAD_NORM_COEFFICIENT",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


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
class DistSummary:
    name: str
    n: int
    mean: float
    std: float
    min_: float
    p50: float
    p90: float
    p95: float
    p97: float
    p99: float
    max_: float
    current_constant: Optional[float]
    recommended_constant: float
    mean_plus_15_std: float


@dataclass
class EloSummary:
    n: int
    mean: float
    std: float
    min_: float
    p01: float
    p05: float
    p50: float
    p95: float
    p99: float
    max_: float
    recommended_min: float
    recommended_max: float


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


def _valid_stat(v: object) -> bool:
    return v is not None and v != -1 and not (isinstance(v, float) and math.isnan(v))


def _get_match_value(m: FSMatch, key: str) -> Optional[float]:
    if key == "home_goals":
        return float(m.home_goals) if m.home_goals is not None else None
    if key == "away_goals":
        return float(m.away_goals) if m.away_goals is not None else None

    v = m.stats.get(key, -1)
    return float(v) if _valid_stat(v) else None


def _summary_from_values(
    name: str, values: List[float], current_constant: Optional[float], percentile: float
) -> DistSummary:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError(f"No values for {name}")

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    return DistSummary(
        name=name,
        n=int(arr.size),
        mean=mean,
        std=std,
        min_=float(np.min(arr)),
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p97=float(np.percentile(arr, 97)),
        p99=float(np.percentile(arr, 99)),
        max_=float(np.max(arr)),
        current_constant=current_constant,
        recommended_constant=float(np.percentile(arr, percentile)),
        mean_plus_15_std=float(mean + 1.5 * std),
    )


def collect_distribution_summaries(league_matches_sorted: List[FSMatch], percentile: float) -> List[DistSummary]:
    out: List[DistSummary] = []

    for display_name, const_name, (home_key, away_key) in TEAM_LEVEL_STAT_SPECS:
        vals: List[float] = []
        for m in league_matches_sorted:
            hv = _get_match_value(m, home_key)
            av = _get_match_value(m, away_key)
            if hv is not None:
                vals.append(hv)
            if av is not None:
                vals.append(av)

        current_constant = getattr(sett, const_name) if const_name else None
        out.append(_summary_from_values(display_name, vals, current_constant, percentile))

    for display_name, const_name, (home_key, away_key) in TOTAL_LEVEL_STAT_SPECS:
        vals: List[float] = []
        for m in league_matches_sorted:
            hv = _get_match_value(m, home_key)
            av = _get_match_value(m, away_key)
            if hv is not None and av is not None:
                vals.append(hv + av)
        current_constant = getattr(sett, const_name)
        out.append(_summary_from_values(display_name, vals, current_constant, percentile))

    # Match load: reconstruct the raw count/day ratio instead of using already-normalized features.
    index = build_team_match_index(league_matches_sorted)
    match_load_vals: List[float] = []
    for m in league_matches_sorted:
        for team in (m.home_team, m.away_team):
            if team is None:
                continue
            for days in (10, 25):
                prev_matches = get_n_previous_matches(index, team.id, m, n=200)
                cnt = 0
                for pm in reversed(prev_matches):
                    if is_within_days(m, pm, days):
                        cnt += 1
                    else:
                        break
                match_load_vals.append(float(cnt) / float(days))

    out.append(
        _summary_from_values(
            "match load per day",
            match_load_vals,
            getattr(sett, "MATCH_LOAD_NORM_COEFFICIENT"),
            percentile,
        )
    )

    return out


def collect_elo_summary(league_matches_sorted: List[FSMatch]) -> Optional[EloSummary]:
    vals: List[float] = []
    for m in league_matches_sorted:
        if m.home_elo_after_match_raw is not None:
            vals.append(float(m.home_elo_after_match_raw))
        if m.away_elo_after_match_raw is not None:
            vals.append(float(m.away_elo_after_match_raw))

    if not vals:
        return None

    arr = np.asarray(vals, dtype=float)
    return EloSummary(
        n=int(arr.size),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=0)),
        min_=float(np.min(arr)),
        p01=float(np.percentile(arr, 1)),
        p05=float(np.percentile(arr, 5)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        max_=float(np.max(arr)),
        recommended_min=float(np.percentile(arr, 1)),
        recommended_max=float(np.percentile(arr, 99)),
    )


def build_summary_table(summaries: List[DistSummary]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        rows.append(
            {
                "stat_name": s.name,
                "setting_name": NAME_TO_SETTING.get(s.name),
                "n": s.n,
                "mean": s.mean,
                "std": s.std,
                "min": s.min_,
                "p50": s.p50,
                "p90": s.p90,
                "p95": s.p95,
                "p97": s.p97,
                "p99": s.p99,
                "max": s.max_,
                "current_constant": s.current_constant,
                "mean_plus_1_5_std": s.mean_plus_15_std,
                "recommended_constant": s.recommended_constant,
            }
        )
    return pd.DataFrame(rows)


def write_recommendations(
    log: TeeLogger, summaries: List[DistSummary], elo_summary: Optional[EloSummary], percentile: float
) -> None:
    log.log("=" * 88)
    log.log("RAW MATCH STATISTICS NORMALIZATION ANALYSIS")
    log.log("=" * 88)
    log.log(f"Chosen global rule for clip constants: empirical p{percentile:.0f}")
    log.log("Reason: normalization constants are upper clipping scales in value/constant normalization,")
    log.log("and with a dataset of more than 30,000 league matches, p99 remains robust while preserving more")
    log.log("legitimate upper-tail variation than p95. The same less-strict philosophy is used for Elo by")
    log.log("recommending p01/p99 bounds instead of p02/p98.")
    log.log()

    for s in summaries:
        log.log(f"[{s.name}]")
        log.log(
            f"n={s.n}, mean={s.mean:.4f}, std={s.std:.4f}, min={s.min_:.4f}, "
            f"p50={s.p50:.4f}, p90={s.p90:.4f}, p95={s.p95:.4f}, p97={s.p97:.4f}, p99={s.p99:.4f}, max={s.max_:.4f}"
        )
        if s.current_constant is not None:
            log.log(
                f"current={s.current_constant:.4f}, recommended_p{percentile:.0f}={s.recommended_constant:.4f}, "
                f"mean+1.5std={s.mean_plus_15_std:.4f}"
            )
        else:
            log.log(f"recommended_p{percentile:.0f}={s.recommended_constant:.4f}")
        log.log()

    log.log("Suggested fs_settings.py replacements:")
    for s in summaries:
        if s.name == "ball possession":
            continue
        const_name = NAME_TO_SETTING[s.name]
        log.log(f"{const_name} = {s.recommended_constant:.4f}")

    if elo_summary is not None:
        log.log()
        log.log("Suggested Elo normalization bounds (keeping Elo dynamics constants unchanged):")
        log.log(
            f"normalize_elo(raw_elo, min_elo={elo_summary.recommended_min:.1f}, "
            f"max_elo={elo_summary.recommended_max:.1f})"
        )
        log.log(
            f"Elo stats: n={elo_summary.n}, mean={elo_summary.mean:.2f}, std={elo_summary.std:.2f}, "
            f"min={elo_summary.min_:.2f}, p01={elo_summary.p01:.2f}, p05={elo_summary.p05:.2f}, "
            f"p50={elo_summary.p50:.2f}, p95={elo_summary.p95:.2f}, p99={elo_summary.p99:.2f}, "
            f"max={elo_summary.max_:.2f}"
        )


def make_distribution_plot(league_matches_sorted: List[FSMatch], out_png: Path, out_pdf: Path) -> None:
    records = []
    short_labels = {
        "goals scored": "Goals",
        "shots on target": "Shots on target",
        "shots off target": "Shots off target",
        "total shots": "Total shots",
        "corner kicks": "Corners",
        "fouls committed": "Fouls",
        "ball possession": "Possession [%]",
        "attacks": "Attacks",
        "dangerous attacks": "Dangerous attacks",
        "in-match xG": "xG",
        "pre-match xG": "Pre-match xG",
    }

    for display_name, _, (home_key, away_key) in TEAM_LEVEL_STAT_SPECS:
        for m in league_matches_sorted:
            hv = _get_match_value(m, home_key)
            av = _get_match_value(m, away_key)
            if hv is not None:
                records.append({"stat": short_labels[display_name], "value": hv})
            if av is not None:
                records.append({"stat": short_labels[display_name], "value": av})

    df = pd.DataFrame(records)
    if df.empty:
        print("Skipping raw match statistics distribution plot because there are no values.")
        return

    compact_order = [
        "Goals",
        "Shots on target",
        "Shots off target",
        "Total shots",
        "Corners",
        "Fouls",
        "xG",
        "Pre-match xG",
    ]

    large_order = [
        "Possession [%]",
        "Dangerous attacks",
        "Attacks",
    ]

    compact_df = df[df["stat"].isin(compact_order)].copy()
    large_df = df[df["stat"].isin(large_order)].copy()

    compact_df["stat"] = pd.Categorical(compact_df["stat"], categories=compact_order, ordered=True)
    large_df["stat"] = pd.Categorical(large_df["stat"], categories=large_order, ordered=True)

    compact_df = compact_df.sort_values("stat")
    large_df = large_df.sort_values("stat")

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(13.5, 9.5),
        gridspec_kw={"height_ratios": [3.2, 1.5]},
        sharex=False,
    )

    sns.boxenplot(
        data=compact_df,
        y="stat",
        x="value",
        order=compact_order,
        orient="h",
        linewidth=0.8,
        k_depth="trustworthy",
        showfliers=False,
        ax=ax1,
    )

    ax1.set_title("Distributions of raw match statistics across all league matches")
    ax1.set_xlabel("Raw value")
    ax1.set_ylabel("")
    ax1.grid(axis="x", linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    sns.boxenplot(
        data=large_df,
        y="stat",
        x="value",
        order=large_order,
        orient="h",
        linewidth=0.8,
        k_depth="trustworthy",
        showfliers=False,
        ax=ax2,
    )

    ax2.set_xlabel("Raw value")
    ax2.set_ylabel("")
    ax2.grid(axis="x", linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax1.set_facecolor("#fafafa")
    ax2.set_facecolor("#fafafa")

    if not compact_df.empty:
        compact_xmax = float(np.nanpercentile(compact_df["value"], 99.5))
        ax1.set_xlim(left=0, right=max(1.0, compact_xmax * 1.03))

    if not large_df.empty:
        large_xmax = float(np.nanpercentile(large_df["value"], 99.5))
        ax2.set_xlim(left=0, right=max(1.0, large_xmax * 1.03))

    fig.tight_layout(h_pad=2.0)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    out_dir: Path,
    summaries: List[DistSummary],
    elo_summary: Optional[EloSummary],
    league_matches_sorted: List[FSMatch],
    logger: TeeLogger,
    timestamp: str,
) -> None:
    summary_df = build_summary_table(summaries)
    summary_csv = out_dir / f"normalization_constants_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.log(f"Saved table: {summary_csv}")

    if elo_summary is not None:
        elo_df = pd.DataFrame(
            [
                {
                    "n": elo_summary.n,
                    "mean": elo_summary.mean,
                    "std": elo_summary.std,
                    "min": elo_summary.min_,
                    "p01": elo_summary.p01,
                    "p05": elo_summary.p05,
                    "p50": elo_summary.p50,
                    "p95": elo_summary.p95,
                    "p99": elo_summary.p99,
                    "max": elo_summary.max_,
                    "recommended_min": elo_summary.recommended_min,
                    "recommended_max": elo_summary.recommended_max,
                }
            ]
        )
        elo_csv = out_dir / f"elo_normalization_summary_{timestamp}.csv"
        elo_df.to_csv(elo_csv, index=False)
        logger.log(f"Saved table: {elo_csv}")

    plot_png = out_dir / f"raw_match_stat_distributions_{timestamp}.png"
    plot_pdf = out_dir / f"raw_match_stat_distributions_{timestamp}.pdf"
    make_distribution_plot(league_matches_sorted, plot_png, plot_pdf)
    logger.log(f"Saved plot: {plot_png}")
    logger.log(f"Saved plot: {plot_pdf}")


def run_analysis(*, out_dir: Path, apply_clean_filter: bool, percentile: float) -> None:
    load_data_into_globals()
    league_matches_sorted = get_league_matches(apply_clean_filter=apply_clean_filter)

    if not league_matches_sorted:
        print("No league matches available for normalization analysis.")
        return

    _ensure_dir(out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"analysis_raw_match_stats_normalization_{timestamp}.log"

    logger = TeeLogger(log_path)
    try:
        logger.log(f"Output directory: {out_dir}")
        logger.log(f"League matches analyzed: {len(league_matches_sorted)}")
        logger.log(f"Apply clean filter: {apply_clean_filter}")
        logger.log(f"Season range: {sett.FIRST_SEASON}..{sett.LAST_SEASON - 1}")
        logger.log(f"Chosen percentile rule: p{percentile:.0f}")
        logger.log()

        summaries = collect_distribution_summaries(league_matches_sorted, percentile=percentile)
        elo_summary = collect_elo_summary(league_matches_sorted)
        write_recommendations(logger, summaries, elo_summary, percentile=percentile)
        save_outputs(out_dir, summaries, elo_summary, league_matches_sorted, logger, timestamp)
    finally:
        logger.close()

    print(f"Saved log: {log_path}")


def main() -> None:
    out_dir = Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "thesis_match_stats_normalization"
    run_analysis(
        out_dir=out_dir,
        apply_clean_filter=APPLY_CLEAN_FILTER,
        percentile=CLIP_PERCENTILE,
    )


if __name__ == "__main__":
    main()
