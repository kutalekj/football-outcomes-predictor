from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

import football_outcomes.config.fs_settings as sett
from football_outcomes.application.snapshot_selection import resolve_snapshot_path
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.snapshots import try_load_snapshot
from football_outcomes.data.state import apply_bundle_to_global
from football_outcomes.experiments.bookmaker import (
    DEFAULT_THRESHOLDS,
    TemporalThresholdConfig,
    build_betting_opportunities,
    build_competition_summary,
    build_coverage_curve,
    build_ev_coverage_curve,
    build_fixed_edge_bins,
    build_margin_band_summary,
    build_odds_band_summary,
    build_probability_method_betting_summary,
    build_probability_method_metrics,
    build_quantile_edge_bins,
    build_quantile_ev_bins,
    build_season_summary,
    build_temporal_threshold_summary,
    build_threshold_summary,
    build_walk_forward_probability_rows,
    select_best_opportunity_per_match,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HISTORICAL_OOS_PATH = Path(sett.DATA_DIR) / "tensorboard_logs" / "selected_mlp_binary_u25" / "oos_predictions.csv"


def _safe_float_odds(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result <= 1.0:
        return None
    return result


def _prediction_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    raise KeyError(f"Prediction CSV is missing all of these columns: {names}")


def load_match_odds_by_id(snapshot_path: Path) -> dict[int, dict]:
    cache = try_load_snapshot(snapshot_path)
    if cache is None:
        raise RuntimeError(f"Could not load snapshot: {snapshot_path}")

    apply_bundle_to_global(cache)
    global_state = Global.get_instance()

    output: dict[int, dict] = {}
    for match in global_state.all_matches:
        odds = getattr(match, "odds", None) or {}
        under = _safe_float_odds(odds.get("under25"))
        over = _safe_float_odds(odds.get("over25"))
        if under is None or over is None:
            continue

        raw_under = 1.0 / under
        raw_over = 1.0 / over
        total = raw_under + raw_over
        if total <= 0.0:
            continue

        match_datetime = getattr(match, "datetime", None)
        output[int(match.id)] = {
            "match_datetime": (
                match_datetime.isoformat()
                if hasattr(match_datetime, "isoformat")
                else str(match_datetime) if match_datetime is not None else ""
            ),
            "under25_odds": under,
            "over25_odds": over,
            "book_raw_p_under": raw_under,
            "book_raw_p_over": raw_over,
            "book_fair_p_under": raw_under / total,
            "book_fair_p_over": raw_over / total,
            "book_margin": total - 1.0,
        }

    return output


def read_joined_rows(predictions_path: Path, snapshot_path: Path) -> list[dict]:
    odds_by_id = load_match_odds_by_id(snapshot_path)
    joined: list[dict] = []
    missing_odds = 0

    with predictions_path.open("r", encoding="utf-8", newline="") as file:
        for raw in csv.DictReader(file):
            match_id = int(float(_prediction_value(raw, "match_id")))
            odds = odds_by_id.get(match_id)
            if odds is None:
                missing_odds += 1
                continue

            y_true_under = int(float(_prediction_value(raw, "y_true")))
            p_model_under = float(
                _prediction_value(
                    raw,
                    "y_prob_under25",
                    "probability_under_2_5",
                )
            )

            joined.append(
                {
                    "round_idx": int(float(_prediction_value(raw, "round_idx", "round_index"))),
                    "match_id": match_id,
                    "match_datetime": odds["match_datetime"],
                    "season": int(float(_prediction_value(raw, "season"))),
                    "competition": _prediction_value(raw, "competition"),
                    "y_true_under25": y_true_under,
                    "y_true_over25": 1 - y_true_under,
                    "p_model_under": p_model_under,
                    "p_model_over": 1.0 - p_model_under,
                    **odds,
                }
            )

    print(f"[utility] joined predictions with odds: {len(joined)}")
    print(f"[utility] predictions without valid U/O 2.5 odds: {missing_odds}")
    return joined


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_plot_values(rows: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            values.append(np.nan)
        else:
            values.append(float(value))
    return values


def _bounded_sqrt_widths(counts: list[int]) -> np.ndarray:
    values = np.sqrt(np.asarray(counts, dtype=np.float64))
    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    spread = float(np.ptp(values))
    if spread <= 1e-12:
        return np.full(len(values), 0.70, dtype=np.float64)
    scaled = (values - float(np.min(values))) / spread
    return 0.45 + 0.45 * scaled


def _annotate_bar_count(ax: plt.Axes, bar: object, count: int, value: float) -> None:
    y_min, y_max = ax.get_ylim()
    y_span = max(y_max - y_min, 1e-9)
    x_pos = bar.get_x() + bar.get_width() / 2
    y_pos = bar.get_height()

    if value >= 0:
        text_y = min(y_pos + 0.018 * y_span, y_max - 0.03 * y_span)
        va = "bottom"
    else:
        text_y = max(y_pos - 0.025 * y_span, y_min + 0.02 * y_span)
        va = "top"

    ax.text(
        x_pos,
        text_y,
        f"n={count}",
        ha="center",
        va=va,
        fontsize=7,
        clip_on=False,
    )


def _annotate_bar_count_with_win_share(
    ax: plt.Axes,
    bar: object,
    count: int,
    value: float,
    win_share_pct: float,
) -> None:
    y_min, y_max = ax.get_ylim()
    y_span = max(y_max - y_min, 1e-9)
    x_pos = bar.get_x() + bar.get_width() / 2
    y_pos = bar.get_height()

    if value >= 0:
        text_y = min(y_pos + 0.018 * y_span, y_max - 0.035 * y_span)
        box_alignment = (0.5, 0.0)  # bottom-aligned
    else:
        text_y = max(y_pos - 0.028 * y_span, y_min + 0.025 * y_span)
        box_alignment = (0.5, 1.0)  # top-aligned

    packed = HPacker(
        children=[
            TextArea(
                f"n={count}",
                textprops=dict(color="black", fontsize=8),
            ),
            TextArea(
                f" ({win_share_pct:.0f}%)",
                textprops=dict(color=WIN_SHARE_COLOR, fontsize=8),
            ),
        ],
        align="center",
        pad=0,
        sep=0,
    )

    annotation = AnnotationBbox(
        packed,
        (x_pos, text_y),
        xycoords="data",
        box_alignment=box_alignment,
        frameon=False,
        pad=0.0,
    )
    ax.add_artist(annotation)


def _bounded_marker_areas(
    counts: list[int],
    *,
    minimum: float = 35.0,
    maximum: float = 150.0,
) -> np.ndarray:
    values = np.sqrt(np.asarray(counts, dtype=np.float64))
    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    spread = float(np.ptp(values))
    if spread <= 1e-12:
        return np.full(len(values), (minimum + maximum) / 2.0, dtype=np.float64)
    scaled = (values - float(np.min(values))) / spread
    return minimum + (maximum - minimum) * scaled


def _save_figure(fig: plt.Figure, figure_root: Path, filename: str) -> None:
    fig.savefig(figure_root / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(figure_root / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


WIN_SHARE_COLOR = "#2ca02c"


def _extract_win_share_percent(row: dict) -> float:
    hit_rate = row.get("hit_rate")
    if hit_rate not in (None, ""):
        return 100.0 * float(hit_rate)

    wins = float(row.get("wins", 0.0))
    num_bets = float(row.get("num_bets", 0.0))
    if num_bets <= 0.0:
        return float("nan")
    return 100.0 * wins / num_bets


def plot_edge_selectivity_bar_only(
    edge_bins: list[dict],
    figure_root: Path,
    *,
    total_match_count: int,
    positive_candidate_count: int,
) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.6, 6.3))

    labels = [row["edge_bin"] for row in edge_bins]
    roi = _finite_plot_values(edge_bins, "roi")
    counts = [int(row["num_bets"]) for row in edge_bins]
    win_share_pct = [_extract_win_share_percent(row) for row in edge_bins]

    x = np.arange(len(labels))
    widths = _bounded_sqrt_widths(counts)

    bars = ax.bar(
        x,
        roi,
        width=widths,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_title("Realized ROI by Positive Best-Side Edge Bin", pad=20)
    ax.text(
        0.5,
        1.005,
        (
            f"Non-negative best-side edge: {positive_candidate_count:,} of "
            f"{total_match_count:,} matches "
            f"({positive_candidate_count / max(1, total_match_count):.1%}). "
            "Bar width ∝ √(number of bets)."
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_xlabel("Best-side model probability − corresponding bookmaker break-even probability")
    ax.set_ylabel("ROI per unit stake")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    # Add a little headroom/footroom so annotations do not touch borders.
    finite_roi = [value for value in roi if np.isfinite(value)]
    if finite_roi:
        y_min = min(min(finite_roi), 0.0)
        y_max = max(max(finite_roi), 0.0)
        y_span = max(y_max - y_min, 0.05)
        ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.14 * y_span)

    # n=... annotations
    for bar, value, count, pct in zip(bars, roi, counts, win_share_pct):
        if np.isfinite(value) and np.isfinite(pct):
            _annotate_bar_count_with_win_share(ax, bar, count, value, pct)

    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.08, right=0.98)
    _save_figure(fig, figure_root, "bookmaker_edge_selectivity_bar_only")


def plot_edge_selectivity(
    edge_bins: list[dict],
    coverage: list[dict],
    figure_root: Path,
    *,
    total_match_count: int,
    positive_candidate_count: int,
) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14.6, 9.4),
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.42},
    )

    labels = [row["edge_bin"] for row in edge_bins]
    roi = _finite_plot_values(edge_bins, "roi")
    counts = [int(row["num_bets"]) for row in edge_bins]
    x = np.arange(len(labels))
    widths = _bounded_sqrt_widths(counts)

    bars = axes[0].bar(
        x,
        roi,
        width=widths,
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0].set_title("Realized ROI by Positive Best-Side Edge Bin", pad=26)
    axes[0].text(
        0.5,
        1.015,
        (
            f"Non-negative best-side edge: {positive_candidate_count:,} of "
            f"{total_match_count:,} matches "
            f"({positive_candidate_count / max(1, total_match_count):.1%}). "
            "Bar width ∝ √(number of bets)."
        ),
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    axes[0].set_xlabel("Best-side model probability − corresponding bookmaker break-even probability")
    axes[0].set_ylabel("ROI per unit stake")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].grid(axis="y", linestyle=":", alpha=0.45)
    for bar, value, count in zip(bars, roi, counts):
        if np.isfinite(value):
            _annotate_bar_count(axes[0], bar, count, value)

    coverage_x = [100.0 * float(row["actual_positive_candidate_coverage"]) for row in coverage]
    coverage_roi = _finite_plot_values(coverage, "roi")
    coverage_counts = [int(row["num_bets"]) for row in coverage]
    order = np.argsort(coverage_x)
    coverage_x = [coverage_x[i] for i in order]
    coverage_roi = [coverage_roi[i] for i in order]
    coverage_counts = [coverage_counts[i] for i in order]
    marker_areas = _bounded_marker_areas(coverage_counts, minimum=35, maximum=120)

    axes[1].plot(coverage_x, coverage_roi, linewidth=2.0)
    axes[1].scatter(coverage_x, coverage_roi, s=marker_areas, zorder=3)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_title("ROI as Betting Becomes More Selective", pad=24)
    axes[1].text(
        0.5,
        1.015,
        (
            f"Rank only the {positive_candidate_count:,} non-negative-edge candidates; "
            "retain the strongest fraction by edge."
        ),
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    axes[1].set_xlabel("Share of non-negative-edge candidates retained (%)")
    axes[1].set_ylabel("ROI per unit stake")
    axes[1].grid(axis="y", linestyle=":", alpha=0.45)
    for x_value, y_value, count in zip(coverage_x, coverage_roi, coverage_counts):
        if np.isfinite(y_value):
            axes[1].annotate(
                f"n={count}",
                (x_value, y_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    fig.subplots_adjust(top=0.96, bottom=0.08)
    _save_figure(fig, figure_root, "bookmaker_edge_selectivity")


def plot_side_thresholds(
    threshold_rows: list[dict],
    figure_root: Path,
    *,
    total_match_count: int,
) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.2, 9.0),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    annotation_offsets = {"all": 8, "under": 18, "over": -17}

    zero_all = next(row for row in threshold_rows if row["side"] == "all" and abs(float(row["threshold"])) < 1e-12)
    non_negative_count = int(zero_all["num_bets"])

    for side in ("all", "under", "over"):
        rows = sorted(
            [row for row in threshold_rows if row["side"] == side],
            key=lambda row: float(row["threshold"]),
        )
        thresholds = [float(row["threshold"]) for row in rows]
        values = _finite_plot_values(rows, "roi")
        counts = [int(row["num_bets"]) for row in rows]
        marker_areas = _bounded_marker_areas(counts, minimum=38, maximum=145)

        line = axes[0].plot(thresholds, values, linewidth=2.0, label=side.title())[0]
        axes[0].scatter(
            thresholds,
            values,
            s=marker_areas,
            color=line.get_color(),
            zorder=3,
        )
        count_line = axes[1].plot(
            thresholds,
            counts,
            linewidth=2.0,
            label=side.title(),
        )[0]
        axes[1].scatter(
            thresholds,
            counts,
            s=marker_areas,
            color=count_line.get_color(),
            zorder=3,
        )

        for x_value, y_value, count in zip(thresholds, values, counts):
            if np.isfinite(y_value):
                axes[0].annotate(
                    f"n={count}",
                    (x_value, y_value),
                    xytext=(0, annotation_offsets[side]),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
        for x_value, count in zip(thresholds, counts):
            axes[1].annotate(
                f"n={count}",
                (x_value, count),
                xytext=(0, 8 if side != "over" else -14),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    fig.suptitle(
        "Under vs Over ROI by Required Best-Side Edge Threshold",
        fontsize=16,
        y=0.985,
    )
    fig.text(
        0.5,
        0.953,
        (
            "Bet when max(edge_under, edge_over) ≥ t and use the side with the larger "
            f"edge. At t=0, {non_negative_count:,}/{total_match_count:,} matches "
            f"({non_negative_count / max(1, total_match_count):.1%}) are bettable. "
            "Marker area is scaled by bet count."
        ),
        ha="center",
        fontsize=9,
    )

    axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("ROI per unit stake")
    axes[0].grid(axis="y", linestyle=":", alpha=0.45)
    axes[0].legend()

    axes[1].set_xlabel("Required minimum best-side probability edge t")
    axes[1].set_ylabel("Number of selected bets")
    axes[1].grid(axis="y", linestyle=":", alpha=0.45)
    axes[1].legend()

    fig.subplots_adjust(top=0.91, bottom=0.08)
    _save_figure(fig, figure_root, "bookmaker_under_over_thresholds")


def plot_odds_bands(rows: list[dict], figure_root: Path) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    plotted_thresholds = (0.05, 0.07, 0.10)
    all_rows = [row for row in rows if row.get("side", "all") == "all"]
    labels: list[str] = []
    for row in all_rows:
        label = str(row["odds_band"])
        if label not in labels:
            labels.append(label)

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12.0, 5.8))

    for offset_index, threshold in enumerate(plotted_thresholds):
        threshold_rows = [row for row in all_rows if abs(float(row["threshold"]) - threshold) < 1e-12]
        by_label = {str(row["odds_band"]): row for row in threshold_rows}
        values = [float(by_label[label]["roi"]) if by_label[label]["roi"] is not None else np.nan for label in labels]
        counts = [int(by_label[label]["num_bets"]) for label in labels]
        offset = (offset_index - 1) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=f"edge ≥ {threshold:.0%}",
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, count, value in zip(bars, counts, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{count}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4 if value >= 0 else -12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_title("ROI by Decimal-Odds Band")
    ax.set_xlabel("Selected bet odds")
    ax.set_ylabel("ROI per unit stake")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(title="Required best-side edge")
    fig.tight_layout()
    fig.savefig(figure_root / "bookmaker_odds_band_roi.pdf", bbox_inches="tight")
    fig.savefig(
        figure_root / "bookmaker_odds_band_roi.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_temporal_thresholds(
    rows: list[dict],
    figure_root: Path,
    *,
    candidate_thresholds: tuple[float, ...],
    minimum_development_bets: int,
) -> None:
    usable = [row for row in rows if row["selection_status"] == "selected"]
    if not usable:
        return

    figure_root.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(usable))
    development_roi = _finite_plot_values(usable, "development_roi")
    test_roi = _finite_plot_values(usable, "test_roi")
    labels = [str(row["test_season"]) for row in usable]
    width = 0.34

    fig, ax = plt.subplots(figsize=(11.6, 6.3))
    dev_bars = ax.bar(
        x - width / 2,
        development_roi,
        width=width,
        edgecolor="black",
        linewidth=0.6,
        label="Earlier development seasons used to select t",
    )
    test_bars = ax.bar(
        x + width / 2,
        test_roi,
        width=width,
        edgecolor="black",
        linewidth=0.6,
        label="Next unseen season using frozen t",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    fig.suptitle("Walk-Forward Threshold Selection", fontsize=16, y=0.985)
    threshold_text = ", ".join(f"{threshold:.0%}" for threshold in candidate_thresholds)
    fig.text(
        0.5,
        0.947,
        (
            f"Candidate t ∈ {{{threshold_text}}}; require ≥{minimum_development_bets} "
            "development bets; select the highest development ROI and apply that t "
            "unchanged to the next season."
        ),
        ha="center",
        fontsize=9,
    )
    ax.set_xlabel("Future season evaluated after selection on all earlier seasons")
    ax.set_ylabel("ROI per unit stake")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend()

    for dev_bar, test_bar, row in zip(dev_bars, test_bars, usable):
        dev_value = float(row["development_roi"])
        test_value = float(row["test_roi"])
        _annotate_bar_count(ax, dev_bar, int(row["development_num_bets"]), dev_value)
        _annotate_bar_count(ax, test_bar, int(row["test_num_bets"]), test_value)
        y_anchor = max(dev_value, test_value, 0.0)
        ax.annotate(
            f"selected t={float(row['selected_threshold']):.0%}",
            (dev_bar.get_x() + width, y_anchor),
            xytext=(0, 25),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    fig.subplots_adjust(top=0.89, bottom=0.11)
    _save_figure(fig, figure_root, "bookmaker_temporal_threshold_selection")


def plot_edge_deciles(rows: list[dict], figure_root: Path) -> None:
    if not rows:
        return
    figure_root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.4, 8.4),
        sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    x = np.arange(len(rows))
    labels = [
        (f"{row['quantile_label']}\n" f"{100.0 * float(row['edge_min']):.1f}–" f"{100.0 * float(row['edge_max']):.1f}%")
        for row in rows
    ]
    counts = [int(row["num_bets"]) for row in rows]
    roi = _finite_plot_values(rows, "roi")
    residual = [
        100.0 * float(row["realized_market_residual"]) if row["realized_market_residual"] is not None else np.nan
        for row in rows
    ]

    fig.suptitle("Equal-Count Deciles of Positive Best-Side Edge", fontsize=16, y=0.985)
    fig.text(
        0.5,
        0.953,
        "Q1 contains the weakest positive edges; Q10 contains the strongest positive edges.",
        ha="center",
        fontsize=9,
    )

    bars = axes[0].bar(x, roi, edgecolor="black", linewidth=0.6)
    axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("ROI per unit stake")
    axes[0].grid(axis="y", linestyle=":", alpha=0.45)
    for bar, value, count in zip(bars, roi, counts):
        if np.isfinite(value):
            _annotate_bar_count(axes[0], bar, count, value)

    axes[1].bar(x, residual, edgecolor="black", linewidth=0.6)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Observed hit-rate excess over bookmaker fair probability (pp)")
    axes[1].set_xlabel("Edge decile and observed edge range")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", linestyle=":", alpha=0.45)
    axes[1].text(
        0.01,
        0.97,
        (
            "Positive = selected outcomes occurred more often than the bookmaker's "
            "average de-vig fair probability predicted."
        ),
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
    )

    fig.subplots_adjust(top=0.91, bottom=0.10)
    _save_figure(fig, figure_root, "bookmaker_edge_deciles")


def plot_competition_roi(
    rows: list[dict],
    figure_root: Path,
    *,
    threshold: float,
    side: str,
    minimum_bets: int,
    filename: str,
) -> None:
    usable = [
        row
        for row in rows
        if row.get("side", "all") == side
        and abs(float(row["threshold"]) - threshold) < 1e-12
        and row["roi"] is not None
    ]
    if not usable:
        return
    usable.sort(key=lambda row: float(row["roi"]))

    figure_root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.4, 9.3))
    y = np.arange(len(usable))
    values = [float(row["roi"]) for row in usable]
    counts = [int(row["num_bets"]) for row in usable]
    colors = [
        (
            sett.COMPS_LEAGUE_COLORS.get(str(row["competition"]), "#bdbdbd")
            if int(row["num_bets"]) >= minimum_bets
            else "#d9d9d9"
        )
        for row in usable
    ]
    bars = ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.5)

    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    side_label = "All selected sides" if side == "all" else side.title()
    fig.suptitle(
        f"Competition ROI — {side_label}, Required Edge ≥ {threshold:.0%}",
        fontsize=16,
        y=0.985,
    )
    fig.text(
        0.5,
        0.952,
        (
            "Aggregated across all available OOS seasons. Thesis competition colors "
            f"are used for n≥{minimum_bets}; lower-support bars are light gray."
        ),
        ha="center",
        fontsize=9,
    )
    ax.set_xlabel("ROI per unit stake")
    ax.set_yticks(y)
    ax.set_yticklabels([str(row["competition"]) for row in usable])
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    for bar, value, count in zip(bars, values, counts):
        ax.annotate(
            f"n={count}",
            (value, bar.get_y() + bar.get_height() / 2),
            xytext=(5 if value >= 0 else -5, 0),
            textcoords="offset points",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=7,
        )

    fig.subplots_adjust(top=0.91, left=0.25, right=0.97, bottom=0.08)
    _save_figure(fig, figure_root, filename)


def _plot_grouped_band_by_side(
    rows: list[dict],
    figure_root: Path,
    *,
    band_field: str,
    thresholds: tuple[float, ...],
    title: str,
    filename: str,
    x_label: str,
) -> None:
    labels: list[str] = []
    for row in rows:
        label = str(row[band_field])
        if label not in labels:
            labels.append(label)
    if not labels:
        return

    fig, axes = plt.subplots(1, len(thresholds), figsize=(14.5, 5.5), sharey=True)
    if len(thresholds) == 1:
        axes = np.asarray([axes])
    width = 0.24
    x = np.arange(len(labels))

    for ax, threshold in zip(axes, thresholds):
        for side_index, side in enumerate(("all", "under", "over")):
            selected = [
                row
                for row in rows
                if row.get("side", "all") == side and abs(float(row["threshold"]) - threshold) < 1e-12
            ]
            by_label = {str(row[band_field]): row for row in selected}
            values = [
                float(by_label[label]["roi"]) if label in by_label and by_label[label]["roi"] is not None else np.nan
                for label in labels
            ]
            counts = [int(by_label[label]["num_bets"]) if label in by_label else 0 for label in labels]
            offset = (side_index - 1) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=side.title(),
                edgecolor="black",
                linewidth=0.4,
            )
            for bar, value, count in zip(bars, values, counts):
                if np.isfinite(value) and count:
                    ax.annotate(
                        str(count),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3 if value >= 0 else -10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=6,
                    )
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_title(f"Required edge ≥ {threshold:.0%}", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_xlabel(x_label)
        ax.grid(axis="y", linestyle=":", alpha=0.45)

    axes[0].set_ylabel("ROI per unit stake")
    axes[-1].legend(title="Selected side")
    fig.suptitle(title, fontsize=16, y=0.985)
    fig.subplots_adjust(top=0.88, bottom=0.20, wspace=0.05)
    _save_figure(fig, figure_root, filename)


def plot_margin_by_side(rows: list[dict], figure_root: Path) -> None:
    _plot_grouped_band_by_side(
        rows,
        figure_root,
        band_field="margin_band",
        thresholds=(0.07, 0.10),
        title="ROI by Bookmaker-Margin Band and Selected Side",
        filename="bookmaker_margin_band_roi",
        x_label="Two-way bookmaker overround / margin",
    )


def plot_season_stability(rows: list[dict], figure_root: Path) -> None:
    seasons = sorted({int(row["season"]) for row in rows})
    thresholds = (0.07, 0.10)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.3), sharey=True)
    width = 0.24
    x = np.arange(len(seasons))

    for ax, threshold in zip(axes, thresholds):
        for side_index, side in enumerate(("all", "under", "over")):
            selected = [
                row
                for row in rows
                if row.get("side", "all") == side and abs(float(row["threshold"]) - threshold) < 1e-12
            ]
            by_season = {int(row["season"]): row for row in selected}
            values = [
                float(by_season[season]["roi"]) if by_season[season]["roi"] is not None else np.nan
                for season in seasons
            ]
            counts = [int(by_season[season]["num_bets"]) for season in seasons]
            offset = (side_index - 1) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=side.title(),
                edgecolor="black",
                linewidth=0.4,
            )
            for bar, value, count in zip(bars, values, counts):
                if np.isfinite(value):
                    ax.annotate(
                        str(count),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3 if value >= 0 else -10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=6,
                    )
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_title(f"Required edge ≥ {threshold:.0%}")
        ax.set_xticks(x)
        ax.set_xticklabels(seasons)
        ax.grid(axis="y", linestyle=":", alpha=0.45)

    axes[0].set_ylabel("ROI per unit stake")
    axes[-1].legend(title="Selected side")
    fig.suptitle("Season Stability of Selective Betting")
    fig.tight_layout()
    fig.savefig(figure_root / "bookmaker_season_stability.pdf", bbox_inches="tight")
    fig.savefig(
        figure_root / "bookmaker_season_stability.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_under_odds_bands(rows: list[dict], figure_root: Path) -> None:
    under_rows = [row for row in rows if row.get("side") == "under"]
    plotted_thresholds = (0.05, 0.07, 0.10)
    labels: list[str] = []
    for row in under_rows:
        label = str(row["odds_band"])
        if label not in labels:
            labels.append(label)
    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    for offset_index, threshold in enumerate(plotted_thresholds):
        selected = [row for row in under_rows if abs(float(row["threshold"]) - threshold) < 1e-12]
        by_label = {str(row["odds_band"]): row for row in selected}
        values = [float(by_label[label]["roi"]) if by_label[label]["roi"] is not None else np.nan for label in labels]
        counts = [int(by_label[label]["num_bets"]) for label in labels]
        offset = (offset_index - 1) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=f"edge ≥ {threshold:.0%}",
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, value, count in zip(bars, values, counts):
            if np.isfinite(value):
                ax.annotate(
                    str(count),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4 if value >= 0 else -12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )

    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_title("Under 2.5 ROI by Decimal-Odds Band")
    ax.set_xlabel("Under 2.5 odds")
    ax.set_ylabel("ROI per unit stake")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(title="Required Under edge")
    fig.tight_layout()
    fig.savefig(figure_root / "bookmaker_under_odds_band_roi.pdf", bbox_inches="tight")
    fig.savefig(
        figure_root / "bookmaker_under_odds_band_roi.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_ev_deciles_and_selectivity(
    deciles: list[dict],
    coverage: list[dict],
    figure_root: Path,
) -> None:
    if not deciles:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 8.4), gridspec_kw={"hspace": 0.35})
    x = np.arange(len(deciles))
    labels = [
        f"{row['quantile_label']}\n{100*float(row['ev_min']):.1f}–{100*float(row['ev_max']):.1f}%" for row in deciles
    ]
    roi = _finite_plot_values(deciles, "roi")
    counts = [int(row["num_bets"]) for row in deciles]
    bars = axes[0].bar(x, roi, edgecolor="black", linewidth=0.6)
    axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0].set_title("ROI by Equal-Count Estimated-EV Decile", pad=12)
    axes[0].set_ylabel("ROI per unit stake")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", linestyle=":", alpha=0.45)
    for bar, value, count in zip(bars, roi, counts):
        if np.isfinite(value):
            _annotate_bar_count(axes[0], bar, count, value)

    cx = [100 * float(row["actual_positive_candidate_coverage"]) for row in coverage]
    cy = _finite_plot_values(coverage, "roi")
    cc = [int(row["num_bets"]) for row in coverage]
    order = np.argsort(cx)
    cx = [cx[i] for i in order]
    cy = [cy[i] for i in order]
    cc = [cc[i] for i in order]
    axes[1].plot(cx, cy, linewidth=2.0)
    axes[1].scatter(cx, cy, s=_bounded_marker_areas(cc, minimum=35, maximum=120))
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_title("ROI as Betting Becomes More Selective by Estimated EV", pad=12)
    axes[1].set_xlabel("Share of non-negative-EV candidates retained (%)")
    axes[1].set_ylabel("ROI per unit stake")
    axes[1].grid(axis="y", linestyle=":", alpha=0.45)
    for xv, yv, count in zip(cx, cy, cc):
        if np.isfinite(yv):
            axes[1].annotate(
                f"n={count}",
                (xv, yv),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    fig.suptitle(
        "Estimated Expected Return (EV) as an Alternative Opportunity Ranking",
        fontsize=16,
        y=0.985,
    )
    fig.subplots_adjust(top=0.92, bottom=0.09)
    _save_figure(fig, figure_root, "bookmaker_ev_selectivity")


def plot_walk_forward_side_comparison(
    all_rows: list[dict],
    under_rows: list[dict],
    figure_root: Path,
) -> None:
    all_selected = {int(row["test_season"]): row for row in all_rows if row["selection_status"] == "selected"}
    under_selected = {int(row["test_season"]): row for row in under_rows if row["selection_status"] == "selected"}
    seasons = sorted(set(all_selected) | set(under_selected))
    if not seasons:
        return
    x = np.arange(len(seasons))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    all_values = [float(all_selected[s]["test_roi"]) if s in all_selected else np.nan for s in seasons]
    under_values = [float(under_selected[s]["test_roi"]) if s in under_selected else np.nan for s in seasons]
    bars_all = ax.bar(
        x - width / 2, all_values, width=width, label="All selected sides", edgecolor="black", linewidth=0.5
    )
    bars_under = ax.bar(x + width / 2, under_values, width=width, label="Under only", edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_xlabel("Future unseen season")
    ax.set_ylabel("Future-season ROI per unit stake")
    ax.set_title("Walk-Forward Future ROI: All Sides vs Under-Only Threshold Selection", pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend()
    for bars, mapping in ((bars_all, all_selected), (bars_under, under_selected)):
        for bar, season in zip(bars, seasons):
            row = mapping.get(season)
            if row is None or row["test_roi"] is None:
                continue
            value = float(row["test_roi"])
            ax.annotate(
                f"t={float(row['selected_threshold']):.0%}\nn={int(row['test_num_bets'])}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 5 if value >= 0 else -22),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    fig.tight_layout()
    _save_figure(fig, figure_root, "bookmaker_walk_forward_side_comparison")


def plot_probability_method_metrics(rows: list[dict], figure_root: Path) -> None:
    pooled = [row for row in rows if row["scope"] == "pooled"]
    if not pooled:
        return
    order = [
        "bookmaker-fair",
        "model-raw",
        "model-platt",
        "model-isotonic",
        "bookmaker-model-hybrid",
    ]
    by_method = {str(row["method"]): row for row in pooled}
    methods = [method for method in order if method in by_method]
    labels = [
        {
            "bookmaker-fair": "Bookmaker",
            "model-raw": "Raw model",
            "model-platt": "Platt",
            "model-isotonic": "Isotonic",
            "bookmaker-model-hybrid": "Bookmaker + model",
        }[method]
        for method in methods
    ]
    x = np.arange(len(methods))
    auc = [float(by_method[m]["roc_auc"]) for m in methods]
    brier = [float(by_method[m]["brier_score"]) for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    bars_auc = axes[0].bar(x, auc, edgecolor="black", linewidth=0.5)
    axes[0].axhline(0.5, linestyle="--", linewidth=1.0)
    axes[0].set_title("Walk-Forward ROC AUC")
    axes[0].set_ylabel("ROC AUC (higher is better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(axis="y", linestyle=":", alpha=0.45)
    for bar, value in zip(bars_auc, auc):
        axes[0].annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    bars_brier = axes[1].bar(x, brier, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Walk-Forward Brier Score")
    axes[1].set_ylabel("Brier score (lower is better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(axis="y", linestyle=":", alpha=0.45)
    for bar, value in zip(bars_brier, brier):
        axes[1].annotate(
            f"{value:.3f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    fig.suptitle("Chronological Calibration and Simple Bookmaker–Model Hybrid", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, figure_root, "bookmaker_probability_method_metrics")


def plot_probability_method_betting(rows: list[dict], figure_root: Path) -> None:
    if not rows:
        return
    methods_order = ["model-raw", "model-platt", "model-isotonic", "bookmaker-model-hybrid"]
    labels_map = {
        "model-raw": "Raw model",
        "model-platt": "Platt",
        "model-isotonic": "Isotonic",
        "bookmaker-model-hybrid": "Bookmaker + model",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for ax, threshold in zip(axes, (0.07, 0.10)):
        x = np.arange(len(methods_order))
        width = 0.34
        for side_index, side in enumerate(("all", "under")):
            selected = [row for row in rows if row["side"] == side and abs(float(row["threshold"]) - threshold) < 1e-12]
            by_method = {str(row["method"]): row for row in selected}
            values = [
                float(by_method[m]["roi"]) if m in by_method and by_method[m]["roi"] is not None else np.nan
                for m in methods_order
            ]
            counts = [int(by_method[m]["num_bets"]) if m in by_method else 0 for m in methods_order]
            offset = (side_index - 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label="All sides" if side == "all" else "Under only",
                edgecolor="black",
                linewidth=0.5,
            )
            for bar, value, count in zip(bars, values, counts):
                if np.isfinite(value):
                    ax.annotate(
                        f"n={count}",
                        (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 4 if value >= 0 else -12),
                        textcoords="offset points",
                        ha="center",
                        fontsize=6,
                    )
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_title(f"Required edge ≥ {threshold:.0%}")
        ax.set_xticks(x)
        ax.set_xticklabels([labels_map[m] for m in methods_order], rotation=25, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.45)
    axes[0].set_ylabel("Future-only ROI per unit stake")
    axes[-1].legend()
    fig.suptitle("Betting Utility After Chronological Calibration / Hybridization", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, figure_root, "bookmaker_probability_method_betting")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended bookmaker betting-utility analysis.")
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=HISTORICAL_OOS_PATH)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(sett.DATA_DIR) / "comparison" / "bookmaker_utility",
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "bookmaker_utility",
    )
    parser.add_argument("--minimum-development-bets", type=int, default=100)
    parser.add_argument("--competition-minimum-bets", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_path = args.snapshot or resolve_snapshot_path()
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)

    joined_rows = read_joined_rows(args.predictions, snapshot_path)
    if not joined_rows:
        raise RuntimeError("No predictions could be joined with valid Under/Over 2.5 odds.")

    opportunities = build_betting_opportunities(joined_rows)
    best = select_best_opportunity_per_match(opportunities)

    edge_bins = build_fixed_edge_bins(best)
    quantile_bins = build_quantile_edge_bins(best)
    coverage = build_coverage_curve(best)
    threshold_summary = build_threshold_summary(best)
    odds_bands = build_odds_band_summary(best)
    margin_bands = build_margin_band_summary(best)
    season_summary = build_season_summary(best)
    competition_summary = build_competition_summary(best)
    temporal_config = TemporalThresholdConfig(
        thresholds=tuple(DEFAULT_THRESHOLDS),
        minimum_development_bets=args.minimum_development_bets,
    )
    temporal_summary = build_temporal_threshold_summary(best, temporal_config)
    temporal_under_summary = build_temporal_threshold_summary(best, temporal_config, side="under")
    ev_quantile_bins = build_quantile_ev_bins(best)
    ev_coverage = build_ev_coverage_curve(best)
    probability_rows = build_walk_forward_probability_rows(joined_rows)
    probability_metrics = build_probability_method_metrics(probability_rows)
    probability_betting = build_probability_method_betting_summary(probability_rows)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.figure_root.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_root / "bookmaker_utility_joined_predictions.csv", joined_rows)
    write_csv(args.output_root / "bookmaker_utility_opportunities.csv", opportunities)
    write_csv(args.output_root / "bookmaker_utility_best_opportunities.csv", best)
    write_csv(args.output_root / "bookmaker_edge_bins_fixed.csv", edge_bins)
    write_csv(args.output_root / "bookmaker_edge_bins_quantile.csv", quantile_bins)
    write_csv(args.output_root / "bookmaker_coverage_curve.csv", coverage)
    write_csv(args.output_root / "bookmaker_threshold_side_summary.csv", threshold_summary)
    write_csv(args.output_root / "bookmaker_odds_band_summary.csv", odds_bands)
    write_csv(args.output_root / "bookmaker_margin_band_summary.csv", margin_bands)
    write_csv(args.output_root / "bookmaker_season_summary.csv", season_summary)
    write_csv(args.output_root / "bookmaker_competition_summary.csv", competition_summary)
    write_csv(args.output_root / "bookmaker_temporal_threshold_summary.csv", temporal_summary)
    write_csv(
        args.output_root / "bookmaker_temporal_under_threshold_summary.csv",
        temporal_under_summary,
    )
    write_csv(args.output_root / "bookmaker_ev_bins_quantile.csv", ev_quantile_bins)
    write_csv(args.output_root / "bookmaker_ev_coverage_curve.csv", ev_coverage)
    write_csv(
        args.output_root / "bookmaker_walk_forward_probability_rows.csv",
        probability_rows,
    )
    write_csv(
        args.output_root / "bookmaker_probability_method_metrics.csv",
        probability_metrics,
    )
    write_csv(
        args.output_root / "bookmaker_probability_method_betting.csv",
        probability_betting,
    )

    positive_candidate_count = sum(float(row["probability_edge"]) >= 0.0 for row in best)
    metadata = {
        "snapshot": str(snapshot_path.resolve()),
        "predictions": str(args.predictions.resolve()),
        "joined_match_count": len(joined_rows),
        "opportunity_count": len(opportunities),
        "best_opportunity_count": len(best),
        "positive_best_side_candidate_count": positive_candidate_count,
        "positive_best_side_candidate_share": (positive_candidate_count / max(1, len(best))),
        "thresholds": list(DEFAULT_THRESHOLDS),
        "minimum_development_bets": args.minimum_development_bets,
        "competition_minimum_bets": args.competition_minimum_bets,
    }
    with (args.output_root / "bookmaker_utility_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    plot_edge_selectivity(
        edge_bins,
        coverage,
        args.figure_root,
        total_match_count=len(best),
        positive_candidate_count=positive_candidate_count,
    )
    plot_edge_selectivity_bar_only(
        edge_bins,
        args.figure_root,
        total_match_count=len(best),
        positive_candidate_count=positive_candidate_count,
    )
    plot_side_thresholds(
        threshold_summary,
        args.figure_root,
        total_match_count=len(best),
    )
    plot_odds_bands(odds_bands, args.figure_root)
    plot_temporal_thresholds(
        temporal_summary,
        args.figure_root,
        candidate_thresholds=tuple(DEFAULT_THRESHOLDS),
        minimum_development_bets=args.minimum_development_bets,
    )
    plot_edge_deciles(quantile_bins, args.figure_root)
    plot_competition_roi(
        competition_summary,
        args.figure_root,
        threshold=0.07,
        side="all",
        minimum_bets=args.competition_minimum_bets,
        filename="bookmaker_competition_roi_7pct",
    )
    plot_competition_roi(
        competition_summary,
        args.figure_root,
        threshold=0.10,
        side="all",
        minimum_bets=args.competition_minimum_bets,
        filename="bookmaker_competition_roi_10pct",
    )
    plot_competition_roi(
        competition_summary,
        args.figure_root,
        threshold=0.07,
        side="under",
        minimum_bets=args.competition_minimum_bets,
        filename="bookmaker_under_competition_roi_7pct",
    )
    plot_competition_roi(
        competition_summary,
        args.figure_root,
        threshold=0.10,
        side="under",
        minimum_bets=args.competition_minimum_bets,
        filename="bookmaker_under_competition_roi_10pct",
    )
    plot_margin_by_side(margin_bands, args.figure_root)
    plot_season_stability(season_summary, args.figure_root)
    plot_under_odds_bands(odds_bands, args.figure_root)
    plot_ev_deciles_and_selectivity(ev_quantile_bins, ev_coverage, args.figure_root)
    plot_walk_forward_side_comparison(temporal_summary, temporal_under_summary, args.figure_root)
    plot_probability_method_metrics(probability_metrics, args.figure_root)
    plot_probability_method_betting(probability_betting, args.figure_root)

    print("[utility] analysis: PASS")
    print(f"[utility] joined matches: {len(joined_rows)}")
    print(f"[utility] candidate side opportunities: {len(opportunities)}")
    print(f"[utility] one best side per match: {len(best)}")
    print(f"[saved] {args.output_root / 'bookmaker_edge_bins_fixed.csv'}")
    print(f"[saved] {args.output_root / 'bookmaker_coverage_curve.csv'}")
    print(f"[saved] {args.output_root / 'bookmaker_threshold_side_summary.csv'}")
    print(f"[saved] {args.output_root / 'bookmaker_odds_band_summary.csv'}")
    print(f"[saved] {args.output_root / 'bookmaker_competition_summary.csv'}")
    print(f"[saved] {args.output_root / 'bookmaker_temporal_threshold_summary.csv'}")
    print(f"[saved] {args.figure_root / 'bookmaker_edge_selectivity.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_under_over_thresholds.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_odds_band_roi.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_temporal_threshold_selection.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_edge_deciles.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_competition_roi_7pct.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_competition_roi_10pct.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_under_competition_roi_10pct.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_margin_band_roi.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_season_stability.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_under_odds_band_roi.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_under_competition_roi_7pct.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_ev_selectivity.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_walk_forward_side_comparison.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_probability_method_metrics.png'}")
    print(f"[saved] {args.figure_root / 'bookmaker_probability_method_betting.png'}")


if __name__ == "__main__":
    main()
