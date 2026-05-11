from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from football_outcomes.config import fs_settings as sett

matplotlib.use("Agg")

OOS_PATH = Path(sett.DATA_DIR) / "tensorboard_logs" / "selected_mlp_binary_u25" / "oos_predictions.csv"

OUT_DIR = Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "per_competition"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMOOTH_WINDOW = 45
MIN_ROUND_MATCHES_FOR_AUC = 4
MIN_MATCHES_FOR_TREND = 500

ATTA_MILLS_COMPS = {
    "Belgium Pro League",
    "Netherlands Eredivisie",
    "Scotland Premiership",
}

FORCE_EXCLUDE_FROM_TREND = {
    "India Indian Super League",
}


def moving_average(values: list[float], window: int = SMOOTH_WINDOW) -> list[float]:
    if len(values) < window:
        return values

    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        vals = [v for v in values[lo:hi] if not np.isnan(v)]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


def safe_auc(y_true: list[int], y_prob: list[float]) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def read_oos_rows() -> list[dict]:
    with OOS_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_summary(
    rows: list[dict],
) -> tuple[list[dict], dict[str, list[dict]], dict[tuple[str, int], list[dict]]]:
    by_comp: dict[str, list[dict]] = defaultdict(list)
    by_comp_round: dict[tuple[str, int], list[dict]] = defaultdict(list)

    for r in rows:
        comp = r["competition"]
        round_idx = int(float(r["round_idx"]))
        by_comp[comp].append(r)
        by_comp_round[(comp, round_idx)].append(r)

    summary = []

    for comp, comp_rows in by_comp.items():
        y_true = [int(float(r["y_true"])) for r in comp_rows]
        y_prob = [float(r["y_prob_under25"]) for r in comp_rows]
        y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

        summary.append(
            {
                "competition": comp,
                "matches": len(comp_rows),
                "under_rate": float(np.mean(y_true)),
                "over_rate": float(1.0 - np.mean(y_true)),
                "auc": safe_auc(y_true, y_prob),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "brier": float(brier_score_loss(y_true, y_prob)),
            }
        )

    summary.sort(
        key=lambda r: (
            -999 if np.isnan(r["auc"]) else -r["auc"],
            -r["accuracy"],
        )
    )

    return summary, by_comp, by_comp_round


def choose_highlight_competitions(summary: list[dict]) -> set[str]:
    valid = [
        r
        for r in summary
        if not np.isnan(r["auc"])
        and r["matches"] >= MIN_MATCHES_FOR_TREND
        and r["competition"] not in FORCE_EXCLUDE_FROM_TREND
    ]

    top3_non_atta = {
        r["competition"] for r in valid[:4] if r["competition"] not in ATTA_MILLS_COMPS
    }  # should result in picking top 3 non Atta

    bottom1 = {r["competition"] for r in valid[-1:]}

    return top3_non_atta | bottom1 | ATTA_MILLS_COMPS


def write_summary(summary: list[dict], highlight_comps: set[str]) -> None:
    rows = []
    for r in summary:
        rr = dict(r)
        rr["highlighted_in_trend"] = r["competition"] in highlight_comps
        rr["excluded_from_trend"] = r["competition"] in FORCE_EXCLUDE_FROM_TREND
        rows.append(rr)

    csv_path = OUT_DIR / "per_competition_selected_mlp_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUT_DIR / "per_competition_selected_mlp_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


def plot_combined(
    summary: list[dict],
    by_comp_round: dict[tuple[str, int], list[dict]],
    highlight_comps: set[str],
) -> None:
    comps = [r["competition"] for r in summary]
    aucs = [r["auc"] for r in summary]

    colors = [sett.COMPS_LEAGUE_COLORS.get(comp, "#bdbdbd") for comp in comps]

    fig, (ax_bar, ax_line) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(15.5, 11.5),
        gridspec_kw={"height_ratios": [1.0, 1.8], "hspace": 1.05},
    )

    # ------------------------------------------------------------------
    # Bar chart: pooled AUC by competition
    # ------------------------------------------------------------------
    bars = []

    for i, (comp, auc, color) in enumerate(zip(comps, aucs, colors)):
        is_highlight = comp in highlight_comps

        bar = ax_bar.bar(
            i,
            auc,
            color=color,
            edgecolor="#000000" if is_highlight else "#555555",
            linewidth=2.0 if is_highlight else 0.5,
        )[0]

        bars.append(bar)

    ax_bar.axhline(0.5, linestyle="--", color="#555555", linewidth=1.0, alpha=0.75)
    ax_bar.set_title("Selected MLP Performance by Competition", fontsize=20)
    ax_bar.set_ylabel("Pooled AUC", fontsize=14)
    ax_bar.set_xlabel("Competition", fontsize=14)
    ax_bar.set_ylim(0.40, max(aucs) + 0.045)
    ax_bar.set_xticks(range(len(comps)))
    ax_bar.set_xticklabels(comps, rotation=55, ha="right")
    ax_bar.grid(axis="y", linestyle=":", alpha=0.55)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    for bar, value, comp in zip(bars, aucs, comps):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
            color="#000000" if comp in highlight_comps else "#222222",
            fontweight="bold" if comp in highlight_comps else "normal",
        )

    # ------------------------------------------------------------------
    # Line chart: selected competition AUC progression
    # ------------------------------------------------------------------
    for comp in sorted(highlight_comps):
        round_indices = sorted(round_idx for (c, round_idx) in by_comp_round.keys() if c == comp)

        xs = []
        ys = []

        for round_idx in round_indices:
            rr = by_comp_round[(comp, round_idx)]

            if len(rr) < MIN_ROUND_MATCHES_FOR_AUC:
                continue

            y_true = [int(float(r["y_true"])) for r in rr]
            y_prob = [float(r["y_prob_under25"]) for r in rr]

            auc = safe_auc(y_true, y_prob)
            if np.isnan(auc):
                continue

            xs.append(round_idx)
            ys.append(auc)

        if not xs:
            continue

        ys_smooth = moving_average(ys, SMOOTH_WINDOW)
        color = sett.COMPS_LEAGUE_COLORS.get(comp, None)

        ax_line.plot(xs, ys, color=color, alpha=0.08, linewidth=0.5)
        ax_line.plot(xs, ys_smooth, color=color, linewidth=2.0, label=comp)

    ax_line.axhline(0.5, linestyle="--", color="#555555", linewidth=1.0, alpha=0.75)
    ax_line.set_title(
        "Round-Level AUC Progression for Highlighted Competitions",
        fontsize=20,
    )
    ax_line.set_ylabel("Validation AUC per competition", fontsize=14)
    ax_line.set_xlabel("Validation round", fontsize=14)
    ax_line.set_ylim(0.4, 0.75)
    ax_line.grid(axis="y", linestyle=":", alpha=0.55)
    ax_line.legend(ncol=2, fontsize=9)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)

    fig.savefig(OUT_DIR / "per_competition_auc_combined.pdf", bbox_inches="tight")
    fig.savefig(
        OUT_DIR / "per_competition_auc_combined.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {OUT_DIR / 'per_competition_auc_combined.pdf'}")
    print(f"Saved: {OUT_DIR / 'per_competition_auc_combined.png'}")


def main() -> None:
    rows = read_oos_rows()
    summary, _by_comp, by_comp_round = build_summary(rows)
    highlight_comps = choose_highlight_competitions(summary)

    write_summary(summary, highlight_comps)
    plot_combined(summary, by_comp_round, highlight_comps)

    print("Highlighted competitions:")
    for comp in sorted(highlight_comps):
        print(f"  - {comp}")


if __name__ == "__main__":
    main()
