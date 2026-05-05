from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

import football_outcomes.config.fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.fs_io import try_load_snapshot
from football_outcomes.data.fs_retrieve import fill_globals_with_cache

matplotlib.use("Agg")

OOS_PATH = Path(sett.DATA_DIR) / "tensorboard_logs" / "selected_mlp_binary_u25" / "oos_predictions.csv"

OUT_DIR = Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "bookmaker_odds"
OUT_DATA_DIR = Path(sett.DATA_DIR) / "comparison" / "bookmaker_odds"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.00, 0.02, 0.03, 0.05, 0.07, 0.10]

SERIES_COLORS = {
    "model": "#6f8fbf",
    "bookmaker": "#8a7fa6",
    "num_bets": "#b8c7dc",
}

BAR_EDGE_COLOR = "#000000"
BAR_EDGE_WIDTH = 0.6


def _safe_float(x) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v) or v <= 1.0:
        return None
    return v


def load_match_odds_by_id() -> dict[int, dict]:
    cache = try_load_snapshot()
    if cache is None:
        raise RuntimeError("Could not load snapshot. Odds benchmark requires cached FSMatch objects.")

    fill_globals_with_cache(cache, update_leagues_list=False)
    g = Global.get_instance()

    out = {}
    for m in g.all_matches:
        odds = getattr(m, "odds", None) or {}
        under = _safe_float(odds.get("under25"))
        over = _safe_float(odds.get("over25"))

        if under is None or over is None:
            continue

        raw_under = 1.0 / under
        raw_over = 1.0 / over
        total = raw_under + raw_over

        if total <= 0:
            continue

        out[int(m.id)] = {
            "under25_odds": under,
            "over25_odds": over,
            "book_raw_p_under": raw_under,
            "book_raw_p_over": raw_over,
            "book_fair_p_under": raw_under / total,
            "book_fair_p_over": raw_over / total,
            "book_margin": total - 1.0,
        }

    return out


def read_joined_rows() -> list[dict]:
    odds_by_id = load_match_odds_by_id()

    joined = []
    missing_odds = 0

    with OOS_PATH.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            match_id = int(float(r["match_id"]))
            odds = odds_by_id.get(match_id)

            if odds is None:
                missing_odds += 1
                continue

            y_true_under = int(float(r["y_true"]))
            p_model_under = float(r["y_prob_under25"])

            joined.append(
                {
                    "round_idx": int(float(r["round_idx"])),
                    "match_id": match_id,
                    "season": int(float(r["season"])),
                    "competition": r["competition"],
                    "y_true_under25": y_true_under,
                    "y_true_over25": 1 - y_true_under,
                    "p_model_under": p_model_under,
                    "p_model_over": 1.0 - p_model_under,
                    **odds,
                }
            )

    print(f"[odds] joined predictions with odds: {len(joined)}")
    print(f"[odds] OOS predictions without valid U/O 2.5 odds: {missing_odds}")

    return joined


def probability_metrics(rows: list[dict]) -> dict:
    y = np.asarray([r["y_true_under25"] for r in rows], dtype=np.int32)

    p_model = np.asarray([r["p_model_under"] for r in rows], dtype=np.float32)
    p_book_fair = np.asarray([r["book_fair_p_under"] for r in rows], dtype=np.float32)
    p_book_raw = np.asarray([r["book_raw_p_under"] for r in rows], dtype=np.float32)

    return {
        "num_matches_with_valid_odds": int(len(rows)),
        "avg_book_margin": float(np.mean([r["book_margin"] for r in rows])),
        "model_auc": float(roc_auc_score(y, p_model)),
        "model_accuracy": float(accuracy_score(y, (p_model >= 0.5).astype(np.int32))),
        "model_brier": float(brier_score_loss(y, p_model)),
        "book_fair_auc": float(roc_auc_score(y, p_book_fair)),
        "book_fair_accuracy": float(accuracy_score(y, (p_book_fair >= 0.5).astype(np.int32))),
        "book_fair_brier": float(brier_score_loss(y, p_book_fair)),
        "book_raw_brier": float(brier_score_loss(y, p_book_raw)),
    }


def simulate_betting(rows: list[dict], threshold: float) -> dict:
    bets = []
    cumulative_profit = []
    profit_sum = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for r in rows:
        candidates = []

        under_edge = r["p_model_under"] - r["book_raw_p_under"]
        over_edge = r["p_model_over"] - r["book_raw_p_over"]

        if under_edge >= threshold:
            candidates.append(("under", under_edge, r["under25_odds"], r["y_true_under25"] == 1))

        if over_edge >= threshold:
            candidates.append(("over", over_edge, r["over25_odds"], r["y_true_over25"] == 1))

        if not candidates:
            continue

        side, edge, odds, won = max(candidates, key=lambda x: x[1])

        profit = (odds - 1.0) if won else -1.0
        profit_sum += profit
        cumulative_profit.append(profit_sum)

        peak = max(peak, profit_sum)
        max_drawdown = max(max_drawdown, peak - profit_sum)

        bets.append(
            {
                **r,
                "bet_side": side,
                "edge": edge,
                "bet_odds": odds,
                "won": int(won),
                "profit": profit,
                "cumulative_profit": profit_sum,
            }
        )

    num_bets = len(bets)
    total_staked = float(num_bets)

    return {
        "threshold": threshold,
        "num_bets": num_bets,
        "coverage": num_bets / max(1, len(rows)),
        "hit_rate": float(np.mean([b["won"] for b in bets])) if bets else 0.0,
        "total_profit": float(profit_sum),
        "roi": float(profit_sum / total_staked) if total_staked > 0 else 0.0,
        "max_drawdown": float(max_drawdown),
        "bets": bets,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_probability_comparison(metrics: dict) -> None:
    labels = ["AUC", "Accuracy", "Brier score"]
    model = [metrics["model_auc"], metrics["model_accuracy"], metrics["model_brier"]]
    book = [metrics["book_fair_auc"], metrics["book_fair_accuracy"], metrics["book_fair_brier"]]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.8, 5.2))

    ax.bar(
        x - width / 2,
        model,
        width=width,
        label="Selected MLP",
        color=SERIES_COLORS["model"],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )
    ax.bar(
        x + width / 2,
        book,
        width=width,
        label="Bookmaker implied probabilities",
        color=SERIES_COLORS["bookmaker"],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )

    ax.set_title("Model vs bookmaker implied probabilities", fontsize=16)
    ax.set_xlabel("Metric", fontsize=13)
    ax.set_ylabel("Metric value", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(max(model), max(book)) + 0.08)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for offset, vals in [(-width / 2, model), (width / 2, book)]:
        for i, value in enumerate(vals):
            ax.text(
                i + offset,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "bookmaker_probability_comparison.pdf", bbox_inches="tight")
    fig.savefig(
        OUT_DIR / "bookmaker_probability_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_betting_thresholds(betting_summaries: list[dict]) -> None:
    thresholds = [r["threshold"] for r in betting_summaries]
    roi = [r["roi"] for r in betting_summaries]
    num_bets = [r["num_bets"] for r in betting_summaries]

    fig, ax1 = plt.subplots(figsize=(10.8, 5.2))

    ax1.plot(
        thresholds,
        roi,
        marker="o",
        linewidth=2.0,
        color=SERIES_COLORS["model"],
        label="ROI",
    )
    ax1.axhline(
        0.0,
        linestyle="--",
        color="#555555",
        linewidth=1.0,
        alpha=0.75,
    )
    ax1.set_title("Betting simulation by edge threshold", fontsize=16)
    ax1.set_xlabel("Minimum model edge over break-even probability", fontsize=13)
    ax1.set_ylabel("ROI per unit stake", fontsize=13)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.bar(
        thresholds,
        num_bets,
        width=0.008,
        alpha=0.45,
        color=SERIES_COLORS["num_bets"],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
        label="Number of bets",
    )
    ax2.set_ylabel("Number of bets", fontsize=13)
    ax2.spines["top"].set_visible(False)

    lines, line_labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, line_labels + bar_labels, loc="best")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "bookmaker_betting_thresholds.pdf", bbox_inches="tight")
    fig.savefig(
        OUT_DIR / "bookmaker_betting_thresholds.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    rows = read_joined_rows()

    if not rows:
        raise RuntimeError("No OOS predictions could be joined with valid Under/Over 2.5 odds.")

    write_csv(OUT_DATA_DIR / "bookmaker_joined_predictions.csv", rows)

    metrics = probability_metrics(rows)

    betting_outputs = []
    betting_summaries = []

    for threshold in THRESHOLDS:
        result = simulate_betting(rows, threshold)
        bets = result.pop("bets")
        betting_summaries.append(result)

        for b in bets:
            b["threshold"] = threshold
            betting_outputs.append(b)

    with (OUT_DATA_DIR / "bookmaker_probability_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    write_csv(OUT_DATA_DIR / "bookmaker_betting_summary.csv", betting_summaries)
    write_csv(OUT_DATA_DIR / "bookmaker_betting_bets.csv", betting_outputs)

    plot_probability_comparison(metrics)
    plot_betting_thresholds(betting_summaries)

    print("[saved]", OUT_DATA_DIR / "bookmaker_probability_metrics.json")
    print("[saved]", OUT_DATA_DIR / "bookmaker_betting_summary.csv")
    print("[saved]", OUT_DATA_DIR / "bookmaker_betting_bets.csv")
    print("[saved]", OUT_DIR / "bookmaker_probability_comparison.pdf")
    print("[saved]", OUT_DIR / "bookmaker_betting_thresholds.pdf")


if __name__ == "__main__":
    main()
