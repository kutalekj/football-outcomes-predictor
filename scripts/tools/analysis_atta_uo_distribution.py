from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from football_outcomes.config import fs_settings as sett

matplotlib.use("Agg")

FULL_BINARY_OOS_PATH = Path(sett.DATA_DIR) / "tensorboard_logs" / "selected_mlp_binary_u25" / "oos_predictions.csv"

OUT_DIR = Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "atta_mills_contextual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATTA_MILLS_COMPETITIONS = {
    "Belgium Pro League",
    "Netherlands Eredivisie",
    "Scotland Premiership",
}

OTHER_COLOR = "#bdbdbd"


def main() -> None:
    counts: dict[str, dict[str, int]] = {}

    with FULL_BINARY_OOS_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        comp = r["competition"]
        y = int(float(r["y_true"]))  # 1 = Under 2.5, 0 = Over 2.5

        counts.setdefault(comp, {"under": 0, "over": 0})

        if y == 1:
            counts[comp]["under"] += 1
        else:
            counts[comp]["over"] += 1

    records = []
    for comp, c in counts.items():
        total = c["under"] + c["over"]
        if total == 0:
            continue

        records.append(
            {
                "competition": comp,
                "under_2_5": c["under"],
                "over_2_5": c["over"],
                "total": total,
                "over_rate": c["over"] / total,
                "is_atta_mills_competition": comp in ATTA_MILLS_COMPETITIONS,
            }
        )

    records.sort(key=lambda r: r["over_rate"], reverse=True)

    comps = [r["competition"] for r in records]
    values = [100.0 * r["over_rate"] for r in records]

    colors = [
        sett.COMPS_LEAGUE_COLORS.get(r["competition"], OTHER_COLOR) if r["is_atta_mills_competition"] else OTHER_COLOR
        for r in records
    ]

    edge_colors = ["black" if r["is_atta_mills_competition"] else "#777777" for r in records]

    line_widths = [1.4 if r["is_atta_mills_competition"] else 0.4 for r in records]

    fig, ax = plt.subplots(figsize=(14.5, 6.2))

    bars = ax.bar(
        range(len(records)),
        values,
        color=colors,
        edgecolor=edge_colors,
        linewidth=line_widths,
        alpha=0.95,
    )

    ax.axhline(
        50.0,
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
    )

    ax.set_title("Share of Over 2.5 Matches by Competition", fontsize=22)
    ax.set_ylabel("Over 2.5 matches (%)", fontsize=16)
    ax.set_xlabel("Competition", fontsize=16)
    ax.set_ylim(0, max(values) + 9)

    ax.set_xticks(range(len(records)))
    ax.set_xticklabels(comps, rotation=55, ha="right", fontsize=10)

    ax.grid(axis="y", linestyle=":", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, r, value in zip(bars, records, values):
        label = f"{value:.1f}%"
        fontweight = "bold" if r["is_atta_mills_competition"] else "normal"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.7,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=0,
            fontweight=fontweight,
        )

    # Manual legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=OTHER_COLOR, ec="#777777", label="Other competitions"),
    ]

    for comp in sorted(ATTA_MILLS_COMPETITIONS):
        legend_handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                color=sett.COMPS_LEAGUE_COLORS.get(comp, OTHER_COLOR),
                ec="black",
                label=comp,
            )
        )

    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    fig.tight_layout()

    png_path = OUT_DIR / "all_competitions_over25_distribution.png"
    pdf_path = OUT_DIR / "all_competitions_over25_distribution.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = OUT_DIR / "all_competitions_over25_distribution.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "competition",
                "under_2_5",
                "over_2_5",
                "total",
                "over_rate",
                "is_atta_mills_competition",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
