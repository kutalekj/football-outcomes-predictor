import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATS_TEXT_PATH = "missing_stats_output.txt"
FIG_HEATMAP_PATH = "missing_values_heatmap.png"
FIG_BAR_PATH = "missing_values_by_competition.png"

COMP_INFO = {
    141: ("Segunda División", "tomato"),
    42: ("League Two", "dodgerblue"),
    41: ("League One", "cornflowerblue"),
    188: ("A-League", "silver"),
    323: ("Indian Super League", "darkkhaki"),
    79: ("2. Bundesliga", "goldenrod"),
    119: ("Danish Superliga", "thistle"),
    78: ("Bundesliga", "gold"),
    144: ("Jupiler Pro League", "aquamarine"),
    40: ("Championship", "royalblue"),
    61: ("Ligue 1", "blueviolet"),
    140: ("La Liga", "red"),
    203: ("Süper Lig", "rosybrown"),
    307: ("Saudi Pro League", "darkseagreen"),
    106: ("Ekstraklasa", "lightpink"),
    39: ("Premier League", "blue"),
    136: ("Serie B", "lightgreen"),
    62: ("Ligue 2", "mediumorchid"),
    135: ("Serie A", "limegreen"),
    179: ("Premiership", "lavender"),
    88: ("Eredivisie", "orange"),
    218: ("Austrian Bundesliga", "lavenderblush"),
    207: ("Swiss Super League", "paleturquoise"),
    94: ("Primeira Liga", "honeydew"),
}

ATTRIBUTES = [
    "home_team_id",
    "away_team_id",
    "home_team_goals",
    "away_team_goals",
    "home_team_points",
    "away_team_points",
    "home_team_xg",
    "away_team_xg",
    "total_xg",
    "home_team_pre_match_xg",
    "away_team_pre_match_xg",
    "total_pre_match_xg",
    "home_team_shots_on_target",
    "away_team_shots_on_target",
    "home_team_total_shots",
    "away_team_total_shots",
    "home_team_shots_inside_box",
    "away_team_shots_inside_box",
    "home_team_corner_kicks",
    "away_team_corner_kicks",
    "home_team_ball_possession",
    "away_team_ball_possession",
    "home_team_passes_acc",
    "away_team_passes_acc",
]


# If you want to *exclude* comp+round groups with too few matches,
# set this to e.g. 10 or 20. Default 0 = keep all.
MIN_ROWS_PER_GROUP = 0


def parse_stats_text(path: str) -> pd.DataFrame:
    """
    Parse the text output produced by step1_compute_missing_stats.py
    and return a DataFrame with:
      comp_name, comp_id, round_group, attribute, valid, total, percent
    """
    header_re = re.compile(
        r"^===\s+(?P<comp_name>.+?)\s+\(comp_id=(?P<comp_id>\d+)\),\s+round_group:\s+(?P<round_group>.+?)\s+===$"
    )
    attr_re = re.compile(r"^(?P<attr>[^:]+):\s+(?P<valid>\d+)/(?P<total>\d+)\s+\((?P<percent>[\d\.]+)%\)")

    records = []
    comp_name = None
    comp_id = None
    round_group = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m_header = header_re.match(line)
            if m_header:
                comp_name = m_header.group("comp_name")
                comp_id = int(m_header.group("comp_id"))
                round_group = m_header.group("round_group")
                continue

            m_attr = attr_re.match(line)
            if m_attr:
                records.append(
                    {
                        "comp_name": comp_name,
                        "comp_id": comp_id,
                        "round_group": round_group,
                        "attribute": m_attr.group("attr"),
                        "valid": int(m_attr.group("valid")),
                        "total": int(m_attr.group("total")),
                        "percent": float(m_attr.group("percent")),
                    }
                )

    return pd.DataFrame(records)


def make_heatmap(df_stats: pd.DataFrame):
    """
    Heatmap: rows = competition + round_group, columns = attributes,
    values = % of non-missing values.

    Extra features:
    1) y-labels include total n of matches per comp+round group.
    2) cells with <97.5% are annotated with the percentage.
    3) y-labels colored by competition color with dark background box.
    4) ability to filter out groups with too few matches via MIN_ROWS_PER_GROUP.
    """
    # build comp_round label
    df_stats["comp_round"] = df_stats["comp_name"] + " – " + df_stats["round_group"].astype(str)

    # total matches per comp_round (same "total" for all attrs in group,
    # so max/mean are equivalent)
    totals_by_comp_round = df_stats.groupby("comp_round")["total"].max()  # Series: comp_round -> total rows

    # optional filtering by MIN_ROWS_PER_GROUP
    if MIN_ROWS_PER_GROUP > 0:
        valid_comp_rounds = totals_by_comp_round[totals_by_comp_round >= MIN_ROWS_PER_GROUP].index
        df_stats = df_stats[df_stats["comp_round"].isin(valid_comp_rounds)].copy()
        totals_by_comp_round = totals_by_comp_round.loc[valid_comp_rounds]

    # comp_round -> comp_id (for coloring y-labels)
    comp_round_to_id = df_stats.groupby("comp_round")["comp_id"].first()

    pivot = df_stats.pivot_table(index="comp_round", columns="attribute", values="percent")
    pivot = pivot.reindex(columns=ATTRIBUTES)

    fig, ax = plt.subplots(figsize=(len(ATTRIBUTES) * 0.6, max(6, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=100, cmap="viridis")

    # x ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)

    # y ticks with color + "n=" info
    ax.set_yticks(range(len(pivot.index)))
    y_labels = []
    for comp_round in pivot.index:
        n = totals_by_comp_round.get(comp_round, np.nan)
        if pd.isna(n):
            label_text = comp_round
        else:
            label_text = f"{comp_round} (n={int(n)})"
        y_labels.append(label_text)
    ax.set_yticklabels(y_labels, fontsize=8)

    # color each y-tick label according to competition color and give it
    # a dark background box for better contrast
    for i, label in enumerate(ax.get_yticklabels()):
        comp_round = pivot.index[i]
        comp_id = comp_round_to_id.get(comp_round)
        color = "white"
        if comp_id in COMP_INFO:
            color = COMP_INFO[comp_id][1]

        label.set_color(color)
        # label.set_bbox(dict(facecolor="black", edgecolor="none", alpha=0.7, pad=1))

    ax.set_xlabel("Attribute")
    ax.set_ylabel("Competition – Round group (n = number of matches)")
    ax.set_title("Percentage of non-missing values per attribute / competition / round group")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% non-missing values")

    # annotate cells with % if < 95.0
    data = pivot.values
    nrows, ncols = data.shape
    for i in range(nrows):
        for j in range(ncols):
            val = data[i, j]
            if np.isnan(val):
                continue
            if val < 95.0:
                txt_color = "white" if val < 80 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_color,
                )

    plt.tight_layout()
    plt.savefig(FIG_HEATMAP_PATH, dpi=200)
    # plt.show()


def make_competition_bar_chart(df_stats: pd.DataFrame):
    """
    Bar chart of average completeness per competition (across all groups+attributes).

    Extra features:
    - y-axis scaled between 90 and 100
    - labels with actual values on top of each bar
    """
    comp_means = df_stats.groupby(["comp_id", "comp_name"])["percent"].mean().sort_values(ascending=False)

    comp_ids = [idx[0] for idx in comp_means.index]
    comp_names = [idx[1] for idx in comp_means.index]
    values = comp_means.values

    colors = []
    for cid in comp_ids:
        info = COMP_INFO.get(cid)
        colors.append(info[1] if info is not None else "gray")

    fig, ax = plt.subplots(figsize=(max(8, len(comp_names) * 0.5), 5))
    ax.bar(range(len(comp_names)), values, color=colors)

    ax.set_xticks(range(len(comp_names)))
    ax.set_xticklabels(comp_names, rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("Average % non-missing values")
    ax.set_title("Average match statistics completeness by competition")

    # y-axis between 90 and 100
    ax.set_ylim(90, 100)

    # add value labels on top of each bar
    for i, v in enumerate(values):
        ax.text(
            i,
            v + 0.1,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(FIG_BAR_PATH, dpi=200)
    # plt.show()


def main():
    df_stats = parse_stats_text(STATS_TEXT_PATH)
    if df_stats.empty:
        raise ValueError("No data parsed from stats text output.")

    make_heatmap(df_stats)
    make_competition_bar_chart(df_stats)


if __name__ == "__main__":
    main()
