import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# CONFIG: INPUT FILE
# -----------------------------
TXT_PATH = Path("mp_out.txt")  # adjust if needed


# -----------------------------
# COMPETITION METADATA & COLORS
# -----------------------------
COMP_INFO = {
    39: {
        "name": "Premier League",
        "fs_alias": "Premier League",
    },
    40: {
        "name": "Championship",
        "fs_alias": "Championship",
    },
    41: {
        "name": "League One",
        "fs_alias": "EFL League One",
    },
    42: {
        "name": "League Two",
        "fs_alias": "EFL League Two",
    },
    61: {
        "name": "Ligue 1",
        "fs_alias": "Ligue 1",
    },
    62: {
        "name": "Ligue 2",
        "fs_alias": "Ligue 2",
    },
    78: {
        "name": "Bundesliga",
        "fs_alias": "Bundesliga",
    },
    79: {
        "name": "2. Bundesliga",
        "fs_alias": "2. Bundesliga",
    },
    88: {
        "name": "Eredivisie",
        "fs_alias": "Eredivisie",
    },
    94: {
        "name": "Primeira Liga",
        "fs_alias": "Liga NOS",
    },
    106: {
        "name": "Ekstraklasa",
        "fs_alias": "Ekstraklasa",
    },
    119: {
        "name": "Superliga",
        "fs_alias": "Superliga",
    },
    135: {
        "name": "Serie A",
        "fs_alias": "Serie A",
    },
    136: {
        "name": "Serie B",
        "fs_alias": "Serie B",
    },
    140: {
        "name": "La Liga",
        "fs_alias": "La Liga",
    },
    141: {
        "name": "Segunda División",
        "fs_alias": "Segunda División",
    },
    144: {
        "name": "Jupiler Pro League",
        "fs_alias": "Pro League",
    },
    179: {
        "name": "Premiership",
        "fs_alias": "Premiership",
    },
    188: {
        "name": "A-League",
        "fs_alias": "A-League",
    },
    203: {
        "name": "Süper Lig",
        "fs_alias": "Süper Lig",
    },
    207: {
        "name": "Super League",
        "fs_alias": "Super League",
    },
    218: {
        "name": "Bundesliga",
        "fs_alias": "Bundesliga",
    },
    307: {
        "name": "Pro League",
        "fs_alias": "Professional League",
    },
    323: {
        "name": "Indian Super League",
        "fs_alias": "Indian Super League",
    },
}

# Consistent colors per competition (by ID)
COMP_COLORS = {
    141: "tomato",  # Segunda División
    42: "dodgerblue",  # League Two
    41: "cornflowerblue",  # League One
    188: "silver",  # A-League
    323: "darkkhaki",  # Indian Super League
    79: "goldenrod",  # 2. Bundesliga
    119: "thistle",  # Danish Superliga
    78: "gold",  # Bundesliga
    144: "aquamarine",  # Jupiler Pro League
    40: "royalblue",  # Championship
    61: "blueviolet",  # Ligue 1
    140: "red",  # La Liga
    203: "rosybrown",  # Süper Lig
    307: "darkseagreen",  # Saudi Pro League
    106: "lightpink",  # Ekstraklasa
    39: "blue",  # Premier League
    136: "lightgreen",  # Serie B
    62: "mediumorchid",  # Ligue 2
    135: "limegreen",  # Serie A
    179: "lavender",  # Premiership
    88: "orange",  # Eredivisie
    218: "lavenderblush",  # Austrian Bundesliga
    207: "paleturquoise",  # Swiss Super League
    94: "honeydew",  # Primeira Liga
}


# -----------------------------
# PARSING mp_out.txt
# -----------------------------
def parse_mp_out(path: Path):
    """
    Parse mp_out.txt into a dict of dicts:
    {
        "mp3_all_players_involved_in_team_strength_calculation": {comp_id: value, ...},
        "mp5_team_strength_DOB_missing": {...},
        ...
    }
    """
    metrics = {}
    current_key = None

    key_pattern = re.compile(r"^(mp[0-9a-zA-Z_]+):\s*$")
    entry_pattern = re.compile(r"^\s+(\d+):\s+(\d+)\s*$")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # New metric header
            m_key = key_pattern.match(line)
            if m_key:
                current_key = m_key.group(1)
                metrics[current_key] = {}
                continue

            # Entries under current metric
            if current_key is not None:
                m_entry = entry_pattern.match(line)
                if m_entry:
                    comp_id = int(m_entry.group(1))
                    value = int(m_entry.group(2))
                    metrics[current_key][comp_id] = value

    return metrics


metrics = parse_mp_out(TXT_PATH)

# Convenience references
mp3 = metrics["mp3_all_players_involved_in_team_strength_calculation"]
mp5 = metrics["mp5_team_strength_DOB_missing"]
mp6 = metrics["mp6_team_strength_FS_SF_matching"]
mp7 = metrics["mp7_team_strength_imitated_skills_as_no_CSV_data"]
mp8b = metrics["mp8b_team_strength_imitated_players_as_no_CSV_data"]
mp9_field_to_gk = metrics["mp9_team_strength_balancing_field_to_gk"]
mp9_gk_to_def = metrics["mp9_team_strength_balancing_gk_to_def"]
mp9_gk_to_mid = metrics["mp9_team_strength_balancing_gk_to_mid"]
mp9_gk_to_att = metrics["mp9_team_strength_balancing_gk_to_att"]

# Use mp3 as the master set of competitions and a consistent order
comp_ids = sorted(mp3.keys())


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_comp_label(comp_id: int) -> str:
    """Use the competition 'name' for axis labels."""
    info = COMP_INFO.get(comp_id, {})
    return info.get("name", str(comp_id))


def annotate_bar(ax, x, height, text, y_offset_factor=0.01, fontsize=8, color="black"):
    """Annotate a single bar value above the bar, horizontally."""
    if height <= 0:
        return
    ylim_top = ax.get_ylim()[1]
    ax.text(
        x,
        height + y_offset_factor * ylim_top,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=color,
        rotation=0,
    )


def annotate_segment_center(ax, x, bottom, height, text, fontsize=7, color="white"):
    """Annotate inside a stacked segment (e.g., missing part), horizontally."""
    if height <= 0:
        return
    y = bottom + height / 2
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        rotation=0,
    )


def annotate_total_sum(ax, total_value: int, fontsize=10):
    """Annotate a single sum value somewhere at the top part of the graph."""
    ax.text(
        0.99,
        0.97,
        f"Total = {total_value:,}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
    )


def setup_axes(ax, title: str):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(comp_ids)))
    ax.set_xticklabels(
        [get_comp_label(cid) for cid in comp_ids],
        rotation=45,  # 45 degrees clockwise
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Number of players")
    ax.grid(axis="y", linestyle="--", alpha=0.3)


# -----------------------------
# 1. Total players per competition (mp3)
# -----------------------------
def plot_chart_1():
    fig, ax = plt.subplots(figsize=(14, 6))

    totals = [mp3[cid] for cid in comp_ids]
    x = np.arange(len(comp_ids))

    ax.bar(
        x,
        totals,
        color=[COMP_COLORS.get(cid, "gray") for cid in comp_ids],
        edgecolor="black",
    )

    # Ensure a bit of headroom for annotations
    max_val = max(totals) if totals else 0
    ax.set_ylim(0, max_val * 1.1 if max_val > 0 else 1)

    for xi, h in zip(x, totals):
        annotate_bar(ax, xi, h, str(h))

    setup_axes(ax, "1) Total players involved in team strength calculation (mp3)")
    annotate_total_sum(ax, sum(totals))

    fig.tight_layout()
    return fig


# -----------------------------
# 2. Dates of birth missing vs total (mp5 vs mp3) – stacked
# -----------------------------
def plot_chart_2():
    fig, ax = plt.subplots(figsize=(14, 6))

    total = [mp3[cid] for cid in comp_ids]
    missing_dob = [mp5.get(cid, 0) for cid in comp_ids]
    known = [t - m for t, m in zip(total, missing_dob)]

    x = np.arange(len(comp_ids))

    ax.bar(
        x,
        known,
        color=[COMP_COLORS.get(cid, "gray") for cid in comp_ids],
        edgecolor="black",
        label="Date of birth known",
    )
    ax.bar(
        x,
        missing_dob,
        bottom=known,
        color="black",
        edgecolor="black",
        label="Dates of birth missing (mp5)",
    )

    max_val = max(total) if total else 0
    ax.set_ylim(0, max_val * 1.1 if max_val > 0 else 1)

    for xi, k, m, t in zip(x, known, missing_dob, total):
        annotate_bar(ax, xi, t, str(t))
        if m > 0:
            annotate_segment_center(ax, xi, k, m, str(m), color="white")

    setup_axes(ax, "2) Players with dates of birth missing vs total (mp5 vs mp3)")
    annotate_total_sum(ax, sum(total))
    ax.legend()

    fig.tight_layout()
    return fig


# -----------------------------
# 3. FS–SF matching misses vs total (mp6 vs mp3) – stacked
# -----------------------------
def plot_chart_3():
    fig, ax = plt.subplots(figsize=(14, 6))

    total = [mp3[cid] for cid in comp_ids]
    missing_match = [mp6.get(cid, 0) for cid in comp_ids]
    known = [t - m for t, m in zip(total, missing_match)]

    x = np.arange(len(comp_ids))

    ax.bar(
        x,
        known,
        color=[COMP_COLORS.get(cid, "gray") for cid in comp_ids],
        edgecolor="black",
        label="Matched players",
    )
    ax.bar(
        x,
        missing_match,
        bottom=known,
        color="black",
        edgecolor="black",
        label="FS–SF name matching misses (mp6)",
    )

    max_val = max(total) if total else 0
    ax.set_ylim(0, max_val * 1.1 if max_val > 0 else 1)

    for xi, k, m, t in zip(x, known, missing_match, total):
        annotate_bar(ax, xi, t, str(t))
        if m > 0:
            annotate_segment_center(ax, xi, k, m, str(m), color="white")

    setup_axes(ax, "3) FS–SF matching misses vs total (mp6 vs mp3)")
    annotate_total_sum(ax, sum(total))
    ax.legend()

    fig.tight_layout()
    return fig


# -----------------------------
# 4. Combined dates-of-birth + matching misses vs total (mp5 + mp6 vs mp3)
# -----------------------------
def plot_chart_4():
    fig, ax = plt.subplots(figsize=(14, 6))

    total = [mp3[cid] for cid in comp_ids]
    missing_dob = [mp5.get(cid, 0) for cid in comp_ids]
    missing_match = [mp6.get(cid, 0) for cid in comp_ids]

    missing_total = [d + m for d, m in zip(missing_dob, missing_match)]
    known = [t - mt for t, mt in zip(total, missing_total)]

    x = np.arange(len(comp_ids))

    ax.bar(
        x,
        known,
        color=[COMP_COLORS.get(cid, "gray") for cid in comp_ids],
        edgecolor="black",
        label="Players used",
    )
    ax.bar(
        x,
        missing_dob,
        bottom=known,
        color="dimgray",
        edgecolor="black",
        label="Dates of birth missing (mp5)",
    )
    bottom_after_dob = [k + d for k, d in zip(known, missing_dob)]
    ax.bar(
        x,
        missing_match,
        bottom=bottom_after_dob,
        color="darkgray",
        edgecolor="black",
        label="FS–SF matching miss (mp6)",
    )

    max_val = max(total) if total else 0
    # a bit more headroom for annotations to avoid overlap with bar tops
    ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1)

    for xi, k, d, m, t in zip(x, known, missing_dob, missing_match, total):
        # Slightly larger upward offset for totals on this crowded stacked chart
        annotate_bar(ax, xi, t, str(t), y_offset_factor=0.02)
        if d > 0:
            annotate_segment_center(ax, xi, k, d, str(d), color="white")
        if m > 0:
            annotate_segment_center(ax, xi, k + d, m, str(m), color="black")

    setup_axes(
        ax,
        "4) Combined missing players: dates of birth + matching (mp5 + mp6) vs total (mp3)",
    )
    annotate_total_sum(ax, sum(total))
    ax.legend()

    fig.tight_layout()
    return fig


# -----------------------------
# 5. Imputed players (mp8b) vs total (mp3) – stacked
# -----------------------------
def plot_chart_5():
    fig, ax = plt.subplots(figsize=(14, 6))

    total = [mp3[cid] for cid in comp_ids]
    imputed = [mp8b.get(cid, 0) for cid in comp_ids]
    non_imputed = [t - i for t, i in zip(total, imputed)]

    x = np.arange(len(comp_ids))

    ax.bar(
        x,
        non_imputed,
        color=[COMP_COLORS.get(cid, "gray") for cid in comp_ids],
        edgecolor="black",
        label="Non-imputed players",
    )
    ax.bar(
        x,
        imputed,
        bottom=non_imputed,
        color="black",
        edgecolor="black",
        label="Imputed players (mp8b)",
    )

    max_val = max(total) if total else 0
    ax.set_ylim(0, max_val * 1.1 if max_val > 0 else 1)

    for xi, n, i, t in zip(x, non_imputed, imputed, total):
        annotate_bar(ax, xi, t, str(t))
        if i > 0:
            annotate_segment_center(ax, xi, n, i, str(i), color="white")
            annotate_segment_center(ax, xi, n / 10, 100, f"{(i / t * 100):.2f}%", color="black")

    setup_axes(ax, "5) Imputed players due to missing data (mp8b) vs total (mp3)")
    annotate_total_sum(ax, sum(total))
    ax.legend()

    fig.tight_layout()
    return fig


# -----------------------------
# 6. Causes of missing data (mp5 + mp6 + mp7) vs total (mp3) – stacked
# -----------------------------
def plot_chart_6():
    fig, ax = plt.subplots(figsize=(14, 6))

    total = [mp3[cid] for cid in comp_ids]
    missing_dob = [mp5.get(cid, 0) for cid in comp_ids]
    missing_match = [mp6.get(cid, 0) for cid in comp_ids]
    missing_skill = [mp7.get(cid, 0) for cid in comp_ids]

    missing_total = [d + m + s for d, m, s in zip(missing_dob, missing_match, missing_skill)]
    used_players = [t - mt for t, mt in zip(total, missing_total)]

    x = np.arange(len(comp_ids))

    ax.bar(
        x,
        used_players,
        color=[COMP_COLORS.get(cid, "gray") for cid in comp_ids],
        edgecolor="black",
        label="Players used",
    )
    ax.bar(
        x,
        missing_dob,
        bottom=used_players,
        color="dimgray",
        edgecolor="black",
        label="Dates of birth missing (mp5)",
    )
    bottom_after_dob = [u + d for u, d in zip(used_players, missing_dob)]
    ax.bar(
        x,
        missing_match,
        bottom=bottom_after_dob,
        color="darkgray",
        edgecolor="black",
        label="FS–SF matching miss (mp6)",
    )
    bottom_after_match = [u + d + m for u, d, m in zip(used_players, missing_dob, missing_match)]
    ax.bar(
        x,
        missing_skill,
        bottom=bottom_after_match,
        color="white",
        edgecolor="black",
        label="No CSV in range (mp7)",
    )

    max_val = max(total) if total else 0
    ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1)

    for xi, u, d, m, s, t in zip(x, used_players, missing_dob, missing_match, missing_skill, total):
        annotate_bar(ax, xi, t, str(t), y_offset_factor=0.02)
        if d > 0:
            annotate_segment_center(ax, xi, u, d, str(d), color="white")
        if m > 0:
            annotate_segment_center(ax, xi, u + d, m, str(m), color="black")
        if s > 0:
            annotate_segment_center(ax, xi, u + d + m, s, str(s), color="black")

    setup_axes(
        ax,
        "6) Causes of missing data (dates of birth + matching + no CSV) vs total (mp3)",
    )
    annotate_total_sum(ax, sum(total))
    ax.legend()

    fig.tight_layout()
    return fig


# -----------------------------
# 7. Balancing between positions (mp9_*) – stacked, 4 colors, no mp3
# -----------------------------
def plot_chart_7():
    fig, ax = plt.subplots(figsize=(14, 6))

    # Values
    field_to_gk = [mp9_field_to_gk.get(cid, 0) for cid in comp_ids]
    gk_to_def = [mp9_gk_to_def.get(cid, 0) for cid in comp_ids]
    gk_to_mid = [mp9_gk_to_mid.get(cid, 0) for cid in comp_ids]
    gk_to_att = [mp9_gk_to_att.get(cid, 0) for cid in comp_ids]

    x = np.arange(len(comp_ids))

    # Custom colors NOT linked to competition colors
    c_field_to_gk = "#1f77b4"
    c_gk_to_def = "#ff7f0e"
    c_gk_to_mid = "#2ca02c"
    c_gk_to_att = "#d62728"

    ax.bar(
        x,
        field_to_gk,
        color=c_field_to_gk,
        edgecolor="black",
        label="field → GK",
    )
    bottom_after_f2g = field_to_gk
    ax.bar(
        x,
        gk_to_def,
        bottom=bottom_after_f2g,
        color=c_gk_to_def,
        edgecolor="black",
        label="GK → DEF",
    )
    bottom_after_g2d = [f + d for f, d in zip(field_to_gk, gk_to_def)]
    ax.bar(
        x,
        gk_to_mid,
        bottom=bottom_after_g2d,
        color=c_gk_to_mid,
        edgecolor="black",
        label="GK → MID",
    )
    bottom_after_g2m = [f + d + m for f, d, m in zip(field_to_gk, gk_to_def, gk_to_mid)]
    ax.bar(
        x,
        gk_to_att,
        bottom=bottom_after_g2m,
        color=c_gk_to_att,
        edgecolor="black",
        label="GK → ATT",
    )

    # Annotate each stacked segment and total per bar
    totals_per_bar = []
    for i, cid in enumerate(comp_ids):
        f2g = field_to_gk[i]
        g2d = gk_to_def[i]
        g2m = gk_to_mid[i]
        g2a = gk_to_att[i]
        total_bar = f2g + g2d + g2m + g2a
        totals_per_bar.append(total_bar)

        bottom = 0
        if f2g > 0:
            annotate_segment_center(ax, i, bottom, f2g, str(f2g), color="white")
        bottom += f2g
        if g2d > 0:
            annotate_segment_center(ax, i, bottom, g2d, str(g2d), color="white")
        bottom += g2d
        if g2m > 0:
            annotate_segment_center(ax, i, bottom, g2m, str(g2m), color="white")
        bottom += g2m
        if g2a > 0:
            annotate_segment_center(ax, i, bottom, g2a, str(g2a), color="white")

        if total_bar > 0:
            annotate_bar(ax, i, total_bar, str(total_bar))

    max_val = max(totals_per_bar) if totals_per_bar else 0
    ax.set_ylim(0, max_val * 1.1 if max_val > 0 else 1)

    ax.set_title(
        "7) Balancing between positions after imputation (mp9_* stacks)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(range(len(comp_ids)))
    ax.set_xticklabels(
        [get_comp_label(cid) for cid in comp_ids],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Number of players moved between positions")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    total_sum = sum(field_to_gk) + sum(gk_to_def) + sum(gk_to_mid) + sum(gk_to_att)
    annotate_total_sum(ax, total_sum)
    ax.legend()

    fig.tight_layout()
    return fig


# -----------------------------
# MAIN: generate all charts & SAVE to files
# -----------------------------
if __name__ == "__main__":
    figs_and_names = [
        (plot_chart_1(), "01_total_players_involved.png"),
        (plot_chart_2(), "02_dates_of_birth_missing_vs_total.png"),
        (plot_chart_3(), "03_matching_misses_vs_total.png"),
        (plot_chart_4(), "04_combined_missing_vs_total.png"),
        (plot_chart_5(), "05_imputed_players_vs_total.png"),
        (plot_chart_6(), "06_missing_causes_vs_total.png"),
        (plot_chart_7(), "07_position_balancing.png"),
    ]

    for fig, fname in figs_and_names:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
