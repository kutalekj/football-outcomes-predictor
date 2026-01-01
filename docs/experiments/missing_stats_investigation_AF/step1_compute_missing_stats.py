import datetime

import pandas as pd

CSV_PATH = "m_25-11-23_full.csv"
OUTPUT_TEXT_PATH = "missing_stats_output.txt"

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

COMP_ROUND_LIST = [
    {"id": 39, "name": "Premier League", "regular_round_keywords": ["Regular Season"]},
    {"id": 40, "name": "Championship", "regular_round_keywords": ["Regular Season"]},
    {"id": 41, "name": "League One", "regular_round_keywords": ["Regular Season"]},
    {"id": 42, "name": "League Two", "regular_round_keywords": ["Regular Season"]},
    {"id": 61, "name": "Ligue 1", "regular_round_keywords": ["Regular Season"]},
    {"id": 62, "name": "Ligue 2", "regular_round_keywords": ["Regular Season"]},
    {"id": 78, "name": "Bundesliga", "regular_round_keywords": ["Regular Season"]},
    {"id": 79, "name": "2. Bundesliga", "regular_round_keywords": ["Regular Season"]},
    {"id": 88, "name": "Eredivisie", "regular_round_keywords": ["Regular Season"]},
    {"id": 94, "name": "Primeira Liga", "regular_round_keywords": ["Regular Season"]},
    {"id": 106, "name": "Ekstraklasa", "regular_round_keywords": ["Regular Season"]},  # POL
    {
        "id": 119,
        "name": "Superliga",
        "regular_round_keywords": ["Regular Season", "Championship Round", "Relegation Round"],
    },  # DEN
    {"id": 135, "name": "Serie A", "regular_round_keywords": ["Regular Season"]},
    {"id": 136, "name": "Serie B", "regular_round_keywords": ["Regular Season"]},
    {"id": 140, "name": "La Liga", "regular_round_keywords": ["Regular Season"]},
    {"id": 141, "name": "Segunda División", "regular_round_keywords": ["Regular Season"]},
    {
        "id": 144,
        "name": "Jupiler Pro League",
        "regular_round_keywords": [
            "Regular Season",
            "Championship Round",
            "Conference League Play-off Group",
        ],
    },  # BEL
    {
        "id": 179,
        "name": "Premiership",
        "regular_round_keywords": ["1st Phase", "Championship Round", "Relegation Round -"],
    },  # SCO
    {
        "id": 188,
        "name": "A-League",
        "regular_round_keywords": [
            "Regular Season",
            "Elimination Finals",
            "Semi-finals",
            "Grand Final",
        ],
    },  # AUS
    {"id": 203, "name": "Süper Lig", "regular_round_keywords": ["Regular Season"]},  # TUR
    {
        "id": 207,
        "name": "Super League",
        "regular_round_keywords": ["Regular Season", "Championship Round", "Relegation Round -"],
    },  # SUI
    {
        "id": 218,
        "name": "Bundesliga",
        "regular_round_keywords": ["Regular Season", "Championship Round", "Relegation Round -"],
    },  # AUT
    {"id": 307, "name": "Pro League", "regular_round_keywords": ["Regular Season"]},  # SA
    {
        "id": 323,
        "name": "Indian Super League",
        "regular_round_keywords": ["Regular Season", "Qualifying Finals", "Championship -"],
    },
]

# Turn list into a dict for easy lookup: comp_id -> config
COMP_ROUND_CONFIG = {c["id"]: c for c in COMP_ROUND_LIST}


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


def get_round_group(comp_id: int, round_name: str) -> str | None:
    """
    Map a raw round_name (e.g. 'Regular Season - 23') to a round *group*
    (e.g. 'Regular Season'), using COMP_ROUND_CONFIG[comp_id]['regular_round_keywords'].

    Returns the keyword used as group name, or None if no keyword matches.
    """
    cfg = COMP_ROUND_CONFIG.get(comp_id)
    if cfg is None:
        return None

    rn = str(round_name).lower()
    for kw in cfg["regular_round_keywords"]:
        if kw.lower() in rn:
            return kw
    return None


def get_comp_display_name(comp_id: int) -> str:
    """Use the display name from COMP_INFO."""
    info = COMP_INFO.get(comp_id)
    if info is None:
        return f"Comp {comp_id}"
    return info[0]


def print_data_summary(df: pd.DataFrame) -> None:
    df["round_group"] = df.apply(
        lambda row: (row["country"], row["comp_id"], row["season"]),
        axis=1,
    )

    matches = {}
    min_date = datetime.date(1970, 1, 1)
    max_date = datetime.date(2025, 12, 2)
    for index, row in df.iterrows():
        if row["round_group"] not in matches:
            matches[row["round_group"]] = (0, max_date, min_date)

        count, earliest, latest = matches[row["round_group"]]
        count += 1

        # row_date = datetime.datetime.strptime(row["datetime"], '%Y-%m-%d %H:%M').date()
        row_date = datetime.datetime.fromisoformat(row["datetime"]).date()
        if row_date < earliest:
            earliest = row_date
        if row_date > latest:
            latest = row_date

        matches[row["round_group"]] = (count, earliest, latest)

    # matches_sorted = sorted(matches, key=lambda x: (x[0], x[1], x[2]))

    # Turkey Turkish Cup (2025):    117 matches (from 2025-01-08 00:00:00+00:00 to 2025-10-30 00:00:00+00:00)
    out_list = []
    for k, v in matches.items():
        country, comp_id, season = k
        num_matches, min_d, max_d = v

        if country == "World":
            country = "Europe"
        out_line = (
            f"{country} {str(comp_id)} ({str(season)}):\t\t{str(num_matches)} matches "
            f"(from {str(min_d)} 00:00:00+00:00 to {str(max_d)} 00:00:00+00:00)"
        )
        out_list.append(out_line)

    out_list.sort()

    for line in out_list:
        print(line)

    pass


def main():
    df = pd.read_csv(CSV_PATH)

    print_data_summary(df)

    # keep only competitions that are in the color/name mapping
    valid_comp_ids = set(COMP_INFO.keys())
    df = df[df["comp_id"].isin(valid_comp_ids)].copy()

    # compute round group (purpose) for each row
    df["round_group"] = df.apply(
        lambda row: get_round_group(row["comp_id"], row["round_name"]),
        axis=1,
    )

    # drop matches where we couldn't map the round to any group (e.g. some cups)
    df = df[df["round_group"].notna()].copy()

    # sanity check: all attributes present
    missing_cols = [c for c in ATTRIBUTES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns missing in CSV: {missing_cols}")

    df_sorted = df.sort_values(["comp_id", "round_group"])

    lines: list[str] = []

    grouped = df_sorted.groupby(["comp_id", "round_group"], dropna=False)

    for (comp_id, round_group), group in grouped:
        comp_name = get_comp_display_name(comp_id)
        header = f"=== {comp_name} (comp_id={comp_id}), round_group: {round_group} ==="
        print(header)
        lines.append(header)

        total_rows = len(group)

        for attr in ATTRIBUTES:
            series = group[attr]

            # values >= 0 are treated as NOT missing; -1 or NaN = missing
            valid_mask = series >= 0
            valid_count = int(valid_mask.sum())
            total_count = total_rows
            percent = 100.0 * valid_count / total_count if total_count > 0 else 0.0

            line = f"{attr}: {valid_count}/{total_count} ({percent:.1f}%)"
            print(line)
            lines.append(line)

        print()
        lines.append("")

    if OUTPUT_TEXT_PATH:
        with open(OUTPUT_TEXT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    main()
