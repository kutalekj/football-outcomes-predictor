import csv
import sys

KEEP_COLS = [
    "player_id",
    "version",
    "name",
    "full_name",
    "dob",
    "club_id",
    "club_name",
    "club_league_id",
    "club_league_name",
    "country_id",
    "country_name",
    "country_league_id",
    "country_league_name",
]


def load_by_id(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        data = {}
        for row in reader:
            pid = (row.get("player_id") or "").strip()
            if not pid:
                continue
            data[pid] = row
        return data


def write_subset(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEEP_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k, "") if r.get(k, "") is not None else "") for k in KEEP_COLS})


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <first.csv> <second.csv>")
        sys.exit(1)

    first_path, second_path = sys.argv[1], sys.argv[2]

    a = load_by_id(first_path)
    b = load_by_id(second_path)

    only_a_ids = sorted(set(a.keys()) - set(b.keys()), key=int)
    only_b_ids = sorted(set(b.keys()) - set(a.keys()), key=int)

    only_a_rows = [a[pid] for pid in only_a_ids]
    only_b_rows = [b[pid] for pid in only_b_ids]

    write_subset("only_in_first.csv", only_a_rows)
    write_subset("only_in_second.csv", only_b_rows)

    print(f"only_in_first.csv:  {len(only_a_rows)} rows")
    print(f"only_in_second.csv: {len(only_b_rows)} rows")


if __name__ == "__main__":
    main()
