from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from football_outcomes.application.snapshot_selection import (
    resolve_snapshot_path,
)
from football_outcomes.config import fs_settings as sett
from football_outcomes.config.fs_globals import Global
from football_outcomes.data.snapshots import try_load_snapshot
from football_outcomes.data.state import (
    apply_bundle_to_global,
)

matplotlib.use("Agg")

EXPECTED_FS_POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
EXPECTED_SOFIFA_POSITIONS = [
    "GK",
    "CB",
    "LB",
    "RB",
    "LWB",
    "RWB",
    "CDM",
    "CM",
    "CAM",
    "LM",
    "RM",
    "LW",
    "RW",
    "CF",
    "ST",
]


def normalize_fs_position(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return s


def split_sofifa_positions(value: object) -> list[str]:
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def write_counter_csv(counter: Counter, out_path: Path, key_name: str, value_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([key_name, value_name])
        for key, val in counter.most_common():
            writer.writerow([key, val])


def analyze_fs_positions(out_dir: Path) -> None:
    cache = try_load_snapshot(resolve_snapshot_path())
    if cache is None:
        raise RuntimeError("No cached snapshot available.")

    apply_bundle_to_global(cache)
    g = Global.get_instance()

    # Try a few common collection names safely
    fs_players = []
    for attr in ["all_players", "players", "fs_players"]:
        if hasattr(g, attr):
            candidate = getattr(g, attr)
            if isinstance(candidate, (list, tuple, set)):
                fs_players = list(candidate)
                break
            if isinstance(candidate, dict):
                fs_players = list(candidate.values())
                break

    if not fs_players:
        raise RuntimeError("Could not find a player collection in Global instance.")

    total = len(fs_players)
    pos_counter = Counter()
    unexpected_counter = Counter()
    missing = 0

    for p in fs_players:
        pos = normalize_fs_position(getattr(p, "position", ""))

        if not pos:
            missing += 1
            continue

        pos_counter[pos] += 1
        if pos not in EXPECTED_FS_POSITIONS:
            unexpected_counter[pos] += 1

    # Save summaries
    summary_path = out_dir / "fs_positions_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Total FS players: {total}\n")
        f.write(f"Missing position: {missing} ({missing / total:.2%})\n")
        f.write(f"Non-missing position: {total - missing} ({(total - missing) / total:.2%})\n\n")
        f.write("Counts of expected FS positions:\n")
        for pos in EXPECTED_FS_POSITIONS:
            f.write(f"  {pos}: {pos_counter.get(pos, 0)}\n")
        f.write("\nUnexpected FS position values:\n")
        if unexpected_counter:
            for pos, cnt in unexpected_counter.most_common():
                f.write(f"  {pos}: {cnt}\n")
        else:
            f.write("  none\n")

    write_counter_csv(pos_counter, out_dir / "fs_position_counts.csv", "fs_position", "count")
    write_counter_csv(unexpected_counter, out_dir / "fs_position_unexpected.csv", "unexpected_fs_position", "count")

    # Plot expected FS positions
    vals = [pos_counter.get(pos, 0) for pos in EXPECTED_FS_POSITIONS]
    plt.figure(figsize=(8, 4.5))
    plt.bar(EXPECTED_FS_POSITIONS, vals)
    plt.title("FootyStats player positions")
    plt.xlabel("Position")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "fs_position_counts.png", dpi=200)
    plt.savefig(out_dir / "fs_position_counts.pdf")
    plt.close()


def analyze_sofifa_positions(snapshot_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime(2021, 5, 1)
    end_date = datetime(2025, 5, 1)

    total_players = 0
    missing_positions = 0

    position_counter = Counter()
    num_positions_per_player = Counter()

    snapshot_files = sorted(snapshot_dir.glob("*.csv"))

    for csv_path in snapshot_files:
        # --- FILTER BY DATE ---
        try:
            date = datetime.strptime(csv_path.stem, "%Y-%m-%d")
        except ValueError:
            print(f"[SKIP] Invalid filename format: {csv_path.name}")
            continue

        if not (start_date <= date <= end_date):
            continue

        print(f"Processing {csv_path.name}")

        try:
            df = pd.read_csv(
                csv_path,
                engine="python",
                on_bad_lines="warn",
            )
        except Exception as e:
            print(f"[ERROR] Failed to parse {csv_path.name}: {e}")
            continue

        if "positions" not in df.columns:
            print(f"[WARNING] No 'positions' column in {csv_path.name}")
            continue

        for pos_str in df["positions"]:
            total_players += 1

            if pd.isna(pos_str) or pos_str.strip() == "":
                missing_positions += 1
                continue

            positions = [p.strip() for p in pos_str.split(",") if p.strip()]

            num_positions_per_player[len(positions)] += 1

            for p in positions:
                position_counter[p] += 1

    # SAVE SUMMARY
    summary_path = out_dir / "sofifa_positions_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Total player entries: {total_players}\n")
        f.write(f"Missing positions: {missing_positions} " f"({missing_positions / total_players:.2%})\n\n")

        f.write("Position counts:\n")
        for pos, count in position_counter.most_common():
            f.write(f"  {pos}: {count}\n")

        f.write("\nPositions per player distribution:\n")
        for k, v in sorted(num_positions_per_player.items()):
            f.write(f"  {k} positions: {v}\n")

    # SAVE CSV (GLOBAL COUNTS)
    df_counts = pd.DataFrame([{"position": p, "count": c} for p, c in position_counter.items()]).sort_values(
        "count", ascending=False
    )

    df_counts.to_csv(out_dir / "sofifa_position_counts.csv", index=False)

    # SAVE NORMALIZED DISTRIBUTION
    total_positions = sum(position_counter.values())
    df_counts["percentage"] = df_counts["count"] / total_positions

    df_counts.to_csv(out_dir / "sofifa_position_distribution.csv", index=False)

    print(f"Saved SoFIFA summary to: {out_dir}")


def main() -> None:
    sofifa_snapshot_dir = Path(sett.SOFIFA_CSV_DIR)

    out_dir = Path(sett.PROJECT_ROOT) / "docs" / "experiments" / "thesis_missing_player_positions"
    out_dir.mkdir(parents=True, exist_ok=True)

    analyze_fs_positions(out_dir / "footystats")
    analyze_sofifa_positions(sofifa_snapshot_dir, out_dir / "sofifa")

    print(f"Saved analysis outputs to: {out_dir}")


if __name__ == "__main__":
    main()
