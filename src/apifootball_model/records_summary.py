import csv
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as m_dates
import numpy as np

plt.switch_backend('TkAgg')


def plot_winning_graph(records_file):
    model_updates = ["2025-03-27 00:00 UTC", "2025-03-30 00:00 UTC", "2025-04-04 00:00 UTC"]

    # Load records
    values = []
    with open(records_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append((datetime.strptime(row["match_start_datetime_utc"], "%Y-%m-%d %H:%M UTC"),
                           float(row["bet_placed"]), float(row["won"])))
    values.sort(key=lambda x: x[0])  # sort by datetime

    # Plot
    x = [x[0] for x in values]
    y_bet = np.cumsum([x[1] for x in values])
    y_won = np.cumsum([x[2] for x in values])

    plt.gca().xaxis.set_major_formatter(m_dates.DateFormatter("%Y-%m-%d"))
    plt.plot(x, y_bet, label="Bet")
    plt.plot(x, y_won, label="Won")
    plt.gcf().autofmt_xdate()
    for update in model_updates:
        plt.axvline(datetime.strptime(update, "%Y-%m-%d %H:%M UTC"), color='gray', linestyle='--', linewidth=1,
                    label='model update' if update == model_updates[0] else None)

    plt.title("Cumulative sums of bet and won values")
    plt.xlabel("Date of match played")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Print summary plots based on records CSV")
    parser.add_argument("input_path", type=Path, help="Path to the input CSV file with records")
    args = parser.parse_args()
    plot_winning_graph(args.input_path)
