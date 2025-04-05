import csv
from datetime import datetime
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as m_dates
import numpy as np

plt.switch_backend('TkAgg')


def plot_winning_graph(records_file):

    # Load the data
    df = pd.read_csv(records_file, encoding="utf-8")
    df["match_start_datetime_utc"] = pd.to_datetime(df["match_start_datetime_utc"], format="%Y-%m-%d %H:%M UTC")
    df = df.sort_values(by="match_start_datetime_utc").reset_index(drop=True)  # sort by datetime (asc.)

    df["cumulative_bet"] = df["bet_placed"].astype(float).cumsum()
    df["cumulative_won"] = df["won"].astype(float).cumsum()

    model_updates = [datetime(2025, 3, 27), datetime(2025, 3, 30), datetime(2025, 4, 4)]

    x = list(range(len(df)))
    x_labels = df["match_start_datetime_utc"].dt.strftime("%Y-%m-%d")  # x-axis as indices (to get linear spacing)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, df["cumulative_bet"], label="Bet", color="red")
    plt.plot(x, df["cumulative_won"], label="Won", color="green")

    for i in range(len(x)):  # color fill areas between curves
        if df["cumulative_won"].iloc[i] > df["cumulative_bet"].iloc[i]:
            plt.fill_between([x[i - 1], x[i]] if i > 0 else [x[i], x[i]],
                             df["cumulative_bet"].iloc[i - 1:i + 1] if i > 0 else [df["cumulative_bet"].iloc[i]],
                             df["cumulative_won"].iloc[i - 1:i + 1] if i > 0 else [df["cumulative_won"].iloc[i]],
                             color="lawngreen", alpha=0.3)
        else:
            plt.fill_between([x[i - 1], x[i]] if i > 0 else [x[i], x[i]],
                             df["cumulative_won"].iloc[i - 1:i + 1] if i > 0 else [df["cumulative_won"].iloc[i]],
                             df["cumulative_bet"].iloc[i - 1:i + 1] if i > 0 else [df["cumulative_bet"].iloc[i]],
                             color="orangered", alpha=0.3)

    for update in model_updates:  # add model update v-lines and segment annotations
        closest_index = df["match_start_datetime_utc"].searchsorted(update)
        if 0 <= closest_index < len(df):
            plt.axvline(x=closest_index, color="gray", linestyle="--", linewidth=1)
            diff = df["cumulative_won"].iloc[closest_index] - df["cumulative_bet"].iloc[closest_index]
            text_color = 'green' if diff >= 0 else 'red'
            plt.text(closest_index + 0.5,
                     df[["cumulative_bet", "cumulative_won"]].max().max() * 0.95,
                     f"{diff:+.2f}",
                     rotation=90,
                     verticalalignment="top",
                     fontsize=9,
                     color=text_color)

    final_idx = len(df["cumulative_bet"]) - 1  # final annotation
    final_diff = df["cumulative_won"][final_idx] - df["cumulative_bet"][final_idx]
    final_color = 'green' if final_diff >= 0 else 'red'

    plt.text(
        final_idx,
        max(df["cumulative_bet"][final_idx], df["cumulative_won"][final_idx]) + 70,
        f"{final_diff:+.2f}",
        ha="center",
        fontsize=14,
        color=final_color
    )

    plt.xticks(ticks=x[::max(1, len(x) // 10)], labels=x_labels[::max(1, len(x) // 10)], rotation=45, ha="right")
    plt.xlabel("Date of match played", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.title("Cumulative sums of bet and won values", fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Print summary plots based on records CSV")
    parser.add_argument("input_path", type=Path, help="Path to the input CSV file with records")
    args = parser.parse_args()
    plot_winning_graph(args.input_path)
