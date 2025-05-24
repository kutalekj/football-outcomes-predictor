import csv
from datetime import datetime
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('TkAgg')


def plot_winning_graph(records_file):

    # Load and preprocess data
    df = pd.read_csv(records_file, encoding="utf-8")
    df["match_start_datetime_utc"] = pd.to_datetime(df["match_start_datetime_utc"], format="%Y-%m-%d %H:%M UTC")
    df = df.sort_values(by="match_start_datetime_utc").reset_index(drop=True)

    df["cumulative_bet"] = df["bet_placed"].astype(float).cumsum()
    df["cumulative_won"] = df["won"].astype(float).cumsum()

    x = list(range(len(df)))  # index as x-axis
    x_labels = df["match_start_datetime_utc"].dt.strftime("%Y-%m-%d")

    model_updates = [datetime(2025, 3, 27), datetime(2025, 3, 30), datetime(2025, 4, 4), datetime(2025, 4, 7),
                     datetime(2025, 4, 11), datetime(2025, 4, 18), datetime(2025, 4, 21), datetime(2025, 4, 26),
                     datetime(2025, 4, 29), datetime(2025, 5, 5), datetime(2025, 5, 11), datetime(2025, 5, 17),
                     datetime(2025, 5, 22)]

    # Plot
    plt.figure(figsize=(13, 8))
    plt.plot(x, df["cumulative_bet"], label="Bet", color="red")
    plt.plot(x, df["cumulative_won"], label="Won", color="green")

    # Color area between curves
    for i in range(1, len(df)):
        x_range = [x[i - 1], x[i]]
        y1 = df["cumulative_bet"].iloc[i - 1:i + 1]
        y2 = df["cumulative_won"].iloc[i - 1:i + 1]
        if y2.iloc[1] > y1.iloc[1]:
            plt.fill_between(x_range, y1, y2, color="lawngreen", alpha=0.3)
        else:
            plt.fill_between(x_range, y2, y1, color="orangered", alpha=0.3)

    # Alternate background shading by day
    df["date_only"] = df["match_start_datetime_utc"].dt.date
    unique_dates = df["date_only"].unique()  # group by date for day separation

    shade = True
    for date in unique_dates:
        day_indices = df.index[df["date_only"] == date].tolist()
        if not day_indices:
            continue
        start_idx = day_indices[0]
        end_idx = day_indices[-1]
        if shade:
            plt.axvspan(start_idx - 0.5, end_idx + 0.5, facecolor="darkgray", alpha=0.2, zorder=0)
        shade = not shade

    # Add vertical lines and segment progress labels
    for update in model_updates:
        idx = df["match_start_datetime_utc"].searchsorted(update)
        if 0 <= idx < len(df):
            plt.axvline(x=idx, color="gray", linestyle="--", linewidth=1.7)
            diff = df["cumulative_won"].iloc[idx] - df["cumulative_bet"].iloc[idx]
            color = "green" if diff >= 0 else "red"
            plt.text(idx + 0.5, df[["cumulative_bet", "cumulative_won"]].max().max() * 0.95, f"{diff:+.2f}",
                     color=color, rotation=90, verticalalignment="top", fontsize=10)

    # Add final state label
    final_idx = len(df) - 1
    final_diff = df["cumulative_won"].iloc[final_idx] - df["cumulative_bet"].iloc[final_idx]
    final_perc_gain = final_diff / df["cumulative_bet"].iloc[final_idx]
    final_color = "green" if final_diff >= 0 else "red"
    plt.text(final_idx, max(df["cumulative_bet"].iloc[final_idx], df["cumulative_won"].iloc[final_idx]) + 450,
             f"{final_diff:+.2f} ({final_perc_gain:+.2%})", ha="center", fontsize=14, color=final_color)

    # Total number of bets label
    total_bets = len(df)
    plt.text(0.99, 0.05, f"Total bets: {total_bets}",
             transform=plt.gca().transAxes, ha="right", va="bottom", fontsize=12, color="black")
    plt.text(0.99, 0.01, f"Total spent: {df['cumulative_bet'].iloc[final_idx]:.2f}",
             transform=plt.gca().transAxes, ha="right", va="bottom", fontsize=10, color="black")

    # Formatting axes and labels
    tick_step = max(1, len(x) // 10)
    plt.xticks(ticks=x[::tick_step], labels=x_labels[::tick_step], rotation=45, ha="right")
    plt.xlabel("Date of match played", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.title("Cumulative sums of bet and won values", fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Print summary plots based on records CSV")
    parser.add_argument("input_path", type=Path, help="Path to the input CSV file with records")
    args = parser.parse_args()
    plot_winning_graph(args.input_path)
