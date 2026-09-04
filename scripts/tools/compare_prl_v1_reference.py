from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _auc(y_true: pd.Series, probability: pd.Series) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true.astype(int), probability.astype(float)))


def _describe(name: str, frame: pd.DataFrame, probability_column: str) -> None:
    probability = frame[probability_column].astype(float)
    print(name)
    print(f"  rows: {len(frame)}")
    print(f"  rounds: {frame['round_index'].nunique()} ({frame['round_index'].min()}..{frame['round_index'].max()})")
    print(f"  probability mean: {probability.mean():.9f}")
    print(f"  probability std:  {probability.std(ddof=1):.9f}")
    print(f"  probability min:  {probability.min():.9f}")
    print(f"  probability max:  {probability.max():.9f}")
    print(f"  AUC: {_auc(frame['y_true'], probability)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Compare repaired PRL proposed-v1 OOS predictions with the " "historical selected-v1 reference.")
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    candidate = pd.read_csv(args.candidate_run / "predictions.csv")
    candidate = candidate.loc[candidate["model_name"].eq("proposed-v1")].copy()

    reference = reference.rename(
        columns={
            "round_idx": "round_index",
            "y_prob_under25": "reference_probability",
        }
    )
    candidate = candidate.rename(columns={"probability_under_2_5": "candidate_probability"})

    reference_columns = ["round_index", "match_id", "y_true", "reference_probability"]
    candidate_columns = ["round_index", "match_id", "y_true", "candidate_probability"]
    reference = reference[reference_columns].copy()
    candidate = candidate[candidate_columns].copy()

    _describe("REFERENCE", reference, "reference_probability")
    _describe("CANDIDATE", candidate, "candidate_probability")

    merged = candidate.merge(
        reference,
        on=["round_index", "match_id"],
        suffixes=("_candidate", "_reference"),
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError("No common candidate/reference prediction rows were found.")

    target_mismatches = int((merged["y_true_candidate"].astype(int) != merged["y_true_reference"].astype(int)).sum())
    delta = merged["candidate_probability"].astype(float) - merged["reference_probability"].astype(float)
    correlation = float(merged["candidate_probability"].corr(merged["reference_probability"]))

    print("COMMON")
    print(f"  rows: {len(merged)}")
    print(f"  candidate coverage: {len(merged) / len(candidate):.6f}")
    print(f"  reference coverage: {len(merged) / len(reference):.6f}")
    print(f"  target mismatches: {target_mismatches}")
    print(f"  probability correlation: {correlation:.9f}")
    print(f"  mean absolute probability difference: {np.abs(delta).mean():.9f}")
    print(f"  max absolute probability difference:  {np.abs(delta).max():.9f}")
    print(f"  candidate common AUC: {_auc(merged['y_true_candidate'], merged['candidate_probability'])}")
    print(f"  reference common AUC: {_auc(merged['y_true_reference'], merged['reference_probability'])}")

    by_round = []
    for round_index, group in merged.groupby("round_index", sort=True):
        by_round.append(
            {
                "round": int(round_index),
                "n": len(group),
                "candidate_mean": float(group["candidate_probability"].mean()),
                "candidate_std": float(group["candidate_probability"].std(ddof=1)),
                "reference_mean": float(group["reference_probability"].mean()),
                "reference_std": float(group["reference_probability"].std(ddof=1)),
                "candidate_auc": _auc(group["y_true_candidate"], group["candidate_probability"]),
                "reference_auc": _auc(group["y_true_reference"], group["reference_probability"]),
            }
        )

    round_frame = pd.DataFrame(by_round)
    print("\nPER-ROUND")
    print(round_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
