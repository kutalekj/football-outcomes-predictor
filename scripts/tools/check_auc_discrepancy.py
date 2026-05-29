from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import football_outcomes.config.fs_settings as sett
from football_outcomes.data.fs_io import load_snapshot
from football_outcomes.data.fs_retrieve import fill_globals_with_cache
from football_outcomes.training.fs_training_utils import build_categorical_maps
from football_outcomes.training.train_mlp_rolling import train_rolling
from scripts.main_footystats import prepare_matches, selected_model_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ATTA_COMPETITIONS = [
    "Scotland Premiership",
    "Belgium Pro League",
    "Netherlands Eredivisie",
]

GLOBAL_RUN_NAME = "integrity_check_global_24_selected_mlp_binary_u25"
ATTA_RUN_NAME = "integrity_check_atta3_selected_mlp_binary_u25"


def _load_explicit_snapshot_into_globals(snapshot_path: Path) -> None:
    snapshot_path = Path(snapshot_path)

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot path does not exist: {snapshot_path}")

    print(f"[load explicit snapshot] {snapshot_path}")
    bundle = load_snapshot(snapshot_path)
    fill_globals_with_cache(bundle, update_leagues_list=False)


def _summary_from_predictions(df: pd.DataFrame) -> dict:
    y_true = df["y_true"].astype(float).to_numpy()
    y_prob = df["y_prob_under25"].astype(float).to_numpy()
    y_pred = (y_prob >= 0.5).astype(float)

    out = {
        "n": int(len(df)),
        "positive_rate_under25": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
    }

    if len(np.unique(y_true)) < 2:
        out["auc"] = None
    else:
        out["auc"] = float(roc_auc_score(y_true, y_prob))

    return out


def _read_oos(run_name: str) -> pd.DataFrame:
    path = Path(sett.DATA_DIR) / "tensorboard_logs" / run_name / "oos_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing OOS predictions: {path}")
    return pd.read_csv(path)


def _read_saved_summary(run_name: str) -> dict:
    path = Path(sett.DATA_DIR) / "tensorboard_logs" / run_name / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_selected_mlp(matches, run_name: str) -> None:
    cat_maps = build_categorical_maps(matches)
    cfg = selected_model_config(run_name=run_name)

    # Keep this diagnostic check light; logs are still saved, but branch diagnostics are disabled.
    cfg.enable_branch_diagnostics = False
    cfg.save_oos_predictions = True

    print("=" * 80)
    print(f"[RUN] {run_name}")
    print(f"[DATA] matches={len(matches)} competitions={len({m.comp_name for m in matches})}")
    print("=" * 80)

    _ = train_rolling(matches, cat_maps, cfg)


def _extract_competition_summaries(global_oos: pd.DataFrame) -> dict:
    out = {}
    for comp in ATTA_COMPETITIONS:
        part = global_oos[global_oos["competition"] == comp].copy()
        out[comp] = _summary_from_predictions(part)
    return out


def _write_results(results: dict) -> None:
    out_dir = Path(sett.DATA_DIR) / "integrity_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "auc_discrepancy_check.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"[SAVED] {out_path}")
    print("=" * 80)


def _print_compact_results(results: dict) -> None:
    print("\n" + "=" * 80)
    print("[INTEGRITY CHECK RESULTS]")
    print("=" * 80)

    g = results["global_24_training"]["recomputed_from_oos"]
    print(
        f"Global 24-league training: "
        f"AUC={g['auc']:.4f}, Acc={g['accuracy']:.4f}, Brier={g['brier']:.4f}, n={g['n']}"
    )

    print("\nPer-competition slices from the GLOBAL 24-league model:")
    for comp, s in results["global_model_per_competition_slices"].items():
        print(f"  {comp}: " f"AUC={s['auc']:.4f}, Acc={s['accuracy']:.4f}, Brier={s['brier']:.4f}, n={s['n']}")

    a = results["atta3_training"]["recomputed_from_oos"]
    print(
        f"\nThree-league-only training: "
        f"AUC={a['auc']:.4f}, Acc={a['accuracy']:.4f}, Brier={a['brier']:.4f}, n={a['n']}"
    )

    print("\nInterpretation check:")
    print(
        "  If the global AUC is close to 0.581 and the per-competition slices are close "
        "to the thesis values, but the three-league-only run remains around 0.557, "
        "then the discrepancy is reproducible and is probably not caused by metric "
        "calculation or accidental filtering."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun the selected MLP on the full 24-league dataset and on the "
            "three Atta Mills competitions to verify the AUC discrepancy."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Optional full snapshot path. If omitted, fs_settings.LOAD_SNAPSHOT_PATH is used.",
    )
    parser.add_argument(
        "--skip-global-training",
        action="store_true",
        help="Reuse existing global integrity-check log if present.",
    )
    parser.add_argument(
        "--skip-atta-training",
        action="store_true",
        help="Reuse existing three-league integrity-check log if present.",
    )
    args = parser.parse_args()

    # Force full thesis mode, not submission EPL mode.
    sett.SUBMISSION_MODE = False
    sett.ALL_LOAD = True
    sett.ALL_GET_NEW = False
    sett.ALL_STORE = False

    if args.snapshot is not None:
        sett.LOAD_SNAPSHOT_PATH = Path(args.snapshot)

    print("=" * 80)
    print("[AUC DISCREPANCY INTEGRITY CHECK]")
    print("=" * 80)
    print(f"[snapshot] {sett.LOAD_SNAPSHOT_PATH}")
    print(f"[submission mode] {sett.SUBMISSION_MODE}")

    _load_explicit_snapshot_into_globals(sett.LOAD_SNAPSHOT_PATH)
    all_matches = prepare_matches()

    from football_outcomes.config.fs_globals import Global

    g = Global.get_instance()
    print(
        f"[loaded snapshot sanity] comp_seasons={len(g.all_comp_seasons)} "
        f"teams={len(g.all_teams)} players={len(g.all_players)} matches={len(g.all_matches)}"
    )

    all_comps = sorted({m.comp_name for m in all_matches})
    print(f"[prepared] matches={len(all_matches)} competitions={len(all_comps)}")

    atta_matches = [m for m in all_matches if m.comp_name in ATTA_COMPETITIONS]
    atta_comps = sorted({m.comp_name for m in atta_matches})
    print(f"[atta subset] matches={len(atta_matches)} competitions={atta_comps}")

    if len(atta_comps) != 3:
        raise RuntimeError(f"Expected exactly 3 Atta competitions, got: {atta_comps}")

    if not args.skip_global_training:
        _run_selected_mlp(all_matches, GLOBAL_RUN_NAME)
    else:
        print(f"[SKIP] global training; using existing logs for {GLOBAL_RUN_NAME}")

    if not args.skip_atta_training:
        _run_selected_mlp(atta_matches, ATTA_RUN_NAME)
    else:
        print(f"[SKIP] three-league training; using existing logs for {ATTA_RUN_NAME}")

    global_oos = _read_oos(GLOBAL_RUN_NAME)
    atta_oos = _read_oos(ATTA_RUN_NAME)

    results = {
        "snapshot": str(sett.LOAD_SNAPSHOT_PATH),
        "competitions_checked": ATTA_COMPETITIONS,
        "global_24_training": {
            "run_name": GLOBAL_RUN_NAME,
            "saved_summary": _read_saved_summary(GLOBAL_RUN_NAME),
            "recomputed_from_oos": _summary_from_predictions(global_oos),
            "num_competitions_in_oos": int(global_oos["competition"].nunique()),
            "competitions_in_oos": sorted(global_oos["competition"].dropna().unique().tolist()),
        },
        "global_model_per_competition_slices": _extract_competition_summaries(global_oos),
        "global_model_three_competitions_pooled_slice": _summary_from_predictions(
            global_oos[global_oos["competition"].isin(ATTA_COMPETITIONS)].copy()
        ),
        "atta3_training": {
            "run_name": ATTA_RUN_NAME,
            "saved_summary": _read_saved_summary(ATTA_RUN_NAME),
            "recomputed_from_oos": _summary_from_predictions(atta_oos),
            "num_competitions_in_oos": int(atta_oos["competition"].nunique()),
            "competitions_in_oos": sorted(atta_oos["competition"].dropna().unique().tolist()),
        },
    }

    _write_results(results)
    _print_compact_results(results)


if __name__ == "__main__":
    main()
