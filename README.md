# Predicting Goal Count from Multi-Source Football Match Data

The submitted version contains a reduced English Premier League (EPL) sample snapshot so that the pipeline can be executed locally without downloading the full dataset.

## Submitted Project Structure

```text
football-outcomes-predictor/
├── README.md
├── pyproject.toml
├── src/
│   └── football_outcomes/
│       ├── cli/
│       │   └── main.py
│       ├── config/
│       │   ├── fs_settings.py
│       │   └── fs_globals.py
│       ├── data/
│       │   ├── fs_io.py
│       │   ├── fs_models.py
│       │   └── fs_retrieve.py
│       ├── training/
│       │   ├── fs_classical_baselines.py
│       │   ├── fs_training_utils.py
│       │   └── train_mlp_rolling.py
│       └── utils/
│           ├── fs_common.py
│           ├── fs_feature_utils.py
│           └── fs_player_skill_utils.py
├── scripts/
│   ├── main_footystats.py
│   └── tools/
│       ├── create_submission_epl_snapshot.py
│       ├── analysis_baselines_comparison.py
│       ├── analysis_per_competition_selected_mlp.py
│       ├── analysis_bookmaker_odds_benchmark.py
│       └── plot_thesis_section_51_figures.py
├── data/
│   └── submission/
│       └── epl_sample_snapshot.pkl
└── docs/
    ├── thesis.pdf
    └── experiments/
        └── final_figures/
```

#### What Is Included

The submitted sample dataset contains English Premier League matches from all the four seasons 2021/2022 to 2024/2025 (`data/submission/epl_sample_snapshot.pkl`).
It includes cached FootyStats match/team/player objects and the corresponding reduced SoFIFA player-skill snapshot data required to run the pipeline.
The sample is intended to verify that the implementation runs correctly, not to reproduce all experiments from the thesis.

#### What Is Not Included

The full dataset, raw API downloads, raw SoFIFA CSV snapshots, TensorBoard logs, temporary experiment outputs, and archived implementations.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install the project:

```bash
pip install -e .
```

If dependencies are not installed automatically from `pyproject.toml`, install the required packages manually:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib rapidfuzz requests
```

### Running the Submitted Pipeline

Run:

```bash
python -m football_outcomes.cli.main pipeline
```

This command executes: `scripts/main_footystats.py`.

The script performs the following steps:

1. Loads the reduced EPL sample snapshot.
2. Links matches to competition-seasons.
3. Initializes league-table state.
4. Links FootyStats entities to SoFIFA entities.
5. Computes pre-match features.
6. Trains the selected MLP model on the binary Under/Over 2.5 objective.
7. Saves round-level metrics and out-of-sample predictions.

## Expected Outputs

After a successful run, outputs are written to:

`data/tensorboard_logs/submission_epl_selected_mlp_binary_u25/`

Expected files include:

* `summary.json` contains pooled out-of-sample metrics computed over all validation predictions, including: accuracy, AUC, Brier score, basic round statistics
* `round_metrics.csv` contains validation metrics computed separately for each rolling validation round
* `oos_predictions.csv` contains individual out-of-sample predictions, with each row corresponding to a validation match predicted by a model trained only on previous rolling rounds

The script also writes a dataset summary to:

`data/submission_outputs/submission_dataset_summary.json`

### Main Source Files

* `scripts/main_footystats.py`: main entry-point script.

* `src/football_outcomes/training/train_mlp_rolling.py`: neural network architecture definitions and training pipeline.

* `src/football_outcomes/config/fs_settings.py`: global project configuration, paths, runtime flags, and experiment settings.

* `src/football_outcomes/data/fs_models.py`: core data model definitions for matches, teams, players, competitions, seasons, and features.

* `src/football_outcomes/training/fs_classical_baselines.py`: implementations of baseline models used for comparison in the experiments.

* `src/football_outcomes/training/fs_training_utils.py`: utilities for training-related stuff.

* `src/football_outcomes/utils/fs_feature_utils.py`: utilities for feature-related stuff.

* `src/football_outcomes/utils/fs_player_skill_utils.py`: player-skill handling utilities.
