import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score


def evaluate_model(model, inputs, y_true):
    # Predict and return accuracy given the inputs and true labels
    y_pred = model.predict(inputs)
    y_pred_binary = (y_pred > 0.5).astype(int)
    return accuracy_score(y_true, y_pred_binary)


def permutation_importance_for_branch(model, inputs, y_true, branch_index, n_repeats=5):
    # Compute permutation importance for one input branch (by shuffling each feature/column and measuring accuracy drop)
    baseline = evaluate_model(model, inputs, y_true)
    importance = {}

    X_orig = inputs[branch_index]
    n_samples, n_features = X_orig.shape

    for col in range(n_features):
        scores = []

        for _ in range(n_repeats):
            X_permuted = X_orig.copy()

            np.random.shuffle(X_permuted[:, col])  # shuffle values in the current column

            permuted_inputs = list(inputs)  # create new list of inputs with the permuted branch
            permuted_inputs[branch_index] = X_permuted

            score = evaluate_model(model, permuted_inputs, y_true)
            scores.append(score)

        mean_score = np.mean(scores)
        importance[col] = baseline - mean_score

    return importance  # dict mapping each column index to its importance (baseline = mean accuracy after permutation)


def load_validation_data(file_path):
    # Load validation data from an NPZ file
    data = np.load(file_path)
    X_num = data['X_num']
    X_home = data['X_home']
    X_away = data['X_away']
    X_comp = data['X_comp']
    X_home_strength = data['X_home_strength']
    X_away_strength = data['X_away_strength']
    y_val = data['y_val']

    inputs = [X_num, X_home, X_away, X_comp, X_home_strength, X_away_strength]  # order expected by model
    return inputs, y_val


def get_creation_time(file_path):
    return os.path.getctime(file_path)


def main():
    model = load_model('C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\apifootball_model\\learned_models\\main_model_ann.keras')

    validation_dir = 'C:\\Users\\kutalekj\\PycharmProjects\\MyFlashscoreScraper\\src\\validation_data'
    npz_files = sorted([os.path.join(validation_dir, f) for f in os.listdir(validation_dir) if f.endswith('.npz')])

    # Sort files by creation time (asc.)
    npz_files.sort(key=get_creation_time)

    # Prepare containers to aggregate importance values across rounds for each input branch
    aggregate_importance = {
        'num': {},  # numerical features
        'home': {},  # home team ID embeddings
        'away': {},  # away team ID embeddings
        'comp': {},  # competition ID embeddings
        'home_strength': {},  # home team strength embeddings
        'away_strength': {}  # away team strength embeddings
    }

    round_count = 0
    for file_path in npz_files:
        print("Processing validation file:", file_path)
        inputs, y_val = load_validation_data(file_path)
        baseline = evaluate_model(model, inputs, y_val)
        print("Baseline Accuracy:", baseline)

        # Compute permutation importance for each branch:
        imp_num = permutation_importance_for_branch(model, inputs, y_val, branch_index=0, n_repeats=5)
        imp_home = permutation_importance_for_branch(model, inputs, y_val, branch_index=1, n_repeats=5)
        imp_away = permutation_importance_for_branch(model, inputs, y_val, branch_index=2, n_repeats=5)
        imp_comp = permutation_importance_for_branch(model, inputs, y_val, branch_index=3, n_repeats=5)
        imp_home_strength = permutation_importance_for_branch(model, inputs, y_val, branch_index=4, n_repeats=5)
        imp_away_strength = permutation_importance_for_branch(model, inputs, y_val, branch_index=5, n_repeats=5)

        # Helper function: update aggregate dictionary by appending the current importance values.
        def update_agg(agg, current_imp):
            for key, value in current_imp.items():
                if key in agg:
                    agg[key].append(value)
                else:
                    agg[key] = [value]

        update_agg(aggregate_importance['num'], imp_num)
        update_agg(aggregate_importance['home'], imp_home)
        update_agg(aggregate_importance['away'], imp_away)
        update_agg(aggregate_importance['comp'], imp_comp)
        update_agg(aggregate_importance['home_strength'], imp_home_strength)
        update_agg(aggregate_importance['away_strength'], imp_away_strength)

        round_count += 1

    # Average the permutation importances over all rounds
    def average_importance(agg):
        avg = {}
        for key, values in agg.items():
            avg[key] = np.mean(values)
        return avg

    avg_imp = {
        'num': average_importance(aggregate_importance['num']),
        'home': average_importance(aggregate_importance['home']),
        'away': average_importance(aggregate_importance['away']),
        'comp': average_importance(aggregate_importance['comp']),
        'home_strength': average_importance(aggregate_importance['home_strength']),
        'away_strength': average_importance(aggregate_importance['away_strength']),
    }

    # Display averaged results:
    print("\nAverage Permutation Importance over", round_count, "rounds:")
    print("\nNumerical Features:")
    for idx, imp in avg_imp['num'].items():
        print(f"  Feature {idx}: Importance = {imp:.4f}")

    print("\nHome Team ID Embeddings:")
    for idx, imp in avg_imp['home'].items():
        print(f"  Dimension {idx}: Importance = {imp:.4f}")

    print("\nAway Team ID Embeddings:")
    for idx, imp in avg_imp['away'].items():
        print(f"  Dimension {idx}: Importance = {imp:.4f}")

    print("\nCompetition ID Embeddings:")
    for idx, imp in avg_imp['comp'].items():
        print(f"  Dimension {idx}: Importance = {imp:.4f}")

    print("\nHome Team Strength Embeddings:")
    for idx, imp in avg_imp['home_strength'].items():
        print(f"  Dimension {idx}: Importance = {imp:.4f}")

    print("\nAway Team Strength Embeddings:")
    for idx, imp in avg_imp['away_strength'].items():
        print(f"  Dimension {idx}: Importance = {imp:.4f}")


if __name__ == '__main__':
    main()
