from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, Flatten, Concatenate, BatchNormalization, Lambda, Activation
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.backend as K
import os
import shutil
from settings import NUM_NUMERICAL_FEATURES, NUM_CATEGORICAL_FEATURES
from globals import Global


NUM_TRAINING_ROUNDS = 25
EMBEDDING_OUT_SIZE_TEAM = 9
EMBEDDING_OUT_SIZE_COMP = 2

team_encoder = LabelEncoder()
comp_encoder = LabelEncoder()


def train(regular_matches_in_rounds):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  # TODO: Acceleration
    global_instance = Global.get_instance()

    # Extract all unique team and comp IDs globally
    all_home_team_ids, all_away_team_ids, all_comp_ids = extract_all_unique_ids(regular_matches_in_rounds)
    all_team_ids = np.concatenate([all_home_team_ids, all_away_team_ids])

    # Map categorical IDs to zero-indexed values using LabelEncoder
    team_encoder.fit(all_team_ids)
    comp_encoder.fit(all_comp_ids)

    # Set the number of unique team IDs and comp IDs in the global instance
    global_instance.num_unique_regular_teams_for_training = len(team_encoder.classes_)
    global_instance.num_unique_regular_comps_for_training = len(comp_encoder.classes_)
    print(f"\t\t\t{global_instance.num_unique_regular_teams_for_training} "
          f"different regular teams are going to participate in the training process")
    print(f"\t\t\t{global_instance.num_unique_regular_comps_for_training} "
          f"different regular comps are going to participate in the training process")

    # Step 3: Train the final model
    train_main_model(regular_matches_in_rounds)


def train_main_model(regular_matches_in_rounds):
    weighted_accuracy = []
    num_validation_matches = 0
    total_rounds = len(regular_matches_in_rounds)

    # Callbacks
    log_dir = os.path.join("logs", "fit" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), "rounds")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Define the inputs
    numerical_input = Input(shape=(NUM_NUMERICAL_FEATURES,), name='numerical_input')
    home_team_input = Input(shape=(1,), dtype='int32', name='home_team_input')
    away_team_input = Input(shape=(1,), dtype='int32', name='away_team_input')
    comp_input = Input(shape=(1,), dtype='int32', name='comp_input')

    # Embedding layers for team IDs
    team_embedding_layer = Embedding(input_dim=Global.get_instance().num_unique_regular_teams_for_training,
                                     output_dim=EMBEDDING_OUT_SIZE_TEAM, name='team_embedding')

    home_team_embedding = Flatten()(team_embedding_layer(home_team_input))
    away_team_embedding = Flatten()(team_embedding_layer(away_team_input))

    # Embedding layer for comp IDs
    comp_embedding_layer = Embedding(input_dim=Global.get_instance().num_unique_regular_comps_for_training,
                                     output_dim=EMBEDDING_OUT_SIZE_COMP, name='comp_embedding')

    comp_embedding = Flatten()(comp_embedding_layer(comp_input))

    # Concatenate all features
    merged = Concatenate()([numerical_input, home_team_embedding, away_team_embedding, comp_embedding])

    # Build the rest of the model
    x = Dense(256, activation='relu')(merged)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Define the model
    model = Model(inputs=[numerical_input, home_team_input, away_team_input, comp_input], outputs=output)

    model_optimizer = Adam(learning_rate=0.00007)
    model.compile(optimizer=model_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    for round_number in range(NUM_TRAINING_ROUNDS + 1, total_rounds):
        # Extract features and labels (training data)
        train_numerical_features, train_labels = get_data_for_window(regular_matches_in_rounds, round_number,
                                                                     NUM_TRAINING_ROUNDS)
        train_home_team_input_data, train_away_team_input_data, train_comp_id_input_data, _ = extract_embedding_inputs(
            regular_matches_in_rounds, round_number, NUM_TRAINING_ROUNDS)

        # Map categorical IDs to zero-indexed values using LabelEncoder
        train_home_team_input_data_mapped = team_encoder.transform(train_home_team_input_data)
        train_away_team_input_data_mapped = team_encoder.transform(train_away_team_input_data)
        train_comp_id_input_data_mapped = comp_encoder.transform(train_comp_id_input_data)

        # Similarly for the validation data...
        val_numerical_features, val_labels = get_data_for_round(regular_matches_in_rounds, round_number)
        val_home_team_input_data, val_away_team_input_data, val_comp_id_input_data, _ = extract_embedding_inputs(
            regular_matches_in_rounds, round_number + 1, 1)

        val_home_team_input_data_mapped = team_encoder.transform(val_home_team_input_data)
        val_away_team_input_data_mapped = team_encoder.transform(val_away_team_input_data)
        val_comp_id_input_data_mapped = comp_encoder.transform(val_comp_id_input_data)

        print(f"\t\t\t\t\t\t\tRound {str(round_number)}: {str(train_numerical_features.shape)} train and"
              f" {str(val_numerical_features.shape)} val. data")
        num_validation_matches += val_numerical_features.shape[0]

        # Train the main model
        model.fit(
            [train_numerical_features, train_home_team_input_data_mapped, train_away_team_input_data_mapped,
             train_comp_id_input_data_mapped],
            train_labels,
            epochs=10,
            batch_size=32,
            validation_data=(
                [val_numerical_features, val_home_team_input_data_mapped, val_away_team_input_data_mapped,
                 val_comp_id_input_data_mapped],
                val_labels
            ),
            callbacks=[early_stopping, tensorboard_callback]
        )

        loss, accuracy = model.evaluate(
            [val_numerical_features, val_home_team_input_data_mapped, val_away_team_input_data_mapped,
             val_comp_id_input_data_mapped],
            val_labels
        )

        print(f"\tRound {str(round_number)} - Loss: {str(loss)}, Accuracy: {str(accuracy)}")

        # Access embedding weights
        team_embedding_weights = model.get_layer('team_embedding').get_weights()[0]
        comp_embedding_weights = model.get_layer('comp_embedding').get_weights()[0]

        # DEBUG PRINTS
        sample_team_ids = team_encoder.classes_[:5]
        for team_id in sample_team_ids:
            idx = team_encoder.transform([team_id])[0]
            embedding_vector = team_embedding_weights[idx]
            print(f"Team ID: {team_id}, Embedding Index: {idx}, Embedding Vector: {embedding_vector}")

        sample_team_ids = team_encoder.classes_[300:305]
        for team_id in sample_team_ids:
            idx = team_encoder.transform([team_id])[0]
            embedding_vector = team_embedding_weights[idx]
            print(f"Team ID: {team_id}, Embedding Index: {idx}, Embedding Vector: {embedding_vector}")

        sample_comp_ids = comp_encoder.classes_[:3]
        for comp_id in sample_comp_ids:
            idx = comp_encoder.transform([comp_id])[0]
            embedding_vector = comp_embedding_weights[idx]
            print(f"Comp ID: {comp_id}, Embedding Index: {idx}, Embedding Vector: {embedding_vector}")

        sample_comp_ids = comp_encoder.classes_[10:13]
        for comp_id in sample_comp_ids:
            idx = comp_encoder.transform([comp_id])[0]
            embedding_vector = comp_embedding_weights[idx]
            print(f"Comp ID: {comp_id}, Embedding Index: {idx}, Embedding Vector: {embedding_vector}")
        # TODO: Add output log saving with debug outputs, weighted acc (from last N epochs only) and...
        # TODO: The model still learns to map embedding values close to zero... Maybe weights init close to 0?
        # TODO: ...and with highest/lowest embedding value - see if really all values close to 0
        # TODO: Try cumulative training, without sliding window (non RNN)
        # TODO: Find out which features contribute more and which less to the training

        weighted_accuracy.append(accuracy * val_numerical_features.shape[0])

    final_weighted_acc = float(np.sum(weighted_accuracy) / num_validation_matches)
    print(f"\tWeighted validation accuracy = "f"{final_weighted_acc}")

    if final_weighted_acc < 0.56:
        print(f"Weighted accuracy ({final_weighted_acc}) < 0.56. Deleting log folder {log_dir}...")
        try:
            shutil.rmtree(log_dir)
        except OSError as e:
            print(f"Error deleting directory {log_dir}: {e}")


def extract_all_unique_ids(regular_matches_in_rounds):
    all_home_team_ids = []
    all_away_team_ids = []
    all_comp_ids = []

    for matches in regular_matches_in_rounds:
        for match in matches:
            all_home_team_ids.append(match.home_team.id)
            all_away_team_ids.append(match.away_team.id)
            all_comp_ids.append(match.comp.id)

    return np.array(all_home_team_ids), np.array(all_away_team_ids), np.array(all_comp_ids)


def get_data_for_window(regular_matches_in_rounds, round_number, window_size):
    start_round = round_number - window_size - 1
    end_round = round_number - 1

    data = []
    labels = []

    for r in range(start_round, end_round):
        matches = regular_matches_in_rounds[r]

        for match in matches:
            total_goals = match.home_team_goals + match.away_team_goals
            label = 1 if total_goals < 2.5 else 0

            data.append(match.feature_vector_before_match_played)
            labels.append(label)

    data = np.array(data)  # shape (num_matches, num_features)
    labels = np.array(labels)  # shape (num_matches,)

    return data, labels


def extract_embedding_inputs(regular_matches_in_rounds, round_number, window_size):
    start_round = round_number - window_size - 1
    end_round = round_number - 1

    home_team_input = []
    away_team_input = []
    comp_id_input = []
    labels = []

    for r in range(start_round, end_round):
        matches = regular_matches_in_rounds[r]

        for match in matches:
            total_goals = match.home_team_goals + match.away_team_goals
            label = 1 if total_goals < 2.5 else 0

            home_team_input.append(match.home_team.id)  # home team ID (integer)
            away_team_input.append(match.away_team.id)  # away team ID (integer)
            comp_id_input.append(match.comp.id)  # comp ID (integer)
            labels.append(label)

    return np.array(home_team_input), np.array(away_team_input), np.array(comp_id_input), np.array(labels)


def get_data_for_round(regular_matches_in_rounds, round_number):
    matches = regular_matches_in_rounds[round_number - 1]

    data = []
    labels = []

    for match in matches:
        total_goals = match.home_team_goals + match.away_team_goals
        label = 1 if total_goals < 2.5 else 0

        data.append(match.feature_vector_before_match_played)
        labels.append(label)

    data = np.array(data)  # shape (num_matches, num_features)
    labels = np.array(labels)  # shape (num_matches,)

    return data, labels


def scale_to_0_1(x):
    return (x - K.min(x)) / (K.max(x) - K.min(x))