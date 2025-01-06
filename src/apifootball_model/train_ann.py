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

PAR_NEURONS = {'neu_1': 32, 'neu_2': 64, 'neu_3': 128}
PAR_TEAM_EMB = {'teamem_1': 6, 'teamem_2': 9, 'teamem_3': 12}
PAR_COMP_EMB = {'compem_1': 2, 'compem_2': 3, 'compem_3': 4}
PAR_LR = {'lr_1': 0.00002, 'lr_2': 0.00007, 'lr_3': 0.0002, 'lr_4': 0.0007}
PAR_DROPOUT = {'drp_1': 0.15, 'drp_2': 0.3, 'drp_3': 0.45}
PAR_ACTIV = {'act_1': 'relu', 'act_2': 'leaky_relu'}
PAR_ITER = {'it_1': 1, 'it_2': 2, 'it_3': 3, 'it_4': 4}

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
    for par_iter_k, par_iter_v in PAR_ITER.items():

        weighted_accuracy = []
        accuracies = []
        num_validation_matches = 0
        total_rounds = len(regular_matches_in_rounds)

        # Lists to accumulate all-round validation data for computing comp-specific statistic
        val_numerical_features_all = []
        val_home_team_input_mapped_all = []
        val_away_team_input_mapped_all = []
        val_comp_id_input_mapped_all = []
        val_labels_all = []

        # Callbacks
        # log_dir = os.path.join("logs", "fit" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_' + str(par_neurons_k) + '_' + str(par_team_emb_k) + '_' + str(par_comp_emb_k) + '_' + str(par_drop_k) + '_' + str(par_act_k) + '_' + str(par_lr_k) + '_' + str(par_iter_k))
        log_dir = os.path.join("logs", "fit" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_' + str(par_iter_k))
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
        x = Dropout(0.6)(x)
        x = Dense(126, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.4)(x)
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
            accuracies.append(accuracy)

            # Store this round's validation data for later comp-specific analysis
            val_numerical_features_all.append(val_numerical_features)
            val_home_team_input_mapped_all.append(val_home_team_input_data_mapped)
            val_away_team_input_mapped_all.append(val_away_team_input_data_mapped)
            val_comp_id_input_mapped_all.append(val_comp_id_input_data_mapped)
            val_labels_all.append(val_labels)

        final_weighted_acc = float(np.sum(weighted_accuracy) / num_validation_matches)
        print(f"\tWeighted validation accuracy = "f"{final_weighted_acc}")

        last_100_rounds_acc = float(np.sum(accuracies[-100:]) / 100)
        print(f"\tAverage accuracy in last 100 rounds = "f"{last_100_rounds_acc:.3%}")

        # === ADDITION: Compute per-comp ratio across *all* validation sets ===
        # 1) Concatenate all stored validation sets
        if len(val_numerical_features_all) > 0:
            all_val_numerical_features = np.concatenate(val_numerical_features_all, axis=0)
            all_val_home_team = np.concatenate(val_home_team_input_mapped_all, axis=0)
            all_val_away_team = np.concatenate(val_away_team_input_mapped_all, axis=0)
            all_val_comp = np.concatenate(val_comp_id_input_mapped_all, axis=0)
            all_val_labels = np.concatenate(val_labels_all, axis=0)

            # 2) Predict on entire concatenated set
            preds = model.predict([all_val_numerical_features,
                                   all_val_home_team,
                                   all_val_away_team,
                                   all_val_comp])
            preds_bin = (preds >= 0.5).astype(int).flatten()

            # 3) Group by comp (mapped) and compute accuracy
            comp_correct = {}
            comp_total = {}

            for i in range(len(all_val_labels)):
                mapped_id = all_val_comp[i]
                if mapped_id not in comp_correct:
                    comp_correct[mapped_id] = 0
                    comp_total[mapped_id] = 0
                if preds_bin[i] == all_val_labels[i]:
                    comp_correct[mapped_id] += 1
                comp_total[mapped_id] += 1

            # 4) Print final ratio for each comp across *all* rounds
            print("\nPer-Comp Prediction Accuracy (All Validation Rounds):")
            for mapped_id, correct_count in comp_correct.items():
                total_count = comp_total[mapped_id]
                ratio = correct_count / total_count if total_count > 0 else 0.0
                original_comp_id = comp_encoder.inverse_transform([mapped_id])[0]
                print(f"Comp {original_comp_id}: {correct_count}/{total_count} = {ratio:.3%}")
            print("============================================\n")
        # === END ADDITION ===


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
