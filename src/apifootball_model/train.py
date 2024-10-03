from datetime import datetime
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, Flatten, Concatenate, BatchNormalization, Lambda, Activation
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.backend as K
import os
from settings import NUM_NUMERICAL_FEATURES, NUM_CATEGORICAL_FEATURES
from globals import Global


NUM_TRAINING_ROUNDS = 25
EMBEDDING_OUT_SIZE_TEAM = 6
EMBEDDING_OUT_SIZE_COMP = 3


def train(regular_matches_in_rounds):
    global_instance = Global.get_instance()

    # Extract all unique team and comp IDs globally
    all_home_team_ids, all_away_team_ids, all_comp_ids = extract_all_unique_ids(regular_matches_in_rounds)
    all_team_ids = np.concatenate([all_home_team_ids, all_away_team_ids])

    # Map categorical IDs to zero-indexed values using LabelEncoder
    team_encoder = LabelEncoder()
    all_team_ids_mapped = team_encoder.fit_transform(all_team_ids)

    comp_encoder = LabelEncoder()
    all_comp_ids_mapped = comp_encoder.fit_transform(all_comp_ids)

    # Set the number of unique team IDs and comp IDs in the global instance
    global_instance.num_unique_regular_teams_for_training = len(np.unique(all_team_ids_mapped))
    global_instance.num_unique_regular_comps_for_training = len(np.unique(all_comp_ids_mapped))

    # TensorBoard callbacks for the Embedding models
    log_dir_team = os.path.join("logs", "embedding_team_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    tensorboard_callback_team = TensorBoard(log_dir=log_dir_team, histogram_freq=1)
    log_dir_comp = os.path.join("logs", "embedding_comp_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    tensorboard_callback_comp = TensorBoard(log_dir=log_dir_comp, histogram_freq=1)

    # Step 1a: Pre-train Team embedding model
    team_input = Input(shape=(1,), dtype='int32', name='team_input')  # one input for both home and away teams

    # Embedding layers for the team IDs categorical features
    team_embedding = Embedding(input_dim=global_instance.num_unique_regular_teams_for_training,
                               output_dim=EMBEDDING_OUT_SIZE_TEAM)(team_input)
    team_embedding = BatchNormalization()(team_embedding)
    team_embedding = Lambda(scale_to_0_1)(team_embedding)
    team_flat = Flatten()(team_embedding)

    team_embedding_model = Model(inputs=team_input, outputs=team_flat)
    team_embedding_model_optimizer = Adam()
    team_embedding_model.compile(optimizer=team_embedding_model_optimizer, loss='mse')
    team_embedding_model.summary()

    # Step 2a: Train the model using both home and away team IDs
    team_embedding_model.fit(all_team_ids_mapped, np.zeros((len(all_team_ids_mapped), EMBEDDING_OUT_SIZE_TEAM)),
                             epochs=10, batch_size=32, callbacks=[tensorboard_callback_team])

    # Step 1b: Pre-train Comp embedding model
    comp_id_input = Input(shape=(1,), dtype='int32', name='comp_id_input')

    # Embedding layer for the comp ID categorical feature
    comp_id_embedding = Embedding(input_dim=global_instance.num_unique_regular_comps_for_training,
                                  output_dim=EMBEDDING_OUT_SIZE_COMP)(comp_id_input)
    comp_id_embedding = BatchNormalization()(comp_id_embedding)
    comp_id_embedding = Lambda(scale_to_0_1)(comp_id_embedding)
    comp_id_flat = Flatten()(comp_id_embedding)

    comp_embedding_model = Model(inputs=comp_id_input, outputs=comp_id_flat)
    comp_embedding_model_optimizer = Adam()
    comp_embedding_model.compile(optimizer=comp_embedding_model_optimizer, loss='mse')
    comp_embedding_model.summary()

    # Step 2b: Train the comp embedding model using all comp IDs
    comp_embedding_model.fit(all_comp_ids_mapped, np.zeros((len(all_comp_ids_mapped), EMBEDDING_OUT_SIZE_COMP)),
                             epochs=10, batch_size=32, callbacks=[tensorboard_callback_comp])

    """
    for round_number in range(NUM_TRAINING_ROUNDS + 1, len(regular_matches_in_rounds)):
        home_team_input_data, away_team_input_data, comp_id_input_data, _ = extract_embedding_inputs(
            regular_matches_in_rounds, round_number, NUM_TRAINING_ROUNDS)

        # Map categorical IDs to zero-indexed values using LabelEncoder
        home_team_input_data_mapped, away_team_input_data_mapped, comp_id_input_data_mapped \
            = map_categorical_to_zero_indexed(home_team_input_data, away_team_input_data, comp_id_input_data)

        # Train the Team embedding model using both home and away team inputs
        combined_team_input_data = np.concatenate([home_team_input_data_mapped, away_team_input_data_mapped])
        team_embedding_model.fit(combined_team_input_data,
                                 np.zeros((len(combined_team_input_data), EMBEDDING_OUT_SIZE_TEAM)),
                                 epochs=10, batch_size=16, callbacks=[tensorboard_callback_team])

        # Train the Comp embedding model
        comp_embedding_model.fit(comp_id_input_data_mapped,
                                 np.zeros((len(comp_id_input_data_mapped), EMBEDDING_OUT_SIZE_COMP)),
                                 epochs=10, batch_size=16, callbacks=[tensorboard_callback_comp])
    """

    # Step 3: Train the final model with precomputed embeddings
    train_main_model(regular_matches_in_rounds, team_embedding_model, comp_embedding_model)


def train_main_model(regular_matches_in_rounds, team_embedding_model, comp_embedding_model):
    total_rounds = len(regular_matches_in_rounds)

    # Callbacks
    log_dir = os.path.join("logs", "fit" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), "rounds")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # The main model
    main_model = Sequential()
    main_model.add(Dense(128, input_dim=NUM_NUMERICAL_FEATURES + EMBEDDING_OUT_SIZE_TEAM * 2 + EMBEDDING_OUT_SIZE_COMP,
                         activation='relu'))
    main_model.add(Dense(64, activation='relu'))
    main_model.add(Dropout(0.3))
    main_model.add(Dense(1, activation='sigmoid'))

    main_mode_optimizer = Adam()
    main_model.compile(optimizer=main_mode_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    main_model.summary()

    for round_number in range(NUM_TRAINING_ROUNDS + 1, total_rounds):
        # Extract features and labels (training data)
        train_numerical_features, train_labels = get_data_for_window(regular_matches_in_rounds, round_number,
                                                                     NUM_TRAINING_ROUNDS)
        train_home_team_input_data, train_away_team_input_data, train_comp_id_input_data, _ = extract_embedding_inputs(
            regular_matches_in_rounds, round_number, NUM_TRAINING_ROUNDS)

        # Map categorical inputs to zero-indexed values using LabelEncoder
        train_home_team_input_data_mapped, train_away_team_input_data_mapped, train_comp_id_input_data_mapped, \
            = map_categorical_to_zero_indexed(
                train_home_team_input_data, train_away_team_input_data, train_comp_id_input_data)

        # Predict embeddings for the round (training data)
        train_home_team_embeddings = team_embedding_model.predict(train_home_team_input_data_mapped)
        train_away_team_embeddings = team_embedding_model.predict(train_away_team_input_data_mapped)
        train_comp_embeddings = comp_embedding_model.predict(train_comp_id_input_data_mapped)

        # Concatenate embeddings with other features (training data)
        train_input = np.concatenate([train_numerical_features, train_home_team_embeddings,
                                      train_away_team_embeddings, train_comp_embeddings], axis=1)

        # Similarly for the validation data...
        val_numerical_features, val_labels = get_data_for_round(regular_matches_in_rounds, round_number)
        val_home_team_input_data, val_away_team_input_data, val_comp_id_input_data, _ = extract_embedding_inputs(
            regular_matches_in_rounds, round_number + 1, 1)

        val_home_team_input_data_mapped, val_away_team_input_data_mapped, val_comp_id_input_data_mapped, \
            = map_categorical_to_zero_indexed(
                val_home_team_input_data, val_away_team_input_data, val_comp_id_input_data)

        val_home_team_embeddings = team_embedding_model.predict(val_home_team_input_data_mapped)
        val_away_team_embeddings = team_embedding_model.predict(val_away_team_input_data_mapped)
        val_comp_embeddings = comp_embedding_model.predict(val_comp_id_input_data_mapped)

        val_input = np.concatenate([val_numerical_features, val_home_team_embeddings,
                                    val_away_team_embeddings, val_comp_embeddings], axis=1)
        print(f"\t\t\t\t\t\t\tRound {str(round_number)}: {str(train_input.shape[0])} train and"
              f" {str(val_input.shape[0])} val. data")

        # Train the main model
        main_model.fit(train_input, train_labels, epochs=10, batch_size=16, validation_data=(val_input, val_labels),
                       callbacks=[early_stopping, tensorboard_callback])

        loss, accuracy = main_model.evaluate(val_input, val_labels)
        print(f"\tRound {str(round_number)} - Loss: {str(loss)}, Accuracy: {str(accuracy)}")


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


def map_categorical_to_zero_indexed(home_team_input_data, away_team_input_data, comp_id_input_data):
    home_team_encoder = LabelEncoder()
    away_team_encoder = LabelEncoder()
    comp_id_encoder = LabelEncoder()

    home_team_input_data_mapped = home_team_encoder.fit_transform(home_team_input_data)
    away_team_input_data_mapped = away_team_encoder.fit_transform(away_team_input_data)
    comp_id_input_data_mapped = comp_id_encoder.fit_transform(comp_id_input_data)

    return home_team_input_data_mapped, away_team_input_data_mapped, comp_id_input_data_mapped


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
