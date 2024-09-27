import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os
from settings import ONE_HOT_ENCODED_VECTOR_LENGTH, NUM_NUMERICAL_FEATURES, NUM_CATEGORICAL_FEATURES


NUM_TRAINING_ROUNDS = 25
EMBEDDING_SIZE = 8


def train_embedding(regular_matches_in_rounds):
    # Step 1: Pre-train Embedding model
    home_team_input = Input(shape=(1,), dtype='int32', name='home_team_input')
    away_team_input = Input(shape=(1,), dtype='int32', name='away_team_input')
    comp_id_input = Input(shape=(1,), dtype='int32', name='comp_id_input')

    # Embedding layers for the categorical features
    home_team_embedding = Embedding(input_dim=ONE_HOT_ENCODED_VECTOR_LENGTH,
                                    output_dim=EMBEDDING_SIZE)(home_team_input)
    away_team_embedding = Embedding(input_dim=ONE_HOT_ENCODED_VECTOR_LENGTH,
                                    output_dim=EMBEDDING_SIZE)(away_team_input)
    comp_id_embedding = Embedding(input_dim=ONE_HOT_ENCODED_VECTOR_LENGTH,
                                  output_dim=EMBEDDING_SIZE)(comp_id_input)

    # Flatten embeddings to create dense vectors
    home_team_flat = Flatten()(home_team_embedding)
    away_team_flat = Flatten()(away_team_embedding)
    comp_id_flat = Flatten()(comp_id_embedding)

    # Concatenate all embeddings
    combined_embeddings = Concatenate()([away_team_flat, comp_id_flat, home_team_flat])

    # Define and compile the embedding model
    embedding_model = Model(inputs=[away_team_input, comp_id_input, home_team_input], outputs=combined_embeddings)
    embedding_model.compile(optimizer='adam', loss='mse')  # Embedding model doesn't need to classify
    embedding_model.summary()

    # Step 2: Train the embedding model
    for round_number in range(NUM_TRAINING_ROUNDS + 1, len(regular_matches_in_rounds)):
        away_team_input_data, comp_id_input_data, home_team_input_data, _ = extract_embedding_inputs(
            regular_matches_in_rounds, round_number, NUM_TRAINING_ROUNDS)

        embedding_model.fit([away_team_input_data, comp_id_input_data, home_team_input_data],
                            np.zeros((len(away_team_input_data), EMBEDDING_SIZE * 3)), epochs=10, batch_size=16)

    # Step 3: Train the final model with precomputed embeddings
    train_final_model(regular_matches_in_rounds, embedding_model)


def train_final_model(regular_matches_in_rounds, embedding_model):
    total_rounds = len(regular_matches_in_rounds)
    log_dir = os.path.join("logs", "fit", "rounds")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # The main model
    main_model = Sequential()
    main_model.add(Dense(64, input_dim=NUM_NUMERICAL_FEATURES + (EMBEDDING_SIZE * NUM_CATEGORICAL_FEATURES),
                         activation='relu'))
    main_model.add(Dense(32, activation='relu'))
    main_model.add(Dropout(0.3))
    main_model.add(Dense(1, activation='sigmoid'))
    main_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    main_model.summary()

    for round_number in range(NUM_TRAINING_ROUNDS + 1, total_rounds):
        # Extract features and labels (training data)
        train_numerical_features, train_labels = get_data_for_window(regular_matches_in_rounds, round_number,
                                                                     NUM_TRAINING_ROUNDS)
        train_home_team_input_data, train_away_team_input_data, train_comp_id_input_data, _ = extract_embedding_inputs(
            regular_matches_in_rounds, round_number, NUM_TRAINING_ROUNDS)

        # Predict embeddings for the round (training data)
        train_precomputed_embeddings = embedding_model.predict([train_away_team_input_data, train_comp_id_input_data,
                                                                train_home_team_input_data])

        # Concatenate embeddings with other features (training data)
        train_input = np.concatenate([train_numerical_features, train_precomputed_embeddings], axis=1)

        # Similarly for the validation data...
        val_numerical_features, val_labels = get_data_for_round(regular_matches_in_rounds, round_number)
        val_home_team_input_data, val_away_team_input_data, val_comp_id_input_data, _ = extract_embedding_inputs(
            regular_matches_in_rounds, round_number, 1)

        val_precomputed_embeddings = embedding_model.predict(
            [val_home_team_input_data, val_away_team_input_data, val_comp_id_input_data])
        val_input = np.concatenate([val_numerical_features, val_precomputed_embeddings], axis=1)

        # Train the main model
        main_model.fit(train_input, train_labels, epochs=10, batch_size=16, validation_data=(val_input, val_labels),
                       callbacks=[early_stopping, tensorboard_callback])

        loss, accuracy = main_model.evaluate(val_input, val_labels)
        print(f"Round {round_number} - Loss: {loss}, Accuracy: {accuracy}")


def train(regular_matches_in_rounds):
    # Model
    model = Sequential()
    model.add(Dense(64, input_dim=142, activation='relu'))  # input layer matching the feature vector size
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # output layer for binary classification

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Training
    total_rounds = len(regular_matches_in_rounds)

    # TensorBoard
    log_dir = os.path.join("logs", "fit", "rounds")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    for round_number in range(NUM_TRAINING_ROUNDS + 1, total_rounds):
        train_data, train_labels = get_data_for_window(regular_matches_in_rounds, round_number, NUM_TRAINING_ROUNDS)
        val_data, val_labels = get_data_for_round(regular_matches_in_rounds, round_number)

        model.fit(train_data, train_labels, epochs=10, batch_size=16, validation_data=(val_data, val_labels),
                  callbacks=[early_stopping, tensorboard_callback])

        loss, accuracy = model.evaluate(val_data, val_labels)
        print(f"Round {round_number} - Loss: {loss}, Accuracy: {accuracy}")
        print(f"{len(train_data)} training data and {len(val_data)} validation data were used in this round training")


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

    away_team_input = []
    comp_id_input = []
    home_team_input = []
    labels = []

    for r in range(start_round, end_round):
        matches = regular_matches_in_rounds[r]

        for match in matches:
            home_team_input.append(match.home_team.id)  # home team ID (integer)
            away_team_input.append(match.away_team.id)  # away team ID (integer)
            comp_id_input.append(match.comp.id)  # comp ID (integer)

            total_goals = match.home_team_goals + match.away_team_goals
            label = 1 if total_goals < 2.5 else 0

            labels.append(label)

    return np.array(away_team_input), np.array(comp_id_input), np.array(home_team_input), np.array(labels)


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
