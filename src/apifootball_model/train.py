import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os


NUM_TRAINING_ROUNDS = 25


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
    # train_data, train_labels = get_data_up_to_round(regular_matches_in_rounds, NUM_TRAINING_ROUNDS)
    for round_number in range(NUM_TRAINING_ROUNDS + 1, total_rounds):
        train_data, train_labels = get_data_for_window(regular_matches_in_rounds, round_number, NUM_TRAINING_ROUNDS)
        val_data, val_labels = get_data_for_round(regular_matches_in_rounds, round_number)

        model.fit(train_data, train_labels, epochs=10, batch_size=16, validation_data=(val_data, val_labels),
                  callbacks=[early_stopping, tensorboard_callback])

        loss, accuracy = model.evaluate(val_data, val_labels)
        print(f"Round {round_number} - Loss: {loss}, Accuracy: {accuracy}")
        print(f"{len(train_data)} training data and {len(val_data)} validation data were used in this round training")

        # train_data, train_labels = append_to_training_data(train_data, train_labels, val_data, val_labels)


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


def get_data_up_to_round(regular_matches_in_rounds, round_number):
    data = []
    labels = []

    for r in range(round_number - 1):  # round_number is 1-based, so exclude it
        matches = regular_matches_in_rounds[r]

        for match in matches:
            total_goals = match.home_team_goals + match.away_team_goals
            label = 1 if total_goals < 2.5 else 0

            data.append(match.feature_vector_before_match_played)
            labels.append(label)

    data = np.array(data)  # shape (num_matches, num_features)
    labels = np.array(labels)  # shape (num_matches,)

    return data, labels


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


def append_to_training_data(t_data, t_labels, v_data, v_labels):
    t_data = np.concatenate((t_data, v_data), axis=0)
    t_labels = np.concatenate((t_labels, v_labels), axis=0)

    return t_data, t_labels
