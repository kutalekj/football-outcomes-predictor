from datetime import datetime, timezone
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, Flatten, Concatenate, BatchNormalization, Lambda, Activation, LSTM
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras.backend as K
import os
from settings import NUM_NUMERICAL_FEATURES, NUM_CATEGORICAL_FEATURES
from globals import Global
from utils import get_n_previous_matches


NUM_TRAINING_ROUNDS = 25
EMBEDDING_OUT_SIZE_TEAM = 9
EMBEDDING_OUT_SIZE_COMP = 2
SEQUENCE_LENGTH = 10

team_encoder = LabelEncoder()
comp_encoder = LabelEncoder()


def train(regular_matches_in_rounds):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    global_instance = Global.get_instance()

    # Extract all unique team and comp IDs globally
    all_home_team_ids, all_away_team_ids, all_comp_ids = extract_all_unique_ids(regular_matches_in_rounds)
    all_team_ids = np.concatenate([all_home_team_ids, all_away_team_ids, np.array([0])])
    all_comp_ids = np.concatenate([all_comp_ids, np.array([0])])  # Allows encoders to correctly transform 0 values

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
    train_rnn_model(regular_matches_in_rounds)


def train_rnn_model(regular_matches_in_rounds):
    # Callbacks
    log_dir = os.path.join("logs", "fit" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), "rounds")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Define the model
    model = build_rnn_model(
        num_unique_teams=Global.get_instance().num_unique_regular_teams_for_training,
        num_unique_comps=Global.get_instance().num_unique_regular_comps_for_training,
        embedding_out_size_team=EMBEDDING_OUT_SIZE_TEAM,
        embedding_out_size_comp=EMBEDDING_OUT_SIZE_COMP,
        num_numerical_features=NUM_NUMERICAL_FEATURES,
        sequence_length=SEQUENCE_LENGTH
    )

    # Prepare sequences and labels
    train_sequences, train_labels = prepare_sequences(regular_matches_in_rounds, is_training=True)
    val_sequences, val_labels = prepare_sequences(regular_matches_in_rounds, is_training=False)

    # Unpack sequences
    train_home_numerical_sequences, train_away_numerical_sequences, \
        train_home_team_sequences, train_away_team_sequences, \
        train_home_comp_sequences, train_away_comp_sequences = train_sequences

    val_home_numerical_sequences, val_away_numerical_sequences, \
        val_home_team_sequences, val_away_team_sequences, \
        val_home_comp_sequences, val_away_comp_sequences = val_sequences

    # Map categorical IDs to zero-indexed values using LabelEncoder
    train_home_team_sequences_mapped = team_encoder.transform(train_home_team_sequences.flatten()).reshape(
        train_home_team_sequences.shape) + 1
    train_away_team_sequences_mapped = team_encoder.transform(train_away_team_sequences.flatten()).reshape(
        train_away_team_sequences.shape) + 1
    train_home_comp_sequences_mapped = comp_encoder.transform(train_home_comp_sequences.flatten()).reshape(
        train_home_comp_sequences.shape) + 1
    train_away_comp_sequences_mapped = comp_encoder.transform(train_away_comp_sequences.flatten()).reshape(
        train_away_comp_sequences.shape) + 1

    val_home_team_sequences_mapped = team_encoder.transform(val_home_team_sequences.flatten()).reshape(
        val_home_team_sequences.shape) + 1
    val_away_team_sequences_mapped = team_encoder.transform(val_away_team_sequences.flatten()).reshape(
        val_away_team_sequences.shape) + 1
    val_home_comp_sequences_mapped = comp_encoder.transform(val_home_comp_sequences.flatten()).reshape(
        val_home_comp_sequences.shape) + 1
    val_away_comp_sequences_mapped = comp_encoder.transform(val_away_comp_sequences.flatten()).reshape(
        val_away_comp_sequences.shape) + 1

    print(f"Training data shapes:")
    print(f"Home numerical sequences: {train_home_numerical_sequences.shape}")
    print(f"Away numerical sequences: {train_away_numerical_sequences.shape}")
    print(f"Home team sequences: {train_home_team_sequences_mapped.shape}")
    print(f"Away team sequences: {train_away_team_sequences_mapped.shape}")
    print(f"Home comp sequences: {train_home_comp_sequences_mapped.shape}")
    print(f"Away comp sequences: {train_away_comp_sequences_mapped.shape}")
    print(f"Labels: {train_labels.shape}")

    # Train the model
    model.fit(
        [
            train_home_numerical_sequences, train_away_numerical_sequences,
            train_home_team_sequences_mapped, train_away_team_sequences_mapped,
            train_home_comp_sequences_mapped, train_away_comp_sequences_mapped
        ],
        train_labels,
        epochs=1000,
        batch_size=32,
        validation_data=(
            [
                val_home_numerical_sequences, val_away_numerical_sequences,
                val_home_team_sequences_mapped, val_away_team_sequences_mapped,
                val_home_comp_sequences_mapped, val_away_comp_sequences_mapped
            ],
            val_labels
        ),
        callbacks=[early_stopping, tensorboard_callback]
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(
        [val_home_numerical_sequences, val_away_numerical_sequences, val_home_team_sequences_mapped,
         val_away_team_sequences_mapped, val_home_comp_sequences_mapped, val_away_comp_sequences_mapped],
        val_labels
    )
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

    # TODO: Add output log saving with debug outputs, weighted acc (from last N epochs only) and...
    # TODO: The model still learns to map embedding values close to zero... Maybe weights init close to 0?
    # TODO: ...and with highest/lowest embedding value - see if really all values close to 0
    # TODO: Try cumulative training, without sliding window
    # TODO: Try RNN
    # TODO: Find out which features contribute more and which less to the training


def build_rnn_model(num_unique_teams, num_unique_comps, embedding_out_size_team,
                    embedding_out_size_comp, num_numerical_features, sequence_length):
    # Inputs
    home_numerical_input = Input(shape=(sequence_length, num_numerical_features), name='home_numerical_input')
    away_numerical_input = Input(shape=(sequence_length, num_numerical_features), name='away_numerical_input')
    home_team_input = Input(shape=(sequence_length,), name='home_team_input', dtype='int32')
    away_team_input = Input(shape=(sequence_length,), name='away_team_input', dtype='int32')
    home_comp_input = Input(shape=(sequence_length,), name='home_comp_input', dtype='int32')
    away_comp_input = Input(shape=(sequence_length,), name='away_comp_input', dtype='int32')

    # Embedding layers without masking
    team_embedding_layer = Embedding(
        input_dim=num_unique_teams + 1,  # adjusted for shifted IDs
        output_dim=embedding_out_size_team,
        mask_zero=False,
        name='team_embedding'
    )
    comp_embedding_layer = Embedding(
        input_dim=num_unique_comps + 1,
        output_dim=embedding_out_size_comp,
        mask_zero=False,
        name='comp_embedding'
    )

    # Embedded sequences
    home_team_embedded = team_embedding_layer(home_team_input)
    away_team_embedded = team_embedding_layer(away_team_input)
    home_comp_embedded = comp_embedding_layer(home_comp_input)
    away_comp_embedded = comp_embedding_layer(away_comp_input)

    # Concatenate numerical features with embeddings
    home_sequence = Concatenate(axis=-1)([home_numerical_input, home_team_embedded, home_comp_embedded])
    away_sequence = Concatenate(axis=-1)([away_numerical_input, away_team_embedded, away_comp_embedded])

    # LSTM layers for home and away sequences
    home_lstm_out = LSTM(64)(home_sequence)
    away_lstm_out = LSTM(64)(away_sequence)

    # Combine outputs
    combined = Concatenate()([home_lstm_out, away_lstm_out])

    # Dense layers
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Define model
    model = Model(inputs=[
        home_numerical_input, away_numerical_input,
        home_team_input, away_team_input,
        home_comp_input, away_comp_input
    ], outputs=output)

    # Compile model
    model_optimizer = Adam(learning_rate=0.00007)
    model.compile(optimizer=model_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def prepare_sequences(regular_matches_in_rounds, is_training=True):
    # Prepare sequences of data for the RNN model
    sequences = []
    labels = []

    for matches in regular_matches_in_rounds:
        for match in matches:
            if match.datetime.tzinfo is None:
                match.datetime = match.datetime.replace(tzinfo=timezone.utc)

            # Get previous matches for home and away teams
            home_team_prev_matches = get_n_previous_matches(
                SEQUENCE_LENGTH, match, match.home_team.id, regular=True
            )
            away_team_prev_matches = get_n_previous_matches(
                SEQUENCE_LENGTH, match, match.away_team.id, regular=True
            )

            # Replace None values with dummy matches
            home_team_prev_matches = [m if m is not None else create_dummy_match() for m in home_team_prev_matches]
            away_team_prev_matches = [m if m is not None else create_dummy_match() for m in away_team_prev_matches]

            # After replacing None values, check sequence lengths
            if len(home_team_prev_matches) < SEQUENCE_LENGTH:
                home_team_prev_matches = pad_matches(home_team_prev_matches, SEQUENCE_LENGTH)
            if len(away_team_prev_matches) < SEQUENCE_LENGTH:
                away_team_prev_matches = pad_matches(away_team_prev_matches, SEQUENCE_LENGTH)

            # For validation, pad sequences if necessary
            if not is_training:
                home_team_prev_matches = pad_matches(home_team_prev_matches, SEQUENCE_LENGTH)
                away_team_prev_matches = pad_matches(away_team_prev_matches, SEQUENCE_LENGTH)

            # Sort matches in ascending order
            home_team_prev_matches.sort(key=lambda x: x.datetime)
            away_team_prev_matches.sort(key=lambda x: x.datetime)

            # Extract features for home and away team sequences
            home_numerical_sequence = [m.feature_vector_before_match_played for m in home_team_prev_matches]
            away_numerical_sequence = [m.feature_vector_before_match_played for m in away_team_prev_matches]

            # Extract team IDs for embedding (we can use match.home_team.id and match.away_team.id)
            home_team_sequence = [match.home_team.id] * SEQUENCE_LENGTH
            away_team_sequence = [match.away_team.id] * SEQUENCE_LENGTH

            # Extract competition IDs for home and away team sequences
            home_comp_sequence = [m.comp.id for m in home_team_prev_matches]
            away_comp_sequence = [m.comp.id for m in away_team_prev_matches]

            sequences.append((
                home_numerical_sequence, away_numerical_sequence,
                home_team_sequence, away_team_sequence,
                home_comp_sequence, away_comp_sequence
            ))

            # Label: outcome of the current match
            total_goals = match.home_team_goals + match.away_team_goals
            label = 1 if total_goals < 2.5 else 0
            labels.append(label)

    # Convert sequences to numpy arrays
    home_numerical_sequences = np.array([seq[0] for seq in sequences])
    away_numerical_sequences = np.array([seq[1] for seq in sequences])
    home_team_sequences = np.array([seq[2] for seq in sequences])
    away_team_sequences = np.array([seq[3] for seq in sequences])
    home_comp_sequences = np.array([seq[4] for seq in sequences])
    away_comp_sequences = np.array([seq[5] for seq in sequences])
    labels = np.array(labels)

    return (
               home_numerical_sequences, away_numerical_sequences,
               home_team_sequences, away_team_sequences,
               home_comp_sequences, away_comp_sequences
           ), labels


def pad_matches(matches, sequence_length):
    # Pad matches with dummy matches if necessary
    padding_needed = sequence_length - len(matches)
    if padding_needed > 0:
        padding = [create_dummy_match()] * padding_needed
        matches = padding + matches
    return matches


def create_dummy_match():
    # Create a dummy match with zeroed features
    class DummyTeam:
        id = 0

    class DummyComp:
        id = 0

    class DummyMatch:
        home_team = DummyTeam()
        away_team = DummyTeam()
        comp = DummyComp()
        datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)
        feature_vector_before_match_played = np.zeros(NUM_NUMERICAL_FEATURES)
        home_team_goals = 0
        away_team_goals = 0

    return DummyMatch()


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
