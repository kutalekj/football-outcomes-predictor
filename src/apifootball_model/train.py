from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, Flatten, Concatenate, BatchNormalization, Lambda, Activation
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow.keras.backend as K
import os
from settings import NUM_NUMERICAL_FEATURES, NUM_CATEGORICAL_FEATURES
from globals import Global


NUM_TRAINING_ROUNDS = 25
EMBEDDING_OUT_SIZE_TEAM = 6
EMBEDDING_OUT_SIZE_COMP = 3

team_encoder = LabelEncoder()
comp_encoder = LabelEncoder()


def train(regular_matches_in_rounds):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    global_instance = Global.get_instance()

    # Extract all unique team and comp IDs globally
    all_home_team_ids, all_away_team_ids, all_comp_ids = extract_all_unique_ids(regular_matches_in_rounds)
    all_team_ids = np.concatenate([all_home_team_ids, all_away_team_ids])

    # Map categorical IDs to zero-indexed values using LabelEncoder
    all_team_ids_mapped = team_encoder.fit_transform(all_team_ids)
    all_comp_ids_mapped = comp_encoder.fit_transform(all_comp_ids)

    # Set the number of unique team IDs and comp IDs in the global instance
    global_instance.num_unique_regular_teams_for_training = len(np.unique(all_team_ids_mapped))
    global_instance.num_unique_regular_comps_for_training = len(np.unique(all_comp_ids_mapped))
    print(f"\t\t\t{global_instance.num_unique_regular_teams_for_training} "
          f"different regular teams are going to participate in the training process")
    print(f"\t\t\t{global_instance.num_unique_regular_comps_for_training} "
          f"different regular comps are going to participate in the training process")

    # TensorBoard callbacks for the Embedding models
    log_dir_team = os.path.join("logs", "embedding_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    tensorboard_callback_embedding = TensorBoard(log_dir=log_dir_team, histogram_freq=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Step 1: Pre-train Embedding Model on Auxiliary Task
    embedding_model = build_embedding_pretrain_model(
        num_unique_teams=global_instance.num_unique_regular_teams_for_training,
        num_unique_comps=global_instance.num_unique_regular_comps_for_training,
        embedding_out_size_team=EMBEDDING_OUT_SIZE_TEAM,
        embedding_out_size_comp=EMBEDDING_OUT_SIZE_COMP
    )

    # Prepare data for embedding model training
    numerical_features, home_team_ids, away_team_ids, comp_ids, labels = get_embedding_training_data(
        regular_matches_in_rounds)
    home_team_ids_mapped = team_encoder.transform(home_team_ids)
    away_team_ids_mapped = team_encoder.transform(away_team_ids)
    comp_ids_mapped = comp_encoder.transform(comp_ids)

    # Reshape categorical IDs to have shape (batch_size, 1)
    home_team_ids_mapped = home_team_ids_mapped.reshape(-1, 1)
    away_team_ids_mapped = away_team_ids_mapped.reshape(-1, 1)
    comp_ids_mapped = comp_ids_mapped.reshape(-1, 1)

    # Train the embedding model
    embedding_model.fit(
        {'numerical_input': numerical_features,
         'home_team_input': home_team_ids_mapped,
         'away_team_input': away_team_ids_mapped,
         'comp_id_input': comp_ids_mapped},
        labels,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping_callback, tensorboard_callback_embedding]
    )

    # Step 2: Extract and Scale Embeddings
    team_embedding_weights = embedding_model.get_layer('team_embedding').get_weights()[0]
    comp_embedding_weights = embedding_model.get_layer('comp_embedding').get_weights()[0]

    # Scale embeddings to (0,1) range
    scaler_team = MinMaxScaler(feature_range=(0, 1))
    scaler_comp = MinMaxScaler(feature_range=(0, 1))

    team_embeddings_scaled = scaler_team.fit_transform(team_embedding_weights)
    comp_embeddings_scaled = scaler_comp.fit_transform(comp_embedding_weights)

    # Step 3: Train the Main Model with Pre-trained Embeddings
    train_main_model(regular_matches_in_rounds, team_embeddings_scaled, comp_embeddings_scaled)

    """
    # Step 1a: Pre-train Team embedding model
    team_input = Input(shape=(1,), dtype='int32', name='team_input')  # one input for both home and away teams

    # Embedding layers for the team IDs categorical features
    team_embedding = Embedding(input_dim=global_instance.num_unique_regular_teams_for_training,
                               output_dim=EMBEDDING_OUT_SIZE_TEAM)(team_input)
    team_embedding = BatchNormalization()(team_embedding)
    team_embedding = Lambda(scale_to_0_1)(team_embedding)
    team_flat = Flatten()(team_embedding)

    team_embedding_model = Model(inputs=team_input, outputs=team_flat)
    team_embedding_model_optimizer = Adam(learning_rate=0.0005)
    team_embedding_model.compile(optimizer=team_embedding_model_optimizer, loss='mse')
    team_embedding_model.summary()

    # Step 2a: Train the model using both home and away team IDs
    team_embedding_model.fit(all_team_ids_mapped, np.zeros((len(all_team_ids_mapped), EMBEDDING_OUT_SIZE_TEAM)),
                             epochs=150, batch_size=32, callbacks=[tensorboard_callback_team])

    # Step 1b: Pre-train Comp embedding model
    comp_id_input = Input(shape=(1,), dtype='int32', name='comp_id_input')

    # Embedding layer for the comp ID categorical feature
    comp_id_embedding = Embedding(input_dim=global_instance.num_unique_regular_comps_for_training,
                                  output_dim=EMBEDDING_OUT_SIZE_COMP)(comp_id_input)
    comp_id_embedding = BatchNormalization()(comp_id_embedding)
    comp_id_embedding = Lambda(scale_to_0_1)(comp_id_embedding)
    comp_id_flat = Flatten()(comp_id_embedding)

    comp_embedding_model = Model(inputs=comp_id_input, outputs=comp_id_flat)
    comp_embedding_model_optimizer = Adam(learning_rate=0.0005)
    comp_embedding_model.compile(optimizer=comp_embedding_model_optimizer, loss='mse')
    comp_embedding_model.summary()

    # Step 2b: Train the comp embedding model using all comp IDs
    comp_embedding_model.fit(all_comp_ids_mapped, np.zeros((len(all_comp_ids_mapped), EMBEDDING_OUT_SIZE_COMP)),
                             epochs=150, batch_size=32, callbacks=[tensorboard_callback_comp])

    # Step 3: Train the final model with precomputed embeddings
    train_main_model(regular_matches_in_rounds, team_embedding_model, comp_embedding_model)
    """


def train_main_model(regular_matches_in_rounds, team_embeddings_scaled, comp_embeddings_scaled):
    total_rounds = len(regular_matches_in_rounds)

    # Callbacks
    log_dir = os.path.join("logs", "fit" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), "rounds")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # The main model
    main_model = Sequential()
    main_model.add(Dense(128, input_dim=NUM_NUMERICAL_FEATURES + EMBEDDING_OUT_SIZE_TEAM * 2 + EMBEDDING_OUT_SIZE_COMP,
                         activation='relu'))
    main_model.add(Dense(64, activation='relu'))
    main_model.add(Dropout(0.3))
    main_model.add(Dense(1, activation='sigmoid'))

    main_mode_optimizer = Adam(learning_rate=0.001)
    main_model.compile(optimizer=main_mode_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    main_model.summary()

    for round_number in range(NUM_TRAINING_ROUNDS + 1, total_rounds):
        # Extract features and labels (training data) - DEPRECATED - just check if same values!
        train_numerical_features, train_labels = get_data_for_window(regular_matches_in_rounds, round_number,
                                                                     NUM_TRAINING_ROUNDS)
        # Extract features and labels (training data)
        train_numerical_features, train_home_team_ids, train_away_team_ids, train_comp_ids, train_labels = get_data_up_to_round(
            regular_matches_in_rounds, round_number)

        """
        train_home_team_input_data, train_away_team_input_data, train_comp_id_input_data, _ = extract_embedding_inputs(
            regular_matches_in_rounds, round_number, NUM_TRAINING_ROUNDS)
        """

        # Map categorical IDs to zero-indexed values using LabelEncoder
        train_home_team_ids_mapped = team_encoder.transform(train_home_team_ids)
        train_away_team_ids_mapped = team_encoder.transform(train_away_team_ids)
        train_comp_ids_mapped = comp_encoder.transform(train_comp_ids)

        # Reshape to (batch_size, 1)
        train_home_team_ids_mapped = train_home_team_ids_mapped.reshape(-1, 1)
        train_away_team_ids_mapped = train_away_team_ids_mapped.reshape(-1, 1)
        train_comp_ids_mapped = train_comp_ids_mapped.reshape(-1, 1)

        # Map IDs to embeddings
        train_home_team_embeddings = team_embeddings_scaled[train_home_team_ids_mapped]
        train_away_team_embeddings = team_embeddings_scaled[train_away_team_ids_mapped]
        train_comp_embeddings = comp_embeddings_scaled[train_comp_ids_mapped]

        # Concatenate embeddings with other features (training data)
        train_input = np.concatenate([train_numerical_features, train_home_team_embeddings,
                                      train_away_team_embeddings, train_comp_embeddings], axis=1)

        # Similarly for the validation data...
        """
        val_numerical_features, val_labels = get_data_for_round(regular_matches_in_rounds, round_number)
        val_home_team_input_data, val_away_team_input_data, val_comp_id_input_data, _ = extract_embedding_inputs(
            regular_matches_in_rounds, round_number + 1, 1)
        """

        val_numerical_features, val_home_team_ids, val_away_team_ids, val_comp_ids, val_labels = get_data_for_round(
            regular_matches_in_rounds, round_number + 1)

        val_home_team_ids_mapped = team_encoder.transform(val_home_team_ids)
        val_away_team_ids_mapped = team_encoder.transform(val_away_team_ids)
        val_comp_ids_mapped = comp_encoder.transform(val_comp_ids)

        val_home_team_ids_mapped = val_home_team_ids_mapped.reshape(-1, 1)
        val_away_team_ids_mapped = val_away_team_ids_mapped.reshape(-1, 1)
        val_comp_ids_mapped = val_comp_ids_mapped.reshape(-1, 1)

        val_home_team_embeddings = team_embeddings_scaled[val_home_team_ids_mapped]
        val_away_team_embeddings = team_embeddings_scaled[val_away_team_ids_mapped]
        val_comp_embeddings = comp_embeddings_scaled[val_comp_ids_mapped]

        val_input = np.concatenate([val_numerical_features, val_home_team_embeddings,
                                    val_away_team_embeddings, val_comp_embeddings], axis=1)
        print(f"\t\t\t\t\t\t\tRound {str(round_number)}: {str(train_input.shape[0])} train and"
              f" {str(val_input.shape[0])} val. data")

        # Train the main model
        main_model.fit(train_input, train_labels, epochs=5, batch_size=32, validation_data=(val_input, val_labels),
                       callbacks=[early_stopping, tensorboard_callback])

        loss, accuracy = main_model.evaluate(val_input, val_labels)
        print(f"\tRound {str(round_number)} - Loss: {str(loss)}, Accuracy: {str(accuracy)}")


def build_embedding_pretrain_model(num_unique_teams, num_unique_comps, embedding_out_size_team, embedding_out_size_comp):
    # Inputs
    numerical_input = Input(shape=(NUM_NUMERICAL_FEATURES,), name='numerical_input')
    home_team_input = Input(shape=(1,), name='home_team_input')
    away_team_input = Input(shape=(1,), name='away_team_input')
    comp_id_input = Input(shape=(1,), name='comp_id_input')

    # Shared team embedding layer
    team_embedding_layer = Embedding(input_dim=num_unique_teams, output_dim=embedding_out_size_team,
                                     name='team_embedding')

    # Team embeddings for home and away teams using the shared embedding layer
    home_team_embedding = team_embedding_layer(home_team_input)
    home_team_embedding = Flatten()(home_team_embedding)

    away_team_embedding = team_embedding_layer(away_team_input)
    away_team_embedding = Flatten()(away_team_embedding)

    comp_embedding = Embedding(input_dim=num_unique_comps, output_dim=embedding_out_size_comp, name='comp_embedding')(comp_id_input)
    comp_embedding = Flatten()(comp_embedding)

    # Concatenate embeddings with numerical input
    concatenated = Concatenate()([numerical_input, home_team_embedding, away_team_embedding, comp_embedding])

    # Dense layers
    x = Dense(128, activation='relu')(concatenated)
    x = Dense(64, activation='relu')(x)

    # Output layer (predict total goals)
    output = Dense(1, activation='linear', name='output')(x)

    # Model
    model = Model(inputs=[numerical_input, home_team_input, away_team_input, comp_id_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    model.summary()
    return model


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


def get_embedding_training_data(regular_matches_in_rounds):
    numerical_features = []
    home_team_ids = []
    away_team_ids = []
    comp_ids = []
    labels = []

    for matches in regular_matches_in_rounds:
        for match in matches:
            numerical_features.append(match.feature_vector_before_match_played)
            home_team_ids.append(match.home_team.id)
            away_team_ids.append(match.away_team.id)
            comp_ids.append(match.comp.id)
            total_goals = match.home_team_goals + match.away_team_goals
            labels.append(total_goals)  # Use total goals as label for regression

    numerical_features = np.array(numerical_features)
    home_team_ids = np.array(home_team_ids)
    away_team_ids = np.array(away_team_ids)
    comp_ids = np.array(comp_ids)
    labels = np.array(labels)

    return numerical_features, home_team_ids, away_team_ids, comp_ids, labels


def get_data_up_to_round(regular_matches_in_rounds, round_number):
    numerical_features = []
    home_team_ids = []
    away_team_ids = []
    comp_ids = []
    labels = []

    for r in range(round_number - NUM_TRAINING_ROUNDS - 1, round_number - 1):
        matches = regular_matches_in_rounds[r]
        for match in matches:
            numerical_features.append(match.feature_vector_before_match_played)
            home_team_ids.append(match.home_team.id)
            away_team_ids.append(match.away_team.id)
            comp_ids.append(match.comp.id)
            total_goals = match.home_team_goals + match.away_team_goals
            label = 1 if total_goals < 2.5 else 0
            labels.append(label)

    numerical_features = np.array(numerical_features)
    home_team_ids = np.array(home_team_ids)
    away_team_ids = np.array(away_team_ids)
    comp_ids = np.array(comp_ids)
    labels = np.array(labels)

    return numerical_features, home_team_ids, away_team_ids, comp_ids, labels


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

    numerical_features = []
    home_team_ids = []
    away_team_ids = []
    comp_ids = []
    labels = []

    for match in matches:
        numerical_features.append(match.feature_vector_before_match_played)
        home_team_ids.append(match.home_team.id)
        away_team_ids.append(match.away_team.id)
        comp_ids.append(match.comp.id)

        total_goals = match.home_team_goals + match.away_team_goals
        label = 1 if total_goals < 2.5 else 0

        # data.append(match.feature_vector_before_match_played)
        labels.append(label)

    numerical_features = np.array(numerical_features)
    home_team_ids = np.array(home_team_ids)
    away_team_ids = np.array(away_team_ids)
    comp_ids = np.array(comp_ids)
    # data = np.array(data)  # shape (num_matches, num_features)
    labels = np.array(labels)  # shape (num_matches,)

    # return data, labels
    return numerical_features, home_team_ids, away_team_ids, comp_ids, labels


def scale_to_0_1(x):
    return (x - K.min(x)) / (K.max(x) - K.min(x))
