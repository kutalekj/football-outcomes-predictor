from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, Flatten, Concatenate, BatchNormalization, \
    Lambda, Activation
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
import os
import settings
import utils as ut

NUM_TRAINING_ROUNDS = 25

PAR_NEURONS = {'neu_1': 32, 'neu_2': 64, 'neu_3': 128}
PAR_TEAM_EMB = {'teamem_1': 6, 'teamem_2': 9, 'teamem_3': 12}
PAR_COMP_EMB = {'compem_1': 2, 'compem_2': 3, 'compem_3': 4}
PAR_LR = {'lr_1': 0.00002, 'lr_2': 0.00007, 'lr_3': 0.0002, 'lr_4': 0.0007}
PAR_DROPOUT = {'drp_1': 0.15, 'drp_2': 0.3, 'drp_3': 0.45}
PAR_ACTIV = {'act_1': 'relu', 'act_2': 'leaky_relu'}
PAR_ITER = {'it_1': 1, 'it_2': 2, 'it_3': 3, 'it_4': 4}


def build_mlp(dense1_neurons=256, dense2_neurons=128, dense3_neurons=64, dropout1=0.6, dropout2=0.5, dropout3=0.4,
              lr=0.00007):
    # Inputs
    numerical_input = Input(shape=(settings.NUM_NUMERICAL_FEATURES,), dtype='float32', name='numerical_input')
    home_team_input = Input(shape=(settings.TEAM_ID_EMBEDDING_SIZE,), dtype='float32', name='home_team_input')
    away_team_input = Input(shape=(settings.TEAM_ID_EMBEDDING_SIZE,), dtype='float32', name='away_team_input')
    comp_input = Input(shape=(settings.COMP_ID_EMBEDDING_SIZE,), dtype='float32', name='comp_input')
    home_team_strength_input = Input(shape=(settings.TEAM_STRENGTH_EMBEDDING_SIZE,), dtype='float32',
                                     name='home_team_strength_input')
    away_team_strength_input = Input(shape=(settings.TEAM_STRENGTH_EMBEDDING_SIZE,), dtype='float32',
                                     name='away_team_strength_input')

    numerical_input = Flatten()(numerical_input)
    home_team_input = Flatten()(home_team_input)
    away_team_input = Flatten()(away_team_input)
    comp_input = Flatten()(comp_input)
    home_team_strength_input = Flatten()(home_team_strength_input)
    away_team_strength_input = Flatten()(away_team_strength_input)

    # Concatenate
    merged = Concatenate()([numerical_input, home_team_input, away_team_input, comp_input,
                            home_team_strength_input, away_team_strength_input])

    # Model
    x = Dense(dense1_neurons, activation='relu')(merged)
    x = Dropout(dropout1)(x)
    x = Dense(dense2_neurons, activation='relu')(x)
    x = Dropout(dropout2)(x)
    x = Dense(dense3_neurons, activation='relu')(x)
    x = Dropout(dropout3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[numerical_input, home_team_input, away_team_input, comp_input,
                          home_team_strength_input, away_team_strength_input], outputs=output)

    model_optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=model_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def train(regular_matches_in_rounds, team_id_map, comp_id_map):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  # TODO: Acceleration

    # Load pre-trained embedding models
    comp_id_embedding_model = load_model(settings.COMP_ID_EMBEDDING_MODEL_PATH)
    team_id_embedding_model = load_model(settings.TEAM_ID_EMBEDDING_MODEL_PATH)
    team_strength_embedding_model = load_model(settings.TEAM_STRENGTH_EMBEDDING_MODEL_PATH)

    comp_id_embedding_model = get_embedding_extractor(comp_id_embedding_model, 'competition_embedding')
    team_id_embedding_model = get_embedding_extractor(team_id_embedding_model, 'team_embedding')

    # Callbacks
    log_dir = os.path.join("logs", "fit" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_ann_')
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Build main model
    model = build_mlp()

    # Train
    weighted_accuracy = []
    accuracies = []
    total_val_matches = 0
    total_rounds = len(regular_matches_in_rounds)

    for round_number in range(NUM_TRAINING_ROUNDS + 1, total_rounds):
        # Extract features and labels (get training and validation data)
        train_numerical_features, train_labels = get_data_for_window(regular_matches_in_rounds, round_number,
                                                                     NUM_TRAINING_ROUNDS)
        train_home_ids, train_away_ids, train_comp_ids, train_home_strengths, train_away_strengths = \
            extract_embeddings(regular_matches_in_rounds, round_number, NUM_TRAINING_ROUNDS,
                               team_id_embedding_model, comp_id_embedding_model, team_strength_embedding_model,
                               team_id_map, comp_id_map)

        val_numerical_features, val_labels = get_data_for_round(regular_matches_in_rounds, round_number)
        val_home_ids, val_away_ids, val_comp_ids, val_home_strengths, val_away_strengths = \
            extract_embeddings(regular_matches_in_rounds, round_number + 1, 1,
                               team_id_embedding_model, comp_id_embedding_model, team_strength_embedding_model,
                               team_id_map, comp_id_map)

        print(f"\t\t\t\t\t\t\tRound {str(round_number)}: {str(train_numerical_features.shape)} train and"
              f" {str(val_numerical_features.shape)} val. data")
        total_val_matches += val_numerical_features.shape[0]

        # Train
        model.fit(
            [train_numerical_features, train_home_ids, train_away_ids, train_comp_ids,
             train_home_strengths, train_away_strengths],
            train_labels,
            epochs=10,
            batch_size=32,
            validation_data=(
                [val_numerical_features, val_home_ids, val_away_ids, val_comp_ids,
                 val_home_strengths, val_away_strengths],
                val_labels
            ),
            callbacks=[early_stopping, tensorboard_callback]
        )

        # Evaluate
        loss, accuracy = model.evaluate([val_numerical_features, val_home_ids, val_away_ids, val_comp_ids,
                                         val_home_strengths, val_away_strengths], val_labels)

        print(f"\tRound {str(round_number)} - Loss: {str(loss)}, Accuracy: {str(accuracy)}")

        weighted_accuracy.append(accuracy * val_numerical_features.shape[0])
        accuracies.append(accuracy)

    # Save encoder
    model_path = settings.TRAINED_MODELS_DIR + "\\main_model_ann.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Overall evaluation
    final_weighted_acc = float(np.sum(weighted_accuracy) / total_val_matches)
    print(f"\tWeighted validation accuracy = "f"{final_weighted_acc}")
    last_100_rounds_acc = float(np.sum(accuracies[-100:]) / 100)
    print(f"\tAverage accuracy in last 100 rounds = "f"{last_100_rounds_acc:.3%}")


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


def extract_embeddings(regular_matches_in_rounds, round_number, window_size,
                       team_id_embedding_model, comp_id_embedding_model, team_strength_embedding_model,
                       team_id_map, comp_id_map):
    start_round = round_number - window_size - 1
    end_round = round_number - 1

    home_id_inputs = []
    away_id_inputs = []
    comp_id_inputs = []
    home_strength_inputs = []
    away_strength_inputs = []

    for r in range(start_round, end_round):
        matches = regular_matches_in_rounds[r]

        for match in matches:
            # Map raw categorical features to indices
            home_id_inputs.append(team_id_map[match.home_team.id])
            away_id_inputs.append(team_id_map[match.away_team.id])
            comp_id_inputs.append(comp_id_map[match.comp.id])

            # Scale team strength features skills to [0,1]
            home_strength = np.array(
                [[z / 100.0 for z in y] for y in match.features_before_match_played.home_team_strength])
            away_strength = np.array(
                [[z / 100.0 for z in y] for y in match.features_before_match_played.away_team_strength])

            # Normalize team strength features (expected shape is (11, 34) - add batch dimension, then remove it)
            home_strength_norm = np.squeeze(
                ut.separate_normalize_gk_and_outfield_skills(np.expand_dims(home_strength, axis=0)), axis=0)
            away_strength_norm = np.squeeze(
                ut.separate_normalize_gk_and_outfield_skills(np.expand_dims(away_strength, axis=0)), axis=0)

            home_strength_inputs.append(home_strength_norm)
            away_strength_inputs.append(away_strength_norm)

    # For categorical features add extra dimension so each input is shape (1,)
    home_id_array = np.expand_dims(np.array(home_id_inputs), axis=-1)  # shape: (num_samples, 1)
    away_id_array = np.expand_dims(np.array(away_id_inputs), axis=-1)
    comp_id_array = np.expand_dims(np.array(comp_id_inputs), axis=-1)

    # For team strength assuming each normalized sample is (11, 34)
    home_strength_array = np.array(home_strength_inputs)  # shape: (num_samples, 11, 34)
    away_strength_array = np.array(away_strength_inputs)

    # Batch predict embeddings
    home_id_embeddings = np.squeeze(team_id_embedding_model.predict(home_id_array), axis=1)  # shape (num_samples, 8)
    away_id_embeddings = np.squeeze(team_id_embedding_model.predict(away_id_array), axis=1)
    comp_id_embeddings = np.squeeze(comp_id_embedding_model.predict(comp_id_array), axis=1)
    home_strength_embeddings = team_strength_embedding_model.predict(home_strength_array)
    away_strength_embeddings = team_strength_embedding_model.predict(away_strength_array)

    # Normalize predicted embeddings to [0,1]
    home_id_embeddings = normalize_embeddings(home_id_embeddings)
    away_id_embeddings = normalize_embeddings(away_id_embeddings)
    comp_id_embeddings = normalize_embeddings(comp_id_embeddings)

    return (home_id_embeddings, away_id_embeddings, comp_id_embeddings,
            home_strength_embeddings, away_strength_embeddings)


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


def get_embedding_extractor(full_model, embedding_layer_name):
    # Construct new model taking same input as full_model and returning activations of specified embedding layer
    return Model(inputs=full_model.input,
                 outputs=full_model.get_layer(embedding_layer_name).output)


def normalize_embeddings(embeddings):
    min_val = np.min(embeddings)
    max_val = np.max(embeddings)
    return (embeddings - min_val) / (max_val - min_val)
