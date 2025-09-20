import os
import random
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, Reshape, Activation, Dropout, \
    BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from football_outcomes.config import settings
from football_outcomes.utils import common as ut
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')


def build_team_strength_autoencoder(conv_filters, dense_neurons, dropout, kernel_size, final_embedding_size,
                                    input_shape=(11, 34)):
    team_skills_input = Input(shape=input_shape, name='team_skills')

    # Encoder
    x = Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', padding='same')(team_skills_input)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)  # aggregate over the 11 players -> vector of size 16
    x = Dropout(dropout)(x)
    team_strength = Dense(final_embedding_size, activation='sigmoid', name='team_strength')(x)

    # Decoder (reconstruct the original 11x34 matrix from the 8-dimensional vector)
    y = Dense(dense_neurons, activation='relu')(team_strength)
    y = BatchNormalization()(y)
    y = Dense(input_shape[0] * input_shape[1], activation='linear')(y)
    reconstructed = Reshape(input_shape)(y)

    autoencoder = Model(inputs=team_skills_input, outputs=reconstructed, name='team_strength_autoencoder')
    autoencoder.compile(optimizer='adam', loss='mse')
    # autoencoder.summary()

    # Build a separate encoder model for later usage
    encoder = Model(inputs=team_skills_input, outputs=team_strength, name='team_strength_encoder')

    return autoencoder, encoder


def train(team_player_skills, batch_size=32, num_epochs=100):
    # Avoid domination of outfield player skills (29/34) over goalkeeper skills (5/34)
    team_player_skills = ut.separate_normalize_gk_and_outfield_skills(team_player_skills)

    # Callbacks
    log_dir = os.path.join("logs", "team_strength_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=int(num_epochs / 2), restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, verbose=1)

    # Data
    train_data, val_data = train_test_split(team_player_skills, test_size=0.2, random_state=42, shuffle=True)

    # Build models
    autoencoder, encoder = build_team_strength_autoencoder(conv_filters=256, dense_neurons=256, dropout=0.2,
                                                           kernel_size=7, final_embedding_size=24)

    # Train autoencoder
    autoencoder.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=num_epochs,
                    batch_size=batch_size, callbacks=[tensorboard_callback, early_stopping, lr_scheduler])

    # Visual inspection of learned embeddings
    cols, rows = 2, 4  # cols x rows grid
    plot_counter = 0
    for _ in range(cols * rows):
        random_number = int(random.uniform(0, val_data.shape[0]))

        sample = val_data[random_number]
        reconstructed_sample = autoencoder.predict(np.expand_dims(sample, axis=0))[0]

        plt.subplot(rows, cols, plot_counter % (rows * cols) + 1)
        plt.title(f"Original Skills (Sample {random_number})", fontsize=10)
        plt.imshow(sample, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()

        plt.subplot(rows, cols, plot_counter % (rows * cols) + 2)
        plt.title(f"Reconstructed Skills (Sample {random_number})", fontsize=10)
        plt.imshow(reconstructed_sample, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar()

        plot_counter += 2

        if plot_counter % (rows * cols) == 0:
            plt.tight_layout()
            plt.show()  # display plots every (rows * cols) iterations

    # Save encoder
    model_path = settings.TRAINED_MODELS_DIR + "\\team_strength_embedding_model_gk_outfield_balanced.keras"
    encoder.save(model_path)
    print(f"Model saved to {model_path}")

    return autoencoder, encoder  # encoder can be used to extract team strength embeddings after training

