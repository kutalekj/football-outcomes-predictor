import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from football_outcomes.config import settings

tf.random.set_seed(42)
tf.keras.utils.set_random_seed(41)


def train(mapped_team_ids, batch_size, num_epochs):
    log_dir = os.path.join("logs", "teamID_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Data
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        np.array(mapped_team_ids),
        np.array(mapped_team_ids),
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    train_labels = to_categorical(train_labels, num_classes=settings.NUM_REGULAR_TEAMS)
    val_labels = to_categorical(val_labels, num_classes=settings.NUM_REGULAR_TEAMS)

    # Model
    team_input = Input(shape=(1,), name="team_id")
    team_embedding = Embedding(
        input_dim=settings.NUM_REGULAR_TEAMS,
        output_dim=settings.TEAM_ID_EMBEDDING_SIZE,
        name="team_embedding",
    )(team_input)
    comp_embed_flat = Flatten()(team_embedding)

    # MLP
    x = Dense(32, activation="relu")(comp_embed_flat)
    x = Dense(16, activation="relu")(x)
    output = Dense(settings.NUM_REGULAR_TEAMS, activation="softmax")(x)

    autoencoder_model = Model(inputs=[team_input], outputs=output)
    autoencoder_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    autoencoder_model.summary()

    # Train
    autoencoder_model.fit(
        [train_ids],
        train_labels,
        validation_data=([val_ids], val_labels),
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[tensorboard_callback],
    )

    # Save trained model
    model_path = settings.TRAINED_MODELS_DIR + "\\team_id_embedding_model.keras"
    autoencoder_model.save(model_path)
    print(f"Model saved to {model_path}")

    # Extract predicted embeddings
    embedding_layer = autoencoder_model.get_layer("team_embedding")
    team_embeddings = embedding_layer.get_weights()[0]
    normalized_embeddings = normalize_embeddings(team_embeddings)
    print(normalized_embeddings)

    return normalized_embeddings


def normalize_embeddings(embeddings):
    min_val = np.min(embeddings)
    max_val = np.max(embeddings)
    return (embeddings - min_val) / (max_val - min_val)
