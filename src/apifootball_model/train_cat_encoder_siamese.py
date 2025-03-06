import os
import random
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
import settings
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

FINAL_EMBEDDING_SIZE = 16


def create_match_embedding_model():
    # Inputs for one match
    home_input = Input(shape=(1,), name='home_team_id')
    away_input = Input(shape=(1,), name='away_team_id')
    comp_input = Input(shape=(1,), name='comp_id')

    # Embedding layers with separate weights for home and away teams
    home_emb = Embedding(input_dim=settings.NUM_REGULAR_TEAMS,
                         output_dim=settings.TEAM_ID_EMBEDDING_SIZE,
                         name='home_team_embedding')(home_input)
    away_emb = Embedding(input_dim=settings.NUM_REGULAR_TEAMS,
                         output_dim=settings.TEAM_ID_EMBEDDING_SIZE,
                         name='away_team_embedding')(away_input)
    comp_emb = Embedding(input_dim=settings.NUM_REGULAR_COMPS,
                         output_dim=settings.COMP_ID_EMBEDDING_SIZE,
                         name='comp_embedding')(comp_input)

    home_flat = Flatten()(home_emb)
    away_flat = Flatten()(away_emb)
    comp_flat = Flatten()(comp_emb)

    # Map to (0,1) range via a Dense layer with sigmoid (helps align scale with other features)
    home_norm = Dense(settings.TEAM_ID_EMBEDDING_SIZE, activation='sigmoid')(home_flat)
    away_norm = Dense(settings.TEAM_ID_EMBEDDING_SIZE, activation='sigmoid')(away_flat)
    comp_norm = Dense(settings.COMP_ID_EMBEDDING_SIZE, activation='sigmoid')(comp_flat)

    joint = Concatenate(name='joint_embedding')([home_norm, away_norm, comp_norm])

    # Learn embedding the match "triple"
    x = Dense(64, activation='relu')(joint)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    embedding = Dense(FINAL_EMBEDDING_SIZE)(x)  # no activation before normalization

    # L2-normalize the final embedding - make distances comparable
    normalized_embedding = Lambda(lambda t: K.l2_normalize(t, axis=1), name='normalized_embedding')(embedding)

    model = Model(inputs=[home_input, away_input, comp_input], outputs=normalized_embedding,
                  name='match_embedding_model')
    return model


def euclidean_distance(vecs):
    x, y = vecs
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def build_siamese_model():
    embedding_model = create_match_embedding_model()

    # Inputs for a pair of matches
    home_input_a = Input(shape=(1,), name='home_team_id_a')
    away_input_a = Input(shape=(1,), name='away_team_id_a')
    comp_input_a = Input(shape=(1,), name='comp_id_a')

    home_input_b = Input(shape=(1,), name='home_team_id_b')
    away_input_b = Input(shape=(1,), name='away_team_id_b')
    comp_input_b = Input(shape=(1,), name='comp_id_b')

    # Process each with shared embedding network
    embedding_a = embedding_model([home_input_a, away_input_a, comp_input_a])
    embedding_b = embedding_model([home_input_b, away_input_b, comp_input_b])

    distance = Lambda(euclidean_distance, name='distance')([embedding_a, embedding_b])  # embeddings Euclidean distance

    siamese_net = Model(inputs=[home_input_a, away_input_a, comp_input_a,
                                home_input_b, away_input_b, comp_input_b],
                        outputs=distance, name='siamese_model')
    return siamese_net


def contrastive_loss(y_true, y_pred):
    margin = 0.6

    # y_true should be 1 for similar pairs and 0 for dissimilar ones
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def evaluate_embeddings(embedding_model, home_ids_a, away_ids_a, comp_ids_a,
                        home_ids_b, away_ids_b, comp_ids_b, labels):

    embeddings_a = embedding_model.predict([home_ids_a, away_ids_a, comp_ids_a])
    embeddings_b = embedding_model.predict([home_ids_b, away_ids_b, comp_ids_b])

    distances = np.sqrt(np.sum((embeddings_a - embeddings_b) ** 2, axis=1))

    similar_distances = distances[labels == 1]
    dissimilar_distances = distances[labels == 0]

    return similar_distances, dissimilar_distances


def train(categorical_features, similarity_labels, batch_size, num_epochs):
    home_ids_a, away_ids_a, comp_ids_a, home_ids_b, away_ids_b, comp_ids_b = categorical_features

    log_dir = os.path.join("logs", "siameseID_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1)

    siamese_model = build_siamese_model()
    siamese_model.compile(optimizer='adam', loss=contrastive_loss)
    siamese_model.summary()

    (train_home_ids_a, val_home_ids_a, train_away_ids_a, val_away_ids_a, train_comp_ids_a, val_comp_ids_a,
     train_home_ids_b, val_home_ids_b, train_away_ids_b, val_away_ids_b, train_comp_ids_b, val_comp_ids_b,
     train_similarity_labels, val_similarity_labels) = train_test_split(
        home_ids_a, away_ids_a, comp_ids_a, home_ids_b, away_ids_b, comp_ids_b,
        similarity_labels, test_size=0.2, random_state=42)

    # Train
    siamese_model.fit([train_home_ids_a, train_away_ids_a, train_comp_ids_a,
                       train_home_ids_b, train_away_ids_b, train_comp_ids_b],
                      train_similarity_labels,
                      batch_size=batch_size, epochs=num_epochs, validation_split=0.2,
                      callbacks=[tensorboard_callback, lr_scheduler])

    # Evaluate the embeddings
    similar_dists, dissimilar_dists = evaluate_embeddings(
        create_match_embedding_model(),
        val_home_ids_a, val_away_ids_a, val_comp_ids_a, val_home_ids_b, val_away_ids_b, val_comp_ids_b,
        val_similarity_labels
    )
    plt.hist(similar_dists, bins=30, alpha=0.5, label='Similar')
    plt.hist(dissimilar_dists, bins=30, alpha=0.5, label='Dissimilar')
    plt.legend()
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of Distances in Embedding Space")
    plt.show()
