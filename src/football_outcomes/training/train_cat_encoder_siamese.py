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
from football_outcomes.config import settings
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


def generate_pairs_random(all_input_data, k=10):
    """
    Generates pairs of matches with 'similar' (1) and 'dissimilar' (0) labels using random sampling.

    all_input_data: List of tuples (home_id, away_id, comp_id, label) where label=1 if <2.5 goals, else 0.
    k: Number of matches to sample per match for similar and dissimilar pools.
    """
    indices_label1 = [i for i, (_, _, _, label) in enumerate(all_input_data) if label == 1]
    indices_label0 = [i for i, (_, _, _, label) in enumerate(all_input_data) if label == 0]

    home_ids_a, away_ids_a, comp_ids_a = [], [], []
    home_ids_b, away_ids_b, comp_ids_b = [], [], []
    similarity_labels = []

    n = len(all_input_data)
    for i in range(n):
        a0, a1, a2, a_label = all_input_data[i]
        if a_label == 1:
            similar_pool = [idx for idx in indices_label1 if idx != i]
            dissimilar_pool = indices_label0
        else:
            similar_pool = [idx for idx in indices_label0 if idx != i]
            dissimilar_pool = indices_label1

        # Sample up to k matches from each pool
        similar_sample = random.sample(similar_pool, min(k, len(similar_pool))) if similar_pool else []
        dissimilar_sample = random.sample(dissimilar_pool, min(k, len(dissimilar_pool))) if dissimilar_pool else []

        # Similar pairs
        for j in similar_sample:
            b0, b1, b2, _ = all_input_data[j]
            home_ids_a.append(a0)
            away_ids_a.append(a1)
            comp_ids_a.append(a2)
            home_ids_b.append(b0)
            away_ids_b.append(b1)
            comp_ids_b.append(b2)
            similarity_labels.append(1)

        # Dissimilar pairs
        for j in dissimilar_sample:
            b0, b1, b2, _ = all_input_data[j]
            home_ids_a.append(a0)
            away_ids_a.append(a1)
            comp_ids_a.append(a2)
            home_ids_b.append(b0)
            away_ids_b.append(b1)
            comp_ids_b.append(b2)
            similarity_labels.append(0)

    return (np.array(home_ids_a), np.array(away_ids_a), np.array(comp_ids_a),
            np.array(home_ids_b), np.array(away_ids_b), np.array(comp_ids_b),
            np.array(similarity_labels))


def generate_pairs_hard_negatives(embedding_model, original_pairs, margin=0.8, fraction_hard_neg=0.5):
    """
    Generates a new set of pairs that includes a higher proportion of hard negatives.

    siamese_model: The trained Siamese model.
    embedding_model: The submodel that maps a single match to an embedding.
    all_input_data: The original list of matches (home_id, away_id, comp_id, label).
    original_pairs: A tuple of (home_ids_a, away_ids_a, comp_ids_a, home_ids_b, away_ids_b, comp_ids_b, similarity_labels).
    margin: The contrastive margin used in training.
    fraction_hard_neg: Proportion of negative pairs in the new dataset that should be "hard".
    """
    (home_ids_a, away_ids_a, comp_ids_a,
     home_ids_b, away_ids_b, comp_ids_b,
     sim_labels) = original_pairs

    # 1. Compute embeddings for each side
    embeddings_a = embedding_model.predict([home_ids_a, away_ids_a, comp_ids_a])
    embeddings_b = embedding_model.predict([home_ids_b, away_ids_b, comp_ids_b])

    # 2. Compute distances
    distances = np.sqrt(np.sum((embeddings_a - embeddings_b) ** 2, axis=1))

    # 3. Identify negative pairs (label=0) that are below the margin => "hard negatives"
    neg_mask = (sim_labels == 0)
    hard_neg_mask = neg_mask & (distances < margin)

    easy_neg_mask = neg_mask & (distances >= margin)
    pos_mask = (sim_labels == 1)

    # 4. Extract arrays for each category
    hard_neg_indices = np.where(hard_neg_mask)[0]
    easy_neg_indices = np.where(easy_neg_mask)[0]
    pos_indices = np.where(pos_mask)[0]

    # 5. Decide how many easy negatives vs. hard negatives to keep
    # For example, keep all positives, keep half easy negs, half hard negs:
    num_neg = len(hard_neg_indices) + len(easy_neg_indices)
    # fraction_hard_neg = fraction of negative pairs that are "hard"
    desired_hard_count = int(num_neg * fraction_hard_neg)
    desired_easy_count = num_neg - desired_hard_count

    # If we have fewer hard negs than desired, we keep them all
    if len(hard_neg_indices) < desired_hard_count:
        chosen_hard_neg = hard_neg_indices
    else:
        chosen_hard_neg = np.random.choice(hard_neg_indices, size=desired_hard_count, replace=False)

    if len(easy_neg_indices) < desired_easy_count:
        chosen_easy_neg = easy_neg_indices
    else:
        chosen_easy_neg = np.random.choice(easy_neg_indices, size=desired_easy_count, replace=False)

    # Keep all positives
    chosen_pos = pos_indices

    # Combine all chosen indices
    final_indices = np.concatenate([chosen_pos, chosen_hard_neg, chosen_easy_neg])
    np.random.shuffle(final_indices)

    # 6. Build final arrays
    return (home_ids_a[final_indices], away_ids_a[final_indices], comp_ids_a[final_indices],
            home_ids_b[final_indices], away_ids_b[final_indices], comp_ids_b[final_indices],
            sim_labels[final_indices])


def train_with_hard_negatives(all_input_data, batch_size=32, num_epochs=10, k=10, margin=0.6, fraction_hard_neg=0.5):
    """
    Demonstration of a simple two-phase training:
      1) Train on random pairs
      2) Mine hard negatives, re-sample the training set, re-train
    """
    # --- 1. Generate initial random pairs ---
    initial_pairs = generate_pairs_random(all_input_data, k=k)
    (home_ids_a, away_ids_a, comp_ids_a,
     home_ids_b, away_ids_b, comp_ids_b,
     sim_labels) = initial_pairs

    # Train/test split
    (train_home_ids_a, val_home_ids_a,
     train_away_ids_a, val_away_ids_a,
     train_comp_ids_a, val_comp_ids_a,
     train_home_ids_b, val_home_ids_b,
     train_away_ids_b, val_away_ids_b,
     train_comp_ids_b, val_comp_ids_b,
     train_similarity_labels, val_similarity_labels) = train_test_split(
        home_ids_a, away_ids_a, comp_ids_a,
        home_ids_b, away_ids_b, comp_ids_b,
        sim_labels, test_size=0.2, random_state=42
    )

    # --- 2. Build and train the Siamese model on random pairs ---
    siamese_model = build_siamese_model()
    siamese_model.compile(optimizer='adam', loss=contrastive_loss)

    log_dir = os.path.join("logs", "siameseID_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1)

    siamese_model.fit([train_home_ids_a, train_away_ids_a, train_comp_ids_a,
                       train_home_ids_b, train_away_ids_b, train_comp_ids_b],
                      train_similarity_labels,
                      batch_size=batch_size, epochs=num_epochs, validation_split=0.2,
                      callbacks=[tensorboard_callback, lr_scheduler])

    # Evaluate on validation
    # You can also do your evaluate_embeddings(...) step here.

    # --- 3. Mine hard negatives and re-sample the training set ---
    embedding_model = siamese_model.get_layer('match_embedding_model')  # The shared embedding submodel
    # Build new pairs that incorporate more hard negatives
    new_train_pairs = generate_pairs_hard_negatives(
        embedding_model=embedding_model,
        original_pairs=(train_home_ids_a, train_away_ids_a, train_comp_ids_a,
                        train_home_ids_b, train_away_ids_b, train_comp_ids_b,
                        train_similarity_labels),
        margin=margin,
        fraction_hard_neg=fraction_hard_neg
    )

    (train_home_ids_a_hn, train_away_ids_a_hn, train_comp_ids_a_hn,
     train_home_ids_b_hn, train_away_ids_b_hn, train_comp_ids_b_hn,
     train_similarity_labels_hn) = new_train_pairs

    # Optionally, you can keep the same validation set:

    # --- 4. Retrain (or continue training) with the new dataset containing more hard negatives ---
    siamese_model.fit([train_home_ids_a_hn, train_away_ids_a_hn, train_comp_ids_a_hn,
                       train_home_ids_b_hn, train_away_ids_b_hn, train_comp_ids_b_hn],
                      train_similarity_labels_hn,
                      batch_size=batch_size, epochs=num_epochs, validation_data=(
            [val_home_ids_a, val_away_ids_a, val_comp_ids_a,
             val_home_ids_b, val_away_ids_b, val_comp_ids_b],
            val_similarity_labels
        ),
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

    # Final evaluation with evaluate_embeddings or your downstream tasks
    return siamese_model

