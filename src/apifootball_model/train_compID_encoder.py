import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import settings


def train(all_team_ids, batch_size, num_epochs):
    log_dir = os.path.join("logs", "compID_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Data
    team_ids = np.array(all_team_ids)
    train_ids, val_ids = train_test_split(team_ids, test_size=0.2, random_state=42, shuffle=True)

    # Model
    comp_input = Input(shape=(1,), name='competition_id')
    comp_embedding = Embedding(input_dim=settings.NUM_REGULAR_COMPS, output_dim=settings.COMP_ID_EMBEDDING_SIZE,
                               name='competition_embedding')(comp_input)
    comp_embed_flat = Flatten()(comp_embedding)

    comp_embed_norm = Activation('sigmoid')(comp_embed_flat)  # normalize values to (0,1)

    # MLP
    x = Dense(16, activation='relu')(comp_embed_norm)
    output = Dense(1, activation='sigmoid')(x)

    embedding_model = Model(
        inputs=[comp_input],
        outputs=output
    )
    embedding_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    embedding_model.summary()

    # Train
    embedding_model.fit(train_ids, train_ids, validation_data=(val_ids, val_ids),
                        epochs=num_epochs, batch_size=batch_size, callbacks=[tensorboard_callback])
