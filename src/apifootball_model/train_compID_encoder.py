import settings
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Activation
from tensorflow.keras.models import Model

comp_input = Input(shape=(1,), name='competition_id')
comp_embedding = Embedding(input_dim=settings.NUM_REGULAR_COMPS, output_dim=settings.COMP_ID_EMBEDDING_SIZE,
                           name='competition_embedding')(comp_input)
comp_embed_flat = Flatten()(comp_embedding)

comp_embed_norm = Activation('sigmoid')(comp_embed_flat)  # normalize values to (0,1)

# MLP
x = Dense(32, activation='relu')(comp_embed_norm)
x = Dense(16, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

embedding_model = Model(
    inputs=[comp_input],
    outputs=output
)
embedding_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
embedding_model.summary()

# TODO: get data and pass them to model training...
