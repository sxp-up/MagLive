from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
from tensorflow import keras
import keras.backend as K
from loss import SupervisedContrastiveLoss
import submodel

# Set GPU visibility (useful when having multiple GPUs)
os.environ [ "CUDA_VISIBLE_DEVICES"] = "1"

# Hyperparameter settings
learning_rate = 0.001
projection_units = 32
temperature = 0.05
input_shape1 = (100, 1)
input_shape2 = (17, 69, 2)

# Add a projection head to the encoder
def add_projection_head(encoder):
    inputs1 = keras.Input(shape=input_shape1)
    inputs2 = keras.Input(shape=input_shape2)
    features = encoder([inputs1, inputs2])
    outputs = keras.layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=[inputs1, inputs2], outputs=outputs, name="encoder_with_projection-head"
    )
    model.summary()
    return model

# Compile the encoder
encoder = submodel.create_encoder()

encoder_with_projection_head = add_projection_head(encoder)

encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature)
)

# Train the model
encoder_with_projection_head.fit([mag_data, stft_data], labels, batch_size=32,epochs=30,validation_split=0.2)

encoder.save('feature_extractor_model.h5')