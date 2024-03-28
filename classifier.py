from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K

# Hyperparameter settings
learning_rate = 0.001
input_shape1 = (100, 1)
input_shape2 = (17, 69, 2)

# Define the classifier model
def create_classifier(encoder):
    inputs1 = keras.Input(shape=input_shape1)
    inputs2 = keras.Input(shape=input_shape2)
    features = encoder([inputs1, inputs2])
    z = Dense(64)(features)
    z = Dense(1, activation='sigmoid')(z)

    model = keras.Model(inputs=[inputs1, inputs2], outputs=z, name="classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate), 
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()]
    )
    model.summary()
    return model
