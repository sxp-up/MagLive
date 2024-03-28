from tensorflow.keras.layers import Conv2D, AveragePooling1D, MaxPooling2D, BatchNormalization, Flatten, Dense, Input, concatenate, Conv1D,  ReLU
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
from tensorflow import keras
import keras.backend as K
import fusion

input_shape1 = (100, 1)
input_shape2 = (17, 69, 2)

# Create the encoder model
def create_encoder():
    input_layer1 = Input(shape=input_shape1, name='input1')
    input_layer2 = Input(shape=input_shape2, name='input2')

    x1 = Conv1D(16, kernel_size=3)(input_layer1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv1D(32, kernel_size=3)(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv1D(16, kernel_size=3)(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = AveragePooling1D(pool_size=2)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(64, activation='relu')(x1)

    x2 = Conv2D(16, kernel_size=(3, 3), activation='relu')(input_layer2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Flatten()(x2)
    x2 = Dense(64, activation='relu')(x2)

    merged = concatenate([x1, x2])

    # Apply squeeze-excite block
    merged = fusion.squeeze_excite_block(merged)

    model = Model(inputs=[input_layer1, input_layer2], outputs=merged, name="double-encoder")
    model.summary()
    return model
