from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
from tensorflow import keras
import keras.backend as K


# Define a squeeze-and-excitation block
def squeeze_excite_block(input, ratio=16):
    filters = input.shape[-1] 

    reshaped_input = Lambda(lambda x: tf.expand_dims(x, axis=1))(input)

    se = tf.keras.layers.GlobalAveragePooling1D()(reshaped_input)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)

    print(se.shape)

    return tf.keras.layers.multiply([input, se])