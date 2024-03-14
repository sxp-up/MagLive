from tensorflow.keras.layers import Conv2D, AveragePooling1D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, Input, concatenate, Conv1D, MaxPooling1D, Lambda, ReLU, GlobalAveragePooling1D,GlobalMaxPooling1D
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
from tensorflow import keras
import keras.backend as K

# Set GPU visibility (useful when having multiple GPUs)
os.environ [ "CUDA_VISIBLE_DEVICES"] = "1"

# Hyperparameter settings
learning_rate = 0.001
projection_units = 32
temperature = 0.05
input_shape1 = (100, 1)
input_shape2 = (17, 69, 2)

# Define a custom loss class for supervised contrastive learning
class SupervisedContrastiveLoss(keras.losses.Loss):
    # Initialize the loss function with temperature parameter
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    # Compute the loss between labels and feature vectors
    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize the feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute the similarity matrix (logits)
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        # Compute the N-pairs loss
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)
    
# Define a squeeze-and-excitation block
def squeeze_excite_block(input, ratio=16):
    filters = input.shape[-1] 

    reshaped_input = Lambda(lambda x: tf.expand_dims(x, axis=1))(input)

    se = tf.keras.layers.GlobalAveragePooling1D()(reshaped_input)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)

    print(se.shape)

    return tf.keras.layers.multiply([input, se])

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
    merged = squeeze_excite_block(merged)

    model = Model(inputs=[input_layer1, input_layer2], outputs=merged, name="double-encoder")
    model.summary()
    return model

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
encoder = create_encoder()

encoder_with_projection_head = add_projection_head(encoder)

encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature)
)

# Train the model
encoder_with_projection_head.fit([mag_data, stft_data], labels, batch_size=32,epochs=30,validation_split=0.2)

encoder.save('encode_model.h5')