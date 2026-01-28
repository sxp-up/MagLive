import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

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
    