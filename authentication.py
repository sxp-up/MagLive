from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
import keras.backend as K
from sklearn.metrics import roc_curve, confusion_matrix
from keras.models import load_model
import classifier

# Set GPU visibility (useful when having multiple GPUs)
os.environ [ "CUDA_VISIBLE_DEVICES"] = "1"

# Hyperparameter settings
learning_rate = 0.001
input_shape1 = (100, 1)
input_shape2 = (17, 69, 2)

# Function to compute metrics based on true and predicted values
def get_metrics(y_true, y_pred):
    threshold = 0.5
    y_pred = (y_pred > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp/(tp+fn)#recall
    tnr = tn/(tn+fp)
    bac= 0.5 *(tpr + tnr)
    fpr = 1 - tnr
    far = fpr
    frr = 1-tpr
    return bac, far, frr

# Keras custom metric for balanced accuracy
def balanced_accuracy(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))

    false_positives = K.sum(K.round(K.clip((1-y_true) * y_pred, 0, 1)))
    false_negatives = K.sum(K.round(K.clip(y_true * (1-y_pred), 0, 1)))

    recall_pos = true_positives / (true_positives + false_negatives + K.epsilon())
    recall_neg = true_negatives / (true_negatives + false_positives + K.epsilon())

    balanced_acc = (recall_pos + recall_neg) / 2
    return balanced_acc

# Keras custom metrics for false acceptance and rejection rates
def false_acceptance_rate(y_true, y_pred):
    false_positives = K.sum(K.round(K.clip((1-y_true) * y_pred, 0, 1)))
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    return false_positives / (false_positives + true_negatives + K.epsilon())

def false_rejection_rate(y_true, y_pred):
    false_negatives = K.sum(K.round(K.clip(y_true * (1-y_pred), 0, 1)))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    return false_negatives / (false_negatives + true_positives + K.epsilon())

# Function to calculate the Equal Error Rate (EER)
def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return EER, eer_threshold

# Load the pre-trained encoder model
encoder = load_model("./feature_extractor_model.h5")

# Initialize and summarize the classifier
classifier = classifier.create_classifier(encoder)
classifier.summary()

# Train the classifier
classifier.fit([mag_train, stft_train], y_train, batch_size=32,epochs=20,validation_split=0.2)

# Predict and compute metrics
y_scores = classifier.predict([mag_test, stft_test]).ravel()
bac, far, frr = get_metrics(y_test, y_scores)
eer, threshold = calculate_eer(y_test, y_scores)
print("BAC:", round(bac * 100, 2))
print("FAR:", round(far * 100, 2))
print("FRR:", round(frr * 100, 2))
print("EER:", round(eer * 100, 2))
print("Threshold for EER:", threshold)