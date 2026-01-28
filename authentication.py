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