import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, sosfiltfilt, hilbert
import math

# Function to apply a Butterworth high-pass filter to the data
def butter_highpass_filter(WK):
    N = 6  
    Wn = 0.1  
    B, A = butter(N, Wn, output='ba', btype='high') 
    smooth_data = filtfilt(B, A, WK) 
    return smooth_data  

# Function to filter magnetic signal data along x, y, z axes
def mag_filt(mag):
    x = butter_highpass_filter(mag['x'])  
    y = butter_highpass_filter(mag['y'])  
    z = butter_highpass_filter(mag['z'])  
    total = x * x + y * y + z * z  
    total_sqrt = [math.sqrt(score) for score in total] 
    return x, y, z, total, np.array(total_sqrt) 

# Function to pad magnetic signal data to a target length (100 points here)
def mag_pad(data):
    target_length = 100
    pad = []
    for i in range(data.shape[0]):
        current_length = data[i,1] - data[i,0]
        padding_needed = target_length - current_length
        pad_before = padding_needed // 2 
        pad_after = padding_needed - pad_before 
        pad.append(np.array([data[i,0]-pad_before, data[i,1]+pad_after]))
    return np.array(pad) 

# Function to normalize the data
def normalization(data):
    _range = np.max(data) - np.min(data)  
    return (data - np.min(data)) / _range 

# Function to calculate the envelope of the signal using Hilbert transform
def envelope(data):
    analytic_signal = hilbert(data) 
    env = np.abs(analytic_signal) 
    return env 
