import numpy as np

def normalize_spectrum(spectrum):

    mean = np.mean(spectrum)
    std = np.std(spectrum)
    
    normalized_spectrum = (spectrum - mean) / std
    
    return normalized_spectrum, mean, std

def normalize_velocity_field(velocity_field):

    mean = np.mean(velocity_field)
    std = np.std(velocity_field)
    
    normalized_velocity_field = (velocity_field - mean) / std
    
    return normalized_velocity_field, mean, std

def mse_velocity_signals(velocity_signal1, velocity_signal2):
    
    assert velocity_signal1.shape == velocity_signal2.shape, "The input signals must have the same shape"

    squared_diff = (velocity_signal1 - velocity_signal2) ** 2

    mse = np.mean(squared_diff)

    return mse