import numpy as np
from scipy.signal import savgol_filter

def max_value(X_init, filter=False):

    if filter:
        X_init = savgol_filter(X_init, 20, 2)

    X_low_dim = np.max(X_init, axis=1).reshape(-1,1)
    X_reconst = np.array([None])

    X_low_dim = (X_low_dim - np.min(X_low_dim)) / (np.max(X_low_dim) - np.min(X_low_dim))
    
    return X_init, X_reconst, X_low_dim