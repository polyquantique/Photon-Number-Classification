import numpy as np
from numpy import trapz
from scipy.signal import savgol_filter


def area(X_init, filter=False):

    X_init[X_init < np.mean(X_init[:,-3:]) + 0.013] = np.mean(X_init[:,-3:]) + 0.013

    if filter:
        X_init = savgol_filter(X_init, 10, 3)
        
    X_low_dim = trapz(X_init, dx=1).reshape(-1,1)
    X_reconst = np.array([None])


    return X_init, X_reconst, X_low_dim