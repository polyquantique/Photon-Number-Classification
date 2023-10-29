import numpy as np
from numpy import trapz
from scipy.signal import savgol_filter
from scipy.integrate import simps


def area(X_init, filter=False, threshold_cst=0.01):

    threshold = np.mean(X_init[:,-3:]) + threshold_cst
    X_init[X_init < threshold] = threshold

    if filter:
        X_init = savgol_filter(X_init, 10, 4)

    import matplotlib.pyplot as plt
    [plt.plot(i) for i in X_init[::10]]
    plt.hlines(threshold, 0, 250)
        
    X_low_dim = simps(X_init).reshape(-1,1)     #, dx=1
    X_reconst = np.array([None])

    
    X_low_dim = (X_low_dim - np.min(X_low_dim)) / (np.max(X_low_dim) - np.min(X_low_dim))
    X_low_dim = X_low_dim[X_low_dim[:,0] > 0.01]

    return X_init, X_reconst, X_low_dim