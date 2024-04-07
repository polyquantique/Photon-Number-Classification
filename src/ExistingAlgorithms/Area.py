import numpy as np
from scipy import signal
from scipy.integrate import simps


def area(X_init, filter=False, threshold_cst=0.01):

    #threshold = np.mean(X_init[:,-3:]) + threshold_cst
    X_init[X_init < threshold_cst] = threshold_cst

    if filter:
        #X_init = signal.savgol_filter(X_init, 10, 3)
        fs = X_init.shape[1]  # Sampling frequency

        fc = 12  # Cut-off frequency of the filter
        w = fc / (fs / 2) # Normalize the frequency
        b, a = signal.butter(5, w, 'low')
        X_init = signal.filtfilt(b, a, X_init)
        
    X_low_dim = simps(X_init).reshape(-1,1)     #, dx=1
    X_reconst = np.array([None])

    
    X_low_dim = (X_low_dim - np.min(X_low_dim)) / (np.max(X_low_dim) - np.min(X_low_dim))
    #X_low_dim = X_low_dim[X_low_dim[:,0] > 0.01]

    return X_init, X_reconst, X_low_dim