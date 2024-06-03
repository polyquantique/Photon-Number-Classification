import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy.integrate import simps

from .Utils import norm, plot_traces


def sklearn_available(X_train : np.array, 
                     X_test : np.array, 
                     path : str, 
                     function, 
                     **param):
    """

    Execute a dimensionality reduction available in the scikit-learn library.
    If the method has a transformation, it will be trained using X_train the 
    output will be the transformation of X_test. Otherwise the X_test will directly
    be transformed using the `fit_transform` method.

    Parameters
    ----------
    X_train : ndarray
        Training dataset.
    X_test : ndarray
        Testing dataset.
    path : str
        Path when the results are stored and loaded.
    function : Sklearn method
        Dimensionality reduction method that is executed.
    param : kwargs
        Parameters used in `function`.
    

    Returns
    -------
    X_low : ndarray
        Low-dimensional representation of the samples.
    
    """
    method = function(**param)
    file_name = f"{path}/{function}_{param}.npy"
    file_name = file_name.replace('<','')
    file_name = file_name.replace('>','')
    file_name = file_name.replace(':','')
    file_name = file_name.replace(r"'",'')
    
    if os.path.isfile(file_name):
        X_low = np.load(file_name)
    else:
        if hasattr(method, 'transform'):
            trained = method.fit(X_train)
            X_low = trained.transform(X_test)
        else:
            print('Method does not have a transform function')
            X_low = method.fit_transform(X_test)
        
        X_low = norm(X_low)
        np.save(file_name, X_low)

    return X_low



def area(X_high : np.array, 
         filter : bool = False,
         cutoff : float = 0.1,
         threshold : float = None,
         plot_filter : bool = False):
    """

    Area of signals.

    Parameters
    ----------
    X_high : ndarray
    filter : bool
        Filter traces.
    plot_traces : bool
        Plot filtered traces (if filtered).

    Returns
    -------
    Area : ndarray
    
    """

    if filter:

        #X_init = signal.savgol_filter(X_init, 20, 2)
        fs = 1                # Sampling frequency
        w = cutoff / (fs / 2) # Normalize the frequency
        b, a = butter(5, w, 'low')
        X_high = filtfilt(b, a, X_high)
    
    if threshold != None:
        X_high[X_high < threshold] = threshold
        
    if plot_filter:
        plot_traces(X_high)

    X_low = simps(X_high).reshape(-1,1)

    return norm(X_low).reshape(-1,1)




def max_value(X_high : np.array, 
              filter : bool = False):
    """

    Maximum value of signals.

    Parameters
    ----------
    X_high : ndarray

    Returns
    -------
    norm : ndarray
    
    """
    if filter:
        #X_init = signal.savgol_filter(X_init, 20, 2)
        fs = X_high.shape[1]  # Sampling frequency

        fc = 15  # Cut-off frequency of the filter
        w = fc / (fs / 2) # Normalize the frequency
        b, a = butter(5, w, 'low')
        X_high = filtfilt(b, a, X_high)
    
    return norm(X_high.max(axis = 1)).reshape(-1,1)


