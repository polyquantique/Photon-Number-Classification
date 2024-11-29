import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from scipy.stats import poisson
import zipfile
import polars as pl

Average = [7.08211260e+06, 5.15056588e+06, 3.72436481e+06, 2.71572185e+06,
            1.97472603e+06, 1.42408918e+06, 1.02232317e+06, 7.32125310e+05,
            5.24090642e+05, 3.74998464e+05, 2.66916821e+05, 1.89078950e+05,
            1.35700261e+05, 9.74077327e+04, 6.94691976e+04, 4.90518881e+04,
            3.50311232e+04, 2.50723929e+04, 1.77261578e+04, 1.26385561e+04,
            9.03448510e+03, 6.34776305e+03, 4.44228612e+03, 3.11561879e+03,
            2.20105919e+03, 1.55472974e+03, 1.10119877e+03, 7.74547737e+02,
            5.46214434e+02, 3.86814856e+02, 2.77270797e+02, 1.97954232e+02,
            1.39667657e+02, 9.96796628e+01, 7.12159618e+01, 5.03875460e+01,
            3.56871128e+01, 2.54771649e+01, 1.80622346e+01, 1.27380046e+01,
            9.02367143e+00, 6.40704597e+00, 4.54269070e+00, 3.20272751e+00,
            2.26309906e+00]

dB = [ 7.0,   7.5,  8.0,   8.5,  9.0,   9.5, 10.,  10.5, 11.,  11.5, 12.,  12.5, 13.,  13.5,
       14.,  14.5, 15.,  15.5, 16.,  16.5, 17.,  17.5, 18.,  18.5, 19.,  19.5, 20.,  20.5,
       21.,  21.5, 22.,  22.5, 23.,  23.5, 24.,  24.5, 25.,  25.5, 26.,  26.5, 27.,  27.5,
       28.,  28.5, 29. ]

def stand(X : np.array):
    """

    Standardize an array

    Parameters
    ----------
    X : ndarray

    Returns
    -------
    norm : ndarray
    
    """
    return (X - X.mean()) / X.std()

def dataset_dat(weights,
                path_data : str,
                path_random_index : str,
                signal_size : int = 8192,
                interval : list = [0,270],
                n_photon_number : int = 100,
                standardize : bool = False,
                plot_traces : bool= False,
                plot_expected : bool = False):

    init_size = len(weights)
    weights = np.array([0]*(len(Average) - init_size) + weights)
    weights = weights / weights.max()
    dB_ = [str(i) for i in dB]

    archive = zipfile.ZipFile(path_data, 'r')
    files = sorted(archive.namelist())

    X_test = []
    X_train = []
    X_dB_test = []
    X_dB_train = []

    for file_ in files:

        if len(file_) > 100:
        
            w = int(weights[dB_.index(file_[84:88].replace(" ", ""))] * 1_024)

            if w > 1:

                if float(file_[111:113]) < 11: # Train 

                    X_train.append(
                        np.frombuffer(
                            archive.read(file_), dtype=np.float16
                            ).reshape(-1,signal_size)[:w,interval[0]:interval[1]]
                        )
                    X_dB_train.append(
                        np.full(w, file_[84:88])
                        )
                else: # Test
                    X_test.append(
                        np.frombuffer(
                            archive.read(file_), dtype=np.float16
                            ).reshape(-1,signal_size)[:w,interval[0]:interval[1]]
                        )
                    X_dB_test.append(
                        np.full(w, file_[84:88])
                        )
            else:
                pass
        else:
            pass
        
    X_test = -1 * np.concatenate(X_test).astype("float")
    X_train = -1 * np.concatenate(X_train).astype("float")
    X_dB_test = np.concatenate(X_dB_test)
    X_dB_train = np.concatenate(X_dB_train)


    if standardize:     
        data_train = stand(X_train)
        data_test = stand(X_test)
    else:   
        data_train = X_train
        data_test = X_test

    if path_random_index != None:
        # Shuffle based on reference index for reproducable results on different hardware
        random_index = np.load(path_random_index)
        data_train = data_train[random_index]
        data_test = data_test[random_index]
        X_dB_train = X_dB_train[random_index]
        X_dB_test = X_dB_test[random_index]

    expected_prob = np.zeros(n_photon_number)
    n_arr = np.arange(n_photon_number)

    for average_, amplitude_ in zip(Average, weights):
        expected_prob += amplitude_ * poisson(mu = average_).pmf(n_arr)

    expected_prob = expected_prob / np.sum(expected_prob)


    if plot_expected:
        with plt.style.context("seaborn-v0_8"):
            plt.figure(figsize=(6,3), dpi=100)
            plt.bar(x = n_arr, 
                    height = expected_prob,
                    alpha = 0.5, 
                    zorder=2)
            plt.ylabel('Counts')
            plt.xlabel('Photon number')
            plt.show()

    if plot_traces:
        with plt.style.context("seaborn-v0_8"):
            plt.figure(figsize=(10,4), dpi=100)
            plt.plot(data_train[::10].T,
                     alpha = 0.05,
                     linewidth = 1)
            plt.plot(data_test[::10].T,
                     alpha = 0.05,
                     linewidth = 1)
            plt.xlabel('Time (a.u.)')
            plt.ylabel('Voltage (a.u.)')
            plt.show()

    return data_train, data_test, expected_prob, X_dB_train, X_dB_test


def dataset_csv(path_data : str, 
                files : Union[list,None] = None,
                plot_traces : bool = False):

    archive = zipfile.ZipFile(path_data, 'r')

    if files is None:
        files = archive.namelist()
    else:
        pass

    files = sorted(files)
    
    data = []
    for file_ in files:

        if len(file_) > 15:
            data_ =  pl.read_csv(
                archive.read(file_), 
                has_header = False, 
                separator = ","
                ).to_numpy()
            
            data.append((data_[:,::3] - data_[:,:10].mean()))

        else:
            pass

    data = np.concatenate(data, axis=0)
    data_train = data[::2]
    data_test = data[1::2]

    if plot_traces:
        with plt.style.context("seaborn-v0_8"):
            plt.figure(figsize=(10,4), dpi=100)
            plt.plot(data_train[::10].T,
                     alpha = 0.05,
                     linewidth = 1)
            plt.plot(data_test[::10].T,
                     alpha = 0.05,
                     linewidth = 1)
            plt.xlabel('Time (a.u.)')
            plt.ylabel('Voltage (a.u.)')
            plt.show()

    return data_train, data_test

