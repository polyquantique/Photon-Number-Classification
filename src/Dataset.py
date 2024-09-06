import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from os import listdir

from .Utils import norm
from .ExistingAlgorithms import area

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

dB = [ 7.,   7.5,  8.,   8.5,  9.,   9.5, 10.,  10.5, 11.,  11.5, 12.,  12.5, 13.,  13.5,
       14.,  14.5, 15.,  15.5, 16.,  16.5, 17.,  17.5, 18.,  18.5, 19.,  19.5, 20.,  20.5,
       21.,  21.5, 22.,  22.5, 23.,  23.5, 24.,  24.5, 25.,  25.5, 26.,  26.5, 27.,  27.5,
       28.,  28.5, 29. ]


def dataset_TES(weights,
                path_test = r'/home/nicolasdc/Documents/publish/src/Datasets/TES/NIST test/',
                path_train = r'/home/nicolasdc/Documents/publish/src/Datasets/TES/NIST train/',
                path_random_index = r'/home/nicolasdc/Documents/publish/src/Results TES (Uniform)/randomIndexUniform.npy',
                signal_size = 8192,
                interval = [0,270],
                n_photon_number = 100,
                order_dB = False,
                normalize = False,
                plot_traces = False,
                plot_expected = False,
                return_db = False):

    init_size = len(weights)
    weights = np.array([0]*(len(Average) - init_size) + weights)
    weights = weights / weights.max()
    dB_ = [str(i) for i in dB]
    files_test = listdir(path_test)
    files_train = listdir(path_train)
    
    file_weight_test = [int(weights[dB_.index(i[67:71])] * 1_024) for i in files_test]
    file_weight_train = [int(weights[dB_.index(i[67:71])] * 1_024) for i in files_train]
    
    X_test = []
    X_train = []
    X_dB_test = []
    X_dB_train = []

    for w, fileName in zip(file_weight_test, files_test):
        X_test_ = np.fromfile(f"{path_test}{fileName}",dtype=np.float16).reshape(-1,signal_size)[:w,interval[0]:interval[1]]
        X_test.append(X_test_)
        X_dB_test.append(np.full(X_test_.shape[0], fileName[67:71]))

    for w, fileName in zip(file_weight_train, files_train):
        X_train_ = np.fromfile(f"{path_train}{fileName}",dtype=np.float16).reshape(-1,signal_size)[:w,interval[0]:interval[1]]
        X_train.append(X_train_)
        X_dB_train.append(np.full(X_train_.shape[0], fileName[67:71]))
    
    X_test = -1*np.concatenate(X_test).astype("float")
    X_train = -1*np.concatenate(X_train).astype("float")
    X_dB_test = np.concatenate(X_dB_test)
    X_dB_train = np.concatenate(X_dB_train)

    if order_dB:
        
        min_ = X_test.min()
        max_ = X_test.max()
        data_test = []
        data_train = []

        for dB_it in dB_[-init_size:]:

            x = files_test[0][67:71]
            file_weight_test = [i for i in files_test if i[67:71] == dB_it]
            file_weight_train = [i for i in files_train if i[67:71] == dB_it]

            test = -1 * np.concatenate([np.fromfile(f"{path_test}{fileName}",dtype=np.float16) \
                                        .reshape(-1,signal_size)[:,interval[0]:interval[1]] \
                                            for fileName in file_weight_test]).astype("float")
            train = -1 * np.concatenate([np.fromfile(f"{path_train}{fileName}",dtype=np.float16) \
                                        .reshape(-1,signal_size)[:,interval[0]:interval[1]] \
                                            for fileName in file_weight_train]).astype("float")
            
            if normalize:
                test= (test - min_) / (max_ - min_)
                train = (train - min_) / (max_ - min_)

            data_test.append(test)
            data_train.append(train)
        
    else:

        if normalize:     
            data_train = norm(X_train)
            data_test = norm(X_test)
        else:   
            data_train = X_train
            data_test = X_test
        
        #np.random.seed(42)
        #np.random.shuffle(data_train)
        #np.random.shuffle(data_test)

        if path_random_index != None:
            # Shuffle based on reference index for reproducable results on different hardware
            random_index = np.load(path_random_index)
            data_train = data_train[random_index]
            data_test = data_test[random_index]
            X_dB_train = X_dB_train[random_index]
            X_dB_test = X_dB_test[random_index]
        else:
            pass

    
    expected_prob = np.zeros(n_photon_number)
    n_arr = np.arange(0,n_photon_number)

    for average_, amplitude_ in zip(Average, weights):
        expected_prob += amplitude_ * poisson(mu = average_).pmf(n_arr)

    expected_prob = expected_prob / np.sum(expected_prob)


    if plot_expected:
        with plt.style.context("seaborn-v0_8"):
            plt.figure(figsize=(6,3), dpi=100)
            plt.bar(x = n_arr, 
                    height = expected_prob,
                    alpha = 0.5, 
                    #edgecolor = 'k', 
                    zorder=2)
            plt.ylabel('Counts')
            plt.xlabel('Photon number')
            plt.show()

    if plot_traces:
        with plt.style.context("seaborn-v0_8"):
            plt.figure(figsize=(10,4), dpi=100)
            plt.plot(data_train.T[:,::10],
                     alpha = 0.05,
                     linewidth = 1)
            plt.plot(data_test.T[:,::10],
                     alpha = 0.05,
                     linewidth = 1)
            plt.xlabel('Time (a.u.)')
            plt.ylabel('Voltage (a.u.)')
            plt.show()

    if return_db:
        return data_train, data_test, expected_prob, X_dB_train, X_dB_test
    else:
        return data_train, data_test, expected_prob











def dataset_SNSPD(selected_dB,
                path_test = r'/home/nicolasdc/files/Photon-Number-Classification/src/Datasets/SNSPD/Paderborn/data test/',
                path_train = r'/home/nicolasdc/files/Photon-Number-Classification/src/Datasets/SNSPD/Paderborn/data train/',
                path_dB = r'/home/nicolasdc/files/Photon-Number-Classification/src/Datasets/SNSPD/Paderborn/db_shuffled.npy',
                signal_size = 30_000,
                interval = [3250,4250],
                skip = 1,
                normalize = False):
    

    X_test = [] 
    X_train = [] 
    number_file_test = len(listdir(path_test))
    number_file_train = len(listdir(path_train))

    dB = np.load(path_dB)
    print(dB)

    for file_number in range(number_file_test):
    
        if dB[file_number] in selected_dB:
            data_temp = np.load(f"{path_test}/TracesNr{file_number}.npy").reshape(-1, signal_size)
            data_temp = data_temp[:, interval[0]:interval[1]]
            data_temp = data_temp[::skip]
            X_test.append(data_temp)

    X_test = -1 * np.concatenate(X_test)

    for file_number in range(number_file_test, number_file_train):
    
        if dB[file_number] in selected_dB:
            data_temp = np.load(f"{path_train}/TracesNr{file_number}.npy").reshape(-1, signal_size)
            data_temp = data_temp[:, interval[0]:interval[1]]
            data_temp = data_temp[::skip]
            X_train.append(data_temp)

    X_train = -1 * np.concatenate(X_train)

    if normalize:     
        data_train = norm(X_train)
        data_test = norm(X_test)
    else:   
        data_train = X_train
        data_test = X_test

    
    np.random.seed(42)
    np.random.shuffle(data_train)
    np.random.shuffle(data_test)

    return data_train, data_test
