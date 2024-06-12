import numpy as np
import pickle
import matplotlib.pyplot as plt

from .GaussianMixture import gaussian_mixture


def plot_traces(X_high : np.array):
    """

    Plot signals.

    Parameters
    ----------
    X_high : ndarray

    Returns
    -------
    None
    
    """
    with plt.style.context("seaborn-v0_8"):
        plt.figure(figsize=(10,4), dpi=100)
        [plt.plot(x, alpha = 0.01) for x in X_high[::10]]
        plt.xlabel('Time (a.u.)')
        plt.ylabel('Voltage (a.u.)')
        #plt.xticks([])
        #plt.yticks([])
        plt.show()


def norm(X : np.array):
    """

    Normalize an array

    Parameters
    ----------
    X : ndarray

    Returns
    -------
    norm : ndarray
    
    """
    return (X - X.min()) / (X.max() - X.min())


def save_results(gm, 
                 name_method : str = 'Max', 
                 path : str = r'src/Results'):
    
    # if not all(v == 0 for v in gm.confidence_1D):
    np.save(f'{path}/Confidence/{name_method} 1D.npy', gm.confidence_1D) 
    # if not all(v == 0 for v in gm.confidence_2D):
    np.save(f'{path}/Confidence/{name_method} 2D.npy', gm.confidence_2D)
    np.save(f'{path}/Mean Clusters/{name_method}.npy', gm.cluster_means) 
    np.save(f'{path}/Trustworthiness Cosine/{name_method}.npy', gm.trustworthiness_cos) 
    np.save(f'{path}/Trustworthiness Euclidian/{name_method}.npy', gm.trustworthiness_eucl) 


def get_means(name_method : str = 'Max', 
              path : str = r'src/Results TES'):
    try:
        means = np.load(f'{path}/Mean Clusters/{name_method}.npy') 
    except:
        means = None
    return means


def open_object(file_name):
    """
    Open a file using the pickle library. 

    Parameters
    ----------
    file_name : str
        Name of the file to open and read.

    Returns
    -------
    None
    
    """
    try:
        with open(file_name, 'rb') as f:
            object_ = pickle.load(f)
    except Exception as ex:
        print("Error when loading file : ", ex)

    return object_


def load_mean_std(path):

    path = f"{path}"
    config = open_object(f"{path}/log.bin")

    return config['internal']['mean'], config['internal']['std']


def plot_poisson_confidence(average_list : list,
                            n_cluster_list : list,
                            X_low : list,
                            X_high : list,
                            confidence_dimension : int = 1):

    confidence_poisson = []

    for index, average in enumerate(average_list):

        gm = gaussian_mixture(X_low[index], 
                            X_high[index],
                            number_cluster = n_cluster_list[index],
                            cluster_iter = 20,
                            info_sweep = 10,
                            plot_sweep = True,
                            dpi = 100)

        gm.plot_density(bw_adjust = 0.1)
        gm.plot_cluster(plot_kde = True)
                
        if confidence_dimension == 2:
            gm.plot_confidence_2d(average_poisson = average)
            confidence_poisson.append(gm.confidence_2D)
        else:
            gm.plot_confidence_1d(axis = 0, average_poisson = average) 
            confidence_poisson.append(gm.confidence_1D)

    return confidence_poisson

