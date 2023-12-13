from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import numpy as np


def principal_component(X_init, filter=False):

    X_init[X_init < np.mean(X_init[:,-3:]) + 0.013] = np.mean(X_init[:,-3:]) + 0.013

    if filter:
        X_init = savgol_filter(X_init, 20, 2)

    pca = PCA(n_components=1)
    X_low_dim = pca.fit_transform(X_init)
    X_reconst = pca.inverse_transform(X_low_dim)

    return X_init, X_reconst, X_low_dim