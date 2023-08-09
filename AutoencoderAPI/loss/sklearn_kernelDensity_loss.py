import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from torch import nn

class sklearn_kernelDensity_loss():

    def __init__(self):
        pass

    def forward(self, output, data, network, X):

        feature = network(X, encoding=True).detach().numpy().reshape(-1,1)
        min_ = np.min(feature)
        max_ = np.max(feature)

        kde = KernelDensity(kernel='cosine', bandwidth=(max_-min_)/100).fit(feature)

        s = np.linspace(min_,max_,100)
        e = kde.score_samples(s.reshape(-1,1))
        mi = argrelextrema(e, np.less)[0], 
        ma = argrelextrema(e, np.greater)[0]

        mse = nn.MSELoss()
        loss = mse(output, data)

        return loss