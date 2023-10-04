import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from matplotlib.pyplot import cm

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV



class kernel_density():
    """
    Use kernel density estimation to separate the feature space into regions
    of high density associated to photon events.

    The user can use bandwidth selection rules by modifying the bandwidth array.
    Scott’s rule of thumb and Silverman’s rule of thumb for example can be implemented.
    However these basis rules did not show sufficient precision during testing.
    Giving an array of possible bandwidth showed better results.

    Bandwidth selection :

    - small   -> rough mapping
    - big     -> smooth mapping 

    Parameters
    ----------
    X_low : numpy.array
        Array containing all the samples in their low-dimensional representation.
    bw : tuple or numpy.array
        If bw is a tuple, it represents the parameters inside np.logspace(\*bw).
        Otherwise, an array can be used, this represents an array containing all 
        the possible bandwidth used in the kernel density estimation.
    flip : bool
        If `True` flips the feature space. This can be used to re-ordered the labels (right to left). 
    skip : int
        Skip a number of elements to define the feature space separation following the [::skip] structure.

    Returns
    -------
    None

    """
    def __init__(self, X_low, bw = (-5, -2, 20), flip = False, skip = 0):
        
        X_low = X_low.reshape(-1,1)
        
        if flip:
            self.flip = True
            X_low = -1 * X_low
        else:
            self.flip = False

        if skip < 2: skip = 1 

        min_ = np.min(X_low)
        max_ = np.max(X_low)

        if type(bw) == tuple:
            bw = np.logspace(*bw)

        params = {"bandwidth": bw}
        grid = GridSearchCV(KernelDensity(), params).fit(X_low[::skip])
        kd = grid.best_estimator_

        #print("Bandwidth : {0}".format(grid.best_estimator_.bandwidth))
        
        self.space = np.linspace(min_, max_, 5_000).reshape(-1,1)
        self.density = kd.score_samples(self.space)
        self.mins = self.space[argrelextrema(self.density, np.less)[0]].flatten()

        self.labels = np.digitize(X_low, bins=self.mins).reshape(-1)    
        self.bins = np.linspace(min(X_low), max(X_low), 50_000).reshape(-1)
        
        self.clusters_low = []
        self.condition = []
        for label in range(len(self.mins)+1):
            condition = self.labels == label
            self.clusters_low.append(X_low.flatten()[condition])
            self.condition.append(condition)

    def plot_density(self):
        """
        Plot the kernel density estimation over the feature space.
        This can be useful to evaluate if the bandwidth is appropriate 

        Bandwidth selection :
        
        - small   -> rough mapping
        - big     -> smooth mapping 

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        plt.figure(figsize=(30,4), dpi=200)
        plt.plot(self.space, self.density)
        plt.xlabel("Feature")
        plt.ylabel("Density")
        plt.show()



    def plot_cluster(self, xlim=None):
        """
        Plot a histogram of the samples in the feature space.
        Each sample is also labels using the kernel density estimation.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        plt.figure(figsize=(30,4), dpi=200)
        n =len(self.clusters_low)
        color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*n))))
        for index_cluster, cluster in enumerate(self.clusters_low):
            c = next(color)
            plt.hist(cluster.flatten() , self.bins,color=c, label=f"{index_cluster}", fill=True, histtype='step')
        plt.xlabel("Feature Space")
        plt.ylabel("Counts")
        if xlim != None:
            plt.xlim(xlim[0],xlim[1])
        #plt.legend(ncol=3)
        plt.show()



    def plot_traces(self, X, xlim=None):
        """
        Plot the traces `X` and labels them by following the order of the low-dimensional representation
        given in the initialization process.  

        Parameters
        ----------
        X : numpy.array
            Array containing all the samples.

        Returns
        -------
        None

        """
        plt.figure(figsize=(10,4), dpi=100)
        n =len(self.condition)
        color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*n)))) 
        
        for condition in self.condition:
            cluster = X[condition]
            c = next(color)

            if len(cluster) > 1000:
                cluster = cluster[:1000]

            for i, _ in enumerate(cluster):
                plt.plot(cluster[i], alpha=0.05, c=c)
                
        if xlim != None:
            plt.xlim(xlim[0],xlim[1])

        plt.xlabel("Time (a.u.)")
        plt.ylabel("Voltage (a.u.)")
        plt.show()



    def fit(self, X_low):
        """
        Assign a label to low-dimensional representations of samples based on the
        initialized feature space separation.

        Parameters
        ----------
        X_low : numpy.array
            Array containing the low-dimensional representation of samples.

        Returns
        -------
        labels : numpy.array
            Array containing the labels of X_low.

        """
        if self.flip:
            X_low = -1*X_low
        return np.digitize(X_low, bins=self.mins)
