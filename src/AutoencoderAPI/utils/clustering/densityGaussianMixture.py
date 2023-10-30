import numpy as np
import torch
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV



class density_gaussianMixture():

    def __init__(self, X_low, 
                 bw = (-5, -2, 20), 
                 bins_plot = 5000,
                 flip = False, 
                 skip = 1):

        if flip:
            X_low = -1*X_low
        if skip < 2:
            skip = 1

        X_low = np.array(X_low).reshape(-1,1)
        self.min_ = np.min(X_low)
        self.max_ = np.max(X_low)

        params = {"bandwidth": bw}
        grid = GridSearchCV(KernelDensity(), params).fit(X_low[::skip])
        kd = grid.best_estimator_

        self.space = np.linspace(self.min_, self.max_, 2*bins_plot).reshape(-1,1)
        self.density = kd.score_samples(self.space)
        self.maxs = self.space[argrelextrema(self.density, np.greater)[0]].reshape(-1,1)
        self.mins = torch.tensor(self.space[argrelextrema(self.density, np.less)[0]].flatten())

        bgm = GaussianMixture(n_components=len(self.maxs), tol=1e-3, max_iter=100, means_init=self.maxs)
        fit_ = bgm.fit(X_low)

        # Map labels based on their position in latent space
        cluster_means = fit_.means_
        cluster_covariance = fit_.covariances_
        weights = fit_.weights_
        labels = fit_.predict(X_low)

        #Sort labels/means
        unique_labels = range(len(cluster_means))
        centroids, mapping = zip(*sorted(zip(cluster_means, unique_labels)))
        sorted_labels = np.array([mapping[i] for i in labels])

        self.style_name = "seaborn-v0_8"
        self.clusters_low = []
        self.condition = []
        for label in mapping:
            condition = labels == label
            self.clusters_low.append(X_low.flatten()[condition])
            self.condition.append(condition)

        self.bins = np.linspace(min(X_low), max(X_low), bins_plot).reshape(-1)
        self.cluster_means = cluster_means
        self.cluster_covariance = cluster_covariance
        self.weights = weights
        self.labels = sorted_labels
        self.predict_ = fit_.predict
        self.flip = flip
        self.mapping = mapping


    def predict(self, X_low):
        #if self.flip:
            #X_low = -1*X_low
        #labels = self.predict_(X_low)
        #labels = np.array([self.mapping[i] for i in labels])
        #return labels
        if self.flip:
            X_low = -1*X_low
        return torch.searchsorted(self.mins, X_low)
     
    
    

    def plot_cluster(self):
        """
        Plot a histogram of the samples in the latent space.
        Each sample is also labels using the kernel density estimation.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        def gaussian_function(x, mean, variance):
            return (2*np.pi*variance)**(-1/2) * np.exp(-(x - mean)**2/(2*variance))
        
        x = np.linspace(self.min_, self.max_, 1000).reshape(-1,1)
        n =len(self.clusters_low)
        color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*n))))

        #with plt.rc_context({'axes.edgecolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'white'}):
        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4)) #, dpi=100
            for index_cluster, cluster in enumerate(self.clusters_low):
                c = next(color)
                plt.hist(cluster.flatten() , self.bins, label=f"{index_cluster}", fill=True, histtype='step',color=c)#"#8dd3c7")
            for index, mean_value in enumerate(self.cluster_means):
                plt.plot(x, gaussian_function(x, mean_value, self.cluster_covariance[index]), color="k")
            plt.xlabel("Latent Space")
            plt.ylabel("Counts")

            plt.legend(ncol=3)
            plt.show()
        #plt.savefig('cluster.svg',format="svg", transparent=True)


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
        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4)) #, dpi=100
            n =len(self.condition)
            color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*n)))) 
            
            for condition in self.condition:
                cluster = X[condition]
                c = next(color)

                if len(cluster) > 1000:
                    cluster = cluster[:1000]

                for i, _ in enumerate(cluster):
                    plt.plot(cluster[i], alpha=0.05, c=c)# c="#8dd3c7")
                    
            if xlim != None:
                plt.xlim(xlim[0],xlim[1])

            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()
        #plt.savefig('traces.svg',format="svg", transparent=True)


    def plot_traces_average(self, X, xlim=None):
        """
        Plot the traces average and labels them by following the order of the low-dimensional representation
        given in the initialization process.  

        Parameters
        ----------
        X : numpy.array
            Array containing all the samples.

        Returns
        -------
        None

        """
        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4)) #, dpi=100
            n =len(self.condition)
            color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*n)))) 
            
            for condition in self.condition:
                cluster = X[condition]
                c = next(color)

                if len(cluster) > 1000:
                    cluster = cluster[:1000]

                plt.plot(np.mean(cluster, axis=0), c=c)
                    
            if xlim != None:
                plt.xlim(xlim[0],xlim[1])

            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()
            #plt.savefig('average_traces.svg',format="svg", transparent=True)