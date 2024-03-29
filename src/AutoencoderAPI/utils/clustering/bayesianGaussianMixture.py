import numpy as np
import matplotlib.pyplot as plt 
from sklearn.mixture import BayesianGaussianMixture
from matplotlib.pyplot import cm



class bayesian_gaussianMixture():
    """
    Use Bayesian Gaussian mixture to separate the latent space into regions
    of high density associated with photon events.

    Parameters
    ----------
    X_low : numpy.array
        Array containing all the samples in their low-dimensional representation.
    cluster_max : float
        Maximum number of clusters to consider in the Bayesian Gaussian Mixture.
    flip : bool
        If `True` flips the latent space. This can be used to re-order the labels (right to left). 

    Returns
    -------
    None

    """
    def __init__(self, X_low, cluster_max, flip=False):

        if flip:
            self.flip = -1
            X_low = -1*X_low
        else:
            self.flip = 1

        X_low = np.array(X_low).reshape(-1,1)

        bgm = BayesianGaussianMixture(n_components=cluster_max, tol=1e-3, max_iter=300, init_params="kmeans")
        fit_ = bgm.fit(X_low)

        # Map labels based on their position in latent space
        cluster_means = fit_.means_
        cluster_covariance = fit_.covariances_
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

        number_bins = 5000
        self.bins = np.linspace(min(X_low), max(X_low), number_bins).reshape(-1)
        self.cluster_means = cluster_means
        self.cluster_covariance = cluster_covariance
        self.labels = sorted_labels
        self.predict_ = fit_.predict
        self.mapping = mapping


    def predict(self, X_low):
        """
        Predict the label of samples in `X_low` based on initial latent space separation.

        Parameters
        ----------
        X_low : numpy.array
            Array containing all the samples in their low-dimensional representation.

        Returns
        -------
        None

        """
        X_low = self.flip*X_low
        labels = self.predict_(X_low)
        labels = np.array([self.mapping[i] for i in labels])
        return labels
    
    

    def plot_cluster(self):
        """
        Plot a histogram of the samples in the latent space.
        Each sample is also labeled using the Bayesian Gaussian Mixture.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        #with plt.rc_context({'axes.edgecolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'white'}):
        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4), dpi=100)
            n =len(self.clusters_low)
            color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*n))))

            for index_cluster, cluster in enumerate(self.clusters_low):
                c = next(color)
                plt.hist(cluster.flatten() , self.bins, label=f"{index_cluster}", fill=True, histtype='step',color=c)#"#8dd3c7")
            
            plt.xlabel("Latent Space")
            plt.ylabel("Counts")
            plt.legend(ncol=3)
            plt.show()
            #plt.savefig('cluster.svg',format="svg", transparent=True)


    def plot_traces(self, X):
        """
        Plot the traces `X` and label them by following the order of the low-dimensional representation
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
            plt.figure(figsize=(10,4), dpi=100)
            n =len(self.condition)
            color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*n)))) 
            
            for condition in self.condition:
                cluster = X[condition]
                c = next(color)

                if len(cluster) > 1000:
                    cluster = cluster[:1000]

                for i, _ in enumerate(cluster):
                    plt.plot(cluster[i], alpha=0.05, c=c)# c="#8dd3c7")

            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()
            #plt.savefig('traces.svg',format="svg", transparent=True)


    def plot_traces_average(self, X):
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
            plt.figure(figsize=(10,4), dpi=100)
            n =len(self.condition)
            color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*n)))) 
            
            for condition in self.condition:
                cluster = X[condition]
                c = next(color)

                if len(cluster) > 1000:
                    cluster = cluster[:1000]

                plt.plot(np.mean(cluster, axis=0), c=c)

            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()
            #plt.savefig('average_traces.svg',format="svg", transparent=True)