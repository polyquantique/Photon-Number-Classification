import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from tqdm.notebook import tqdm
from matplotlib.pyplot import cm



class silhouette_gaussianMixture():

    def __init__(self, X_low, cluster_interval, flip=False):

        min_cluster = cluster_interval[0]
        max_cluster = cluster_interval[1]

        if flip:
            X_low = -1*X_low

        X_low = np.array(X_low).reshape(-1,1)
        X_low_approx = np.array(X_low).reshape(-1,1)[::10]
        scores = []

        # Number of cluster approximation
        
        for cluster_number in tqdm(range(min_cluster,max_cluster+1) , desc="Clusters"):
            init = np.linspace(min(X_low), max(X_low), cluster_number)
            predict = GaussianMixture(n_components=cluster_number, 
                                      tol=1e-3, 
                                      max_iter=200, 
                                      means_init=init).fit_predict(X_low_approx)
            predict = GaussianMixture(n_components=cluster_number, tol=1e-3, max_iter=300, init_params="k-means++").fit_predict(X_low_approx)
            if len(np.unique(predict)) != 1:
                scores.append(silhouette_score(X_low_approx, predict))
            else:
                scores.append(0)

        # Clustering using optimal number of cluster
        optimal_cluster = np.argmax(scores) + min_cluster
        km = GaussianMixture(n_components=optimal_cluster, tol=1e-3, max_iter=200, init_params="k-means++")
        fit_ = km.fit(X_low)

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
        self.n_cluster = range(min_cluster, max_cluster+1)
        self.scores = scores
        self.cluster_means = cluster_means
        self.cluster_covariance = cluster_covariance
        self.labels = sorted_labels
        self.predict_ = fit_.predict
        self.flip = flip
        self.mapping = mapping


    def predict(self, X_low):
        if self.flip:
            X_low = -1*X_low
        labels = self.predict_(X_low)
        labels = np.array([self.mapping[i] for i in labels])
        return labels
    

    def plot_silhouette(self):
        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4), dpi=100)
            plt.plot(self.n_cluster, self.scores)
            plt.xlabel("Number of cluster")
            plt.ylabel("Silhouette score")
            plt.show()
    

    def plot_cluster(self, xlim=None):
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
            if xlim != None:
                plt.xlim(xlim[0],xlim[1])
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
            plt.figure(figsize=(10,4), dpi=100)
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