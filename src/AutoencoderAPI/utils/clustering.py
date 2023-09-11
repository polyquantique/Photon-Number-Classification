import matplotlib.pyplot as plt
import numpy as np

from .metrics import silhouette_kmean



class clustering():

    def __init__(self, X, min_cluster, max_cluster):

        self.min_cluster = min_cluster
        self.max_cluster = max_cluster

        sk = silhouette_kmean(X, min_cluster, max_cluster)
        scores, optimal_cluster, optimal_score, clusters = sk.get_informations()
        
        self.bins = np.linspace(min(X), max(X), 1_000).reshape(-1)

        self.labels_ = sk.get_labels()
        self.optimal_cluster = optimal_cluster
        self.optimal_score = optimal_score
        self.clusters = clusters
        self.scores = scores
        self.fit = sk.fit

    def labels(self):
        return self.labels_

    def plot_clustering(self):

        print("Number of clusters : ", self.optimal_cluster)
        print("Silhouette score : ", self.optimal_score)

        plt.figure()
        for index_cluster, cluster in enumerate(self.clusters):
            plt.hist(cluster , self.bins, alpha = 0.5, label=f"{index_cluster}")
        plt.xlabel("feature")
        plt.ylabel("counts")
        plt.legend(ncol=3)

    def plot_silhouette(self):

        print("Number of clusters : ", self.optimal_cluster)
        print("Silhouette score : ", self.optimal_score)

        plt.figure()
        plt.plot(range(self.min_cluster, self.max_cluster+1), self.scores)
        plt.xlabel("Number of cluster")
        plt.ylabel("Silhouette score")


