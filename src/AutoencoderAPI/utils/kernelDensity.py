import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from bisect import bisect_right, bisect_left
from tqdm.notebook import tqdm
from matplotlib.pyplot import cm

from sklearn.neighbors import KernelDensity



class kernel_density():

    def __init__(self, X, X_low, bw_cst=4):

        X_low = X_low.reshape(-1,1)
        min_ = np.min(X_low)
        max_ = np.max(X_low)
        bw = len(X_low) **(-1./(1+4)) / bw_cst#max_ - min_ / 1000
        kd = KernelDensity(kernel='gaussian', bandwidth=bw).fit(X_low)
        self.space = np.linspace(min_, max_, 1000).reshape(-1,1)
        self.density = kd.score_samples(self.space)

        self.mins = self.space[argrelextrema(self.density, np.less)[0]].flatten()

        labels = np.array([bisect_right(self.mins, i) for i in X_low])
        self.bins = np.linspace(min(X_low), max(X_low), 1_000).reshape(-1)
        
        clusters_low = []
        clusters_traces = []
        for label in range(len(self.mins)):
            condition = labels == label
            clusters_low.append(X_low.flatten()[condition])
            clusters_traces.append(X[condition])
        self.clusters_low = clusters_low
        self.clusters_traces = clusters_traces

    def plot_density(self):

        plt.figure(figsize=(8,5))
        plt.plot(self.space, self.density)
        plt.xlabel("Feature")
        plt.ylabel("Density")
        plt.show()

    def plot_cluster(self):

        plt.figure(figsize=(8,5))
        n =len(self.clusters_low)
        color = iter(cm.GnBu_r(np.linspace(0, 1, n)))
        for index_cluster, cluster in enumerate(self.clusters_low):
            c = next(color)
            plt.hist(cluster.flatten() , self.bins,color=c, label=f"{index_cluster}", fill=True, histtype='step') #alpha = 0.5
        plt.xlabel("Feature")
        plt.ylabel("Counts")
        plt.legend(ncol=3)
        plt.show()

    def plot_traces(self):

        plt.figure(figsize=(8,5))#figsize=(12,8))
        n =len(self.clusters_traces)
        color = iter(cm.GnBu_r(np.linspace(0, 1, n))) #GnBu_r
        labels = []
        for index_cluster, cluster in enumerate(self.clusters_traces):#tqdm(enumerate(self.clusters_traces), total=n):
            c = next(color)
            labels.append(index_cluster)
            for i, _ in enumerate(cluster):
                plt.plot(cluster[i], alpha=0.01, c=c)
        plt.xlabel("Time (a.u.)")
        plt.ylabel("Voltage (a.u.)")
        #plt.legend(labels=labels, ncol=3, loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.legend()
        plt.show()

    def fit(self, X):

        return np.array([bisect_left(self.mins, i) for i in X])
