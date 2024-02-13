import numpy as np
import torch
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt 
from matplotlib import colors
from matplotlib.pyplot import cm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
#from math import sqrt
from scipy.special import erf
from numpy import sqrt, log

from scipy.stats import norm
from scipy.integrate import  quad
from scipy.special import factorial

#from scipy.integrate import quad


class gaussian_mixture():
    """


    Parameters
    ----------
 

    Returns
    -------
    None

    """
    def __init__(self, X_low, 
                 number_cluster = 1,
                 flip = False, 
                 size_plot = 10,
                 dx = 1e-4,
                 label_shift = 0):
        
        self.flip = -1 if flip else 1
        
        X_low = self.flip * np.array(X_low).reshape(-1,1)

        self.style_name = "seaborn-v0_8"
        self.color = cm.GnBu_r(np.linspace(0, 1, int(1.5*number_cluster)))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.size_plot = size_plot
        self.number_cluster = number_cluster
        self.label_shift = label_shift
        self.min_ = np.min(X_low)
        self.max_ = np.max(X_low)
        self.s = np.arange(self.min_, self.max_, dx)
        self.len_s = len(self.s)
        self.p_s = None 
        self.p_n = None
        self.p_sn = None
        self.p_ns = np.zeros((number_cluster, self.len_s))
        

        fit_ = GaussianMixture(n_components=number_cluster, 
                               tol=1e-3, 
                               max_iter=200).fit(X_low)
        
        means_init = np.sort(fit_.means_.reshape(-1,1), axis=0)

        fit_ = GaussianMixture(n_components=number_cluster, 
                               tol=1e-3, 
                               max_iter=200,
                               means_init=means_init).fit(X_low)
                
        self.cluster_means = fit_.means_
        self.cluster_covariances = fit_.covariances_
        self.cluster_weights = fit_.weights_
        self.predict_ = fit_.predict
        self.clusters_low = []
        self.condition = []
        self.labels = self.predict_(X_low) + self.label_shift
        self.unique_labels = np.unique(self.labels)

        for label in self.unique_labels:
            condition = self.labels == label
        
            cluster_low = X_low[condition]
 
            self.clusters_low.append(cluster_low)
            self.condition.append(condition)


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
        X_low = self.flip * X_low
        return self.predict_(X_low) + self.label_shift
        
    def multi_gaussian(self, x):
        mean = self.cluster_means.reshape(-1,1)
        variance = self.cluster_covariances.reshape(-1,1)
        weights = self.cluster_weights.reshape(-1,1)
        x = x.reshape(1,-1)
        return weights * (2*np.pi*variance)**(-0.5) * np.exp(-(x - mean)**2/(2*variance))
        

    def plot_cluster(self,
                     number_bins = 5000):
        """
        Plot a histogram of the samples in the latent space.
        Each sample is also labeled using the kernel density estimation.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        color = iter(self.color)
        bins = np.linspace(self.min_, self.max_, number_bins)

        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.size_plot,4))
            for index_cluster, cluster in enumerate(self.clusters_low):
                plt.hist(cluster.flatten() , bins, 
                         label=f"{index_cluster + self.label_shift}", 
                         fill=True, 
                         histtype='step',
                         color=next(color))
            plt.xlabel("Latent Space")
            plt.ylabel("Counts")
            plt.legend(ncol=3)
            plt.show()
        #plt.savefig('cluster.svg',format="svg", transparent=True)
            

    def plot_psn(self,
                 n_average):
        """
        
        """
        color = iter(self.color)
        self.p_sn = self.multi_gaussian(self.s)
        self.p_n = (np.exp(-n_average) * (n_average**self.unique_labels) / factorial(self.unique_labels)).reshape(-1,1)
        self.p_s = np.sum(self.p_sn * self.p_n, axis = 0) 

        with plt.style.context(self.style_name):
            plt.figure(figsize = (self.size_plot,4))
            for index, gaussian in enumerate(self.p_sn):
                plt.plot(self.s, gaussian,
                         color = next(color),
                         label = f'{index + self.label_shift}')
        
            plt.plot(self.s, np.sum(self.p_sn, axis = 0) , 
                     label = 'Mixture',
                     alpha = 0.3)
            plt.xlabel("Latent Space")
            plt.ylabel(r"$p(s|n)$")
            plt.legend(ncol=3)
            plt.show()

    
    def plot_pns(self):
        """
        
        """
        color = iter(self.color)
        #self.p_n = (np.exp(-n_average) * (n_average**self.unique_labels) / factorial(self.unique_labels)).reshape(-1,1)
        self.p_ns = self.p_sn * self.p_n / self.p_s

        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.size_plot,4))
            for index, p_ns_ in enumerate(self.p_ns):
                plt.plot(self.s, p_ns_, 
                         color=next(color),
                         label = f'{index + self.label_shift}')
            plt.xlabel("Latent Space")
            plt.ylabel(r"$p(n|s)$")
            plt.legend(ncol=3)
            plt.show()


    def plot_confidence(self):
        """
        
        """
        color = iter(self.color)

        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.size_plot,4))
            for index, (p_ns_, p_sn) in enumerate(zip(self.p_ns, self.p_sn)):
                plt.plot(self.s, p_ns_ * p_sn, 
                         color=next(color),
                         label = f'{index + self.label_shift}')
            plt.xlabel("Latent Space")
            plt.ylabel(r"$p(n|s)$")
            plt.legend(ncol=3)
            plt.show()


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
            plt.figure(figsize=(self.size_plot,4)) #, dpi=100
            n =len(self.condition)
            color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*n)))) 
            
            for condition in self.condition:
                cluster = X[condition]
                c = next(color)
                if len(cluster) > 1000: cluster = cluster[:1000]
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
            plt.figure(figsize=(self.size_plot,4)) #, dpi=100
            n =len(self.condition)
            color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*n)))) 
            
            for condition in self.condition:
                cluster = X[condition]
                c = next(color)
                if len(cluster) > 1000: cluster = cluster[:1000]
                plt.plot(np.mean(cluster, axis=0), c=c)

            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()
            #plt.savefig('average_traces.svg',format="svg", transparent=True)


    def normalize_latent(self, X_l, number):

        try: 
            return X_l / (self.cluster_means[number] - self.cluster_means[0])
        except:
            return X_l 
        