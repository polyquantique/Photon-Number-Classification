import numpy as np
import torch

import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import colors
from matplotlib import cm
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
#from math import sqrt
from scipy.special import factorial
from numpy import sqrt, log

from scipy.stats import multivariate_normal, poisson 
from scipy.integrate import  trapezoid, dblquad
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import peak_widths
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness

#from scipy.integrate import quad


class gaussian_mixture_2d():
    """
    TODO:
    - add label shift for SNSPDs
    - add plot for p_sn and p_ns

    Parameters
    ----------
 

    Returns
    -------
    None

    """
    def __init__(self, X_low, 
                 X_high,
                 number_cluster = 10,
                 size_plot = 10,
                 dx = 1e-2,
                 label_shift = 0,
                 cluster_iter = 10,
                 info_sweep = 10,
                 plot_sweep = False):
        
        # Style
        self.style_name = "seaborn-v0_8"
        self.size_plot = size_plot
        self.color = cm.GnBu_r(np.linspace(0, 1, int(1.5*number_cluster))) 

        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Dataset
        self.X_high = np.array(X_high)
        self.X_low = np.array(X_low).reshape(-1,2)
        self.X_low[:,0] = (self.X_low[:,0] - self.X_low[:,0].min()) / (self.X_low[:,0].max() - self.X_low[:,0].min())
        self.X_low[:,1] = (self.X_low[:,1] - self.X_low[:,1].min()) / (self.X_low[:,1].max() - self.X_low[:,1].min())
        #self.s1 = np.arange(-0.1, 1.1, dx)
        #self.s2 = np.arange(-0.1, 1.1, dx)
        #self.dx = dx
        self.num = 1000
        self.esp = 1e-10
        self.number_cluster = number_cluster

        # Clustering iteration metrics
        self.aic = np.zeros(2*info_sweep)
        self.bic = np.zeros(2*info_sweep)
        self.silhouette = np.zeros(2*info_sweep)

        # Initial clustering iteration (find optimal number of cluster)
        self.clustering_iter(info_sweep = info_sweep,
                            cluster_iter = cluster_iter,
                            plot_sweep = plot_sweep)

        # Initial clustering parameters
        self.cluster_means = np.zeros((self.number_cluster, 2))
        self.cluster_covariances = np.zeros((self.number_cluster, 2, 2))
        self.cluster_weights = np.zeros(self.number_cluster)
        self.predict_ = None

        # Ordered clustering with initialized means
        self.clustering_order(cluster_iter = cluster_iter)
        
        # Labels
        self.label_shift = label_shift
        self.labels = self.predict(self.X_low)
        self.unique_labels = np.arange(self.number_cluster) + label_shift

        # Confidence metrics
        #self.p_s = np.zeros((self.s1.size, self.s2.size))
        #self.p_n = np.zeros(self.number_cluster)
        #self.p_sn = np.zeros((self.number_cluster, self.s1.size, self.s2.size))
        #self.p_ns = np.zeros((self.number_cluster, self.s1.size, self.s2.size))
        #self.mixture = self.p_s = np.zeros((self.s1.size, self.s2.size))
        self.confidence = np.zeros(self.number_cluster)

        # Trustworthiness
        self.trustworthiness_eucl = np.zeros(self.number_cluster)
        self.trustworthiness_cos = np.zeros(self.number_cluster)

    def clustering_iter(self, info_sweep = 10,
                            cluster_iter = 10,
                            plot_sweep = False):

        number_array = np.arange(self.number_cluster-info_sweep,
                                 self.number_cluster+info_sweep)

        for index, n_cluster in enumerate(number_array):

            fit_ = GaussianMixture(n_components = n_cluster, 
                                tol = 1e-4, 
                                max_iter = 300,
                                n_init = -(cluster_iter//-2),
                                init_params='k-means++').fit(self.X_low)
            
            self.aic[index] = fit_.aic(self.X_low)
            self.bic[index] = fit_.bic(self.X_low)
            self.silhouette[index] = silhouette_score(self.X_low, fit_.predict(self.X_low))
        
        number_cluster = number_array[np.argmax(self.silhouette)]

        if plot_sweep:

            with plt.style.context("seaborn-v0_8"):
                plt.figure(figsize=(self.size_plot,4))
                plt.plot(number_array, self.aic, label='AIC')
                plt.plot(number_array, self.bic, label='BIC')
                plt.vlines(number_cluster, np.min(self.bic), np.max(self.bic))
                plt.legend()
                plt.show()

                plt.figure(figsize=(self.size_plot,4))
                plt.plot(number_array, self.silhouette, label='Silhouette')
                plt.vlines(number_cluster, np.min(self.silhouette), np.max(self.silhouette))
                plt.legend()
                plt.show()

    
    def clustering_order(self, cluster_iter = 10):

                
        fit_ = GaussianMixture(n_components=self.number_cluster, 
                                tol = 1e-4, 
                                max_iter = 300,
                                n_init = cluster_iter,
                                init_params='k-means++').fit(self.X_low)
                                
        # Get area and labels
        X_Area = trapezoid(self.X_high, axis=1).reshape(-1,1)
        predict_init = fit_.predict(self.X_low)
        # Get average area of the clusters
        labels = np.arange(self.number_cluster)
        means_area = [np.mean(X_Area[predict_init == label_]) for label_ in labels]
        # Order clusters
        means_area , labels = zip(*sorted(zip(means_area, labels)))
        means_init = fit_.means_[list(labels)]
        
        fit_ = GaussianMixture(n_components = self.number_cluster, 
                            tol = 1e-4,
                            max_iter = 300,
                            n_init = cluster_iter,
                            init_params = 'k-means++',
                            means_init = means_init).fit(self.X_low)
        
        self.cluster_means[:,:] = fit_.means_
        self.cluster_covariances[:,:,:] = fit_.covariances_
        self.cluster_weights[:] = fit_.weights_
        self.predict_ = fit_.predict


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
        return self.predict_(X_low) + self.label_shift
        

    def plot_density(self, bw_adjust = 1):
        """

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        #sns.set_theme(rc={"figure.dpi": 200})
        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.size_plot,4))
            sns.kdeplot(x = self.X_low[:,0], 
                        y = self.X_low[:,1],
                        cmap="Blues",   #"magma",#
                        fill = True,
                        bw_adjust = bw_adjust,
                        thresh = 0,
                        levels = 30)
        #kde.tick_params(left=False, bottom=False)
        plt.show()
      

    def plot_cluster(self):
        """

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        #sns.color_palette("pastel", as_cmap=True)
        with plt.style.context(self.style_name):
            fig = plt.figure(figsize=(self.size_plot,4)) #, dpi=100
            ax = fig.add_subplot()
            for index, (label, mean) in enumerate(zip(self.unique_labels, self.cluster_means)):
                X = self.X_low[self.labels == label]
                ax.scatter(x = X[:,0], 
                           y = X[:,1],
                           s = 1,
                           alpha = 0.1)
                ax.text(mean[0]-0.01,mean[1]-0.01, index)
            plt.ylabel(r'$s_2$')
            plt.xlabel(r'$s_1$')
            plt.show()
            

    def multi_gaussian(self, x, y):
        """
        
        """
        multi = np.zeros((self.number_cluster, x.size, y.size))
        x, y = np.meshgrid(x, y)
        pos = np.dstack((x, y))
        for index, (mean, covariance, weight) in enumerate(zip(self.cluster_means,
                                                               self.cluster_covariances,
                                                               self.cluster_weights)):
            #self.p_sn[index,:,:] = weight * multivariate_normal(mean = mean, cov = covariance).pdf(pos)
            multi[index,:,:] = weight * multivariate_normal(mean = mean, cov = covariance).pdf(pos)

        return multi
    

    def trapezoid_2d(self, x, y, Z):
        """
        
        """
        return trapezoid(trapezoid(Z, x), y)
        

    def plot_confidence(self, plot_int = True):
        """
        
        """
        
            
        #for index , (p_ns, p_sn) in enumerate(zip(self.p_ns, self.p_sn)):
        for index , (mean, covariance, weight) in enumerate(zip(self.cluster_means, 
                                                        self.cluster_covariances,
                                                        self.cluster_weights)):
            x = np.linspace(mean[0] - 1000*covariance[0,0],
                            mean[0] + 1000*covariance[0,0],
                            self.num)
            y = np.linspace(mean[1] - 1000*covariance[1,1],
                            mean[1] + 1000*covariance[1,1],
                            self.num)
            p_sn = self.multi_gaussian(x, y)
            p_s = np.sum(p_sn, axis = 0)
            conf_integral = p_sn[index] / (p_s + self.esp) * p_sn[index]

            if plot_int:
                with plt.style.context(self.style_name):
                    plt.figure(figsize=(self.size_plot,4))
                    plt.imshow(conf_integral,
                            cmap = 'Blues')
                    plt.colorbar()
                    plt.show()

            self.confidence[index] = self.trapezoid_2d(x, y, conf_integral) / weight
  
            plt.show()
            #(self.s1, self.s2, p_ns * p_sn) / self.cluster_weights[index]

        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.size_plot,4))
            plt.plot(self.unique_labels[:-1], self.confidence[:-1])
            plt.xlabel("Photon number")
            plt.ylabel("Confidence")
            plt.show()

    
    def plot_trustworthiness(self):
        
        for index, label in enumerate(self.unique_labels):
            X_low = self.X_low[self.labels <= label]
            X_high = self.X_high[self.labels <= label]

            self.trustworthiness_eucl[index] = trustworthiness(X_high, X_low, metric="euclidean")
            self.trustworthiness_cos[index] = trustworthiness(X_high, X_low, metric="cosine")
        
        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.size_plot,4))
            plt.plot(self.unique_labels, self.trustworthiness_eucl, label='Euclidean')
            plt.plot(self.unique_labels, self.trustworthiness_eucl, label='Cosine')
            plt.xlabel("Photon number")
            plt.ylabel("Trustworthiness")
            plt.show()


    def plot_traces(self):
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
            plt.figure(figsize=(self.size_plot,4))
            color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*self.number_cluster)))) 
            
            for label in self.unique_labels:
                cluster = self.X_high[self.labels == label]
                c = next(color)
                if len(cluster) > 1000: cluster = cluster[:1000]
                plt.plot(cluster, alpha=0.05, c=c)

            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()
        #plt.savefig('traces.svg',format="svg", transparent=True)


    def plot_traces_average(self):
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
            plt.figure(figsize=(self.size_plot,4))
            color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*self.number_cluster)))) 
            
            for label in self.unique_labels:
                cluster = self.X_high[self.labels == label]
                c = next(color)
                plt.plot(np.mean(cluster, axis=0), c=c)

            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()
            #plt.savefig('average_traces.svg',format="svg", transparent=True)

    
    def plot_FWHM_average(self, X):
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
        #width = []#
        width = np.zeros(self.number_cluster)
        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.size_plot,4)) 
            
            for index, label in enumerate(self.unique_labels):
                cluster = self.X_high[self.labels == label]
                average = np.mean(cluster, axis=0)
                half = peak_widths((average-np.max(average)/2).flatten(), peaks=[np.argmax(average)], rel_height=0.5)
                #r1, r2 = spline.roots()
                #width.append(spline(np.arange(average.shape[0])))
                width[index] = half[0]#r2-r1


            #plt.scatter(np.arange(len(self.condition)), width)
            plt.plot(width) #for i in width]
            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()
            #plt.savefig('average_traces.svg',format="svg", transparent=True)
