import numpy as np
import torch
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt 
from matplotlib import colors
from matplotlib.pyplot import cm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from math import erf, sqrt

from scipy.integrate import quad


class density_square_distribution():
    """

    Parameters
    ----------

    Returns
    -------
    None

    """
    def __init__(self, X_low, 
                 bw = [0.01], 
                 bins_plot = 5000,
                 flip = False, 
                 skip = 1):
        
        self.flip = -1 if flip else 1
   
        X_low = self.flip * np.array(X_low).reshape(-1,1)
        if skip < 2: skip = 1
        kd = GridSearchCV(KernelDensity(), {"bandwidth": bw}).fit(X_low[::skip]).best_estimator_

        self.style_name = "seaborn-v0_8"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.min_ = np.min(X_low)
        self.max_ = np.max(X_low)
        self.bins = np.linspace(self.min_, self.max_, bins_plot)
        self.density = kd.score_samples(self.bins.reshape(-1,1))
        self.maxs = self.bins[argrelextrema(self.density, np.greater)[0]].reshape(-1,1)
        self.mins = torch.tensor(self.bins[argrelextrema(self.density, np.less)[0]].flatten()).to(self.device)

        fit_ = GaussianMixture(n_components=len(self.maxs), 
                               tol=1e-3, 
                               max_iter=100, 
                               means_init=self.maxs).fit(X_low)

        # Map labels based on their position in latent space
        self.cluster_means = fit_.means_
        self.cluster_covariance = fit_.covariances_
        self.weights = fit_.weights_
        self.labels = fit_.predict(X_low)
        self.predict_ = fit_.predict
        self.crossTalk = None

        #Sort labels/means
        unique_labels = range(len(self.cluster_means))
        centroids, self.mapping = zip(*sorted(zip(self.cluster_means, unique_labels)))
        #sorted_labels = np.array([self.mapping[i] for i in self.labels])

        self.clusters_low = []
        self.condition = []
        self.labels_samples = []

        for label in self.mapping:
            condition = self.labels == label
        
            cluster_low = X_low.flatten()[condition]
            self.labels_samples.append(len(cluster_low))
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
        return torch.searchsorted(self.mins, X_low)
    
     
    def plot_cross_talk(self):


        """
        sig = self.cluster_covariance
        u = self.cluster_means

        num_distributions = len(u)
        crossTalk = np.zeros((num_distributions, num_distributions))


        
        for i in range(num_distributions):
            for j in range(i+1):
                if i==j:
                    crossTalk[j][i] = 1
                else:
                    u1 = u[j]
                    sig1 = sqrt(sig[j])
                    u2 = u[i]
                    sig2 = sqrt(sig[i])

                    c = (u2*sig1**2 - sig2*(u1*sig2 + sig1*np.sqrt((u1-u2)**2 + 2*(sig1**2-sig2**2)*np.log(sig1/sig2))))/(sig1**2 - sig2**2)
                    crossTalk[j][i] = 1 - (1/2)*erf((c-u1)/(sqrt(2)*sig1)) + (1/2)*erf((c-u2)/(sqrt(2)*sig2))

        crossTalk = crossTalk + crossTalk.T - np.diagflat(np.ones((num_distributions)))

        im = plt.imshow(crossTalk, cmap="GnBu_r")#, norm=colors.LogNorm())
        plt.colorbar(im)
        plt.show()
        """

        num_distributions = len(self.cluster_means)
        
        crossTalk = np.zeros((num_distributions, num_distributions))

        #plt.figure()
        for i in range(num_distributions):
            for j in range(num_distributions):
                #if i == j:
                #    crossTalk[j][i] = self.weights[i]/2
                #else:
                    
                w = np.linspace(self.cluster_means[i],self.cluster_means[j],1_000)

                distr1 = self.gaussian_function(w, self.cluster_means[i], self.cluster_covariance[i], self.weights[i])
                distr2 = self.gaussian_function(w, self.cluster_means[j], self.cluster_covariance[j], self.weights[j])
                inter = w[np.argmin(np.abs(distr2-distr1))]

                #plt.plot(w, self.gaussian_function(w, self.cluster_means[i], self.cluster_covariance[i], self.weights[i]).reshape(-1,1), label='1')
                #plt.plot(w, self.gaussian_function(w, self.cluster_means[j], self.cluster_covariance[j], self.weights[j]).reshape(-1,1), label='2')

                if i > j:
                    x1, x2 = -1, inter
                    x3, x4 = inter, 1
                else:
                    x1, x2 = inter, 1
                    x3, x4 = -1, inter


                integral1 = quad(self.gaussian_function, x1, x2, args=(self.cluster_means[i], self.cluster_covariance[i], self.weights[i]))
                integral2 = quad(self.gaussian_function, x3, x4, args=(self.cluster_means[j], self.cluster_covariance[j], self.weights[j]))
                integral3 = quad(self.gaussian_function, -1, 1, args=(self.cluster_means[i], self.cluster_covariance[i], self.weights[i]))

                crossTalk[j][i] = (integral1[0] + integral2[0]) / integral3[0]
        
        #plt.legend()
        #plt.show()
        self.crossTalk = crossTalk
                
        im = plt.imshow(self.crossTalk, cmap="GnBu_r")#, norm=colors.LogNorm())
        plt.colorbar(im)
        plt.show()



    def confidence(self, zeros=0):

        if zeros == 0:
            cond = True
        else:
            cond = False


    
        average = np.sum([i*(index+1) for index, i in enumerate(self.labels_samples)]) / (np.sum(self.labels_samples) + zeros)
        confidence = []
        number = []

        for photon_index, _ in enumerate(self.labels_samples):

            confidence_temp = 0
            for s, _ in enumerate(self.labels_samples):

                p_sn = self.crossTalk[s,photon_index]
                
                if cond:
                    photon_number = photon_index
                else:
                    photon_number = photon_index + 1
                    
                p_n =  np.exp(- average) * average**photon_number / np.math.factorial(photon_number)
                denum = 0

                for  k, _ in enumerate(self.labels_samples):

                    denum += self.crossTalk[s,k] * self.weights[k]

                confidence_temp += (p_sn**2 * p_n) / denum

            confidence.append(confidence_temp)
            number.append(photon_index+1)
            

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4), dpi=100)
            plt.plot(number, confidence)
            plt.xlabel("Photon number")
            plt.ylabel("Confidence")
            plt.show()
    
    
    def plot_density(self):
        """
        Plot the kernel density estimation over the latent space.
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
        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4), dpi=100)
            plt.plot(self.bins, self.density)
            plt.xlabel("Latent Space")
            plt.ylabel("Density")
            plt.show()

    
    def gaussian_function(self, x, mean, variance, weights):
            return weights * (2*np.pi*variance)**(-1/2) * np.exp(-(x - mean)**2/(2*variance))
        

    def plot_cluster(self):
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
                plt.plot(x, self.gaussian_function(x, mean_value, self.cluster_covariance[index], self.weights[index]), color="k")
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
            plt.figure(figsize=(10,4)) #, dpi=100
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
            plt.figure(figsize=(10,4)) #, dpi=100
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