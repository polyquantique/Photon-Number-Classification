import numpy as np
import torch 
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.integrate import  quad, simpson
from scipy.stats import norm

from .clustering.GaussianMixture import gaussian_mixture

class confidence:

    def __init__(self, X_low,
                number_cluster = 1,
                flip = False, 
                size_plot = 10,
                label_shift = 0):

        # Global clustering for photon number assignment
        kd = gaussian_mixture(X_low, 
                            number_cluster = number_cluster,
                            flip = flip, 
                            size_plot = size_plot,
                            label_shift = label_shift)

        # Verification of clusters
        kd.plot_cluster()

        self.style_name = "seaborn-v0_8"
        self.label_shift = label_shift
        self.possible_photon_number = kd.unique_labels
        self.predict = kd.predict
        self.labels = kd.labels

        self.label_dB = []
        self.density = []
        self.bins = []
        self.average_number = []
        self.n_arr = []
        self.confidence = []

        self.weight_ = None
        self.mean_ = None
        self.variance_ = None
        self.cluster_weights = None
        self.cluster_means = None
        self.cluster_variances = None
        self.p_n = None

    
    def fit(self, X_low,
            number_cluster = 1,
            flip = False, 
            label_dB = None):

        # Global clustering for photon number assignment
        kd = gaussian_mixture(X_low, 
                            number_cluster = number_cluster,
                            flip = flip, 
                            size_plot = 10,
                            label_shift = self.label_shift)

        if label_dB != None:
            self.label_dB.append(label_dB)

        labels = self.predict(X_low)
        unique_labels = np.unique(labels)
        average_number = np.sum(kd.cluster_weights * sorted(unique_labels[:len(kd.cluster_weights)]))
     
        kd.plot_cluster()
        kd.plot_psn(average_number)
        kd.plot_pns()
        kd.plot_confidence()
        n_arr = kd.unique_labels
        confidence = np.zeros(number_cluster)
        self.cluster_weights = kd.cluster_weights.astype('double')
        self.cluster_means = kd.cluster_means.astype('double')
        self.cluster_covariances = kd.cluster_covariances.astype('double')
        self.p_n = kd.p_n.astype('double')

        for index , (p_n_, weight, mean, covariance) in enumerate(zip(kd.p_n,
                                                                kd.cluster_weights, 
                                                                kd.cluster_means, 
                                                                kd.cluster_covariances)):
    

            self.weight_ = weight.astype('double')
            self.mean_ = mean.astype('double')
            self.variance_ = covariance.astype('double')

            integ = lambda s : self.p_sn_func(s)**2 * p_n_ / self.p_s_func(s)

            confidence[index] = quad(integ, -0.4, 0)[0]

            #confidence[index] = simpson(x = s, y = p_ns * p_sn)

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4))
            plt.plot(n_arr,confidence)
            plt.xlabel("Photon number")
            plt.ylabel("Confidence")
            plt.show()


    def p_sn_func(self, x):
        mean = self.mean_#.reshape(-1,1)
        variance = self.variance_#.reshape(-1,1)
        weights = self.weight_#.reshape(-1,1)
        #x = x#.reshape(1,-1)
        return weights * (2*np.pi*variance)**(-1/2) * np.exp(-(x - mean)**2/(2*variance))
    
    def p_s_func(self, x):
        mean = self.cluster_means.reshape(-1,1)
        variance = self.cluster_covariances.reshape(-1,1)
        weights = self.cluster_weights.reshape(-1,1)
        p_n = self.p_n
        p_sn = self.gaussian(x, weights, mean, variance) 
        return np.sum(p_sn * p_n, axis=0)
    
    def gaussian(self, x, w, m, v):
        return w * (2*np.pi*v)**(-1/2) * np.exp(-(x - m)**2/(2*v))    

    def plot_all_confidence(self):

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4))
            for label_dB, n_arr, confidence in zip(self.label_dB, self.n_arr, self.confidence):
                plt.plot(n_arr,confidence, label=f'{label_dB}')
            plt.xlabel("Photon number")
            plt.ylabel("Confidence")
            plt.legend()
            plt.show()

    def plot_all_density(self,xlim=None, flip=False):

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4), dpi=200)
            for label_dB, bins, density in zip(self.label_dB, self.bins, self.density):
                #density = (density - np.min(density)) / (np.max(density) - np.min(density))
                if flip:
                    x = -1*bins#np.abs(bins)**(1/2)
                else:
                    x = bins#1np.abs(bins)**(1/2)
                #print(np.trapz(density.ravel(), bins.ravel()))
                plt.plot(x,density, label=f'{label_dB} dB', linewidth=0.8)
            if xlim != None:
                plt.xlim(xlim)
            plt.xlabel("Latent Space")
            plt.ylabel("Density")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
    

    def plot_total_density(self,xlim=None):

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4), dpi=200)
            density_tot = np.zeros(len(self.density[0]))

            for bins, density in zip(self.bins, self.density):
                density_tot = density_tot + (density - np.min(density)) / (np.max(density) - np.min(density))
            
            #density_tot = (density_tot - np.min(density_tot)) / (np.max(density_tot) - np.min(density_tot))
            plt.plot(bins,density_tot)
            #plt.yscale('log')
            if xlim != None:
                plt.xlim(xlim)
            plt.xlabel("Latent Space")
            plt.ylabel("Density")
            #plt.legend()
            plt.show()

    def plot_mean_cluster(self,xlim=None):


        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4), dpi=200)
            #mean_tot = np.zeros(len(self.means[0]))

            for mean, label in zip(self.average_number, self.label_dB):
                #density_tot = density_tot + (density - np.min(density)) / (np.max(density) - np.min(density))
                plt.scatter(range(len(mean)),mean, label=f'{label}')
            #density_tot = (density_tot - np.min(density_tot)) / (np.max(density_tot) - np.min(density_tot))
            
            #plt.yscale('log')
            if xlim != None:
                plt.xlim(xlim)
            plt.xlabel("Photon number")
            plt.ylabel("Mean of clusters")
            plt.show()  


    
    