import numpy as np
import torch 
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.integrate import  simpson
from scipy.stats import norm

from .clustering.GaussianMixture import gaussian_mixture

class confidence:

    def __init__(self, X_low,
                number_cluster = 1,
                flip = False, 
                size_plot = 10,
                scaling = 1):

        # Global clustering for photon number assignment
        kd = gaussian_mixture(X_low, 
                            number_cluster = number_cluster,
                            flip = flip, 
                            size_plot = size_plot)

        # Verification of clusters
        kd.plot_cluster()

        self.style_name = "seaborn-v0_8"
        self.possible_photon_number = np.unique(kd.labels)
        self.predict = kd.predict
        self.labels = kd.labels

        self.label_dB = []
        self.density = []
        self.bins = []
        self.means = []
        self.n_arr = []
        self.confidence = []

    
    def fit(self, X_low,
            number_cluster = 1,
            flip = False, 
            scaling = 1,
            zeros_number = 0,
            label_dB = None):

        # Global clustering for photon number assignment
        kd = gaussian_mixture(X_low, 
                            number_cluster = number_cluster,
                            flip = flip, 
                            size_plot = 10)

        if label_dB != None:
            self.label_dB.append(label_dB)

        labels = self.predict(X_low)
        unique_labels = np.unique(labels)
        weights = kd.cluster_weights
        average_number = np.sum(weights * sorted(unique_labels[:len(weights)]))
     
        kd.plot_cluster()
        kd.plot_psn()
        kd.plot_pns(average_number)
    
        s = kd.s
        p_ns = kd.p_ns
        p_sn = kd.p_sn
        n_arr = kd.unique_labels
        confidence = np.zeros(number_cluster)

        #print('mean',n_average)

        
        for n , (p_ns_, p_sn_) in enumerate(zip(p_ns, p_sn)):

            #p_sn = lambda s : weights[n] * (2*np.pi*variance[n])**(-1/2) * np.exp(-(s - means[n])**2/(2*variance[n]))
            #p_s = lambda s : np.sum(weights * (2*np.pi*variance)**(-1/2) * np.exp(-(s - means)**2/(2*variance)))
            #p_ns = lambda s : p_sn(s) * p_n[n] / p_s(s)
            #integ = lambda s : p_sn(s) * p_ns(s) 
                               #norm.pdf(s, loc = means[n], scale = variance[n]))**2
            #p_sn = lambda s : (weights[n] * norm.pdf(s, loc = means[n], scale = variance[n]))

            #confidence[n] = p_n[n] *  quad(integ, -np.inf, np.inf)[0]#/ p_s)

            confidence[n] = simpson(x = s, y = p_ns_ * p_sn_)

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4))
            plt.plot(n_arr,confidence)
            plt.xlabel("Photon number")
            plt.ylabel("Confidence")
            plt.show()

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

            for mean, label in zip(self.means, self.label_dB):
                #density_tot = density_tot + (density - np.min(density)) / (np.max(density) - np.min(density))
                plt.scatter(range(len(mean)),mean, label=f'{label}')
            #density_tot = (density_tot - np.min(density_tot)) / (np.max(density_tot) - np.min(density_tot))
            
            #plt.yscale('log')
            if xlim != None:
                plt.xlim(xlim)
            plt.xlabel("Photon number")
            plt.ylabel("Mean of clusters")
            plt.show()  