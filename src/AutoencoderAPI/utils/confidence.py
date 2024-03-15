import numpy as np
import torch 
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.integrate import  quad, simpson, trapezoid
from scipy.stats import norm

from .correlation import second_order
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
        kd.plot_psn()
        kd.plot_pns()
        kd.plot_confidence()

        self.style_name = "seaborn-v0_8"
        self.label_shift = label_shift
        self.possible_photon_number = kd.unique_labels
        self.predict = kd.predict
        self.labels = kd.labels
        self.cluster_means = kd.cluster_means

        self.s = kd.s
        self.label_dB = []
        self.density = []
        self.average_number = []
        self.g2 = []
        self.confidence = []
        self.unique_labels = []
        

    
    def fit(self, X_low,
            number_cluster = 1,
            flip = False, 
            label_dB = None,
            zero_number = None):
        

        if label_dB != None: 
            self.label_dB.append(label_dB)
        labels = self.predict(X_low)
        means_init = self.cluster_means[np.unique(labels) - self.label_shift]
        label_shift = np.min(labels)

        kd = gaussian_mixture(X_low, 
                            number_cluster = number_cluster,
                            flip = flip, 
                            size_plot = 10,
                            label_shift = label_shift)
                            #means_init = means_init)

        labels = kd.predict(X_low)
        unique_labels = np.unique(labels)
        if zero_number != None:
            labels = np.concatenate([labels, np.zeros(zero_number)])
        average_photon = np.sum(kd.cluster_weights * unique_labels)
     
        kd.plot_cluster()
        kd.plot_psn()
        kd.plot_pns()
        kd.plot_confidence()

        self.average_number.append(average_photon)
        self.g2.append(second_order(labels))
        self.confidence.append(kd.confidence)
        self.unique_labels.append(unique_labels)

        print(f'Average photon number {average_photon}')
        print(f'{kd.cluster_weights}')
        print(f'Integral all p(s|n) {simpson(y = kd.mixture, x = kd.s)} (should be 1)')
        
        


    def plot_all_confidence(self):

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4))
            for label_dB, n_arr, confidence in zip(self.label_dB, self.unique_labels, self.confidence):
                plt.plot(n_arr,confidence, label=f'{label_dB}')
            plt.xlabel("Photon number")
            plt.ylabel("Confidence")
            plt.legend()
            plt.show()


    def plot_mean_cluster(self,xlim=None):

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4), dpi=200)
            for mean, label in zip(self.average_number, self.label_dB):
                plt.scatter(range(len(mean)),mean, label=f'{label}')

            if xlim != None:
                plt.xlim(xlim)
            plt.xlabel("Photon number")
            plt.ylabel("Mean of clusters")
            plt.show()  

    """
    def plot_total_density(self,xlim=None):

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4), dpi=200)
            density_tot = np.zeros(len(self.density[0]))

            for bins, density in zip(self.bins, self.density):
                density_tot = density_tot + (density - np.min(density)) / (np.max(density) - np.min(density))
            plt.plot(bins,density_tot)
            if xlim != None:
                plt.xlim(xlim)
            plt.xlabel("Latent Space")
            plt.ylabel("Density")
            plt.show()
    """

    """
    def plot_all_density(self,xlim=None, flip=False):

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4))
            for label_dB, bins, density in zip(self.label_dB, self.bins, self.density):
                if flip: x = -1*bins
                plt.plot(x,density, label=f'{label_dB} dB', linewidth=0.8)
            if xlim != None:
                plt.xlim(xlim)
            plt.xlabel("Latent Space")
            plt.ylabel("Density")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.show()
    """

    
    