import numpy as np
import torch 
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.integrate import  quad, simpson, trapezoid
from scipy.stats import norm

from .correlation import second_order
from .clustering.GaussianMixture2D import gaussian_mixture_2d

class confidence_2d:

    def __init__(self, X_low, X_high,
                number_cluster = 1,
                size_plot = 10,
                label_shift = 0,
                bw_density = 0.24,
                dx = 1e-4,
                cluster_iter = 10):

        # Global clustering for photon number assignment
        gm = gaussian_mixture_2d(X_low, X_high,
                                number_cluster = number_cluster,
                                size_plot = size_plot,
                                label_shift = label_shift,
                                dx = dx,
                                cluster_iter=cluster_iter)


        # Verification of clusters
        gm.plot_density(bw_adjust=bw_density)
        gm.plot_cluster()
        #gm.plot_psn()
        #gm.plot_pns()
        #kd.plot_confidence()
        gm.plot_traces(X_high)
        traces = gm.plot_traces_average(X_high)
        #kd.plot_FWHM_average(X_high)

        self.style_name = "seaborn-v0_8"
        self.label_shift = label_shift
        self.dx = dx
        self.cluster_iter=cluster_iter
        self.unique_labels = gm.unique_labels
        self.predict = gm.predict
        self.labels = gm.labels
        self.cluster_means = gm.cluster_means

        #self.s = gm.s
        self.density = []
        self.average_number = [] #User input in fit
        self.average_number_ = [] #Evaluated with method
        self.g2 = []
        self.confidence = []
        self.cluster_unique_labels = []
        

    
    def fit(self, X_low, X_high,
            average_number,
            number_cluster = 1,
            zero_number = None,
            bw_density = 0.24):
        
        self.average_number.append(average_number)
        labels = self.predict(X_low)
        means_init = self.cluster_means[np.unique(labels) - self.label_shift]
        label_shift = np.min(labels)

        gm = gaussian_mixture_2d(X_low, X_high,
                            number_cluster = number_cluster,
                            size_plot = 10,
                            label_shift = label_shift,
                            means_init = means_init,
                            dx = self.dx,
                            cluster_iter=self.cluster_iter)

        labels = gm.predict(X_low)
        unique_labels = np.unique(labels)
        if zero_number != None:
            labels = np.concatenate([labels, np.zeros(zero_number)])
        average_photon = np.sum(gm.cluster_weights * unique_labels)
     
        gm.plot_density(bw_adjust=bw_density)
        gm.plot_cluster()
        gm.plot_psn(average_number)
        gm.plot_pns()
        gm.plot_confidence()
        gm.plot_traces(X_high)
        traces = gm.plot_traces_average(X_high)

        self.average_number_.append(average_photon)
        self.g2.append(second_order(labels))
        #self.confidence.append(gm.confidence)
        self.cluster_unique_labels.append(unique_labels)

        print(f'Average photon number {average_photon}')
        print(f'{gm.cluster_weights}')
        #print(f'Integral all p(s|n) {simpson(y = gm.mixture, x = kd.s)} (should be 1)')
        
        


    def plot_all_confidence(self):

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4))
            for label_dB, n_arr, confidence in zip(self.label_dB, self.cluster_unique_labels, self.confidence):
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

    
    