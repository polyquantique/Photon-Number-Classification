import numpy as np
import torch 
import matplotlib.pyplot as plt
from scipy.special import factorial

from .clustering.densityGaussianMixture import density_gaussianMixture

class confidence:

    def __init__(self, X_low,
                bw = [0.01], 
                min_cluster_prob = 1e-6,
                bins_plot = 5000,
                density_kernel='gaussian',
                flip = False, 
                skip = 0,
                size_plot = 10,
                scaling = 1):

        # Global clustering for photon number assignment
        kd = density_gaussianMixture(X_low, 
                                    bw = bw, 
                                    min_cluster_prob = min_cluster_prob,
                                    bins_plot = bins_plot,
                                    density_kernel= density_kernel,
                                    flip = flip, 
                                    skip = skip,
                                    size_plot = size_plot)

        # Verification of clusters
        kd.plot_density()
        kd.plot_cluster(scaling)

        self.style_name = "seaborn-v0_8"
        self.possible_photon_number = np.unique(kd.labels)
        self.predict = kd.predict
        self.labels = kd.labels

        self.n_arr = []
        self.confidence = []

    
    def fit(self, X_low,
            bw = [0.01], 
            min_cluster_prob = 1e-6,
            bins_plot = 5000,
            density_kernel='gaussian',
            flip = False, 
            skip = 0,
            size_plot = 10,
            scaling = 1):

        # Global clustering for photon number assignment
        kd = density_gaussianMixture(X_low, 
                                    bw = bw, 
                                    min_cluster_prob = min_cluster_prob,
                                    bins_plot = bins_plot,
                                    density_kernel= density_kernel,
                                    flip = flip, 
                                    skip = skip,
                                    size_plot = size_plot)

        # Verification of clusters
        kd.plot_density()
        kd.plot_cluster(scaling)
        kd.plot_cross_talk()

        cross_talk = kd.crossTalk_
        weights = kd.weights
        detected_photon_number = self.predict(torch.tensor(X_low).to(kd.device)).cpu().numpy()
        n_ = np.mean(detected_photon_number)

        range_ = len(cross_talk)
        confidence = []

        n_arr = np.linspace(0,range_,range_)#.astype('longdouble')
        p_n = np.exp(-n_) * (n_**n_arr) / factorial(n_arr)

        for n in range(range_):

            confidence_temp = 0

            for s in range(range_):

                confidence_temp +=  (cross_talk[s][n]**2 * p_n[n]) / np.sum(cross_talk[s] * p_n)

            confidence.append(confidence_temp)

        self.n_arr.append(n_arr)
        self.confidence.append(confidence)

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4))
            plt.plot(n_arr,confidence)
            plt.xlabel("Photon number")
            plt.ylabel("Confidence")
            plt.show()

    def plot_all(self):

        with plt.style.context(self.style_name):
            plt.figure(figsize=(10,4))
            for n_arr, confidence in zip(self.n_arr, self.confidence):
                plt.plot(n_arr,confidence)
            plt.xlabel("Photon number")
            plt.ylabel("Confidence")
            plt.show()
