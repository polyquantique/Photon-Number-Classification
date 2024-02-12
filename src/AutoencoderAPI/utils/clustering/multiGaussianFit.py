import numpy as np
from scipy.signal import find_peaks, argrelextrema
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class fit_gaussian():

    def __init__(self, bins, density):

        #########
        #plt.figure()
        #plt.plot(bins, density)
        #plt.show()
        #########

        min_ = np.min(bins)
        max_ = np.max(bins)

        peaks_index, _ = find_peaks(density, prominence=0.1)
        peaks_guess = bins[peaks_index]
        variance_guess = [abs(j-i)/2 for i, j in zip(peaks_guess, peaks_guess[1:])]
        variance_guess.append(abs(peaks_guess[-1]-peaks_guess[-2])/2)
        weights_guess = variance_guess / np.sum(variance_guess)

        print(f'Found {len(weights_guess)} clusters')

        guess = np.ravel([[peak, weights, variance] for peak, weights, variance in zip(peaks_guess, weights_guess, variance_guess)])
        # [mean, amp, var]

        upper_bounds = np.ravel([[max_ , np.max(density), (max_-min_)/2] for i in peaks_guess])
        lower_bounds = np.ravel([[min_ , 0, 0] for i in peaks_guess])
        bounds = (lower_bounds , upper_bounds)

        popt, pcov = curve_fit(self.multi_gauss, bins, density, p0=guess, maxfev = 10000, bounds=bounds)
        
        #########
        #plt.figure()
        #plt.plot(bins, self.multi_gauss(bins, *popt))
        #plt.show()
        #########

        popt = np.asarray(popt)

        cluster_means = popt[0::3]
        cluster_weights = popt[1::3]
        cluster_variance = popt[2::3]
        
        cluster_means, cluster_variance, cluster_weights = zip(*sorted(zip(cluster_means, cluster_variance, cluster_weights)))

        self.cluster_means = np.asarray(cluster_means)
        self.cluster_variance = np.asarray(cluster_variance)
        self.cluster_weights = np.asarray(cluster_weights)
        self.mins = self.find_mins()


    def find_mins(self):

        sig = self.cluster_variance
        u = self.cluster_means
        scaling = self.cluster_weights
        num_distributions = len(scaling)
        mins = np.zeros(num_distributions-1)

        for i in range(num_distributions-1):

                    u1 = u[i]
                    coeff1 = scaling[i]
                    sig1 = np.sqrt(sig[i])
                    u2 = u[i+1]
                    coeff2 = scaling[i+1]
                    sig2 = np.sqrt(sig[i+1])

                    mins[i] = (u2*sig1**2 - sig2*(u1*sig2 + sig1*np.sqrt((u1-u2)**2 + 2*(sig1**2-sig2**2)*np.log((sig1* coeff2) / (sig2* coeff1)))))/(sig1**2 - sig2**2)

        return mins

    def multi_gauss(self, x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            mean = params[i]
            amp = params[i+1]
            variance = params[i+2]
            y = y + amp * (2*np.pi*variance)**(-1/2) * np.exp( -(x - mean)**2/(2*variance))
        return y


    def single_gauss1(self, x):
        mean = self.params1[0]
        amp = self.params1[1]
        variance = self.params1[2]
        y = amp * (2*np.pi*variance)**(-1/2) * np.exp( -(x - mean)**2/(2*variance))
        return y
    
    def single_gauss1(self, x):
        mean = self.params2[0]
        amp = self.params2[1]
        variance = self.params2[2]
        y = amp * (2*np.pi*variance)**(-1/2) * np.exp( -(x - mean)**2/(2*variance))
        return y


    def eval_inter(self,x):
        func1 = self.single_gauss1(x)
        func2 = self.single_gauss2(x)
        z = np.array([func1,func2])
        return z
