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

#from scipy.integrate import quad


class density_gaussianMixture():
    """
    Use Kernel density estimation to initialize Gaussian mixture process that separates the latent space into regions
    of high density associated with photon events.

    Parameters
    ----------
    X_low : numpy.array
        Array containing all the samples in their low-dimensional representation.
    cluster_max : float
        Maximum number of clusters to consider in the Bayesian Gaussian Mixture.
    flip : bool
        If `True` flips the latent space. This can be used to re-order the labels (right to left). 

    Returns
    -------
    None

    """
    def __init__(self, X_low, 
                 bw = [0.01], 
                 min_cluster_prob = 1e-6,
                 density_kernel = 'gaussian',
                 bins_plot = 5000,
                 flip = False, 
                 skip = 1,
                 size_plot = 10):
        
        self.crossTalk_ = None
        self.size_plot = size_plot
        self.flip = -1 if flip else 1

        X_low = self.flip * np.array(X_low).reshape(-1,1)
        if skip < 2: skip = 1
        kd = GridSearchCV(KernelDensity(), {"bandwidth": bw, 'kernel': [density_kernel]}).fit(X_low[::skip]).best_estimator_

        self.style_name = "seaborn-v0_8"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.min_ = np.min(X_low)
        self.max_ = np.max(X_low)
        self.bins = np.linspace(self.min_, self.max_, bins_plot)
        self.density = 10**kd.score_samples(self.bins.reshape(-1,1))

        maxs_init = self.bins[argrelextrema(self.density, np.greater)[0]]
        maxs = maxs_init[10**kd.score_samples(maxs_init.reshape(-1,1)) > min_cluster_prob]

        mins_init = self.bins[argrelextrema(self.density, np.less)[0]]

        index = 0
        mins = np.zeros(len(maxs)-1)
        for min_value in mins_init:
            if maxs[index] < min_value < maxs[index+1]:
                mins[index] = min_value
                index += 1
                if index == len(maxs)-1:
                    break

        weights_init = np.zeros(len(maxs))
        labels_init = np.searchsorted(mins, X_low)

        for index in range(len(maxs)):
            weights_init[index] = len(labels_init[labels_init == index]) / len(labels_init)

        fit_ = GaussianMixture(n_components=len(maxs), 
                               tol=1e-5, 
                               max_iter=200, 
                               means_init=maxs.reshape(-1,1),
                               weights_init=weights_init).fit(X_low)
                

        # Map labels based on their position in latent space
        #condition_small_cluster = fit_.weights_ > min_cluster_prob
        self.cluster_means = fit_.means_#[condition_small_cluster]
        self.cluster_covariance = fit_.covariances_#[condition_small_cluster]
        self.weights = fit_.weights_#[condition_small_cluster]
        #self.mins = torch.tensor(mins[condition_small_cluster].flatten())[:-1].to(self.device)

        self.cluster_means, self.cluster_covariance, self.weights  = zip(*sorted(zip(self.cluster_means, self.cluster_covariance, self.weights)))
        self.cluster_means, self.cluster_covariance, self.weights = np.asarray(self.cluster_means), np.asarray(self.cluster_covariance), np.asarray(self.weights)

        self.weights = self.weights #/ np.sum(self.weights)
        l_weights = len(self.weights)

        if l_weights > 1:
            sig = self.cluster_covariance#.astype('longdouble')
            u = self.cluster_means#.astype('longdouble')
            scaling = self.weights#.astype('longdouble')
            num_distributions = l_weights
            mins = np.zeros(num_distributions-1)

            for i in range(num_distributions-1):

                        u1 = u[i]
                        coeff1 = scaling[i]
                        sig1 = sqrt(sig[i])
                        u2 = u[i+1]
                        coeff2 = scaling[i+1]
                        sig2 = sqrt(sig[i+1])

                        mins[i] = (u2*sig1**2 - sig2*(u1*sig2 + sig1*np.sqrt((u1-u2)**2 + 2*(sig1**2-sig2**2)*np.log((sig1* coeff2) / (sig2* coeff1)))))/(sig1**2 - sig2**2)

                        #L = 2*(sig1**2 - sig2**2) * (log(sig1 * coeff2) - log(sig2 * coeff1))
                        #U = u1**2 - u1*u2 + u2**2
                        #delta1 = u1/sig1**2 - u2/sig2**2
                        #delta2 = sig1**-2 - sig2**-2

                        #mins[i] = (sqrt(L+U)/(sig1*sig2)+delta1) / delta2
            self.mins = torch.tensor(mins).to(self.device)

        else:
            self.mins = torch.tensor([self.max_]).to(self.device)
        

        #Sort labels/means
        #unique_labels = range(len(self.cluster_means))
        #centroids, mapping = zip(*sorted(zip(self.cluster_means, unique_labels)))
        #self.sorted_labels = np.array([self.mapping[i] for i in self.labels])

        self.clusters_low = []
        self.condition = []
        self.labels = torch.searchsorted(self.mins, torch.tensor(X_low).to(self.device)).cpu().numpy().flatten()
        

        for label in range(l_weights):
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
        return torch.searchsorted(self.mins, X_low)
    
     
    def plot_cross_talk(self):

        #https://www.wolframalpha.com/input?i2d=true&i=-Divide%5B1%2C2Power%5BSubscript%5B%CF%83%2C1%5D%2C2%5D%5D%5C%2840%29Power%5Bx%2C2%5D-2Subscript%5B%CE%BC%2C1%5Dx%2BPower%5BSubscript%5B%CE%BC%2C1%5D%2C2%5D%5C%2841%29%2Bln%5C%2840%29%CE%B1%5C%2841%29%3D-Divide%5B1%2C2Power%5BSubscript%5B%CF%83%2C2%5D%2C2%5D%5D%5C%2840%29Power%5Bx%2C2%5D-2Subscript%5B%CE%BC%2C2%5Dx%2BPower%5BSubscript%5B%CE%BC%2C2%5D%2C2%5D%5C%2841%29
        
        sig = self.cluster_covariance#.astype('longdouble')
        u = self.cluster_means#.astype('longdouble')
        scaling = self.weights#.astype('longdouble')

        num_distributions = len(u)
        crossTalk = np.zeros((num_distributions+1, num_distributions+1))#.astype('longdouble')
        crossTalk[0][0] = 1

        for i in range(num_distributions):
            for j in range(num_distributions):
                if i == j:
                    crossTalk[j][i] = 1
                elif i < j:
                    u1 = u[i]
                    coeff1 = scaling[i]
                    sig1 = sqrt(sig[i])
                    u2 = u[j]
                    coeff2 = scaling[j]
                    sig2 = sqrt(sig[j])

                    inter = (u2*sig1**2 - sig2*(u1*sig2 + sig1*np.sqrt((u1-u2)**2 + 2*(sig1**2-sig2**2)*np.log((sig1* coeff2) / (sig2* coeff1)))))/(sig1**2 - sig2**2)
                    cross = coeff1*(1 - 1/2*(1 + erf((inter-u1)/(sqrt(2)*sig1)))) + coeff2/2*(1 + erf((inter-u2)/(sqrt(2)*sig2)))

                    crossTalk[j+1][i+1] = cross/coeff2
                else:
                    u1 = u[j]
                    coeff1 = scaling[j]
                    sig1 = sqrt(sig[j])
                    u2 = u[i]
                    coeff2 = scaling[i]
                    sig2 = sqrt(sig[i])

                    inter = (u2*sig1**2 - sig2*(u1*sig2 + sig1*np.sqrt((u1-u2)**2 + 2*(sig1**2-sig2**2)*np.log((sig1* coeff2) / (sig2* coeff1)))))/(sig1**2 - sig2**2)
                    cross = coeff1*(1 - 1/2*(1 + erf((inter-u1)/(sqrt(2)*sig1)))) + coeff2/2*(1 + erf((inter-u2)/(sqrt(2)*sig2)))

                    crossTalk[j+1][i+1] = cross/coeff2
        """
        num_distributions = len(self.cluster_means)
        
        crossTalk = np.zeros((num_distributions, num_distributions))

        #plt.figure()
        for i in range(num_distributions):
            for j in range(num_distributions):
                #if i == j:
                #    crossTalk[j][i] = self.weights[i]/2
                #else:
                    
                w = np.linspace(self.cluster_means[i],self.cluster_means[j],100_000)

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
        #plt.show()
        """
        """
        num_distributions = len(self.cluster_means)
        
        crossTalk = np.zeros((num_distributions, num_distributions))

        #plt.figure()
        for i in range(num_distributions):
            for j in range(num_distributions):


                self.maxs
                self.mins

                w = np.linspace(-1,1,10_000).reshape(-1,1)

                distr1 = 10**self.density_function(w)
                distr2 = 10**self.density_function(w)
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

                
        
        """
        #plt.legend()
        #plt.show()
        self.crossTalk_ = crossTalk
        


        plt.figure()
        im = plt.imshow(self.crossTalk_, cmap="GnBu_r")#, norm=colors.LogNorm())
        plt.colorbar(im)
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
            #plt.yscale('log')
            plt.xlabel("Latent Space")
            plt.ylabel("Density")
            plt.show()

    
    def gaussian_function(self, x, mean, variance, weights):
            return weights * (2*np.pi*variance)**(-1/2) * np.exp(-(x - mean)**2/(2*variance))
        

    def plot_cluster(self,
                     scaling = 1):
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
        
        x = np.linspace(self.min_, self.max_, 1000)
        n =len(self.clusters_low)
        color = iter(cm.GnBu_r(np.linspace(0, 1, int(1.5*n))))

        #with plt.rc_context({'axes.edgecolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'white'}):
        with plt.style.context(self.style_name):
            plt.figure(figsize=(self.size_plot,4))#, dpi=200) #, dpi=100
            for index_cluster, cluster in enumerate(self.clusters_low):
                c = next(color)
                plt.hist(cluster.flatten() , self.bins, label=f"{index_cluster}", fill=True, histtype='step',color=c)#"#8dd3c7")
            gaussian_ = np.zeros(len(x))
            for index, mean_value in enumerate(self.cluster_means):
                gaussian_ = gaussian_ + self.gaussian_function(x, mean_value, self.cluster_covariance[index], scaling*self.weights[index]).flatten()
                plt.plot(x, self.gaussian_function(x.reshape(-1,1), mean_value, self.cluster_covariance[index], scaling*self.weights[index]), color="k")
            plt.plot(x, gaussian_.flatten())
            #plt.vlines(self.mins.cpu(), 0,10)
            #plt.yscale('log')
            plt.xlabel("Latent Space")
            plt.ylabel("Counts")

            plt.legend(ncol=3)
            #plt.xscale('symlog')
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
        