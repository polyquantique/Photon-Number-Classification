import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson 
from scipy.special import factorial
from tqdm.notebook import tqdm


class PGMM_1D:
    """
    

    Parameters
    ----------
    n_cluster : int
        The number of clusters
    pi : array-like, shape (n_cluster,)
        The weight of each mixture component.
    mu : array-like, shape (n_cluster,)
        The means of each mixture component.
    sigma : array-like, shape (n_cluster,)
        The variance of each mixture component.
    gamma :
        Posterior probability that component k was responsible for generating sample x_n
    """
    def __init__(self,  n_cluster = 3, 
                        n_step = 3,
                        n_average = 1,
                        n_number = None,
                        tol = 400,
                        mean_init = None):
        
        if n_number != None:
            n_number = np.array(n_number)
        else:
            n_number = np.arange(n_cluster)

        self.n_cluster = n_cluster
        self.n_sample = None
        self.n_step = n_step
        self.tol = tol
        self.mean_init = mean_init
        self.n_average = n_average
        self.n_number = n_number
        self.pi = None
        self.mu = None
        self.sigma = None
        self.gamma = None
        self.likelihood = []


    def initialize(self, X):
        """
        Initialize means, weights and variance randomly
        
        """
        if self.mean_init != None:
            self.mu = np.array(self.mean_init)
        else:
            self.mu = np.random.choice(X.flatten(), self.n_cluster)

        poisson_ = poisson(mu = self.n_average).pmf(self.n_number)
        self.pi = poisson_ / np.sum(poisson_)
        self.sigma = np.random.random_sample(size = self.n_cluster)

        self.get_likelihood(X)


    def E_step(self, X):
        """
        E Step
        
        Parameters
        ----------
        """
        product = np.zeros((self.n_sample, self.n_cluster))
        for index_cluster in range(self.n_cluster):
            normal = norm(loc = self.mu[index_cluster], 
                          scale = np.sqrt(self.sigma[index_cluster])).pdf(x = X)
            product[:,index_cluster] = self.pi[index_cluster] * normal

        sum_ = np.sum(product, axis = 1).reshape(-1,1)
        self.gamma = product / sum_


    def M_step(self, X):
        """
        M Step
        
        Parameters
        ----------
            
        Returns
        -------
        """
        X = X.reshape(-1,1)
        N_k = np.sum(self.gamma, axis = 0)
        

        self.mu = np.sum(self.gamma * X, axis = 0) / N_k
        self.sigma = np.sum(self.gamma * (X - self.mu)**2, axis = 0) / N_k
        #self.pi = N_k / self.n_sample


    def get_likelihood(self, X):
        """
        
        """
        product = np.zeros((self.n_cluster, self.n_sample))

        for index_cluster in range(self.n_cluster):
            normal = norm(loc = self.mu[index_cluster], 
                          scale = np.sqrt(self.sigma[index_cluster])).pdf(x = X)
            product[index_cluster,:] = self.pi[index_cluster] * normal

        sum1 = np.log(np.sum(product, axis = 0))
        sum2 = np.sum(sum1, axis = 0)
        self.likelihood.append(sum2)


    def fit(self, X):
        """ Training step of the GMM model
        
        Parameters
        ----------
        data : array-like, shape (n_samples,)
            The data.
        n_components : int
            The number of clusters
        n_steps: int
            number of iterations to run
        """
        X = X.reshape(1,-1)
        self.n_sample = X.shape[1]
        self.initialize(X)

        for step in tqdm(range(self.n_step), total=self.n_step):
            self.E_step(X)
            self.M_step(X)
            self.get_likelihood(X)

            #if self.likelihood < self.tol:
            #    break
        
        plt.figure()
        plt.hist(X.flatten(), bins=1000)
        lin = np.arange(-1.5,1,0.01)
        for index_cluster in range(self.n_cluster):
            normal = 20*norm.pdf(x = lin, 
                            loc = self.mu[index_cluster], 
                            scale = np.sqrt(self.sigma[index_cluster]))
            plt.plot(lin, normal)
        plt.show()

        plt.figure()
        plt.plot(self.likelihood)
        plt.show()






