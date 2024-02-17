import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm.notebook import tqdm


class GMM_1D:
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
        The standard deviation of each mixture component.
    gamma :
        Posterior probability that component k was responsible for generating sample x_n
    """
    def __init__(self, n_cluster = 3, 
                        n_step = 3,
                        tol = 400):
        
        self.n_cluster = n_cluster
        self.n_sample = None
        self.n_step = n_step
        self.tol = tol
        self.pi = None
        self.mu = None
        self.sigma = None
        self.gamma = None
        self.likelihood = []


    def initialize(self, X):
        """
        Initialize means, weights and variance randomly
        
        """
        self.pi = np.ones(self.n_cluster) / self.n_cluster
        self.mu = np.random.choice(X, self.n_cluster)
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
            normal = norm.pdf(x = X, 
                            loc = self.mu[index_cluster], 
                            scale = self.sigma[index_cluster])
            product[:,index_cluster] = self.pi[index_cluster] * normal
        
        self.gamma = product / np.sum(product, axis=0)


    def M_step(self, X):
        """M Step
        
        Parameters
        ----------
            
        Returns
        -------
        """
        X = X.reshape(-1,1)
        N_k = np.sum(self.gamma, axis = 0)
        

        self.mu = np.sum(self.gamma * X, axis = 0) / N_k
        self.sigma = np.sum(self.gamma * (X - self.mu) ** 2, axis = 0) / N_k
        self.pi = N_k / self.n_sample


    def get_likelihood(self, X):
        """
        
        """
        product = np.zeros((self.n_cluster, self.n_sample))

        for index_cluster in range(self.n_cluster):
            normal = norm.pdf(x = X, 
                            loc = self.mu[index_cluster], 
                            scale = self.sigma[index_cluster])
            product[index_cluster,:] = self.pi[index_cluster] * normal

        self.likelihood.append(np.sum(np.log(np.sum(product, axis = 0)), axis = 0))


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
        self.initialize(self.n_cluster)

        for step in tqdm(range(self.n_step), total=self.n_step):
            self.E_step(X)
            self.M_step(X)
            self.get_likelihood(X)

            #if self.likelihood < self.tol:
            #    break
        
        plt.figure()
        #plt.hist(X, bins=1000)
        lin = np.arange(-2,1,0.01)
        for index_cluster in range(self.n_cluster):
            normal = norm.pdf(x = lin, 
                            loc = self.mu[index_cluster], 
                            scale = self.sigma[index_cluster])
            plt.plot(lin, normal)
        plt.show()

        plt.figure()
        plt.plot(self.likelihood)
        plt.show()






