import numpy as np
from scipy.stats import multivariate_normal


class GMM_1D:
    def __init__(self, k, max_iter=5):
        self.k = k
        self.max_iter = int(max_iter)

        self.mu = None # Cluster mean
        self.sigma = None # Cluster variance

    def initialize(self, X):
        #self.shape = X.shape
        #self.n, self.m = self.shape
        self.n = len(X) # Number of samples

        self.pi = np.full(shape=self.k, fill_value=1/self.k)
        self.weights = np.full( shape=self.k, fill_value=1/self.k)
        
        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [  X[row_index,:] for row_index in random_row ]
        self.sigma = [ np.cov(X) for _ in range(self.k) ]

    def e_step(self, X):
        # E-Step: update weights and pi holding mu and sigma constant
        self.weights = self.predict_proba(X)
        self.pi = self.weights.mean(axis=0)
    
    def m_step(self, X):
        # M-Step: update mu and sigma holding pi and weights constant
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T, 
                aweights=(weight/total_weight).flatten(), 
                bias=True)

    def fit(self, X):
        self.initialize(X)
        
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            
    def predict_proba(self, X):
        likelihood = np.zeros( (self.n, self.k) )
        for i in range(self.k):
            distribution = multivariate_normal(
                                        mean=self.mu[i], 
                                        cov=self.sigma[i])
            likelihood[:,i] = distribution.pdf(X)
        
        numerator = likelihood * self.pi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights
    
    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)