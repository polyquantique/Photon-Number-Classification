import numpy as np
from scipy.stats import norm

def cross_talk(means, variances):
    
    num_distributions = len(means)
    confusion_matrix = np.zeros((num_distributions, num_distributions))

    for i in range(num_distributions):
        for j in range(num_distributions):
            if i == j:
                confusion_matrix[i][j] = 1.0
            else:
                overlap = norm(loc=means[i], scale=np.sqrt(variances[i] + variances[j])).cdf(means[j])
                confusion_matrix[i][j] = overlap

    return confusion_matrix
