#General import
from torch import nn
import numpy as np



# Class specific import
#from fast_pytorch_kmeans import KMeans as pytorch_kmeans
#from pytorch_adapt.layers import SilhouetteScore as pytorch_silhouette_score

#class pytorch_kmeans_silhouette_loss():

#    def __init__(self):
#        pass

#    def forward(self, output, data, network, X):

#        feature = network(X, encoding=True).reshape(-1,1)

#        kmeans = pytorch_kmeans(n_clusters=23, mode='euclidean', verbose=False)
#        labels = kmeans.fit_predict(feature)

#        score = pytorch_silhouette_score()
#        score = score(feature, labels)
#        mse = nn.MSELoss()
#        loss = mse(output, data)

#        return 1e-4 / score  + loss


# Class specific import
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema


class sklearn_kernelDensity_loss():

    def __init__(self):
        pass

    def forward(self, output, data, network, X):

        feature = network(X, encoding=True).detach().numpy().reshape(-1,1)
        min_ = np.min(feature)
        max_ = np.max(feature)

        kde = KernelDensity(kernel='cosine', bandwidth=(max_-min_)/100).fit(feature)

        s = np.linspace(min_,max_,100)
        e = kde.score_samples(s.reshape(-1,1))
        mi = argrelextrema(e, np.less)[0], 
        ma = argrelextrema(e, np.greater)[0]

        mse = nn.MSELoss()
        loss = mse(output, data)

        return loss


# Class specific import
#import kmeans1d
#from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


#class sklearn_kmeans_silhouette_loss:

#    def __init__(self):
#        pass

#    def forward(self, output, data, network, X):
        
#        feature = network(X, encoding=True).detach().numpy().reshape(-1,1)

#        #labels = KMeans(n_clusters=21, random_state=42, n_init='auto').fit_predict(X_numpy)
#        labels, centroids = kmeans1d.cluster(feature, 23)

#        silhouette_loss = davies_bouldin_score(feature, labels)
#        mse = nn.MSELoss()

#        return silhouette_loss/10 + mse(output, data)
    


class triplet_MSE:

    def __init__(self) -> None:
        pass

    def forward(self, input_, output_, negative, alpha):
        trip = nn.TripletMarginLoss()
        triplet = trip(input_, output_, negative)
        mse = nn.MSELoss()
        MSE = mse(output_, input_)

        return alpha * triplet + MSE
    

from sklearn.manifold import trustworthiness

class trustworthinessLoss:

    def __init__(self):
        pass

    def forward(self, input_, output_, network, X):

        latent = network(X, encoding=True).detach().numpy().reshape(-1,1)
        X_init = X.detach().numpy().reshape(-1,X.size(2))
        t = trustworthiness(X_init, latent, metric="euclidean")

        mse = nn.MSELoss()
        MSE = mse(output_, input_)
        
        return 1/t + MSE
