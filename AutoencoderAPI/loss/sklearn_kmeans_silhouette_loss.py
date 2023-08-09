from torch import nn
#from sklearn.cluster import KMeans
import kmeans1d
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class sklearn_kmeans_silhouette_loss:

    def __init__(self):
        pass

    def forward(self, output, data, network, X):
        
        X_numpy =X.numpy().reshape(-1,250)
        feature = network(X, encoding=True).detach().numpy().reshape(-1,1)

        #labels = KMeans(n_clusters=21, random_state=42, n_init='auto').fit_predict(X_numpy)
        labels, centroids = kmeans1d(X_numpy, 21)

        silhouette_loss = davies_bouldin_score(feature, labels)
        mse = nn.MSELoss()

        return 1e4 / silhouette_loss + mse(output, data)