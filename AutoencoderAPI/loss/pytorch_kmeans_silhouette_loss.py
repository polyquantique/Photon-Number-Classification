
from fast_pytorch_kmeans import KMeans as pytorch_kmeans
from pytorch_adapt.layers import SilhouetteScore as pytorch_silhouette_score
from torch import nn

class pytorch_kmeans_silhouette_loss():

    def __init__(self):
        pass

    def forward(self, output, data, network, X):

        feature = network(X, encoding=True).reshape(-1,1)

        kmeans = pytorch_kmeans(n_clusters=30, mode='euclidean', verbose=False)
        labels = kmeans.fit_predict(feature)

        score = pytorch_silhouette_score()
        score = score(feature, labels)
        mse = nn.MSELoss()
        loss = mse(output, data)

        return 1 / score  + loss