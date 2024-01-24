import torch
import numpy as np
from .utils.files import open_object
from .utils.clustering.clustering import clustering
from .setup.networks.autoencoder import build_autoencoder


class autoencoder_kmeans():

    def __init__(self, model_path):

        config_load = open_object(f"{model_path}/log.bin")

        network = build_autoencoder(config_load)
        network.load_state_dict(torch.load(f"{model_path}/model.pt"))

        self.network = network
        self.config_load = config_load
        self.kmeans = None
        
    
    def fit_cluster(self, X, min_cluster, max_cluster, plot_silhouette=False, plot_clustering=False):

        self.network.eval()
        with torch.no_grad():
            X_pytorch = torch.from_numpy(X).view(-1, 1, self.config_load['files']['input_dimension']).float()
            X_low_dim = self.network(X_pytorch, encoding=True)

            X_low_dim = X_low_dim.detach().numpy().reshape(-1, 1)

            cl = clustering(X_low_dim, min_cluster, max_cluster)
            
            if plot_silhouette:
                cl.plot_silhouette()
            if plot_clustering:
                cl.plot_clustering()

        self.kmeans = cl.fit
        self.label_mapping = cl.label_mapping


    def get_label(self, X):

        X_pytorch = torch.from_numpy(X).view(-1, 1, self.config_load['files']['input_dimension']).float()
        self.network.eval()
        with torch.no_grad():
            X_low_dim = self.network(X_pytorch, encoding=True)
            X_low_dim = X_low_dim.detach().numpy().reshape(-1, 1)
            unsorted_labels = self.kmeans.predict(X_low_dim)
        return np.array([self.label_mapping[i] for i in unsorted_labels])