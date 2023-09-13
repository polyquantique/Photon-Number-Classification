import torch
import numpy as np
from sklearn.neighbors import KernelDensity

from .utils.files import open_object
from .utils.kernelDensity import kernel_density
from .setup.networks.autoencoder import build_autoencoder


class autoencoder_kerneDensity():

    def __init__(self, model_path):

        config_load = open_object(f"{model_path}/log.bin")

        network = build_autoencoder(config_load)
        network.load_state_dict(torch.load(f"{model_path}/model.pt"))

        self.network = network
        self.config_load = config_load
        self.fit = None
        
    
    def fit_cluster(self, X, plot_density=False, plot_cluster=False, plot_traces=False, bw_cst=4):

        self.network.eval()
        traces = np.copy(X)
        with torch.no_grad():
            X_pytorch = torch.from_numpy(X).view(-1, 1, self.config_load['files']['input_dimension']).float()
            X_low_dim = self.network(X_pytorch, encoding=True)

            X_low_dim = X_low_dim.detach().numpy().reshape(-1, 1)

            cl = kernel_density(traces, X_low_dim, bw_cst)
            
        if plot_density:
            cl.plot_density()
        if plot_cluster:
            cl.plot_cluster()
        if plot_traces:
            cl.plot_traces()

        self.fit = cl.fit
        

    def get_label(self, X):

        X_pytorch = torch.from_numpy(X).view(-1, 1, self.config_load['files']['input_dimension']).float()
        self.network.eval()
        with torch.no_grad():
            X_low_dim = self.network(X_pytorch, encoding=True)
            X_low_dim = X_low_dim.detach().numpy().reshape(-1, 1)
        return self.fit(X_low_dim)