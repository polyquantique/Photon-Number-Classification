import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from .utils.files import open_object
from .setup.networks.autoencoder import build_autoencoder

def loadAutoencoder(X_init, model_path, filter=False, threshold=0.0005, flip=False):
    """
    # loadAutoencoder

    load_sweep_results(file_name, parameters)


    Parameters
    ----------
    - file_name : str
            - Name of the file or path to file inside the `Autoencoder Log` folder.
    
    Returns
    -------
    - None

    """
    config_load = open_object(f"{model_path}/log.bin")

    network = build_autoencoder(config_load)
    network.load_state_dict(torch.load(f"{model_path}/model.pt"))
    network.eval()

    X_pytorch = torch.from_numpy(X_init).view(-1, 1, config_load['internal']['size_network']).float()
    X_low_dim = network(X_pytorch, encoding=True)
    X_reconst = network(X_low_dim, decoding=True).view(-1, 1, config_load['internal']['size_network'])
    
    if filter:
        mse = nn.MSELoss()
        index_list = []
        for index, x in enumerate(X_reconst):
            if mse(x, X_pytorch[index]) < threshold:
                index_list.append(index)

        #X_pytorch = X_pytorch[condition]
        X_init = X_init[index_list]
        X_low_dim = X_low_dim[index_list]
        X_reconst = X_reconst[index_list]

    X_low_dim = X_low_dim.detach().numpy().reshape(-1, 1)
    X_reconst = X_reconst.detach().numpy().reshape(-1, config_load['internal']['size_network'])

    X_low_dim = (X_low_dim - np.min(X_low_dim)) / (np.max(X_low_dim) - np.min(X_low_dim))

    if flip:
        X_low_dim = -1*X_low_dim

    return X_init, X_reconst, X_low_dim



def plot_weigths(model_path):

    config_load = open_object(f"{model_path}/log.bin")

    network = build_autoencoder(config_load)
    network.load_state_dict(torch.load(f"{model_path}/model.pt"))

    with plt.style.context("seaborn-v0_8"):
        plt.figure(figsize=(10,4), dpi=100)
        for name, param in network.named_parameters():
            weights = param.detach().numpy()
            if len(np.shape(weights)) > 1 and name[8] != '4':
                plt.plot(np.mean(weights, axis=1), alpha=0.8, label=f'{name}')
        plt.xlabel("Weight Position")
        plt.ylabel("Weight Value")
        plt.legend()
        plt.show()
