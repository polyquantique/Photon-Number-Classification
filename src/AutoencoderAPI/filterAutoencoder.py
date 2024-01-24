import torch
from torch import nn 
import numpy as np
from .setup.networks.autoencoder import build_autoencoder
from .utils.files import open_object


class filterAutoencoder():

    def __init__(self, model_file):

        path = f"Autoencoder Log/{model_file}"
        config_load  = open_object(f"{path}/log.bin")

        network = build_autoencoder(config_load)
        network.load_state_dict(torch.load(f"{path}/model.pt"))
        network.eval()

        self.model = network
        self.config  = config_load 

    def filtering(self, X, threshold):

        input_ = torch.from_numpy(X).view(-1, 1, self.config['files']['input_dimension']).float()
        output_ = self.model(input_)
        mse = nn.MSELoss()
        accepted = []
        filtered = []
        for index, value in enumerate(input_):
            if mse(output_[index], value) < threshold:
                accepted.append(X[index])
            else:
                filtered.append(X[index])

        return np.array(accepted), np.array(filtered)
