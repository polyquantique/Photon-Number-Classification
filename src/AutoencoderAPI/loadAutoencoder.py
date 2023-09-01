import torch
from torch import nn

from .utils.files import open_object
from .setup.networks.autoencoder import build_autoencoder

def loadAutoencoder(X_init, model, filter=False, threshold=0.0005):
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
    path = f"Autoencoder Log/{model}"
    config_load = open_object(f"{path}/log.bin")

    network = build_autoencoder(config_load)
    network.load_state_dict(torch.load(f"{path}/model.pt"))
    network.eval()

    X_pytorch = torch.from_numpy(X_init).view(-1, 1, config_load['files']['input_dimension']).float()
    X_low_dim = network(X_pytorch, encoding=True)
    X_reconst = network(X_low_dim, decoding=True).view(-1, 1, config_load['files']['input_dimension'])
    
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
    X_reconst = X_reconst.detach().numpy().reshape(-1, config_load['files']['input_dimension'])

    return X_init, X_reconst, X_low_dim