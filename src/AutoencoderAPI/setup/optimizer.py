import torch.optim as optim


def build_optimizer(network, config):
    """
    # build_optimizer

    Define the optimizer used in training.

    Parameters
    ----------
    - config : dict
            - Dictionary containing the experiment parameters. 
                See the `autoencoder` class for more details on the config dictionary
    - network : Pytorch sequential : 
            - Autoencoder neural network that is trained to reproduce its input signal.

    Returns
    -------
    - optimizer : Pytorch optimizer
            - Optimizer used to train the autoencoder.
    """

    optimizer_dict = {
        "SGD"  : optim.SGD,
        "Adam" : optim.Adam
    }

    try:
        optimizer = optimizer_dict[config['train']['optimizer']]
    except Exception as ex:
        optimizer = optimizer_dict["adam"]
        print("No optimizer was defined int the configuration dict (was set to adam)")

    return optimizer(network.parameters(), lr=config['train']['learning_rate'])