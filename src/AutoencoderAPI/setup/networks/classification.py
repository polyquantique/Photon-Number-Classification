from torch import nn


class build_classifier(nn.Module):
    """
    Build a Pytorch classifier architecture with desired caracteristics. 

    Parameters
    ----------
    config : dict
        Dictionary containing the CNN desired caracteristics. 

    Returns
    -------
    None
    """
    def __init__(self, config: dict):
        
        super().__init__()

        activation_dict = {
            "ReLU"       : nn.ReLU,
            "Sigmoid"    : nn.Sigmoid,
            "CELU"       : nn.CELU, 
            "Softmax"    : nn.Softmax,
            "Softmin"    : nn.Softmin,
            "Hardshrink" : nn.Hardshrink,
            "LeakyReLU"  : nn.LeakyReLU,
            "ELU"        : nn.ELU,
            "LogSigmoid" : nn.LogSigmoid,
            "PReLU"      : nn.PReLU,
            "GELU"       : nn.GELU,
            "SiLU"       : nn.SiLU,
            "Mish"       : nn.Mish,
            "Softplus"   : nn.Softplus,
            "Softsign"   : nn.Softsign,
            "Tanh"       : nn.Tanh,
            "GLU"        : nn.GLU,
            "Threshold"  : nn.Threshold,
            "MaxPool2d"  : nn.MaxPool2d
        }

        layer_list = config['network']['layer_list'].copy()
        input_ = config['internal']['size_network']
        layer_list = [input_] + layer_list + [input_]
        activation_list = config['network']['activation_list']
        network_type = config['network']['network_type']
        if network_type == 'CNN':
            channel_list = config['network']['CNN_channels']
        
        self.classifier = nn.Sequential()

        for index, activation_type in enumerate(activation_list):

            self.classifier.append(nn.Linear(layer_list[index], layer_list[index+1]))
            if network_type == 'CNN':
                self.classifier.append(nn.Conv1d(channel_list[index], channel_list[index+1], groups=1, kernel_size=10, padding='same'))
            self.classifier.append(activation_dict[activation_type]())
        
        self.classifier.append(nn.MaxPool1d(2, 2))
        self.classifier.append(nn.Conv1d(6, 16, 5))
        self.classifier.append(nn.Linear(16 * 5 * 5, 120))
        self.classifier.append(nn.Linear(120, 84))
        self.classifier.append(nn.Linear(84, 10))


    def forward(self, X) -> any:
        """
        Forward pass of the autoencoder.

        Parameters
        ----------
        X : torch.tensor
            Input signal of the autoencoder.
        encoding : bool
            If `True` the forward pass will return the encoder output.
        decoding : bool
            If `True` the forward pass expects an input X of size equal to 
            the encoder output and returns the encoder output.

        Returns
        -------
        both = `True` (encoder, decoder) : torch.tensor
            Encoder and decoder output.
        encoding = `True` (encoder) : torch.tensor
            Encoder output (size of the middle layer of the autoencoder)
        decoding = `True` (decoder) : torch.tensor
            Decoder output (size of the input layer of the autoencoder)
        encoding = decoding = `False` : torch.tensor
            Autoencoder output (size of the input layer of the autoencoder)
        """
    
        return self.classifier(X)