from torch import nn


class build_autoencoder(nn.Module):
    def __init__(self, config: dict):
        """
        # build_autoencoder

        Build a Pytorch autoencoder architecture with desired caracteristics. 

        Parameters
        ----------
        config : dict
                Dictionary containing the CNN desired caracteristics. 
                See the `autoencoder` class for more details on the config dictionary

        Returns
        -------
        None
        """
        super().__init__()
        
        if config['network']['layer_number'] % 2 != 0:
            print("Invalid number of layer (needs to be even number)")

        # Layer type
        layer_type_dict = {
            "Linear" : nn.Linear
        }
        layer = layer_type_dict[config['network']['layer_type']]

        # Activation function
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
        }

        skip = config['train']["skip_elements"]
        size = config['files']['input_dimension']
        layer_list = config['network']['layer_list']
        layer_list[0] = layer_list[-1] = int(size / skip)
        activation_list = config['network']['activation_list']
        
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        
        #self.encoder.append(nn.Conv1d(1, 1, kernel_size=21, stride=1, padding='same'))
        #self.encoder.append(nn.Conv1d(1, 1, kernel_size=21, stride=1, padding='same'))
        
        for index, activation_type in enumerate(activation_list):
            if index < len(activation_list) // 2 + 1:
                self.encoder.append(layer(layer_list[index], layer_list[index+1]))
                self.encoder.append(activation_dict[activation_type]())
            else:
                self.decoder.append(layer(layer_list[index], layer_list[index+1]))
                self.decoder.append(activation_dict[activation_type]())
            
        self.decoder.append(layer(layer_list[-2], layer_list[-1]))

        #self.decoder.append(nn.Conv1d(1, 1, kernel_size=21, stride=1, padding='same'))
        #self.decoder.append(nn.Conv1d(1, 1, kernel_size=21, stride=1, padding='same'))

    def forward(self, X, encoding=False, decoding=False) -> any:
        """
        # forwarde

        Forward pass of the autoencoder.

        Parameters
        ----------
        - X : torch.tensor
            - Input signal of the autoencoder.
        - encoding : bool
            - If `True` the forward pass will return the encoder output.
        - decoding : bool
            - If `True` the forward pass expects an input X of size equal to 
                the encoder output and returns the encoder output.

        Returns
        -------
        - encoding = `True` (encoder) : torch.tensor
            - Encoder output (size of the middle layer of the autoencoder)
        - decoding = `True` (decoder) : torch.tensor
            - Decoder output (size of the input layer of the autoencoder)
        - encoding = decoding = `False` : torch.tensor
            - Autoencoder output (size of the input layer of the autoencoder)
        """
        if encoding:
            return self.encoder(X)
        elif decoding:
            return self.decoder(X)
        else:
            encode = self.encoder(X)
            decode = self.decoder(encode)
        return decode