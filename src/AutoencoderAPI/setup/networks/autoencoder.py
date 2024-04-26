from torch import nn


class build_autoencoder(nn.Module):
    """
    Build a Pytorch autoencoder architecture with desired caracteristics. 

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

        # Layer type
        #layer_type_dict = {
        #    "Linear" : nn.Linear
        #}
        #layer = layer_type_dict[config['network']['layer_type']]

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
            "MaxPool2d"  : nn.MaxPool2d
        }

        layer_list = config['network']['layer_list'].copy()
        input_ = config['internal']['size_network']
        layer_list = [input_] + layer_list + [input_]
        activation_list = config['network']['activation_list']
        network_type = config['network']['network_type']
        if network_type == 'CNN':
            channel_list = config['network']['CNN_channels']
        
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        switch = True
        #self.encoder.append(nn.Conv1d(1, 1, groups=1, kernel_size=20, padding='same'))
        for index, activation_type in enumerate(activation_list):

            if layer_list[index] == 1:#2
                switch = False

            if switch:
                #elif network_type == 'dropout':
                #   self.encoder.append(nn.Dropout(0.1))
                self.encoder.append(nn.Linear(layer_list[index], layer_list[index+1]))
                if network_type == 'CNN':
                    self.encoder.append(nn.Conv1d(channel_list[index], channel_list[index+1], groups=1, kernel_size=10, padding='same'))
                self.encoder.append(activation_dict[activation_type]())
            else:
                self.decoder.append(nn.Linear(layer_list[index], layer_list[index+1]))
                if network_type == 'CNN':
                    self.decoder.append(nn.ConvTranspose1d(channel_list[index], channel_list[index+1], groups=1, kernel_size=10, padding=0, output_padding=0))
                #elif network_type == 'dropout':
                #   self.encoder.append(nn.Dropout(0.1))
                self.decoder.append(activation_dict[activation_type]())
        
        if network_type == 'CNN':
            self.decoder.append(nn.ConvTranspose1d(channel_list[-2], channel_list[-1], groups=1, kernel_size=10, padding=0, output_padding=0))
        self.decoder.append(nn.Linear(layer_list[-2], layer_list[-1]))
        #self.decoder.append(nn.Conv1d(16, 1, groups=1, kernel_size=20, padding='same'))
        #self.decoder.append(nn.Dropout(0.2))
        print(self.encoder)
        print(self.decoder)


    def forward(self, X, encoding=False, decoding=False, both=False) -> any:
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
        if both:
            encode = self.encoder(X)
            return encode , self.decoder(encode)
        if encoding:
            return self.encoder(X)
        elif decoding:
            return self.decoder(X)
        else:
            encode = self.encoder(X)
            decode = self.decoder(encode)
        return decode
