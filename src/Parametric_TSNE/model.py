from torch import nn


def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear,)):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0.01)
    elif isinstance(m, (nn.BatchNorm1d,)):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)


class Network(nn.Module):

    def __init__(self, dim_input, dim_emb=1, layer_size = 300, n_layer = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_input, layer_size)
        )
        
        for l in range(n_layer):
            self.encoder.append(nn.BatchNorm1d(layer_size))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Linear(layer_size, layer_size))

        self.encoder.append(nn.BatchNorm1d(layer_size))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(layer_size, dim_emb))

        self.decoder = nn.Sequential(
            nn.Linear(dim_emb, layer_size)
        )

        for l in range(n_layer):
            self.decoder.append(nn.BatchNorm1d(layer_size))
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Linear(layer_size, layer_size))

        self.decoder.append(nn.BatchNorm1d(layer_size))
        self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(layer_size, dim_input))

        self.apply(weights_init)

    def forward(self, x, 
                encode = False, 
                decode = False,
                both = False):
        if encode:
            return self.encoder(x)
        elif both:
            encoder_ = self.encoder(x)
            decoder_ = self.decoder(encoder_)
            return encoder_, decoder_
        elif decode:
            return self.decoder(x)



def size_conv(L_in,
              padding : int = 0,
              dilation : int = 1,
              kernel_size : int = 5,
              stride : int = 2):

    return (((L_in + 2*padding - dilation*(kernel_size - 1)) - 1) / stride) + 1

def size_deconv(L_in,
                padding : int = 0,
                dilation : int = 1,
                kernel_size : int = 5,
                stride : int = 2,
                output_padding : int = 0):
    
    return (L_in - 1)*stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1
        

class Network1(nn.Module):

    def __init__(self, dim_input, dim_emb=2):
        super().__init__()

        size_conv1 = size_conv(dim_input, 
                                padding = 0,
                                kernel_size = 4,
                                stride = 2)
        size_conv2 = size_conv(size_conv1,
                                padding = 0,
                                kernel_size = 4,
                                stride = 2)
        size_conv3 = size_conv(size_conv2,
                                padding = 0,
                                kernel_size = 4,
                                stride = 2)

        size_deconv1 = size_deconv(size_conv3,
                                    padding = 0,
                                    kernel_size = 4,
                                    stride = 2,
                                    output_padding = 0)
        size_deconv2 = size_deconv(size_deconv1,
                                    padding = 0,
                                    kernel_size = 4,
                                    stride = 2,
                                    output_padding = 0)
        size_deconv3 = size_deconv(size_deconv2,
                                    padding = 0,
                                    kernel_size = 4,
                                    stride = 2,
                                    output_padding = 0)

        print(size_conv1, size_conv2, size_conv3)
        print(size_deconv1, size_deconv2, size_deconv3)

        self.dim_input = dim_input
        self.dim_emb = dim_emb
        
        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size = 4, stride = 2, padding = 0),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Conv1d(8, 16, kernel_size = 4, stride = 2, padding = 0),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Conv1d(16, 16, kernel_size = 4, stride = 2, padding = 0),
            nn.ReLU(True),

            nn.Flatten(start_dim=1),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(int(16*size_conv3), 128),
            nn.ReLU(True),
            nn.Linear(128, dim_emb)
        )


        self.decoder1 = nn.Sequential(
            nn.Linear(dim_emb, 128),
            nn.ReLU(True),
            nn.Linear(128, int(16*size_conv3)),
            nn.ReLU(True)
        )

        self.decoder2 = nn.Sequential(

            nn.Unflatten(dim=1, unflattened_size=(16, int(size_conv3))),

            nn.ConvTranspose1d(16, 16, kernel_size = 4, stride = 2, padding = 0, output_padding = 0),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, kernel_size = 4, stride = 2, padding = 0, output_padding = 0),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, kernel_size = 4, stride = 2, padding = 0, output_padding = 0)
        )

        self.apply(weights_init)

    def forward(self, x, 
                encode = False, 
                decode = False,
                both = False):
        x = x.view(-1,1,self.dim_input)
        if encode:
            x = self.encoder1(x)
            return self.encoder2(x).view(-1,self.dim_emb)
        elif both:
            x = self.encoder1(x)
            encoder_ = self.encoder2(x)
            decoder_ = self.decoder1(encoder_)
            decoder_ = self.decoder2(decoder_)
            return encoder_.view(-1,self.dim_emb), decoder_.view(-1,self.dim_input)
        elif decode:
            x = self.decoder1(x)
            return self.decoder2(x).view(-1,self.dim_input)


