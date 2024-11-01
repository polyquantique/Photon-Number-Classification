import torch.nn as nn

def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear,)):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0.01)
    elif isinstance(m, (nn.BatchNorm1d,)):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)
   
class default_encoder(nn.Module):
    def __init__(self, dim_input = 350 , dim_emb = 1, layer_size = 300, n_layer = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_input, layer_size)
        )
        
        for l in range(n_layer):
            self.encoder.append(nn.BatchNorm1d(layer_size))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Linear(layer_size, layer_size))

        self.encoder.append(nn.BatchNorm1d(layer_size))
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(layer_size, dim_emb))

        self.apply(weights_init)

    def forward(self, x):
        
        return self.encoder(x)
    
class default_decoder(nn.Module):
    def __init__(self, dim_input = 350, dim_emb = 1, layer_size = 300, n_layer = 4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Flatten(),
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
    
    def forward(self, x):
        return self.decoder(x)
    

    
class conv_encoder(nn.Module):
    def __init__(self, n_components=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=64, kernel_size=4, stride=1, padding=1,
            ),
            nn.Conv1d(
                in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=1,
            ),
            nn.Flatten(),
            nn.Linear(11136, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_components)
        ).cuda()
    def forward(self, X):
        return self.encoder(X)