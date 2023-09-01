from torch import nn
from .transformer.positionalEncoding import PositionalEncoding
from .transformer.transformerEncoderLayer import transformerEncoderLayer
from .transformer.transformerDecoderLayer import transformerDecoderLayer


class build_autoencoder(nn.Module):
    def __init__(self, config: dict) -> None:
        """
        
        """
        super().__init__()

        self.embed_dim = config['network']["embed_dim"]
        self.sequence = config['network']['sequence_len']

        self.pos_encoder = PositionalEncoding(d_model = self.embed_dim, max_len = 5000)
        self.layer1 = transformerEncoderLayer(self.embed_dim, 50, config)
        self.layer2 = transformerEncoderLayer(50, 1, config)
        self.layer3 = transformerDecoderLayer(1, 50, config)
        self.layer4 = transformerDecoderLayer(50, self.embed_dim, config)
        
 

    def forward(self, X, encoding=False, decoding=False):
        
        
        if encoding:
            X = X.view(1, self.sequence, self.embed_dim)
            pos = self.pos_encoder(X)
            l1 = self.layer1(pos)
            l2 = self.layer2(l1)
            return l2
        elif decoding:
            l3 = self.layer3(X)
            l4 = self.layer4(l3)
            return l4
        else:
            X = X.view(1, self.sequence, self.embed_dim)
            pos = self.pos_encoder(X)
            l1 = self.layer1(pos)
            l2 = self.layer2(l1)
            l3 = self.layer3(l2)
            l4 = self.layer4(l3)
        return l4