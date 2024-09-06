from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
import math
import torch

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Network(nn.Module):

    def __init__(self, dim_input, dim_emb=2, layer_size = 300, n_layer = 2):
        super().__init__()

        nhead = 50
        d_hid = dim_input
        d_model = dim_input
        dropout = 0.1
        
        #self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layer)

        #decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_decoder = TransformerEncoder(encoder_layers, n_layer)

        self.d_model = d_model
        self.linear1 = nn.Linear(dim_input, dim_input)
        self.linear2 = nn.Linear(dim_input, dim_emb)
        self.linear3 = nn.Linear(dim_emb, dim_input)
        self.linear4 = nn.Linear(dim_input, dim_input)

    def forward(self, x,
                encode = False, 
                decode = False,
                both = False):
        
        if encode:
            lin = self.linear1(x)
            #pos =  self.pos_encoder(lin)
            enc = self.transformer_encoder(lin)
            return self.linear2(enc)
        elif both:
            lin = self.linear1(x)
            #pos =  self.pos_encoder(lin)
            enc = self.transformer_encoder(lin)
            enc = self.linear2(enc)
            dec = self.linear3(enc)
            dec = self.transformer_encoder(lin)
            dec = self.linear4(dec)
            return enc, dec


        




