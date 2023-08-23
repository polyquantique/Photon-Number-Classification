from torch import nn
from .positionalEncoding import PositionalEncoding

class transformerLayer(nn.Module):
    def __init__(self, input_size, output_size, config):
        """
        
        """
        super().__init__()
        
        skip = config['train']["skip_elements"]
        size = config['files']['input_dimension']
        input_size = int(size / skip)

   
        self.pos_encoder = PositionalEncoding(d_model = config['embed_dim'], max_len = 5000)
        self.attention = nn.MultiheadAttention(embed_dim = config['embed_dim'], num_heads = config['network']['nhead'])
        self.norm = nn.LayerNorm()
        self.feed_forward = nn.GELU()

        
 

    def forward(self, X, encoding=False, decoding=False):
   
        if encoding:
            pos = self.pos_encoder(X)
            return self.transformer(pos)
        elif decoding:
            return self.transformer(X)
        else:
            pos = self.pos_encoder(X)
            encode = self.transformer(pos)
            decode = self.transformer(encode)
        return decode