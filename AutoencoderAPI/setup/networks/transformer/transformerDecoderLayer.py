from torch import nn
import torch
from .positionalEncoding import PositionalEncoding

class transformerDecoderLayer(nn.Module):
    def __init__(self, input_size, output_size, config):
        """
        
        """
        super().__init__()

        sequence = config['network']['sequence_len']

        self.query = nn.Linear(in_features = input_size, out_features = input_size)
        self.key = nn.Linear(in_features = input_size, out_features = input_size)
        self.value = nn.Linear(in_features = input_size, out_features = input_size)
   
        self.attention = nn.MultiheadAttention(embed_dim = input_size, num_heads = input_size)

        self.norm = nn.LayerNorm(normalized_shape=[sequence, input_size])

        self.feed_forward_input2 = nn.Linear(in_features = input_size, out_features = output_size)   # Mention of an hidden layer?
        self.feed_forward_activation2 = nn.GELU()
        self.feed_forward_output2 = nn.Linear(in_features = output_size, out_features = output_size)
        
 
    def forward(self, X):

        #ones = torch.ones((1, self.sequence, self.embed_dim)) * X
   
        query = self.query(X)
        key = self.key(X)
        value = self.value(X)
        
        attention = self.attention(query, key, value)
        add = torch.add(X , attention[0])
        norm = self.norm(add)
        
        in_ff2 = self.feed_forward_input2(norm)
        act_ff2 = self.feed_forward_activation2(in_ff2)
        out_ff2 = self.feed_forward_output2(act_ff2)

        return out_ff2