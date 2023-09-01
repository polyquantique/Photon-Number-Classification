from torch import nn
import torch


class transformerEncoderLayer(nn.Module):
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

        self.feed_forward_input = nn.Linear(in_features = input_size, out_features = input_size)   # Mention of an hidden layer?
        self.feed_forward_activation = nn.GELU()
        self.feed_forward_output = nn.Linear(in_features = input_size, out_features = output_size)

        
    def forward(self, X):

        query = self.query(X)
        key = self.key(X)
        value = self.value(X)
        
        attention = self.attention(query, key, value)
        add = torch.add(X , attention[0])
        norm = self.norm(add)
        
        in_ff = self.feed_forward_input(norm)
        act_ff = self.feed_forward_activation(in_ff)
        out_ff = self.feed_forward_output(act_ff)
        if out_ff.size(2) == 1:
            return torch.mean(out_ff).view(1,1,1)

        return out_ff