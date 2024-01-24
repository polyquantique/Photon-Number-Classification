from torch import nn
import torch
import math


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        # PositionalEncoding

        based on :
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        Parameters
        ----------
        - d_model : int
                - Dimension of the model
        - dropout : float
                - Probability of an element to be zeroed
        - max_len : int
                - 

        Returns
        -------
        - None
        """
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
        # forward

        Positional encoding forward pass.

        Parameters
        ----------
        - x: Tensor
            - shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)