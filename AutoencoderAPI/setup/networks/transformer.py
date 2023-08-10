from .positionalEncoder import PositionalEncoder
import torch.nn as nn 
from torch import nn, Tensor
import torch.nn.functional as F

import torch
import torch.nn as nn 
import math
from torch import nn, Tensor

class TimeSeriesTransformer(nn.Module):

    """
    [1] K. G. A. Ludvigsen, ‘How to make a PyTorch Transformer for time series forecasting’, 
    Medium, Oct. 27, 2022. https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e (accessed Jul. 21, 2023).

    """

    def __init__(self, config): 
        """
        Args:
            input_size: int, number of input variables. 1 if univariate.
            dec_seq_len: int, the length of the input sequence fed to the decoder
            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer 
                                     of the decoder
            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """
        super().__init__() 

        """
        input_size: int,
        dec_seq_len: int,
        batch_first: bool,
        out_seq_len: int=58,
        dim_val: int=512,  
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2, 
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        num_predicted_features: int=1
        """

        self.dec_seq_len = dec_seq_len
        self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.decoder_input_layer = nn.Linear(in_features=num_predicted_features, out_features=dim_val)
        self.linear_mapping = nn.Linear(in_features=dim_val, out_features=num_predicted_features)        
        pe = positionalEncoder(dropout=0.1, 
                                max_seq_len=5000, 
                                d_model=512,
                                batch_first=False)         
        self.positional_encoding_layer = pe.PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val,
                                                    nhead=n_heads,
                                                    dim_feedforward=dim_feedforward_encoder,
                                                    dropout=dropout_encoder,
                                                    batch_first=batch_first
                                                    )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                            num_layers=n_encoder_layers, 
                                            norm=None
                                            )
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val,
                                                    nhead=n_heads,
                                                    dim_feedforward=dim_feedforward_decoder,
                                                    dropout=dropout_decoder,
                                                    batch_first=batch_first
                                                    )
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,
                                            num_layers=n_decoder_layers, 
                                            norm=None
                                            )

    def forward(self, sequence: Tensor, encoding: bool=False):
        """
        Returns a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]
        
        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input, 
                 (S, N, E) if batch_first=False or (N, S, E) if 
                 batch_first=True, where S is the source sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input, 
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if 
                 batch_first=True, where T is the target sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)
            src_mask: the mask for the src sequence to prevent the model from 
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """
        # Input length
        enc_seq_len = 100
        # Output length
        output_sequence_length = 58

        src, trg, trg_y = get_src_trg(sequence, encoder_seq_len, target_seq_len)

      
        tgt_mask = self.generate_square_subsequent_mask(
                                                        dim1=output_sequence_length,
                                                        dim2=output_sequence_length
                                                    )
        src_mask = self.generate_square_subsequent_mask(
                                                        dim1=output_sequence_length,
                                                        dim2=enc_seq_len
                                                        )


        src = self.encoder_input_layer(src) 
        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src)
        decoder_output = self.decoder_input_layer(tgt) 
        decoder_output = self.decoder(tgt=decoder_output,
                                    memory=src,
                                    tgt_mask=tgt_mask,
                                    memory_mask=src_mask
                                    )
        decoder_output = self.linear_mapping(decoder_output)

        if encoding:
            return src
        return decoder_output, trg_y


    def generate_square_subsequent_mask(dim1: int, dim2: int):
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        Source:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        Args:
            dim1: int, for both src and tgt masking, this must be target sequence
                length
            dim2: int, for src masking this must be encoder sequence length (i.e. 
                the length of the input sequence to the model), 
                and for tgt masking, this must be target sequence length 
        Return:
            A Tensor of shape [dim1, dim2]
        """
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


    def get_src_trg(
        self,
        sequence: torch.Tensor, 
        encoder_seq_len: int, 
        target_seq_len: int
        ):

        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence. 
        Args:
            sequence: tensor, a 1D tensor of length n where 
                    n = encoder input length + target sequence length  
            enc_seq_len: int, the desired length of the input to the transformer encoder
            target_seq_len: int, the desired length of the target sequence (the 
                            one against which the model output is compared)
        Return: 
            src: tensor, 1D, used as input to the transformer model
            trg: tensor, 1D, used as input to the transformer model
            trg_y: tensor, 1D, the target sequence against which the model output
                is compared when computing loss. 
        
        """
        src = sequence[:encoder_seq_len] 
        trg = sequence[encoder_seq_len-1:len(sequence)-1]
        trg = trg[:, 0]

        if len(trg.shape) == 1:
            trg = trg.unsqueeze(-1)
        
        trg_y = sequence[-target_seq_len:]
        trg_y = trg_y[:, 0]

        return src, trg, trg_y.squeeze(-1)