import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self, input_size, dim_val, n_heads, n_decoder_layers, num_predicted_features):
        super().__init__()

        self.decoder_input_layer = nn.Linear(
            in_features=input_size,  # the number of features you want to predict. Usually just 1
            out_features=dim_val
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            batch_first=True
        )

        # Stack the decoder layer n times
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        )

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
        )