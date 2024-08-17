import torch
from torch import nn
import PositionalEncoder as pe

class Encoder(nn.Module):

    def __init__(self, input_size, dim_val, dropout_pos_enc=0.1, max_seq_len=5000, n_heads=8, n_encoder_layers=4):
        super().__init__()
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,  # 1
            out_features=dim_val  # 512
        )
        self.position_encoder = pe.PositionalEncoder(
            d_model = dim_val,
            dropout=dropout_pos_enc,
            max_seq_len=max_seq_len
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(dim_val, n_heads, batch_first=True)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )
