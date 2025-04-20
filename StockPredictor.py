import torch
import torch.nn as nn


# 주가 예측을 위한 Transformer 모델 클래스 정의
class StockPredictTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, nhead, output_dim, dropout=0.1):
        super(StockPredictTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, nhead=nhead,
            dropout=dropout
        )
        self.fc_out = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src, tgt):
        src = self.dropout(self.embedding(src))  # 임베딩 후 드롭아웃 적용
        tgt = self.dropout(self.embedding(tgt))
        src = src.permute(1, 0, 2)  # Transformer 입력 차원으로 변경 (seq_len, batch_size, embed_dim)
        tgt = tgt.permute(1, 0, 2)

        transformer_output = self.transformer(src, tgt)
        output = self.fc_out(transformer_output)
        return output.permute(1, 0, 2)  # (batch_size, seq_len, output_dim)로 변환


class StockPredictTransformerNormal(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, nhead, dropout=0.1):
        super().__init__()
        self.embedding = torch.nn.Linear(input_dim, embed_dim)
        self.transformer = torch.nn.Transformer(
            d_model=embed_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, nhead=nhead,
            dropout=dropout
        )
        self.fc_mu = torch.nn.Linear(embed_dim, 1)
        self.fc_sigma = torch.nn.Linear(embed_dim, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, src, tgt):
        src = self.dropout(self.embedding(src))
        tgt = self.dropout(self.embedding(tgt))
        src, tgt = src.permute(1, 0, 2), tgt.permute(1, 0, 2)

        output = self.transformer(src, tgt)
        output = output.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
        last_step = output[:, -1, :]  # (batch, embed_dim)

        mu = self.fc_mu(last_step)  # (batch, 1)
        sigma = torch.exp(self.fc_sigma(last_step))  # Ensure positivity
        return mu.squeeze(-1), sigma.squeeze(-1)