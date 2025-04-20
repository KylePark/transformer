import pandas as pd
from train import train_model
from predict import predict
import torch
from model import StockPredictTransformerNormal

def load_model(model_path, input_dim=8, embed_dim=64, num_layers=2, nhead=4, output_dim=2, dropout=0.1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StockPredictTransformerNormal(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        nhead=nhead,
        output_dim=output_dim,
        dropout=dropout
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def main():
    # 데이터 불러오기 (예: CSV)
    df = pd.read_csv('../XBTUSD_1m_rsi_real.csv')
    df = df.drop(columns=['symbol']).copy()
    df = df.drop(columns=['vwap']).copy()
    df = df.drop(columns=['home_notional']).copy()
    df = df.drop(columns=['foreign_notional']).copy()
    df = df.drop(columns=['bin_size']).copy()
    df = df.drop(columns=['volume']).copy()
    df = df.drop(columns=['change_30min']).copy()

    df = df.loc[df['timestamp'] >= '2017-06-01 00:00:00+00:00']
    df = df.drop(columns=['timestamp']).copy()

    # 데이터 전처리 (정규화 등)
    feature_cols = df.columns[:8]  # 예측열 제외
    df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

    # 학습/검증 분리
    split_idx = int(len(df) * 0.8)
    train_data = df[:split_idx].reset_index(drop=True)
    val_data = df[split_idx:].reset_index(drop=True)

    model = load_model("./model_checkpoint.pth")

    # 학습
    # model = train_model(
    #     train_data=train_data,
    #     val_data=val_data,
    #     window_size=60,
    #     input_dim=8,
    #     embed_dim=64,
    #     num_layers=2,
    #     nhead=4,
    #     output_dim=2,  # 평균, 로그표준편차
    #     dropout=0.1,
    #     batch_size=32,
    #     num_epochs=100,
    #     lr=1e-4,
    #     model_path="model_checkpoint.pth",
    #     log_path="loss_log.csv"
    # )

    # 예측 대상 시퀀스 (마지막 구간)
    test_sequence = torch.tensor(
        df.iloc[-120:, :8].values, dtype=torch.float32
    ).unsqueeze(0)  # (1, 60, 8)
    # 예측 수행
    predict(model, test_sequence)


if __name__ == "__main__":
    main()