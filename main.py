
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import numpy as np
import torch.optim as optim
from StockPredictor import StockPredictTransformer
from StockDataset import StockDataset
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from CustomLoss import CustomLoss
from CustomLoss2 import CustomLoss2

def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def stockPredictTransformer(input_data, test_data):
    # CUDA 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    data = input_data.fillna(0)
    test_data = test_data.fillna(0)
    # 데이터 정규화: 평균 0, 표준편차 1로 표준화
    data = (data - data.mean()) / data.std()
    test_data = (test_data - test_data.mean()) / test_data.std()

    train_data = data

    print(train_data.columns)
    print(test_data.columns)

    # 하이퍼파라미터 설정
    input_dim = data.shape[1]  # 입력 데이터의 차원
    embed_dim = 32   # 임베딩 차원
    num_layers = 2  # Transformer 레이어 수
    nhead = 2  # 멀티 헤드 수
    dropout = 0.2  # 드롭아웃 비율
    output_dim = 1  # 출력 차원
    batch_size = 16  # 배치 크기

    model = StockPredictTransformer(input_dim=input_dim, embed_dim=embed_dim, nhead=nhead, num_layers=num_layers,
                                    output_dim=output_dim, dropout=dropout).to(device)

    # 슬라이딩 윈도우 데이터셋
    window_size = 72
    dataset = StockDataset(train_data, window_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_data_set = StockDataset(test_data, window_size)
    val_loader = DataLoader(test_data_set, batch_size=1, shuffle=False)

    running(model, nn.MSELoss(), train_loader, val_loader, 100, "MSELoss")
    # running(model, CustomLoss(5.0), train_loader, val_loader, 200, "Custom 1")
    # running(model, CustomLoss2(0.5), train_loader, val_loader, 200, "Custom 2")

def running(model, criterion, train_loader, val_loader, num_epochs, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # 하이퍼파라미터 설정
    learning_rate = 0.00001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    patience = 100  # Number of epochs to wait
    min_delta = 0.0000001  # Minimum improvement in validation loss
    best_val_loss = float('inf')  # Track the best validation loss
    early_stop_counter = 0  # Track the number of non-improving epochs

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x, x[:, -1, :].unsqueeze(1))
            # 출력 크기 조정
            y_pred = y_pred.squeeze(-1).squeeze(-1)  # (1, 1, 1) -> (1,)

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x, x[:, -1, :].unsqueeze(1))
                # 출력 크기 조정
                y_pred = y_pred.squeeze(-1).squeeze(-1)  # (1, 1, 1) -> (1,)

                loss = criterion(y_pred, y)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        time = datetime.now().strftime('%Y.%m.%d - %H:%M:%S')
        print(f"{time} Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")

        # Check for Early Stopping
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            early_stop_counter = 0  # Reset counter if validation loss improves
        else:
            early_stop_counter += 1  # Increment counter if no improvement
            time = datetime.now().strftime('%Y.%m.%d - %H:%M:%S')
            print(f"{time} Validation loss did not improve. Counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f"{time} Early stopping triggered.")
                # break

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss: {name}")
    plt.legend()
    plt.grid()
    plt.show()

    predictions, actuals = predict(model, val_loader)

    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Stock Price Prediction using Transformer")
    plt.legend()
    plt.show()

def predict(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for x, y in loader:
            # x, y를 GPU로 이동
            x, y = x.to(device), y.to(device)

            # 예측 수행
            prediction = model(x, x[:, -1, :].unsqueeze(1))

            # 예측 및 실제 값 저장
            predictions.append(prediction.squeeze().cpu().item())
            actuals.append(y.squeeze().cpu().item())

    return predictions, actuals


def calculate_rsi(df, window=14):
    delta = df['closing'].diff()  # 종가의 변화량
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # 상승 평균
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # 하락 평균

    # RSI 계산
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_obv(df):
    obv = np.where(df['closing'].diff() > 0, df['volume'], -df['volume'])  # OBV 계산
    obv = np.cumsum(obv)  # 누적 합산
    return obv


def resample_to_5min(df):
    # 5분봉 데이터프레임 생성
    # df_5min = df.resample('5T').agg({
    #     'opening': 'first',
    #     'high': 'max',
    #     'low': 'min',
    #     'closing': 'last',
    #     'volume': 'sum'
    # })
    # df.dropna(inplace=True)

    # RSI와 OBV 계산하여 컬럼 추가
    df['rsi'] = calculate_rsi(df)
    df['obv'] = calculate_obv(df)

    return df

if __name__ == '__main__':
    df_init = pd.read_csv('./XBTUSD_1m_rsi_real.csv')

    df_train = df_init.loc[df_init['timestamp'] >= '2017-06-01 00:00:00+00:00']
    df_train = df_train.loc[df_train['timestamp'] < '2022-03-22 21:35:00+00:00']
    #
    df_test = df_init.loc[df_init['timestamp'] >= '2022-03-22 21:35:00+00:00']

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    min_max_scaler = MinMaxScaler()

    df_train = df_train.drop(columns=['timestamp']).copy()
    df_train = df_train.drop(columns=['change_30min']).copy()
    #
    df_test = df_test.drop(columns=['timestamp']).copy()
    df_test = df_test.drop(columns=['change_30min']).copy()

    df_train = df_train.drop(columns=['volume']).copy()
    df_test = df_test.drop(columns=['volume']).copy()

    print(df_test.columns)
    print(df_train.columns)
    stockPredictTransformer(df_train, df_test)

    torch.cuda.empty_cache()