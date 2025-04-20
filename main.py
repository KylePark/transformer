
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
from torch.distributions import Normal
from torch.optim import Adam
import os
import math

def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def safe_standardize(df):
    mean = df.mean()
    std = df.std()
    return (df - mean) / std if std != 0 else df - mean

def stockPredictTransformer(input_data, test_data1, test_data2):
    # CUDA 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    data = input_data.fillna(0)
    test_data1 = test_data1.fillna(0)
    test_data2 = test_data2.fillna(0)
    # 데이터 정규화: 평균 0, 표준편차 1로 표준화
    data = (data - data.mean()) / data.std()
    test_data1 = (test_data1 - test_data1.mean()) / test_data1.std()
    test_data2 = (test_data2 - test_data2.mean()) / test_data2.std()
    train_data = data

    print(train_data.columns)
    print(test_data1.columns)
    print(test_data2.columns)

    # 하이퍼파라미터 설정
    input_dim = data.shape[1] - 1  # 입력 데이터의 차원
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

    test_data_set1 = StockDataset(test_data1, window_size)
    val_loader1 = DataLoader(test_data_set1, batch_size=1, shuffle=False)

    test_data_set2 = StockDataset(test_data2, window_size)
    val_loader2 = DataLoader(test_data_set2, batch_size=1, shuffle=False)

    running(model, nn.MSELoss(), train_loader, val_loader1, val_loader2, 500, "MSELoss")
    # running(model, CustomLoss(5.0), train_loader, val_loader1, val_loader2, 200, "Custom 1")
    # running(model, CustomLoss2(0.5), train_loader, val_loader1, val_loader2, 100, "Custom 2")


def running(model, criterion, train_loader, val_loader1, val_loader2, num_epochs, name):

    if (torch.cuda.is_available()):
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    # 하이퍼파라미터 설정
    learning_rate = 0.00001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses1 = []
    val_losses2 = []

    patience = 100  # Number of epochs to wait
    min_delta = 0.0000001  # Minimum improvement in validation loss
    best_val_loss = float('inf')  # Track the best validation loss
    early_stop_counter = 0  # Track the number of non-improving epochs

    #기 저장된 데이터 로드
    df_train = pd.read_csv("./checkpoint_train_loss_1min_259_MSELoss")
    train_losses = df_train["train_loss"].tolist()
    df_train = pd.read_csv("./checkpoint_val1_loss_1min_259_MSELoss")
    val_losses1 = df_train["val_loss1"].tolist()
    df_train = pd.read_csv("./checkpoint_val2_loss_1min_259_MSELoss")
    val_losses2 = df_train["val_loss2"].tolist()
    checkpoint = torch.load("./checkpoint_epoch_1min_259_MSELoss.pth")  # 예제: 50번째 epoch 모델 로드
    model.load_state_dict(checkpoint)

    # for epoch in range(num_epochs):
    #     # Training phase
    #
    #     model.train()
    #     running_loss = 0.0
    #     for x, y in train_loader:
    #         x, y = x.to(device), y.to(device)
    #         optimizer.zero_grad()
    #         y_pred = model(x, x[:, -1, :].unsqueeze(1))
    #         # 출력 크기 조정
    #         y_pred = y_pred.squeeze(-1).squeeze(-1)  # (1, 1, 1) -> (1,)
    #
    #         loss = criterion(y_pred, y)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #     train_losses.append(running_loss / len(train_loader))
    #
    #
    #     # Validation phase
    #     model.eval()
    #     val_loss1 = 0.0
    #     print(len(val_loader1))
    #     print(len(val_loader2))
    #
    #     with torch.no_grad():
    #         for x, y in val_loader1:
    #             x, y = x.to(device), y.to(device)
    #             y_pred = model(x, x[:, -1, :].unsqueeze(1))
    #             # 출력 크기 조정
    #             y_pred = y_pred.squeeze(-1).squeeze(-1)  # (1, 1, 1) -> (1,)
    #
    #             loss = criterion(y_pred, y)
    #             val_loss1 += loss.item()
    #     val_loss1 = val_loss1 / len(val_loader1)
    #     val_losses1.append(val_loss1)
    #     time = datetime.now().strftime('%Y.%m.%d - %H:%M:%S')
    #     print(f"{time} Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss1:.4f}")
    #
    #     # Check for Early Stopping
    #     if best_val_loss - val_loss1 > min_delta:
    #         best_val_loss = val_loss1
    #         early_stop_counter = 0  # Reset counter if validation loss improves
    #     else:
    #         early_stop_counter += 1  # Increment counter if no improvement
    #         time = datetime.now().strftime('%Y.%m.%d - %H:%M:%S')
    #         print(f"{time} Validation loss did not improve. Counter: {early_stop_counter}/{patience}")
    #         if early_stop_counter >= patience:
    #             print(f"{time} Early stopping triggered.")
    #             # break
    #
    #     best_val_loss = float('inf')  # Track the best validation loss
    #     val_loss2 = 0.0
    #     with torch.no_grad():
    #         for x, y in val_loader1:
    #             x, y = x.to(device), y.to(device)
    #             y_pred = model(x, x[:, -1, :].unsqueeze(1))
    #             # 출력 크기 조정
    #             y_pred = y_pred.squeeze(-1).squeeze(-1)  # (1, 1, 1) -> (1,)
    #
    #             loss = criterion(y_pred, y)
    #             val_loss2 += loss.item()
    #     val_loss2 = val_loss2 / len(val_loader1)
    #     val_losses2.append(val_loss2)
    #     time = datetime.now().strftime('%Y.%m.%d - %H:%M:%S')
    #     print(f"{time} Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss2:.4f}")
    #
    #     # Check for Early Stopping
    #     if best_val_loss - val_loss2 > min_delta:
    #         best_val_loss = val_loss2
    #         early_stop_counter = 0  # Reset counter if validation loss improves
    #     else:
    #         early_stop_counter += 1  # Increment counter if no improvement
    #         time = datetime.now().strftime('%Y.%m.%d - %H:%M:%S')
    #         print(f"{time} Validation loss did not improve. Counter: {early_stop_counter}/{patience}")
    #         if early_stop_counter >= patience:
    #             print(f"{time} Early stopping triggered.")
    #             # break
    #
    #     if (epoch+1) % 10 == 0:
    #         save_model_loss(model, 270, epoch, name, train_losses, val_losses1, val_losses2)

    draw_loss(train_losses, val_losses1, val_losses2, name)

    predictions, actuals = predict(model, val_loader1)
    draw_predict(predictions, actuals)

    predictions, actuals = predict(model, val_loader2)
    draw_predict(predictions, actuals)


def save_model_loss(model, num, epoch, name, train_losses, val_losses1, val_losses2):
    torch.save(model.state_dict(), f"checkpoint_epoch_1min_{epoch + num}_{name}.pth")
    print(f"Epoch {epoch + num}: 모델 저장 완료")
    loss_df = pd.DataFrame({"train_loss": train_losses})
    loss_df.to_csv(f"checkpoint_train_loss_1min_{epoch + num}_{name}", index=False)
    loss_df = pd.DataFrame({"val_loss1": val_losses1})
    loss_df.to_csv(f"checkpoint_val1_loss_1min_{epoch + num}_{name}", index=False)
    loss_df = pd.DataFrame({"val_loss2": val_losses2})
    loss_df.to_csv(f"checkpoint_val2_loss_1min_{epoch + num}_{name}", index=False)
def draw_loss(train_losses, val_losses1, val_losses2, name):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, len(val_losses1) + 1), val_losses1, label="Validation Loss1", color="orange")
    plt.plot(range(1, len(val_losses2) + 1), val_losses2, label="Validation Loss2", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss: {name}")
    plt.legend()
    plt.grid()
    plt.show()

def predict(model, loader):
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon (GPU)
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    else:
        device = torch.device("cpu")  # CPU fallback

    model.to(device)
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for x, y in loader:
            # x, y를 GPU로 이동
            x, y = x.to(device), y.to(device)
            print(f"x[:, -1, :]: {x[:, -1, :]}, y: {y}")
            # 예측 수행
            prediction = model(x, x[:, -1, :].unsqueeze(1))

            # 예측 및 실제 값 저장
            predictions.append(prediction.squeeze().cpu().item())
            actuals.append(y.squeeze().cpu().item())

    return predictions, actuals

def draw_predict(predictions, actuals):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Stock Price Prediction using Transformer")
    plt.legend()
    plt.show()

def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, path='checkpoint.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, path)

    # 손실들을 CSV로 저장
    loss_df = pd.DataFrame({'epoch': range(len(train_losses)), 'train_loss': train_losses, 'val_loss': val_losses})
    loss_df.to_csv('losses.csv', index=False)


def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    if not os.path.exists(path):
        return model, optimizer, 0, [], []

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    return model, optimizer, epoch, train_losses, val_losses

def gaussian_nll_loss(y_pred, y_true):
    mu = y_pred[:, 0]
    log_sigma = y_pred[:, 1]
    sigma = torch.exp(log_sigma)

    loss = 0.5 * torch.log(2 * math.pi * sigma ** 2) + ((y_true - mu) ** 2) / (2 * sigma ** 2)
    return loss.mean()

def train_model(model, optimizer, train_loader, val_loader, num_epochs, device, start_epoch=0, train_losses=[], val_losses=[]):
    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            tgt = x[:, -1:, :]
            output = model(x, tgt).squeeze(1)  # (batch, 2)

            loss = gaussian_nll_loss(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                tgt = x[:, -1:, :]
                output = model(x, tgt).squeeze(1)
                val_loss += gaussian_nll_loss(output, y).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # 체크포인트 저장
        save_checkpoint(model, optimizer, epoch+1, train_losses, val_losses)


def predict_with_uncertainty(model, loader, device):
    model.eval()
    model.to(device)

    predictions = []
    uncertainties = []
    actuals = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            tgt = x[:, -1:, :]
            output = model(x, tgt).squeeze(1)  # (batch, 2)

            mu = output[:, 0].cpu()
            sigma = torch.exp(output[:, 1]).cpu()

            predictions.extend(mu.tolist())
            uncertainties.extend(sigma.tolist())
            actuals.extend(y.cpu().tolist())

    return predictions, uncertainties, actuals


def plot_with_uncertainty(preds, sigmas, actuals):
    x = list(range(len(preds)))
    preds = torch.tensor(preds)
    sigmas = torch.tensor(sigmas)

    plt.figure(figsize=(12, 5))
    plt.plot(x, preds, label='Predicted Mean')
    plt.fill_between(x, preds - 1.96 * sigmas, preds + 1.96 * sigmas, color='orange', alpha=0.3,
                     label='95% Confidence Interval')
    plt.plot(x, actuals, label='Actual', color='green', linestyle='--')
    plt.legend()
    plt.title("Prediction with Uncertainty")
    plt.xlabel("Index")
    plt.ylabel("Predicted Value")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    df_init = pd.read_csv('./XBTUSD_1m_rsi_real.csv')
    df_init = df_init.drop(columns=['symbol']).copy()
    df_init = df_init.drop(columns=['vwap']).copy()
    df_init = df_init.drop(columns=['home_notional']).copy()
    df_init = df_init.drop(columns=['foreign_notional']).copy()
    df_init = df_init.drop(columns=['bin_size']).copy()
    df_init = df_init.drop(columns=['volume']).copy()
    df_init = df_init.drop(columns=['change_30min']).copy()


    df_train = df_init.loc[df_init['timestamp'] >= '2017-06-01 00:00:00+00:00']
    df_train = df_train.loc[df_train['timestamp'] < '2022-01-22 21:35:00+00:00']
    #
    df_test = df_init.loc[df_init['timestamp'] >= '2022-03-21 21:35:00+00:00']

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    #
    min_max_scaler = MinMaxScaler()

    df_train = df_train.drop(columns=['timestamp']).copy()
    # df_train = df_train.drop(columns=['change_30min']).copy()
    #
    # df_test = df_test.drop(columns=['change_30min']).copy()

    df_test2 = df_test.loc[df_test['timestamp'] < '2022-03-22 23:35:00+00:00']
    df_test3 = df_test.loc[df_test['timestamp'] >= '2022-03-23 02:35:00+00:00']

    df_test2 = df_test2.drop(columns=['timestamp']).copy()
    df_test3 = df_test3.drop(columns=['timestamp']).copy()

    print(df_test.columns)
    print(df_train.columns)
    # stockPredictTransformer(df_train, df_test2, df_test3)
    torch.cuda.empty_cache()