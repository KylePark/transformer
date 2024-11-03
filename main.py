
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
def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def stockPredictTransformer(input_data):
    # CUDA 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    data = input_data.fillna(0)

    # 데이터 정규화: 평균 0, 표준편차 1로 표준화
    data = (data - data.mean()) / data.std()

    # 학습 데이터와 테스트 데이터 분리 (80% 학습, 20% 테스트)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]


    # 하이퍼파라미터 설정
    input_dim = data.shape[1]  # 입력 데이터의 차원
    d_model = 64  # 임베딩 차원
    num_layers = 4  # Transformer 레이어 수
    nhead = 4  # 멀티 헤드 수
    output_dim = 1  # 출력 차원
    batch_size = 16  # 배치 크기

    model = StockPredictTransformer(input_dim=input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                    output_dim=output_dim).to(device)

    # 하이퍼파라미터 설정
    learning_rate = 0.0001
    epochs = 3

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 슬라이딩 윈도우 데이터셋
    window_size = 30
    dataset = StockDataset(train_data, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 학습 루프
    for epoch in range(epochs):
        model.train()
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # NaN 체크
            if torch.isnan(loss):
                print("Encountered NaN loss. Skipping this batch.")
                continue

            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    test_data_set = StockDataset(test_data, window_size)
    predictions, actuals = predict(model, test_data_set, window_size)

    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label="Actual Price")
    plt.plot(predictions, label="Predicted Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Stock Price Prediction using Transformer")
    plt.legend()
    plt.show()


def predict(model, dataset, window_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for i in range(len(dataset) - window_size):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)  # 배치 차원 추가 및 GPU로 이동

            # 모델 예측
            pred = model(x).cpu().item()  # GPU에서 CPU로 이동 후 스칼라 값 추출
            predictions.append(pred)
            actuals.append(y.item())  # 실제값 추가

    return predictions, actuals
# def beforeTransformer():
#     for i in [2]:
#         ow_size = 8  # 2시간
#         train = df[:-ow_size]
#         train = train.drop(columns = ['datetime']).copy()
#
#         test = df[-ow_size:]
#         test = test.drop(columns = ['datetime']).copy()
#
#         # data_train = train["rate"].to_numpy()
#         # data_test = test["rate"].to_numpy()
#
#         data_train = train
#         data_test = test
#
#         iw = ow_size * 2
#         ow = ow_size
#
#         train_dataset = windowDataset(data_train, input_window=iw, output_window=ow, stride=1)
#         train_loader = DataLoader(train_dataset, ow)
#
#         device = torch.device("cuda")
#         lr = 1e-4
#         model = TFModel(iw, ow_size, 512, 8, 4, data_train.shape[1], 0.1).to(device)
#         #d_model, nhead ?
#         criterion = nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#         epoch = 1
#         model.train()
#         progress = tqdm(range(epoch))
#         for i in progress:
#             batchloss = 0.0
#             for (inputs, outputs) in train_loader:
#                 optimizer.zero_grad()
#                 src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
#                 result = model(inputs.float().to(device), src_mask)
#                 # print(f'result: {result.shape}')
#                 # print(f'output: {outputs.shape}')
#                 loss = criterion(result, outputs.float().to(device))
#                 loss.backward()
#                 optimizer.step()
#                 batchloss += loss
#             progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(train_loader)))
#
#         result = evaluate(iw)
#         print(result)
#         result = min_max_scaler.inverse_transform(result)[0]
#         real = df["rate"].to_numpy()
#         real = min_max_scaler.inverse_transform(real.reshape(-1, 1))[:, 0]
#
#         plt.figure(figsize=(20, 5))
#         plt.plot(range(df_len - (ow_size*2), df_len), real[df_len - (ow_size*2):], label="real")
#         plt.plot(range(df_len - ow_size, df_len), result, label=f'predict {ow}')
#         plt.legend()
#         plt.show()
#         print(f' RESULT : {MAPEval(result, real[-ow_size:])}')
#         torch.cuda.empty_cache()

if __name__ == '__main__':
    df_init = pd.read_csv('./BTCUSD_5m.csv')
    df = df_init.loc[df_init['datetime'] >= '2022-04-30 00:00:00']
    df.reset_index(drop=True, inplace=True)
    df_len = len(df)
    print(len(df))
    for i in range(len(df)):
        df.loc[i, 'rate'] = (df.loc[i, "closing"] - df.loc[i, "opening"])/df.loc[i, "opening"] * 100

    plt.figure(figsize=(20, 5))
    plt.plot(range(len(df)), df['rate'])
    # plt.show()


    min_max_scaler = MinMaxScaler()
    df["rate"] = min_max_scaler.fit_transform(df["rate"].to_numpy().reshape(-1, 1))

    df_train = df.loc[df['datetime'] < '2024-02-01 00:00:00']

    df_train = df_train.drop(columns=['datetime']).copy()
    stockPredictTransformer(df_train)

    torch.cuda.empty_cache()