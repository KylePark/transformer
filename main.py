
import torch
import matplotlib.pyplot as plt
from pykrx import stock
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from WindowDataset import windowDataset
from TransFormerEX import TFModel, PositionalEncoding
from torch import nn
from tqdm import tqdm
import pandas as pd
import TimeSeriesTransformer


if __name__ == '__main__':
    df = stock.get_index_ohlcv_by_date("20100101", "20211231", "1001")
    plt.figure(figsize=(20, 5))
    plt.plot(range(len(df)), df["종가"])
    min_max_scaler = MinMaxScaler()
    df["종가"] = min_max_scaler.fit_transform(df["종가"].to_numpy().reshape(-1, 1))

    data_train = df["종가"]

    iw = 24 * 14
    ow = 24 * 7

    train_dataset = windowDataset(data_train, input_window=iw, output_window=ow, stride=1)
    train_loader = DataLoader(train_dataset, batch_size=64)

    device = torch.device("cuda")
    lr = 1e-4
    model = TFModel(24 * 7 * 2, 24 * 7, 512, 8, 4, 0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch = 500
    model.train()
    progress = tqdm(range(epoch))
    for i in progress:
        batchloss = 0.0
        for (inputs, outputs) in train_loader:
            optimizer.zero_grad()
            src_mask = model.generate_square_subsequent_mask(inputs.shape[1]).to(device)
            result = model(inputs.float().to(device), src_mask)
            loss = criterion(result, outputs[:, :, 0].float().to(device))
            loss.backward()
            optimizer.step()
            batchloss += loss
        progress.set_description("loss: {:0.6f}".format(batchloss.cpu().item() / len(train_loader)))