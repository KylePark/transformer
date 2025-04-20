import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import StockPredictTransformerNormal
from dataset import StockDataset
import pandas as pd
import os

def nll_loss(y_true, y_pred_mean, y_pred_std):
    # 정규분포 기반의 NLL 계산
    var = y_pred_std**2
    log_variance = torch.log(var)
    loss = 0.5 * (log_variance + ((y_true - y_pred_mean) ** 2) / var)
    return loss.mean()

def train_model(
        train_data,
        val_data,
        window_size=60,
        input_dim=8,
        embed_dim=64,
        num_layers=2,
        nhead=4,
        output_dim=2,
        dropout=0.1,
        batch_size=32,
        num_epochs=100,
        lr=1e-4,
        model_path="model_checkpoint.pth",
        log_path="loss_log.csv"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = StockDataset(train_data, window_size)
    val_dataset = StockDataset(val_data, window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = StockPredictTransformerNormal(input_dim, embed_dim, num_layers, nhead, output_dim, dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    log_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            mean_tgt = x[:, -1:, :]
            output = model(x, mean_tgt)
            mean_pred = output[:, -1, 0]
            std_pred = output[:, -1, 1].exp()

            loss = nll_loss(y, mean_pred, std_pred)  # NLL Loss 사용

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                mean_tgt = x[:, -1:, :]
                output = model(x, mean_tgt)
                mean_pred = output[:, -1, 0]
                std_pred = output[:, -1, 1].exp()

                loss = nll_loss(y, mean_pred, std_pred)  # NLL Loss 사용

                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        log_df.loc[len(log_df)] = [epoch, avg_train_loss, avg_val_loss]
        log_df.to_csv(log_path, index=False)

        torch.save(model.state_dict(), model_path)

    print("Training complete. Model saved to", model_path)
    return model
