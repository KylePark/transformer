import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from model import StockPredictTransformerNormal
from dataset import StockDataset
from torch.utils.data import DataLoader


def load_model(model_path, input_dim=8, embed_dim=64, num_layers=2, nhead=4, output_dim=2, dropout=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockPredictTransformerNormal(input_dim, embed_dim, num_layers, nhead, output_dim, dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(model, data, window_size=60, batch_size=1):
    device = next(model.parameters()).device
    dataset = StockDataset(data, window_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds, actuals, stds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            tgt = x[:, -1:, :]
            output = model(x, tgt)
            mean_pred = output[:, -1, 0].cpu().numpy()
            std_pred = output[:, -1, 1].exp().cpu().numpy()

            preds.extend(mean_pred)
            stds.extend(std_pred)
            actuals.extend(y.cpu().numpy())

    return np.array(preds), np.array(stds), np.array(actuals)


def visualize_predictions(preds, stds, actuals, save_path="prediction_plot.png"):
    x = np.arange(len(preds))
    plt.figure(figsize=(12, 6))
    plt.plot(x, actuals, label="Actual")
    plt.plot(x, preds, label="Predicted")
    plt.fill_between(x, preds - 1.96 * stds, preds + 1.96 * stds, color='gray', alpha=0.3, label="95% CI")
    plt.title("Prediction with Uncertainty")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved prediction plot to {save_path}")


def estimate_probability_of_increase(preds, stds):
    from scipy.stats import norm
    prob = norm.sf(0, loc=preds, scale=stds)  # P(X > 0)
    return prob


if __name__ == "__main__":
    model_path = "model_checkpoint.pth"
    df = pd.read_csv('../XBTUSD_1m_rsi_real.csv')
    df = df.drop(columns=['symbol']).copy()
    df = df.drop(columns=['vwap']).copy()
    df = df.drop(columns=['home_notional']).copy()
    df = df.drop(columns=['foreign_notional']).copy()
    df = df.drop(columns=['bin_size']).copy()
    df = df.drop(columns=['volume']).copy()
    df = df.drop(columns=['change_30min']).copy()

    df = df.loc[df['timestamp'] >= '2022-03-23 02:35:00+00:00']
    df = df.drop(columns=['timestamp']).copy()

    model = load_model(model_path)
    preds, stds, actuals = predict(model, df)

    visualize_predictions(preds, stds, actuals)
    prob_up = estimate_probability_of_increase(preds, stds)
    print("Probabilities of increase (first 10):", prob_up[:10])