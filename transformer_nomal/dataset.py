import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = torch.tensor(data.values, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
        self.window_size = window_size

    def __len__(self):
        return max(0, len(self.data) - self.window_size)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size, :8]  # 예측 대상 rate 제외
        y = self.data[idx + self.window_size, 8]       # rate 열 예측
        return x, y
