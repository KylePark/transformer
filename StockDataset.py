import torch
from torch.utils.data import Dataset, DataLoader
class StockDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = torch.tensor(data.values, dtype=torch.float32) if not isinstance(data, torch.Tensor) else data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        #TODO 예측열이 x에 들어가지 않도록
        x = self.data[idx:idx + self.window_size, :8]
        y = self.data[idx + self.window_size, 8]  # rate 열 예측
        return x, y