from torch.utils.data import DataLoader, Dataset
import numpy as np


class windowDataset(Dataset):
    def __init__(self, y, input_window, output_window, stride=5):
        # 총 데이터의 개수
        L = y.shape[0]
        feature_size = y.shape[1]
        # stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1

        # input과 output
        X = np.zeros([input_window, num_samples, feature_size])
        Y = np.zeros([output_window, num_samples])
        y_output = y["rate"].to_numpy()

        for i in np.arange(num_samples):
            start_x = stride * i
            end_x = start_x + input_window
            X[:, i] = y[start_x:end_x]

            start_y = stride * i + input_window
            end_y = start_y + output_window
            Y[:, i] = y_output[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], feature_size).transpose((1, 0, 2))
        Y = Y.reshape(Y.shape[0], Y.shape[1]).transpose((1, 0))

        self.x = X
        self.y = Y

        self.len = len(X)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.len