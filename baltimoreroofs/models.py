import numpy as np
import torch
from torchvision.transforms.functional import rgb_to_grayscale

# Reorder given matplotlib and pytorch have different order of channel, height, width.
# pytorch:    [C, H, W]
# matplotlib: [H, W, C]
TENSOR_TO_NUMPY = [1, 2, 0]
NUMPY_TO_TENSOR = [2, 0, 1]


def tensor_to_numpy(t):
    return t.numpy().transpose(TENSOR_TO_NUMPY)


def numpy_to_tensor(n):
    return torch.from_numpy(n.transpose(NUMPY_TO_TENSOR))


class DarkImageBaseline:
    def __init__(self, dark_threshold):
        self.threshold = dark_threshold

    def fit(self, X, y):
        return self

    def preprocess(self, X):
        numpy_to_tensor(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # How much of each image is darker than the darkness threshold?
        if X.ndim == 0:
            return None
        X = rgb_to_grayscale(numpy_to_tensor(X).unsqueeze(0))
        is_darker = torch.where(X.isnan(), X, X < self.threshold).to(torch.float32)
        return is_darker.nanmean(axis=(2, 3), dtype=torch.float32).squeeze().item()
