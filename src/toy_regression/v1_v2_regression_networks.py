from torch import nn
from torch.nn import functional as F


class MSERegressor(nn.Module):
    def __init__(self, hidden_dim=100):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.network(x)


class GaussianRegressor(nn.Module):
    def __init__(self, hidden_dim=100, variance_floor=1e-6):
        super().__init__()
        self.variance_floor = variance_floor
        self.network = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

    def forward(self, x):
        raw = self.network(x)
        return raw[:, 0:1], F.softplus(raw[:, 1:2]) + self.variance_floor
