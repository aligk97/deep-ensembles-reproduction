"""784 -> 200 -> 200 -> 200 -> 10, with optional MC dropout."""
from torch import nn


class MLP(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        layers = []
        for din, dout in ((784, 200), (200, 200), (200, 200)):
            layers.extend([nn.Linear(din, dout), nn.BatchNorm1d(dout), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(200, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.flatten(1))


def evaluation_mode(model, mc_dropout=False):
    model.eval()
    if mc_dropout:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    return model
