"""Figure 2(a) classifier from section 3.4 of Lakshminarayanan et al."""

from torch import Tensor, nn


class MNISTMLP(nn.Module):
    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        layers = [nn.Flatten()]
        for input_width in (784, 200, 200):
            layers.extend([nn.Linear(input_width, 200), nn.BatchNorm1d(200), nn.ReLU()])
            if dropout_p != 0.0:
                layers.append(nn.Dropout(dropout_p))
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(200, 10)

    def forward(self, images: Tensor) -> Tensor:
        # CrossEntropyLoss takes raw logits and applies log-softmax internally.
        return self.output(self.hidden(images))
