import torch

from src.toy_regression.regression_losses import gaussian_nll, mse_loss
from src.toy_regression.v1_v2_regression_networks import GaussianRegressor, MSERegressor


def test_models_shapes_and_parameters():
    x = torch.randn(7, 1)
    mse, gaussian = MSERegressor(), GaussianRegressor()
    mean, variance = gaussian(x)
    assert mse(x).shape == mean.shape == variance.shape == (7, 1)
    assert torch.isfinite(variance).all() and (variance > 0).all()
    assert sum(p.numel() for p in mse.parameters()) == 301
    assert sum(p.numel() for p in gaussian.parameters()) == 402


def test_gaussian_nll_includes_constant_and_uses_variance():
    mean = torch.tensor([[0.0], [2.0]])
    variance = torch.tensor([[0.25], [3.0]])
    y = torch.tensor([[0.0], [1.0]])
    expected = -torch.distributions.Normal(mean, variance.sqrt()).log_prob(y).mean()
    torch.testing.assert_close(gaussian_nll(mean, variance, y), expected)
    torch.testing.assert_close(mse_loss(mean, y), torch.tensor(0.5))
