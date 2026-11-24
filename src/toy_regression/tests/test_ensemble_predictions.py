import torch

from src.toy_regression.experiment_config import Config
from src.toy_regression.evaluate_predictions import evaluate_noisy, mixture_quantile
from src.toy_regression.regression_losses import gaussian_nll, mixture_nll
from src.toy_regression.ensemble_predictions import aggregate_predictions
from src.toy_regression.train_v1_v2_models import CONDITIONS, initialize_member, member_seeds


def test_identical_gaussians():
    means = torch.tensor([[[1.0]], [[1.0]]])
    variances = torch.full_like(means, 3)
    target = torch.tensor([[2.0]])
    pred = aggregate_predictions(means, variances)
    assert pred.epistemic.item() == 0
    assert pred.total_variance.item() == 3
    torch.testing.assert_close(mixture_nll(means, variances, target),
                               gaussian_nll(means[0], variances[0], target))


def test_population_variance_and_mixture_density():
    means = torch.tensor([[[0.0]], [[2.0]]])
    variances = torch.full_like(means, 3)
    pred = aggregate_predictions(means, variances)
    assert (pred.mean.item(), pred.aleatoric.item(), pred.epistemic.item(),
            pred.total_variance.item(), pred.std.item()) == (1, 3, 1, 4, 2)
    target = torch.tensor([[6.0]])
    members = -torch.distributions.Normal(means, variances.sqrt()).log_prob(target)
    torch.testing.assert_close(mixture_nll(means, variances, target),
                               -(members.neg().exp().mean(dim=0).log()).mean())
    assert not torch.isclose(mixture_nll(means, variances, target), members.mean())
    quantile = mixture_quantile(means, variances, 0.975)
    cdf = torch.special.ndtr((quantile - means) / variances.double().sqrt()).mean()
    torch.testing.assert_close(cdf, torch.tensor(0.975, dtype=torch.double), atol=1e-10, rtol=0)


def test_mse_proxy_floor_preserves_raw_zero():
    prediction = aggregate_predictions(torch.zeros(5, 1, 1))
    kind, result = evaluate_noisy(prediction, torch.zeros(1, 1), CONDITIONS[0], 1e-6)
    assert kind == "proxy_gaussian"
    assert result["near_zero_variance_count"] == 1
    assert result["proxy_gaussian_nll"] < 0
    assert result["mean_aleatoric"] is None
    assert prediction.total_variance.item() == 0


def test_independent_member_parameters():
    config = Config()
    models = [initialize_member(CONDITIONS[3], config, member_seeds(42, index)[0])
              for index in range(config.ensemble_size)]
    weights = [next(model.parameters()) for model in models]
    assert len({weight.data_ptr() for weight in weights}) == 5
    assert all(not torch.equal(weight, other) for i, weight in enumerate(weights) for other in weights[:i])
