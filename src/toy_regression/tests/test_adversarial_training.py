import torch

from src.toy_regression import train_v1_v2_models as train
from src.toy_regression.experiment_config import Config
from src.toy_regression.generate_cubic_data import fgsm_epsilon, generate_train
from src.toy_regression.regression_losses import gaussian_nll
from src.toy_regression.v1_v2_regression_networks import GaussianRegressor


def test_fgsm_matches_dataset_epsilon_gradient_and_targets(monkeypatch):
    data = generate_train(Config(), 42)
    x, y = data.x[:5], data.y_noisy[:5]
    epsilon = fgsm_epsilon(data.x, 0.01)
    assert epsilon == 0.01 * (data.x.max() - data.x.min()).item()
    model = GaussianRegressor()
    source = x.clone().requires_grad_()
    grad, = torch.autograd.grad(gaussian_nll(*model(source), y), source)
    adv = train.fgsm_inputs(model, x, y, epsilon)
    torch.testing.assert_close(adv, x + epsilon * grad.sign())
    assert (adv - x).abs().max() <= epsilon + 1e-6
    assert not adv.requires_grad and all(p.grad is None for p in model.parameters())
    seen = []

    def score(mean, variance, target):
        seen.append(target)
        return gaussian_nll(mean, variance, target)

    monkeypatch.setattr(train, "gaussian_nll", score)
    clean, perturbed = train.training_losses(train.CONDITIONS[2], model, x, y, epsilon)
    assert len(seen) == 3 and all(target is y for target in seen)
    (clean + perturbed).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.equal(x, data.x[:5])


def test_non_at_paths_do_not_call_fgsm(monkeypatch):
    def unexpected(*args):
        raise AssertionError("FGSM must run only in the AT condition")

    monkeypatch.setattr(train, "fgsm_inputs", unexpected)
    for condition in train.CONDITIONS:
        if not condition.adversarial:
            model = train.initialize_member(condition, Config(), 42)
            _, adv = train.training_losses(condition, model, torch.zeros(2, 1), torch.ones(2, 1), 0.08)
            assert adv is None
