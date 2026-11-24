from dataclasses import replace

import pytest
import torch

from src.toy_regression.experiment_config import Config
from src.toy_regression.generate_cubic_data import generate_data, generate_train


def test_training_seed_shape_support_and_noise():
    config = Config()
    train = generate_train(config, 42)
    again = generate_train(config, 42)
    assert train.x.shape == train.y_noisy.shape == (20, 1)
    for name, array in train.arrays().items():
        assert array.tobytes() == again.arrays()[name].tobytes()
    assert train.x.dtype == torch.float32
    assert torch.isfinite(train.y_noisy).all()
    assert not torch.equal(train.x, generate_train(config, 123).x)
    assert ((train.x >= -4) & (train.x <= 4)).all()
    rng = torch.Generator().manual_seed(42)
    torch.rand((20, 1), generator=rng)
    assert torch.equal(train.noise, 3 * torch.randn((20, 1), generator=rng))
    assert torch.equal(train.y_true, train.x.pow(3))
    assert torch.equal(train.y_noisy, train.y_true + train.noise)


def test_eval_split_and_grid():
    data = generate_data(replace(Config(), n_test_in=32, n_test_ood=40), 42)
    assert len(data["test_in"].x) == 32
    assert not torch.equal(data["train"].x, data["test_in"].x[:20])
    assert ((data["test_ood"].x[:20] >= -6) & (data["test_ood"].x[:20] < -4)).all()
    assert ((data["test_ood"].x[20:] > 4) & (data["test_ood"].x[20:] <= 6)).all()
    assert list(data["grid"].x[[0, -1], 0]) == [-6, 6]
    assert data["grid"].x.shape == (601, 1)
    assert data["grid"].y_noisy is None


@pytest.mark.parametrize("key,value", [
    ("batch_size", 0), ("n_test_ood", 3), ("outer_seeds", (42, 42)),
    ("outer_seeds", ()), ("noise_std", 9.0), ("learning_rate", float("nan")),
    ("device", "invalid"), ("gradient_clip_norm", True),
])
def test_invalid_config(key, value):
    with pytest.raises(ValueError, match=key):
        Config(**{key: value})
