from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .experiment_config import Config
from .experiment_io import write_json


@dataclass(frozen=True)
class Split:
    x: torch.Tensor
    y_true: torch.Tensor
    y_noisy: torch.Tensor | None = None
    noise: torch.Tensor | None = None

    def arrays(self):
        return {name: value.numpy() for name, value in vars(self).items() if value is not None}


def uniform(count, low, high, rng):
    return low + (high - low) * torch.rand((count, 1), generator=rng, dtype=torch.float32)


def noisy_split(x, rng, noise_std):
    y_true = x.pow(3)
    noise = noise_std * torch.randn(x.shape, generator=rng, dtype=torch.float32)
    return Split(x, y_true, y_true + noise, noise)


def generate_train(config: Config, seed: int):
    rng = torch.Generator().manual_seed(seed)
    x = uniform(config.train_size, config.train_x_min, config.train_x_max, rng)
    return noisy_split(x, rng, config.noise_std)


def generate_data(config: Config, seed: int):
    rng = torch.Generator().manual_seed(100_000 + seed)
    low, high = config.train_x_min, config.train_x_max
    test_in = noisy_split(uniform(config.n_test_in, low, high, rng), rng, config.noise_std)
    half = config.n_test_ood // 2
    left = uniform(half, config.plot_min, low, rng)
    right = -uniform(half, -config.plot_max, -high, rng)
    # Keep strict inner boundaries even when float32 arithmetic rounds to ±4.
    left.clamp_(max=torch.nextafter(left.new_tensor(low), left.new_tensor(float("-inf"))))
    right.clamp_(min=torch.nextafter(right.new_tensor(high), right.new_tensor(float("inf"))))
    test_ood = noisy_split(torch.cat((left, right)), rng, config.noise_std)
    x_grid = torch.linspace(config.plot_min, config.plot_max, config.plot_points,
                            dtype=torch.float32).reshape(-1, 1)
    return {
        "train": generate_train(config, seed),
        "test_in": test_in,
        "test_ood": test_ood,
        "grid": Split(x_grid, x_grid.pow(3)),
    }


def fgsm_epsilon(train_x, fraction):
    return fraction * (train_x.max() - train_x.min()).item()


def save_data(path: Path, splits, config, seed, epsilon):
    path.mkdir()
    for name, split in splits.items():
        np.savez_compressed(path / f"{name}.npz", **split.arrays())
    write_json(path / "generation.json", {
        "outer_seed": seed, "eval_seed": 100_000 + seed,
        "train_support": [config.train_x_min, config.train_x_max],
        "ood_support": [[config.plot_min, config.train_x_min],
                        [config.train_x_max, config.plot_max]],
        "ood_inner_endpoints_excluded": True,
        "ood_count_per_side": config.n_test_ood // 2,
        "observed_train_range": [splits["train"].x.min().item(), splits["train"].x.max().item()],
        "noise_std": config.noise_std, "noise_variance": config.noise_std ** 2,
        "dtype": "float32", "clean_function": "x**3",
        "sample_counts": {name: len(split.x) for name, split in splits.items()},
        "fgsm_epsilon": epsilon, "fgsm_epsilon_zero": epsilon == 0,
    })
