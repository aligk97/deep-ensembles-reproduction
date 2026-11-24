"""Configuration loading and validation for the Figure 1 experiment."""
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    experiment: str = "toy_regression_figure1"
    preset: str = "practical"
    outer_seeds: tuple[int, ...] = (42, 123, 2026)
    train_size: int = 20
    train_x_min: float = -4.0
    train_x_max: float = 4.0
    noise_std: float = 3.0
    hidden_dim: int = 100
    ensemble_size: int = 5
    optimizer: str = "adam"
    learning_rate: float = 0.001
    max_epochs: int = 3000
    batch_size: int = 5
    shuffle: bool = True
    weight_decay: float = 0.0
    variance_floor: float = 1e-6
    fgsm_epsilon_fraction: float = 0.01
    plot_min: float = -6.0
    plot_max: float = 6.0
    plot_points: int = 601
    plot_sigma_multiplier: float = 3.0
    n_test_in: int = 2000
    n_test_ood: int = 2000
    save_checkpoints: bool = True
    save_per_member_predictions: bool = True
    deterministic: bool = False
    gradient_clip_norm: float | None = None
    device: str = "auto"
    smoke: bool = False

    def __post_init__(self):
        fixed = {
            "experiment": "toy_regression_figure1", "train_size": 20,
            "train_x_min": -4.0, "train_x_max": 4.0, "noise_std": 3.0,
            "hidden_dim": 100, "ensemble_size": 5, "optimizer": "adam",
            "variance_floor": 1e-6, "fgsm_epsilon_fraction": 0.01,
            "plot_min": -6.0, "plot_max": 6.0, "plot_sigma_multiplier": 3.0,
        }
        defaults = {field.name: field.default for field in fields(self)}
        for name, default in defaults.items():
            value = getattr(self, name)
            if type(default) is float:
                if type(value) not in (int, float) or not isfinite(value):
                    raise ValueError(f"{name} must be a finite number")
            elif default is not None and type(value) is not type(default):
                raise ValueError(f"{name} must have type {type(default).__name__}")
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected!r} for Figure 1")
        if self.preset not in {"practical", "paper_default_probe"}:
            raise ValueError("preset must be practical or paper_default_probe")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if not self.outer_seeds or any(
            type(seed) is not int or not 0 <= seed < 1_000_000_000
            for seed in self.outer_seeds
        ) or len(set(self.outer_seeds)) != len(self.outer_seeds):
            raise ValueError("outer_seeds must contain distinct integers in [0, 1e9)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if min(self.max_epochs, self.batch_size, self.n_test_in) < 1:
            raise ValueError("max_epochs, batch_size and n_test_in must be positive")
        if self.n_test_ood < 2 or self.n_test_ood % 2:
            raise ValueError("n_test_ood must be a positive even number")
        if self.plot_points not in (500, 601):
            raise ValueError("plot_points must be 500 or 601")
        clip = self.gradient_clip_norm
        if clip is not None and (type(clip) not in (int, float) or not isfinite(clip) or clip <= 0):
            raise ValueError("gradient_clip_norm must be null or positive and finite")

    def to_dict(self):
        values = asdict(self)
        values["outer_seeds"] = list(self.outer_seeds)
        return values


def load_config(path: Path | str, *, seeds=None, device=None, smoke=False):
    path = Path(path)
    if not path.exists() and path.parent.resolve() == Path(__file__).resolve().parent / "configs":
        # Commands recorded before the preset rename still resolve to the same settings.
        old_names = {"toy_default.yaml": "v1_baseline.yaml", "toy_v2_a.yaml": "v2_a_high_learning_rate.yaml",
                     "toy_v2_b.yaml": "v2_b_full_batch.yaml",
                     "toy_paper_default_probe.yaml": "v2_c_high_learning_rate_full_batch.yaml"}
        path = path.with_name(old_names.get(path.name, path.name))
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Config must be a YAML mapping")
    unknown = set(values) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(map(str, unknown))}")
    if "outer_seeds" in values and isinstance(values["outer_seeds"], list):
        values["outer_seeds"] = tuple(values["outer_seeds"])
    resolved = Config(**values)
    overrides = resolved.to_dict()
    overrides["outer_seeds"] = tuple(seeds) if seeds is not None else resolved.outer_seeds
    if device is not None:
        overrides["device"] = device
    if smoke or resolved.smoke:
        overrides.update(smoke=True, max_epochs=min(resolved.max_epochs, 3),
                         outer_seeds=overrides["outer_seeds"][:1],
                         n_test_in=min(resolved.n_test_in, 80),
                         n_test_ood=min(resolved.n_test_ood, 80))
    return Config(**overrides)
