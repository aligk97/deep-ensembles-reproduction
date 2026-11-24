"""Configuration and reproducible seeds for the MNIST architecture check."""

from dataclasses import dataclass
from pathlib import Path

from ..toy_regression.experiment_io import seed_all as set_seed
from ..toy_regression.train_v1_v2_models import member_seeds


@dataclass(frozen=True)
class Config:
    data_root: Path = Path(__file__).resolve().parents[2] / "data"
    batch_size: int = 100
    experiment_seed: int = 42
    member_index: int = 0
    dropout_p: float = 0.0
    download: bool = True
    device: str = "auto"

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.dropout_p < 1:
            raise ValueError("dropout_p must be in [0, 1)")
        if not 0 <= self.experiment_seed < 1_000_000_000:
            raise ValueError("experiment_seed must be in [0, 1e9)")
        if not 0 <= self.member_index < 1000:
            raise ValueError("member_index must be in [0, 1000) for the repository seed scheme")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")

    @property
    def seeds(self) -> tuple[int, int]:
        """The initialization and shuffle seeds for this zero-based member."""
        return member_seeds(self.experiment_seed, self.member_index)
