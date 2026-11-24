"""Official MNIST splits with pixels scaled to [0, 1]."""

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from .experiment_config import Config


def make_loader(config: Config, device: torch.device, *, train: bool) -> DataLoader:
    dataset = MNIST(root=config.data_root, train=train, transform=ToTensor(), download=config.download)
    generator = torch.Generator().manual_seed(config.seeds[1]) if train else None
    return DataLoader(dataset, batch_size=config.batch_size, pin_memory=device.type == "cuda",
                      shuffle=train, generator=generator)


def make_loaders(config: Config, device: torch.device) -> tuple[DataLoader, DataLoader]:
    return make_loader(config, device, train=True), make_loader(config, device, train=False)
