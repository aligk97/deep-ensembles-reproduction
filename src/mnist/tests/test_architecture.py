from dataclasses import replace
import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from src.mnist import check_architecture, data
from src.mnist.experiment_config import Config, set_seed
from src.mnist.model import MNISTMLP


@pytest.mark.parametrize("dropout_p", [0.0, 0.1])
def test_hidden_blocks_and_raw_logits(dropout_p):
    model = MNISTMLP(dropout_p).eval()
    block = [nn.Linear, nn.BatchNorm1d, nn.ReLU]
    if dropout_p:
        block.append(nn.Dropout)
    assert [type(layer) for layer in model.hidden] == [nn.Flatten] + block * 3
    assert all(layer.p == dropout_p for layer in model.modules() if isinstance(layer, nn.Dropout))
    assert sum(p.numel() for p in model.parameters()) == 240_610
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.copy_(torch.arange(10))
        logits = model(torch.zeros(3, 1, 28, 28))
    torch.testing.assert_close(logits, torch.arange(10.0).expand(3, 10))


def test_seeds_reproduce_weights_and_independent_shuffle(monkeypatch):
    # Exercise the production loaders offline using disjoint, identifiable examples.
    def dataset(*, train, **options):
        return torch.arange(24) if train else torch.arange(24, 32)

    monkeypatch.setattr(data, "MNIST", dataset)

    def initialize(config):
        set_seed(config.seeds[0])
        model = MNISTMLP()
        train, test = data.make_loaders(config, torch.device("cpu"))
        assert isinstance(test.sampler, SequentialSampler)
        assert torch.equal(torch.cat(list(test)), torch.arange(24, 32))
        order = torch.cat(list(train))
        assert torch.equal(order.sort().values, torch.arange(24))
        return model.hidden[1].weight.detach(), order

    base = Config(batch_size=5)
    weight, order = initialize(base)
    repeated_weight, repeated_order = initialize(base)
    assert torch.equal(weight, repeated_weight)
    assert torch.equal(order, repeated_order)
    for config in (replace(base, member_index=1), replace(base, experiment_seed=123)):
        other_weight, other_order = initialize(config)
        assert not torch.equal(weight, other_weight)
        assert not torch.equal(order, other_order)


def test_report_updates_only_after_success(tmp_path, monkeypatch):
    # Expanded views keep this offline persistence test small.
    def make_loaders(config, device):
        def loader(size):
            images = torch.zeros(1, 1, 28, 28).expand(size, -1, -1, -1)
            labels = torch.zeros(1, dtype=torch.long).expand(size)
            return DataLoader(TensorDataset(images, labels), batch_size=config.batch_size)
        return loader(60_000), loader(10_000)

    monkeypatch.setattr(check_architecture, "make_loaders", make_loaders)
    path = tmp_path / "metrics" / "mnist" / "phase1_sanity.json"
    for size in (3, 7):
        config = Config(device="cpu", batch_size=size)
        check_architecture.run(config, path)
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["status"] == "passed"
        assert report["config"]["batch_size"] == size
        assert report["gpu_peak_allocated_bytes"] is None
        for batch in report["batches"].values():
            assert batch["input_shape"] == [size, 1, 28, 28]
            assert batch["logits_shape"] == batch["probabilities_shape"] == [size, 10]
            assert 0 <= batch["max_abs_softmax_sum_deviation"] <= 1e-6

    previous = path.read_bytes()
    monkeypatch.setattr(check_architecture, "make_loaders", lambda *_: (
        DataLoader(range(2)), DataLoader(range(1)),
    ))
    with pytest.raises(ValueError, match="official"):
        check_architecture.run(config, path)
    assert path.read_bytes() == previous
