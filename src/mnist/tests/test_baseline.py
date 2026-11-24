import json
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.mnist import train_baseline as baseline
from src.mnist.classification_metrics import evaluate


def test_classification_metrics_with_unequal_batches():
    # Uniform rows tie at class 0; the third row assigns probability 1/2 to class 1.
    probabilities = torch.full((3, 10), 0.1, dtype=torch.float64)
    probabilities[2].fill_(0.5 / 9)
    probabilities[2, 1] = 0.5
    dataset = TensorDataset(probabilities.log(), torch.tensor([0, 1, 1]))
    result = evaluate(nn.Identity(), DataLoader(dataset, batch_size=2), "cpu")
    assert result["correct"] == 2
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["classification_error"] == pytest.approx(100 / 3)
    assert result["nll"] == pytest.approx((2 * math.log(10) + math.log(2)) / 3)
    assert result["brier_score"] == pytest.approx((0.9 + 0.9 + 0.25 + 0.25 / 9) / 30)


def test_nll_remains_finite_for_extreme_logits():
    logits = torch.zeros(1, 10)
    logits[0, 1] = -1000
    batches = DataLoader(TensorDataset(logits, torch.tensor([1])))
    result = evaluate(nn.Identity(), batches, "cpu")
    assert result["nll"] == pytest.approx(1000 + math.log(9))


def test_probe_replays_initial_weights_and_batches(tmp_path, monkeypatch):
    config = baseline.BaselineConfig(batch_size=2, probe_batches=4, probe_window=2)
    loader = DataLoader(torch.arange(12), shuffle=True, generator=torch.Generator(), batch_size=2)
    starts, orders = [], []

    def measured_batches(model, optimizer, batches, device, criterion):
        starts.append(next(model.parameters()).detach().clone())
        orders.append(torch.cat(list(batches)))
        losses = [1, 1, 1.2, 1.2] if optimizer.param_groups[0]["lr"] == 0.1 else [1, 1, 0.4, 0.2]
        for loss in losses:
            yield loss, 2

    monkeypatch.setattr(baseline, "train_batches", measured_batches)
    assert baseline.probe_rates(config, loader, "cpu", None, tmp_path) == 0.001
    assert torch.equal(starts[0], starts[1])
    assert torch.equal(orders[0], orders[1])
    probe = json.loads((tmp_path / "lr_comparison.json").read_text())
    assert [row["eligible"] for row in probe["candidates"]] == [False, True]
    assert len((tmp_path / "lr_batches.csv").read_text().splitlines()) == 9


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    # Fast offline fixture. Actual MNIST results are generated only by the production command.
    calls = []

    def synthetic_loader(config, device, *, train):
        calls.append(train)
        labels = torch.arange(60_000 if train else 10_000) % 10
        images = nn.functional.one_hot(labels, num_classes=10).float()
        return DataLoader(TensorDataset(images, labels), batch_size=config.batch_size,
                          shuffle=train, generator=torch.Generator().manual_seed(config.seeds[1]))

    monkeypatch.setattr(baseline, "make_loader", synthetic_loader)
    monkeypatch.setattr(baseline, "MNISTMLP", lambda: nn.Linear(10, 10))
    config = baseline.BaselineConfig(device="cpu", epochs=2, batch_size=5000, probe_batches=4, probe_window=2)
    uninterrupted = baseline.run(config, results=tmp_path / "complete")
    writer = baseline.write_csv

    def interrupt_after_checkpoint(path, rows):
        if path.name == "training_history.csv":
            raise InterruptedError("simulated interruption")
        writer(path, rows)

    monkeypatch.setattr(baseline, "write_csv", interrupt_after_checkpoint)
    with pytest.raises(InterruptedError):
        baseline.run(config, results=tmp_path / "interrupted")
    assert calls == [True, False, True]
    checkpoint, = (tmp_path / "interrupted").rglob("*.pth")
    monkeypatch.setattr(baseline, "write_csv", writer)
    resumed = baseline.run(config, resume=checkpoint, results=tmp_path / "interrupted")
    assert calls == [True, False, True, True, False]
    for metric in ("accuracy", "classification_error", "nll", "brier_score"):
        assert resumed[metric] == uninterrupted[metric]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["epoch"] == 2
    assert [r["samples"] for r in state["history"]] == [60_000, 60_000]
    assert state["optimizer_state_dict"]["state"]
    assert (tmp_path / resumed["loss_figure"]).is_file()
