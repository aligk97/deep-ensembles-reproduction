import csv
import json
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.mnist import select_epochs


def test_train_only_partition_is_complete_reproducible_and_disjoint(monkeypatch):
    calls = []

    def official_training_only(config, device, *, train):
        assert train
        calls.append(train)
        return SimpleNamespace(dataset=range(60_000))

    monkeypatch.setattr(select_epochs, "make_loader", official_training_only)
    config = select_epochs.EpochConfig(device="cpu")
    first, validation, train_ids, val_ids = select_epochs.training_partition(config, torch.device("cpu"))
    repeat, _, train_again, val_again = select_epochs.training_partition(config, torch.device("cpu"))
    assert len(train_ids) == 54_000 and len(val_ids) == 6_000
    assert np.intersect1d(train_ids, val_ids).size == 0
    np.testing.assert_array_equal(np.sort(np.concatenate([train_ids, val_ids])), np.arange(60_000))
    np.testing.assert_array_equal(train_again, train_ids)
    np.testing.assert_array_equal(val_again, val_ids)
    assert torch.equal(next(iter(first)), next(iter(repeat)))
    np.testing.assert_array_equal(next(iter(validation)).numpy(), val_ids[:config.batch_size])
    assert calls == [True, True]


def test_selection_uses_only_candidate_nll_and_links_matching_checkpoints(tmp_path, monkeypatch):
    config = select_epochs.EpochConfig(device="cpu", candidates=(1, 3, 4))
    dataset = TensorDataset(torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]))
    loader = DataLoader(dataset, batch_size=2, generator=torch.Generator().manual_seed(42))
    monkeypatch.setattr(select_epochs, "training_partition",
                        lambda *_: (loader, loader, np.array([0, 1]), np.array([2, 3])))
    # Epoch 2 has the lowest NLL overall but is outside the candidate set.
    # Epochs 3 and 4 tie on NLL; the later epoch has better accuracy.
    scores = iter([(0.4, 0.5), (0.01, 1.0), (0.3, 0.5), (0.3, 1.0)])

    def evaluate(*args):
        nll, accuracy = next(scores)
        return dict(test_samples=2, accuracy=accuracy, classification_error=100 * (1 - accuracy),
                    nll=nll, brier_score=0.05)

    monkeypatch.setattr(select_epochs, "evaluate", evaluate)
    select_epochs.run(config, results=tmp_path / "results")
    report_path, = tmp_path.rglob("epoch_selection.json")
    report = json.loads(report_path.read_text())
    assert report["selected_epoch"] == 3
    with (report_path.parent / "epoch_comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["epoch"]) for row in rows] == [1, 3, 4]
    for row in rows:
        checkpoint = torch.load(tmp_path / row["checkpoint"], weights_only=True, map_location="cpu")
        assert checkpoint["epoch"] == int(row["epoch"])
        assert checkpoint["run_id"] == report["run_id"]
        assert checkpoint["config"]["split_indices_sha256"] == report["config"]["split_indices_sha256"]
        assert checkpoint["metrics"]["val_nll"] == float(row["val_nll"])
    assert len(list(tmp_path.rglob("*.png"))) == 4
