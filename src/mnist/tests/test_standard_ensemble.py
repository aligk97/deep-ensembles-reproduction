import math
from statistics import stdev

import pytest
import torch
from torch.utils.data import TensorDataset

from src.mnist import run_ensemble as runner
from src.mnist.ensemble import prefix_scores, summarize


def test_prefixes_match_explicit_softmax_mixtures():
    # First image distinguishes averaging probabilities from softmax of mean logits.
    # Second image distinguishes it from majority voting.
    margins = [(5, 0.2), (5, 0.2), (-20, -5), (1, 1), (-2, -0.4)]
    labels = torch.tensor([0, 1])
    logits = torch.full((5, 2, 10), -50., dtype=torch.float64)
    logits[:, :, 0] = torch.tensor(margins)
    logits[:, :, 1] = 0.
    members, prefixes = prefix_scores(((x.log_softmax(1), labels) for x in logits), (1, 3, 5))
    for size in (1, 3, 5):
        p = logits[:size].softmax(dim=2).mean(dim=0)
        metrics = prefixes[size]
        assert metrics["nll"] == pytest.approx(-p[range(2), labels].log().mean().item())
        target = torch.nn.functional.one_hot(labels, num_classes=10)
        assert metrics["brier_score"] == pytest.approx((p - target).square().mean().item())
        accuracy = (p.argmax(1) == labels).double().mean().item()
        assert metrics["accuracy"] == accuracy
        assert metrics["classification_error"] == 100 * (1 - accuracy)
    assert prefixes[1] == members[0]
    assert prefixes[3]["correct"] == 2
    assert prefixes[3]["nll"] != pytest.approx(sum(m["nll"] for m in members[:3]) / 3)


def test_numerically_stable_mixture_and_alignment_guard():
    logits = torch.zeros(2, 10, dtype=torch.float64)
    logits[:, :2] = -1200
    labels = torch.tensor([0, 1])
    log_p = logits.log_softmax(1)
    _, prefix = prefix_scores([(log_p, labels)] * 3, (1, 3))
    assert prefix[3]["nll"] == pytest.approx(1200 + math.log(8))
    with pytest.raises(ValueError, match="order"):
        prefix_scores([(log_p, labels), (log_p, labels.flip([0]))], (1, 2))


def test_summary_uses_three_outer_seeds_and_sample_std():
    seeds = [42, 123, 2026]
    rows = [dict(experiment_seed=s, ensemble_size=3, accuracy=x, nll=x, brier_score=x,
                 classification_error=x) for s, x in zip(seeds, [1, 2, 6])]
    summary, = summarize(rows, seeds, [3])
    assert summary["repetitions"] == 3
    assert summary["nll_mean"] == 3
    assert summary["nll_std"] == pytest.approx(stdev([1, 2, 6]))
    with pytest.raises(ValueError):
        summarize(rows + rows[:1], seeds, [3])


def test_all_member_initializations_and_shuffle_sequences_are_independent_and_repeatable(tmp_path, monkeypatch):
    settings = runner.Experiment(smoke=True).settings()
    starts = []

    def capture_start(model, optimizer, loader, device, criterion):
        order = tuple(torch.cat(list(loader)).tolist())
        assert sorted(order) == list(range(settings["train_samples"]))
        starts.append((next(model.parameters()).detach().clone(), order))
        yield 0., len(order)

    monkeypatch.setattr(runner, "train_batches", capture_start)
    initializations, orders = set(), set()
    for seed in settings["outer_seeds"]:
        for index in range(5):
            config = runner.member_config(settings, seed, index)
            for repeat in range(2):
                path = tmp_path / f"{seed}_{index}_{repeat}.pth"
                runner.fit_member(settings, config, range(200), torch.device("cpu"),
                                  path, path.with_suffix(".csv"), {})
            weights, order = starts.pop()
            repeated_weights, repeated_order = starts.pop()
            assert torch.equal(weights, repeated_weights)
            assert order == repeated_order
            initializations.add(weights.numpy().tobytes())
            orders.add(order)
    assert len(initializations) == len(orders) == 15


def test_epoch_resume_skip_and_incompatible_checkpoint(tmp_path, monkeypatch):
    settings = dict(runner.Experiment(smoke=True).settings(), epochs=2, train_samples=8, batch_size=4)
    config = runner.member_config(settings, 123, 0)
    cpu = torch.device("cpu")
    images = torch.rand(8, 1, 28, 28, generator=torch.Generator().manual_seed(3))
    dataset = TensorDataset(images, torch.arange(8))
    complete, resumed = tmp_path / "all.pth", tmp_path / "resume.pth"
    def train(path):
        runner.fit_member(settings, config, dataset, cpu, path, path.with_suffix(".csv"), {})
    train(complete)
    writer = runner.write_csv
    def interrupt(*args):
        raise InterruptedError()
    monkeypatch.setattr(runner, "write_csv", interrupt)
    with pytest.raises(InterruptedError):
        train(resumed)
    expected = runner.identity(settings, config)
    with pytest.raises(ValueError, match="completed"):
        runner.load_member(resumed, expected, complete=True)
    monkeypatch.setattr(runner, "write_csv", writer)
    train(resumed)
    reference = runner.load_member(complete, expected, complete=True)
    recovered = runner.load_member(resumed, expected, complete=True)
    for key, weight in reference["model_state_dict"].items():
        assert torch.equal(weight, recovered["model_state_dict"][key])
    before = resumed.read_bytes()
    monkeypatch.setattr(runner, "train_batches", lambda *_: pytest.fail("Retrained complete member"))
    train(resumed)
    assert resumed.read_bytes() == before
    with pytest.raises(ValueError, match="configuration"):
        runner.load_member(resumed, dict(expected, shuffle_seed=10), complete=True)
