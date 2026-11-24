"""Compare Adam learning rates on training batches, then train one MNIST MLP."""

import argparse
from dataclasses import asdict, dataclass
from itertools import islice
import math
from pathlib import Path
from statistics import fmean, stdev
from time import perf_counter

import torch
from torch import nn

from ..toy_regression.experiment_io import configure_torch, new_run_dir, write_csv, write_json
from .check_architecture import print_environment
from .classification_metrics import evaluate
from .data import make_loader
from .experiment_config import Config, set_seed
from .model import MNISTMLP


RESULTS = Path(__file__).resolve().parents[2] / "results"


@dataclass(frozen=True)
class BaselineConfig(Config):
    epochs: int = 10
    rates: tuple[float, ...] = (0.1, 0.001)
    probe_batches: int = 100
    probe_window: int = 20

    def __post_init__(self):
        super().__post_init__()
        if self.epochs < 1 or self.batch_size < 2 or 60_000 % self.batch_size == 1:
            raise ValueError("Use positive epochs and batches of at least two for BatchNorm")
        if not 2 <= self.probe_window <= self.probe_batches // 2:
            raise ValueError("The probe must contain two nonoverlapping loss windows")
        if self.dropout_p != 0 or self.member_index != 0:
            raise ValueError("Phase 2 uses a single baseline model with dropout disabled")


def init_model(config, device, rate):
    set_seed(config.seeds[0])
    model = MNISTMLP().to(device)
    return model, torch.optim.Adam(model.parameters(), lr=rate)


def train_batches(model, optimizer, batches, device, criterion):
    model.train()
    for images, labels in batches:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite training cross-entropy")
        loss.backward()
        optimizer.step()
        yield value, len(labels)


def probe_rates(config, loader, device, criterion, directory):
    rows, summaries = [], []
    if len(loader) < config.probe_batches:
        raise ValueError("Reduce the batch size to provide enough batches for the LR probe")
    for rate in config.rates:
        loader.generator.manual_seed(config.seeds[1])
        model, optimizer = init_model(config, device, rate)
        losses = []
        try:
            for loss, _ in train_batches(model, optimizer, islice(loader, config.probe_batches), device, criterion):
                losses.append(loss)
                rows.append(dict(learning_rate=rate, batch=len(losses), train_loss=loss))
        except FloatingPointError:
            pass  # A divergent candidate is ineligible; keep its finite prefix for diagnosis.
        complete = len(losses) == config.probe_batches
        first = fmean(losses[:config.probe_window]) if complete else None
        last = fmean(losses[-config.probe_window:]) if complete else None
        summary = dict(learning_rate=rate, finite_batches=len(losses), first_mean=first, last_mean=last,
                       last_sample_std=stdev(losses[-config.probe_window:]) if complete else None,
                       eligible=complete and last < first)
        summaries.append(summary)
        print(f"Probe: {summary}", flush=True)
    eligible = [row for row in summaries if row["eligible"]]
    best = min(eligible, key=lambda row: row["last_mean"])["learning_rate"] if eligible else None
    write_csv(directory / "lr_batches.csv", rows)
    write_json(directory / "lr_comparison.json", dict(
        split="official MNIST train", batches=config.probe_batches, window=config.probe_window,
        selection_rule="All batch losses finite, last window mean below first, then lowest last mean.",
        candidates=summaries, selected_learning_rate=best))
    if best is None:
        raise ValueError("Neither LR met the predeclared stability criterion")
    return best


def plot_history(history, path):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7, 4), constrained_layout=True)
    ax = figure.subplots()
    ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], marker="o")
    ax.set(xlabel="Epoch", ylabel="Training cross-entropy", title="MNIST single-model baseline")
    ax.grid(alpha=0.25)
    figure.savefig(path, dpi=180)


def run(config, *, resume=None, results=RESULTS):
    device = configure_torch(config.device, deterministic=True)
    env = print_environment(device)
    if device.type == "cpu":
        print("WARNING: full baseline training will run on CPU.", flush=True)
    settings = {**asdict(config), "data_root": str(config.data_root.resolve())}
    train = make_loader(config, device, train=True)
    if len(train.dataset) != 60_000:
        raise ValueError("Expected all 60,000 official training examples")
    criterion = nn.CrossEntropyLoss()
    if resume is None:
        directory = new_run_dir(results / "metrics" / "mnist", f"phase2_s{config.experiment_seed}")
        print(f"Run directory: {directory}", flush=True)
        start = perf_counter()
        rate = probe_rates(config, train, device, criterion, directory)
        probe_seconds = perf_counter() - start
        history = []
    else:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["config"] != settings:
            raise ValueError("Checkpoint configuration differs from this run")
        if (state["cuda_rng"] is None) != (device.type == "cpu"):
            raise ValueError("Resume on the original device type to restore the recorded RNG state")
        directory = results / "metrics" / "mnist" / state["run_id"]
        rate, history, probe_seconds = state["learning_rate"], state["history"], state["probe_seconds"]
    run_id = directory.name
    checkpoint = results / "raw" / "mnist" / run_id / "checkpoints" / "baseline.pth"
    figure_path = results / "figures" / "mnist" / run_id / "training_loss.png"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    model, optimizer = init_model(config, device, rate)
    train.generator.manual_seed(config.seeds[1])
    if resume is not None:
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        train.generator.set_state(state["shuffle_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        # An interruption can occur between the atomic checkpoint and its CSV export.
        write_csv(directory / "training_history.csv", history)
    for epoch in range(len(history) + 1, config.epochs + 1):
        start = perf_counter()
        total, count = 0.0, 0
        for loss, size in train_batches(model, optimizer, train, device, criterion):
            total += loss * size
            count += size
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append(dict(epoch=epoch, train_loss=total / count, samples=count,
                            seconds=perf_counter() - start))
        state = dict(method="single_model", run_id=run_id, epoch=epoch, config=settings,
                     model_seed=config.seeds[0], shuffle_seed=config.seeds[1], learning_rate=rate,
                     model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     torch_rng=torch.get_rng_state(), shuffle_rng=train.generator.get_state(),
                     cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                     history=history, probe_seconds=probe_seconds)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint)
        write_csv(directory / "training_history.csv", history)
        print(f"Epoch {epoch}/{config.epochs}: train CE {total/count:.5f}, {history[-1]['seconds']:.1f} s", flush=True)

    test = make_loader(config, device, train=False)
    if len(test.dataset) != 10_000:
        raise ValueError("Expected all 10,000 official test examples")
    metrics = evaluate(model, test, device)
    plot_history(history, figure_path)
    report = dict(method="single_model", run_id=run_id, config=settings, environment=env,
                  model_device=str(next(model.parameters()).device), model_seed=config.seeds[0],
                  shuffle_seed=config.seeds[1], learning_rate=rate, train_samples=len(train.dataset),
                  training_time_seconds=sum(row["seconds"] for row in history), probe_seconds=probe_seconds,
                  checkpoint=checkpoint.relative_to(results.parent).as_posix(),
                  loss_figure=figure_path.relative_to(results.parent).as_posix(), **metrics)
    write_json(directory / "test_metrics.json", report)
    print(metrics, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--experiment-seed", type=int, default=BaselineConfig.experiment_seed)
    parser.add_argument("--epochs", type=int, default=BaselineConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=BaselineConfig.batch_size)
    parser.add_argument("--data-root", type=Path, default=BaselineConfig.data_root)
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.add_argument("--resume", type=Path, help="Resume or evaluate a completed baseline checkpoint")
    args = vars(parser.parse_args())
    resume = args.pop("resume")
    try:
        run(BaselineConfig(**args), resume=resume)
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit("GPU memory exhausted. Start a fresh run with a smaller --batch-size.") from exc


if __name__ == "__main__":
    main()
