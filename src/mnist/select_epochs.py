"""Select a fixed MNIST training duration using only the official training split."""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ..toy_regression.experiment_io import configure_torch, new_run_dir, write_csv, write_json
from .check_architecture import print_environment
from .classification_metrics import evaluate
from .data import make_loader
from .experiment_config import Config
from .train_baseline import RESULTS, init_model, train_batches


@dataclass(frozen=True)
class EpochConfig(Config):
    device: str = "cuda"
    download: bool = False
    split_seed: int = 42
    learning_rate: float = 0.001
    candidates: tuple[int, ...] = (10, 15, 20)


def training_partition(config, device):
    dataset = make_loader(config, device, train=True).dataset
    if len(dataset) != 60_000:
        raise ValueError("Expected the official MNIST training split")
    permutation = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config.split_seed))
    train_ids, val_ids = permutation.split([54_000, 6_000])
    if not torch.equal(permutation.sort().values, torch.arange(len(dataset))):
        raise ValueError("Training and validation indices must partition the official training split")
    options = dict(batch_size=config.batch_size, pin_memory=device.type == "cuda")
    train = DataLoader(Subset(dataset, train_ids.tolist()), shuffle=True,
                       generator=torch.Generator().manual_seed(config.seeds[1]), **options)
    validation = DataLoader(Subset(dataset, val_ids.tolist()), shuffle=False, **options)
    return train, validation, train_ids.numpy(), val_ids.numpy()


def save_figures(history, directory, selected):
    from matplotlib.figure import Figure

    plots = {
        "train_loss": ("Training cross-entropy", "training_loss.png"),
        "val_nll": ("Validation NLL", "validation_nll.png"),
        "val_accuracy_pct": ("Validation accuracy (%)", "validation_accuracy.png"),
        "val_brier": ("Validation Brier score (1/K)", "validation_brier.png"),
    }
    for metric, (label, filename) in plots.items():
        figure = Figure(figsize=(7, 4), constrained_layout=True)
        ax = figure.subplots()
        ax.plot([row["epoch"] for row in history], [row[metric] for row in history],
                color="#24557c", marker=".", linewidth=1.7)
        ax.scatter(selected["epoch"], selected[metric], color="#b34a28", zorder=3,
                   label=f"Selected epoch {selected['epoch']}")
        ax.set(xlabel="Epoch", ylabel=label, title=label, xticks=[1, 5, 10, 15, 20])
        ax.grid(alpha=0.25)
        ax.legend()
        figure.savefig(directory / filename, dpi=180)


def run(config, results=RESULTS):
    device = configure_torch(config.device, deterministic=True)
    environment = print_environment(device)
    train, validation, train_ids, val_ids = training_partition(config, device)
    directory = new_run_dir(results / "metrics" / "mnist", "epoch_s42")
    run_id = directory.name
    raw = results / "raw" / "mnist" / run_id
    checkpoints = raw / "checkpoints"
    figures = results / "figures" / "mnist" / run_id
    checkpoints.mkdir(parents=True)
    figures.mkdir(parents=True)
    np.savez_compressed(raw / "split_indices.npz", train=train_ids, validation=val_ids)
    split_hash = sha256(np.concatenate([train_ids, val_ids]).astype("<i8").tobytes()).hexdigest()
    settings = {**asdict(config), "data_root": str(config.data_root.resolve()),
                "dataset": "MNIST train", "train_samples": len(train_ids), "val_samples": len(val_ids),
                "split_indices_sha256": split_hash, "model_seed": config.seeds[0],
                "shuffle_seed": config.seeds[1], "optimizer": "Adam", "loss": "CrossEntropyLoss"}
    model, optimizer = init_model(config, device, config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history, candidates = [], []
    print(f"Run: {directory}\nSplit validated: {len(train_ids)} train, {len(val_ids)} validation, "
          f"0 overlapping indices. Model device: {next(model.parameters()).device}", flush=True)
    start = perf_counter()
    for epoch in range(1, max(config.candidates) + 1):
        total, count = 0.0, 0
        for loss, size in train_batches(model, optimizer, train, device, criterion):
            total += loss * size
            count += size
        metrics = evaluate(model, validation, device)
        if count != len(train_ids) or metrics["test_samples"] != len(val_ids):
            raise ValueError("Incomplete training or validation epoch")
        row = dict(epoch=epoch, train_loss=total / count, val_accuracy_pct=100 * metrics["accuracy"],
                   val_error_pct=metrics["classification_error"], val_nll=metrics["nll"],
                   val_brier=metrics["brier_score"])
        history.append(row)
        if epoch in config.candidates:
            path = checkpoints / f"epoch_{epoch}.pth"
            temporary = path.with_suffix(".tmp")
            torch.save(dict(run_id=run_id, config=settings, epoch=epoch, metrics=row,
                            model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                            shuffle_rng=train.generator.get_state(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                            history=history), temporary)
            temporary.replace(path)
            candidates.append({**row, "checkpoint": path.relative_to(results.parent).as_posix()})
        write_csv(directory / "epoch_history.csv", history)
        print(f"Epoch {epoch:2d} | train CE {row['train_loss']:.5f} | val NLL {row['val_nll']:.5f} "
              f"| accuracy {row['val_accuracy_pct']:.2f}% | Brier {row['val_brier']:.6f}", flush=True)
    seconds = perf_counter() - start
    best = min(candidates, key=lambda row: (row["val_nll"], row["epoch"]))
    write_csv(directory / "epoch_comparison.csv", candidates)
    save_figures(history, figures, best)
    write_json(directory / "epoch_selection.json", dict(
        run_id=run_id, config=settings, environment=environment, model_device=str(next(model.parameters()).device),
        train_validation_seconds=seconds, split_indices=(raw / "split_indices.npz").relative_to(results.parent).as_posix(),
        overlap_count=0, candidates=candidates, selected_epoch=best["epoch"],
        rationale=f"Epoch {best['epoch']} has the lowest validation NLL ({best['val_nll']:.9f}) "
                  f"among {list(config.candidates)}. Exact ties favor the earlier epoch.",
        final_val_nll_change_from_previous_epoch=history[-1]["val_nll"] - history[-2]["val_nll"],
        final_val_nll_change_from_previous_candidate=candidates[-1]["val_nll"] - candidates[-2]["val_nll"],
        phase3_training="Initialize fresh models and train on all 60,000 official training images."))
    print(f"Selected epoch: {best['epoch']} | Results: {directory}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Config.data_root)
    run(EpochConfig(**vars(parser.parse_args())))
