"""Train, resume or evaluate the standard MNIST deep ensemble."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

from filelock import FileLock
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ..toy_regression.experiment_io import configure_torch, write_csv, write_json
from .check_architecture import print_environment
from .data import make_loader
from .ensemble import plot_summary, predict_log_probs, prefix_scores, summarize
from .experiment_config import Config
from .model import MNISTMLP
from .train_baseline import RESULTS, init_model, train_batches


@dataclass(frozen=True)
class Experiment:
    smoke: bool = False
    seeds: tuple = (42, 123, 2026)
    sizes: tuple = (1, 3, 5)
    batch_size: int = 100
    learning_rate: float = 0.001

    @property
    def run_id(self):
        return "phase3_standard_smoke" if self.smoke else "phase3_standard"

    def settings(self):
        return dict(run_id=self.run_id, method="standard_deep_ensemble", architecture="MNISTMLP-3x200-BN-ReLU",
                    outer_seeds=list(self.seeds), ensemble_sizes=list(self.sizes), batch_size=self.batch_size,
                    epochs=1 if self.smoke else 10, learning_rate=self.learning_rate, dropout_p=0.0,
                    optimizer="Adam", loss="CrossEntropyLoss", deterministic=True, shuffle=True,
                    preprocessing="ToTensor", train_samples=200 if self.smoke else 60_000,
                    test_samples=100 if self.smoke else 10_000, dataset="torchvision MNIST official splits",
                    subset="first indices" if self.smoke else "entire splits", smoke=self.smoke)


def get_dataset(config, device, train, size):
    dataset = make_loader(config, device, train=train).dataset
    if len(dataset) != (60_000 if train else 10_000):
        raise ValueError("Unexpected official MNIST split size")
    return dataset if size == len(dataset) else Subset(dataset, range(size))


def member_config(settings, seed, index, data_root=Config.data_root):
    return Config(data_root=data_root, batch_size=settings["batch_size"], experiment_seed=seed,
                  member_index=index, download=False, device="cuda")


def identity(settings, config):
    return {**settings, "experiment_seed": config.experiment_seed, "member_index": config.member_index,
            "model_seed": config.seeds[0], "shuffle_seed": config.seeds[1]}


def load_member(path, expected, *, complete):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("config") != expected:
        raise ValueError(f"Checkpoint configuration or seeds differ: {path}")
    history = state["history"]
    epoch = state["epoch"]
    if not 1 <= epoch <= expected["epochs"] or len(history) != epoch:
        raise ValueError(f"Invalid checkpoint epoch/history: {path}")
    if complete and epoch != expected["epochs"]:
        raise ValueError(f"Evaluation requires a completed checkpoint: {path}")
    if any(r["epoch"] != i or r["samples"] != expected["train_samples"]
           for i, r in enumerate(history, 1)):
        raise ValueError(f"Incomplete training history: {path}")
    if not all(torch.isfinite(t).all() for t in state["model_state_dict"].values()):
        raise ValueError(f"Nonfinite checkpoint weights: {path}")
    return state


def fit_member(settings, config, dataset, device, path, history_path, environment):
    expected = identity(settings, config)
    model, optimizer = init_model(config, device, settings["learning_rate"])
    digest = sha256()
    for p in model.parameters():
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    initial_hash = digest.hexdigest()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seeds[1]), pin_memory=device.type == "cuda")
    history = []
    if path.exists():
        state = load_member(path, expected, complete=False)
        if state["initial_weights_sha256"] != initial_hash:
            raise ValueError(f"Initialization fingerprint differs: {path}")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state["history"]
        write_csv(history_path, history)
        if len(history) == settings["epochs"]:
            print(f"Skip completed: {path.parent.name}/{path.name}", flush=True)
            return
        loader.generator.set_state(state["shuffle_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        del state
    criterion = nn.CrossEntropyLoss()
    for epoch in range(len(history) + 1, settings["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        total, seen = 0.0, 0
        for loss, count in train_batches(model, optimizer, loader, device, criterion):
            total += loss * count
            seen += count
        if seen != expected["train_samples"]:
            raise ValueError("Training epoch did not cover the expected split")
        peak = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
        history.append(dict(epoch=epoch, train_loss=total / seen, samples=seen,
                            seconds=perf_counter() - start, peak_allocated_bytes=peak))
        temporary = path.with_suffix(".tmp")
        torch.save(dict(config=expected, environment=environment, epoch=epoch, history=history,
                        initial_weights_sha256=initial_hash, model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                        shuffle_rng=loader.generator.get_state()), temporary)
        temporary.replace(path)
        write_csv(history_path, history)
        print(f"Seed {config.experiment_seed}, member {config.member_index + 1}, "
              f"epoch {epoch}/{settings['epochs']}, train CE {total/seen:.5f}", flush=True)


def read_predictions(settings, configs, paths, loader, device, records, root):
    model = MNISTMLP().to(device)
    for config, path in zip(configs, paths, strict=True):
        state = load_member(path, identity(settings, config), complete=True)
        model.load_state_dict(state["model_state_dict"])
        history = state["history"]
        records.append(dict(experiment_seed=config.experiment_seed, member_index=config.member_index,
                            model_seed=config.seeds[0], shuffle_seed=config.seeds[1],
                            initial_weights_sha256=state["initial_weights_sha256"],
                            training_time_seconds=sum(r["seconds"] for r in history),
                            peak_allocated_bytes=max((r["peak_allocated_bytes"] or 0) for r in history),
                            training_gpu=state["environment"]["gpu_model"],
                            checkpoint=path.relative_to(root).as_posix()))
        del state
        yield predict_log_probs(model, loader, device)


def run(experiment, evaluate_only, results=RESULTS):
    settings = experiment.settings()
    device = configure_torch("cuda", deterministic=True)
    environment = print_environment(device)
    metrics = results / "metrics" / "mnist" / experiment.run_id
    figures = results / "figures" / "mnist" / experiment.run_id
    raw = results / "raw" / "mnist" / experiment.run_id
    raw.mkdir(parents=True, exist_ok=True)
    with FileLock(raw / "run.lock", timeout=0):
        metrics.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        base = member_config(settings, experiment.seeds[0], 0)
        train = None if evaluate_only else get_dataset(base, device, True, settings["train_samples"])
        test = get_dataset(base, device, False, settings["test_samples"])
        test_loader = DataLoader(test, shuffle=False, batch_size=experiment.batch_size, pin_memory=True)
        rows, members = [], []
        for seed in experiment.seeds:
            configs = [member_config(settings, seed, i) for i in range(max(experiment.sizes))]
            folder = raw / "checkpoints" / f"seed_{seed}"
            paths = [folder / f"member_{c.member_index + 1}.pth" for c in configs]
            if not evaluate_only:
                folder.mkdir(parents=True, exist_ok=True)
                for config, path in zip(configs, paths, strict=True):
                    fit_member(settings, config, train, device, path, path.with_suffix(".csv"), environment)
            records = []
            stream = read_predictions(settings, configs, paths, test_loader, device, records, results.parent)
            individual, prefixes = prefix_scores(stream, experiment.sizes)
            for record, scores in zip(records, individual, strict=True):
                members.append({**record, **scores})
            seed_rows = []
            for size, scores in prefixes.items():
                seed_rows.append(dict(method="Ensemble", experiment_seed=seed, ensemble_size=size,
                                      member_seeds=[c.seeds[0] for c in configs[:size]], epochs=settings["epochs"],
                                      learning_rate=experiment.learning_rate, batch_size=experiment.batch_size,
                                      training_time_seconds=sum(r["training_time_seconds"] for r in records[:size]),
                                      **scores))
            write_json(metrics / f"seed_{seed}.json", seed_rows)
            rows.extend(seed_rows)
        for field in ("model_seed", "shuffle_seed", "initial_weights_sha256"):
            if len({r[field] for r in members}) != len(members):
                raise ValueError(f"Repeated member {field}")
        summary = summarize(rows, experiment.seeds, experiment.sizes)
        write_csv(metrics / "members.csv", members)
        write_csv(metrics / "per_seed.csv", rows)
        write_csv(metrics / "mean_std.csv", summary)
        write_json(metrics / "mean_std.json", summary)
        plot_summary(summary, figures, experiment.smoke)
        write_json(metrics / "experiment.json", dict(config=settings, evaluation_environment=environment,
                   members=members, total_training_seconds=sum(r["training_time_seconds"] for r in members),
                   max_training_peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in members),
                   averaging="arithmetic mean of member softmax probabilities",
                   standard_deviation="sample standard deviation over independent outer seeds (ddof=1)"))
        print(f"Saved results: {metrics}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true", help="Use all MNIST images, 10 epochs per member")
    mode.add_argument("--smoke", action="store_true", help="Use 200 training / 100 test images, one epoch")
    parser.add_argument("--evaluate-only", action="store_true", help="Only score saved, completed members")
    args = parser.parse_args()
    try:
        run(Experiment(smoke=args.smoke), args.evaluate_only)
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit("CUDA ran out of memory. Free GPU memory and repeat to resume.") from exc


if __name__ == "__main__":
    main()
