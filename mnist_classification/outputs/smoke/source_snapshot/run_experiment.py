"""Run smoke/full experiments. All paths are relative to this folder by default."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import prepare
from engine import (METHODS, atomic_json, load_checkpoint, member_seed, predict,
                    seed_all, train_member)
from metrics import classification_metrics
from model import MLP

ROOT = Path(__file__).resolve().parent
CORE_FILES = ("data.py", "model.py", "metrics.py", "engine.py", "run_experiment.py")


def validate_config(c):
    required = {"name", "seeds", "ensemble_sizes", "epochs", "batch_size", "learning_rate",
                "epsilon_fraction", "dropout", "validation_size", "split_seed", "train_limit",
                "test_limit", "eval_batch_size", "threads", "deterministic"}
    if set(c) != required:
        raise ValueError(f"Unknown/missing configuration fields: {set(c) ^ required}")
    assert len(c["seeds"]) == len(set(c["seeds"])) and len(c["seeds"]) >= 3
    assert c["ensemble_sizes"] == sorted(set(c["ensemble_sizes"])) and c["ensemble_sizes"][0] == 1
    assert c["epochs"] > 0 and c["batch_size"] >= 2 and c["learning_rate"] > 0
    assert 0 < c["validation_size"] < 59998 and 0 <= c["dropout"] < 1
    assert c["train_limit"] == 0 or 2 <= c["train_limit"] <= 60000 - c["validation_size"]
    assert 0 <= c["test_limit"] <= 10000 and 0 <= c["epsilon_fraction"] <= 1


def summarize(rows, output):
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "raw_results.csv", index=False)
    seed_folder = output / "seed_results"
    seed_folder.mkdir(exist_ok=True)
    for seed, group in frame.groupby("run_seed"):
        group.to_csv(seed_folder / f"seed_{seed}.csv", index=False)
    metrics = ["classification_error", "accuracy", "nll", "brier"]
    grouped = frame.groupby(["method", "M"], sort=False)
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(item) for item in summary.columns]
    summary["n_seeds"] = grouped.size()
    summary.reset_index().to_csv(output / "summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/full.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    output = (args.output or ROOT / "outputs" / config["name"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CORE_FILES}
    signature = hashlib.sha256(json.dumps({"config": config, "sources": source_hashes},
                                         sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise ValueError("Output belongs to different configuration/source. Choose a NEW --output directory.")
    if args.plots_only:
        from plots import make_plots
        make_plots(output, args.data_dir)
        return
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(config["deterministic"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config["deterministic"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    environment = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                   "pandas": pd.__version__, "platform": platform.platform(),
                   "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "cuda": torch.version.cuda, "deterministic": config["deterministic"]}
    manifest = {"signature": signature, "config": config, "source_sha256": source_hashes,
                "environment": environment, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "resume_policy": "last completed epoch; final epoch evaluated; no test-based selection"}
    if manifest_path.exists():
        manifest["first_started_utc"] = old.get("first_started_utc", old["started_utc"])
    atomic_json(manifest_path, manifest)
    atomic_json(output / "config.json", config)
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in CORE_FILES:
        shutil.copyfile(ROOT / name, snapshot / name)
    print(json.dumps(environment), flush=True)
    start = time.perf_counter()
    try:
        tx, ty, vx, vy, test_x, test_y, epsilon = prepare(args.data_dir, config, output, device)
        paths = {}
        # Train ALL configurations before any test metrics are computed.
        for seed in config["seeds"]:
            for method in METHODS:
                count = 1 if method == "mc_dropout" else max(config["ensemble_sizes"])
                paths[seed, method] = [train_member(config, signature, output, method, seed, member,
                                                  tx, ty, vx, vy, epsilon) for member in range(count)]
        rows = []
        probabilities = output / "probabilities"
        probabilities.mkdir(exist_ok=True)
        cpu_y = test_y.cpu()
        for seed in config["seeds"]:
            for method in METHODS:
                samples = []
                for path in paths[seed, method]:
                    model = MLP(config["dropout"] if method == "mc_dropout" else 0).to(device)
                    state = load_checkpoint(path)
                    assert state["epoch"] == config["epochs"]
                    model.load_state_dict(state["model"])
                    if method == "mc_dropout":
                        # Isolate evaluation RNG from all training RNG streams.
                        seed_all(member_seed(seed, 0, method) + 500000000)
                        for _ in range(max(config["ensemble_sizes"])):
                            samples.append(predict(model, test_x, config["eval_batch_size"], True))
                    else:
                        samples.append(predict(model, test_x, config["eval_batch_size"]))
                    del model
                stack = torch.stack(samples)
                np.savez_compressed(probabilities / f"seed_{seed}_{method}.npz",
                                    probabilities=stack.numpy(), labels=cpu_y.numpy())
                for size in config["ensemble_sizes"]:
                    result = {"run_seed": seed, "method": method, "M": size,
                              "n_test": len(cpu_y), **classification_metrics(stack[:size].mean(0), cpu_y)}
                    rows.append(result)
                    if method == "ensemble" and size == 1:
                        rows.append({**result, "method": "single"})
                summarize(rows, output)
        from plots import make_plots
        make_plots(output, args.data_dir)
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        invocation_seconds=time.perf_counter() - start, result_rows=len(rows),
                        trained_models=sum(map(len, paths.values())),
                        total_training_epoch_seconds=float(sum(pd.read_csv(p)["epoch_seconds"].sum()
                                                               for p in (output / "histories").glob("*.csv"))))
        atomic_json(manifest_path, manifest)
        print(f"COMPLETE: {output}; {len(rows)} result rows", flush=True)
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=repr(exc), invocation_seconds=time.perf_counter() - start)
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
