"""Audit completed output against checkpoints, saved probabilities and raw CSV."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import FILES, md5
from engine import METHODS, atomic_json, load_checkpoint, member_seed
from metrics import classification_metrics

ROOT = Path(__file__).resolve().parent


def audit(output, data_dir):
    output, data_dir = Path(output), Path(data_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete", "Experiment is not complete"
    c = manifest["config"]
    for name, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == expected
    raw = pd.read_csv(output / "raw_results.csv")
    expected_rows = len(c["seeds"]) * (4 * len(c["ensemble_sizes"]) + 1)
    assert len(raw) == expected_rows and not raw.duplicated(["run_seed", "method", "M"]).any()
    assert set(raw.run_seed) == set(c["seeds"])
    summary = pd.read_csv(output / "summary.csv")
    assert (summary.n_seeds == len(c["seeds"])).all()
    for name, expected in FILES.items():
        assert md5(data_dir / name) == expected, name
    with np.load(output / "splits.npz") as split:
        assert len(np.intersect1d(split["train"], split["validation"])) == 0
        assert len(np.unique(split["train"])) == len(split["train"])
        assert len(np.unique(split["validation"])) == c["validation_size"]
        assert len(split["test"]) == (c["test_limit"] or 10000)
    count, training_seeds, nll_floors = 0, set(), []
    for seed in c["seeds"]:
        for method in METHODS:
            n = 1 if method == "mc_dropout" else max(c["ensemble_sizes"])
            member_seeds = []
            for member in range(n):
                path = output / "checkpoints" / f"seed_{seed}" / method / f"member_{member:02d}.pt"
                state = load_checkpoint(path)
                assert state["signature"] == manifest["signature"]
                assert state["epoch"] == c["epochs"] and len(state["history"]) == c["epochs"]
                assert state["member_seed"] == member_seed(seed, member, method)
                assert all(torch.isfinite(t).all() for t in state["model"].values())
                member_seeds.append(state["member_seed"])
                training_seeds.add(state["member_seed"])
                count += 1
            assert len(set(member_seeds)) == n
            with np.load(output / "probabilities" / f"seed_{seed}_{method}.npz") as archive:
                stack = torch.from_numpy(archive["probabilities"])
                y = torch.from_numpy(archive["labels"])
            assert stack.shape == (max(c["ensemble_sizes"]), c["test_limit"] or 10000, 10)
            assert torch.isfinite(stack).all() and (stack >= 0).all()
            for size in c["ensemble_sizes"]:
                average = stack[:size].mean(0)
                true_probability = average[torch.arange(len(y)), y]
                nll_floors.append({"run_seed": seed, "method": method, "M": size,
                                   "n_clamped": int((true_probability < 1e-12).sum()),
                                   "min_true_class_probability": float(true_probability.min())})
                result = classification_metrics(average, y)
                saved = raw[(raw.run_seed == seed) & (raw.method == method) & (raw.M == size)].iloc[0]
                assert saved.n_test == len(y)
                for metric, value in result.items():
                    assert np.isclose(saved[metric], value, atol=1e-10, rtol=1e-9), (seed, method, size, metric)
        single = raw[(raw.run_seed == seed) & (raw.method == "single")].iloc[0]
        first = raw[(raw.run_seed == seed) & (raw.method == "ensemble") & (raw.M == 1)].iloc[0]
        for metric in ("accuracy", "classification_error", "nll", "brier"):
            assert single[metric] == first[metric]
    for (method, size), group in raw.groupby(["method", "M"]):
        row = summary[(summary.method == method) & (summary.M == size)].iloc[0]
        for metric in ("accuracy", "classification_error", "nll", "brier"):
            assert np.isclose(row[metric + "_mean"], group[metric].mean(), atol=1e-12)
            assert np.isclose(row[metric + "_std"], group[metric].std(ddof=1), atol=1e-12)
    for name in ("figure2a_comparison", "training_curves", "mnist_examples",
                 "prediction_probabilities", "high_confidence", "low_confidence"):
        for extension in ("png", "pdf"):
            assert (output / "figures" / f"{name}.{extension}").stat().st_size > 1000
    result = {"status": "passed", "config": c["name"], "checkpoints": count,
              "raw_rows": len(raw), "n_seeds": len(c["seeds"]),
              "M": c["ensemble_sizes"], "probability_metric_recomputation": "passed",
              "seed_sample_sd_recomputation": "passed", "data_md5": "passed",
              "train_validation_disjoint": True,
              "training_seed_policy": "paired across treatments; distinct members and run seeds",
              "unique_initialization_seeds": len(training_seeds), "nll_floor_diagnostics": nll_floors}
    atomic_json(output / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    audit(args.output, args.data_dir)
