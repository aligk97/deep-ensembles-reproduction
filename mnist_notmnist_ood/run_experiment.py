"""Smoke-test, evaluate all prepared ID/OOD images, validate, plot, verify integrity.

No downloads or training. Only writes inside mnist_notmnist_ood. A complete
probability group is the resume unit (seed, method, dataset); M uses prefixes.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch

if __package__:
    from .data import HERE, digest, write_json
    from .inference import (METHOD_NAMES, ORIGINAL, load_config, setup_device, checkpoint_inventory,
                            load_datasets, infer_group, original_api, config_path,
                            cache_path, save_group, read_group, atomic_npy, mc_stream_seed)
    from .metrics import (predictive_entropy, mean_id_entropy, mean_ood_entropy,
                          entropy_gap, ood_auroc, summarize_seed_metrics)
    from .test_inference import run_smoke
    from .plots import make_plots
else:
    from data import HERE, digest, write_json
    from inference import (METHOD_NAMES, ORIGINAL, load_config, setup_device, checkpoint_inventory,
                           load_datasets, infer_group, original_api, config_path,
                           cache_path, save_group, read_group, atomic_npy, mc_stream_seed)
    from metrics import (predictive_entropy, mean_id_entropy, mean_ood_entropy,
                         entropy_gap, ood_auroc, summarize_seed_metrics)
    from test_inference import run_smoke
    from plots import make_plots

METRICS = ("mean_id_entropy", "mean_ood_entropy", "entropy_gap", "ood_auroc")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty results table")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def dataset_metadata(datasets, output):
    records = {}
    for name, dataset in datasets.items():
        x = dataset.tensors[0] if name == "mnist" else dataset.images
        labels = dataset.tensors[1].tolist() if name == "mnist" else dataset.targets.tolist()
        order = ([f"official_mnist_test_index_{i:05d}" for i in range(len(dataset))]
                 if name == "mnist" else dataset.paths)
        record = {"dataset": name, "n_examples": len(dataset), "sample_ids": order,
                  "labels": labels, "tensor_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                  "shape": list(x.shape), "dtype": str(x.dtype)}
        record["order_sha256"] = json_hash(record)
        write_json(output / "sample_order" / f"{name}.json", record)
        records[name] = record
    return records


def group_metadata(seed, method, name, order, inventory, config):
    selected = [row for row in inventory if row["run_seed"] == seed and row["method"] == method]
    return {"seed": seed, "method": method, "dataset": name, "M_max": 5, "prefix_M": [1, 3, 5],
            "n_examples": order["n_examples"], "K": 10, "labels": order["labels"],
            "sample_order_file": f"sample_order/{name}.json", "sample_order_sha256": order["order_sha256"],
            "input_tensor_sha256": order["tensor_sha256"], "checkpoints": selected,
            "M_axis": "stochastic pass 0..4" if method == "mc_dropout" else "member_00..member_04",
            "mc_stream_seed": mc_stream_seed(seed, name, config) if method == "mc_dropout" else None,
            "batch_size": config["batch_size"], "shuffle": False, "drop_last": False,
            "class_order": list(range(10)), "probability_dtype": "float32"}


def compare_previous(stack, labels, seed, method, m, config, old_rows):
    # Preserve original FP32 averaging and original FP64 NLL/Brier implementation.
    current = original_api().classification_metrics(stack[:m].mean(0), labels)
    previous = next(row for row in old_rows if int(row["run_seed"]) == seed
                    and row["method"] == method and int(row["M"]) == m)
    comparison = {"run_seed": seed, "method": method, "M": m,
                  "deterministic_match_required": method != "mc_dropout"}
    all_close = True
    for metric, tolerance in config["classification_atol"].items():
        expected = float(previous[metric])
        delta = current[metric] - expected
        comparison.update({f"{metric}_previous": expected, f"{metric}_current": current[metric],
                           f"{metric}_delta": delta, f"{metric}_atol": tolerance})
        all_close = all_close and abs(delta) <= tolerance
    comparison["within_tolerance"] = all_close
    comparison["interpretation"] = ("same checkpoint/member prefix; numerical consistency required"
                                    if method != "mc_dropout" else
                                    "stochastic scores; legacy MNIST stream seed/batch reproduced; cross-device masks may differ")
    if method != "mc_dropout" and not all_close:
        raise AssertionError(f"Figure 2(a) consistency failed; investigate without changing checkpoints: {comparison}")
    return comparison


def make_summary(rows, config):
    summary = []
    for method in config["methods"]:
        for m in config["ensemble_sizes"]:
            selected = [row for row in rows if row["method"] == method and row["M"] == m]
            if len(selected) != 3 or {row["run_seed"] for row in selected} != set(config["seeds"]):
                raise AssertionError("Summary requires all three run seeds for each method/M")
            stats = summarize_seed_metrics({row["run_seed"]: {metric: row[metric] for metric in METRICS}
                                            for row in selected})
            result = {"method": method, "M": m, "n_seeds": stats["n_seeds"], "ddof": stats["ddof"]}
            for metric in METRICS:
                result[f"{metric}_mean"] = stats["metrics"][metric]["mean"]
                result[f"{metric}_std"] = stats["metrics"][metric]["std"]
            summary.append(result)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/final.json")
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output.resolve()
    if not output.is_relative_to(HERE) or output.is_relative_to(HERE / "data"):
        parser.error("Output must stay inside this OOD folder and outside raw data")
    output.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            raise ValueError("Plot-only mode requires a completed evaluation")
        manifest["plot_settings"] = make_plots(output, manifest["config"])
        manifest.setdefault("evaluation_source_sha256", dict(manifest["source_sha256"]))
        manifest["source_sha256"] = {name: digest(HERE / name) for name in manifest["source_sha256"]}
        manifest["plots_regenerated_utc"] = datetime.now(timezone.utc).isoformat()
        for path in (output / "figures").iterdir():
            if path.suffix in (".png", ".pdf"):
                manifest["artifact_sha256"][str(path.relative_to(output))] = digest(path)
        write_json(output / "experiment_manifest.json", manifest)
        return
    device = setup_device(config, args.device)
    started = time.perf_counter()
    inventory, old_manifest = checkpoint_inventory(config)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": np.__version__, "sklearn": importlib.metadata.version("scikit-learn"),
                   "platform": platform.platform(), "device": str(device), "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "tf32": False, "amp": False, "deterministic_algorithms": True}
    print(json.dumps(environment), flush=True)
    print("Loading prepared datasets (read-only; no downloading)...", flush=True)
    datasets = load_datasets(config)
    order = dataset_metadata(datasets, output)
    sources = {name: digest(HERE / name) for name in (
        "data.py", "metrics.py", "inference.py", "run_experiment.py", "test_inference.py", "plots.py")}
    # Plot/report-only changes do not invalidate fully verified model probabilities.
    signature = json_hash({"config": config, "environment": environment,
                           "inference_sources": {name: sources[name] for name in ("inference.py", "data.py")},
                           "original_manifest_sha256": digest(config_path(config["checkpoint_root"]) / "manifest.json"),
                           "checkpoints": inventory,
                           "datasets": {name: record["order_sha256"] for name, record in order.items()}})
    manifest_path = output / "experiment_manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if old is not None and old["inference_signature"] != signature:
        raise ValueError("Existing run has a different inference signature; select a new --output inside OOD folder")
    manifest = {"status": "smoke_testing", "inference_signature": signature, "config": config,
                "environment": environment, "source_sha256": sources, "checkpoints": inventory,
                "original_training_manifest_sha256": digest(config_path(config["checkpoint_root"]) / "manifest.json"),
                "dataset_info_sha256": digest(HERE / "outputs/dataset_info.json"),
                "dataset_order": {name: {key: value for key, value in record.items()
                                         if key not in ("sample_ids", "labels")} for name, record in order.items()},
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "first_started_utc": old["first_started_utc"] if old else datetime.now(timezone.utc).isoformat(),
                "new_training": False, "resume_unit": "atomic seed/method/dataset NPZ + SHA-256 completion receipt",
                "rng_policy": "One stream seed per MC seed/dataset; 5 sequential passes; never reseed per forward/batch; isolated DataLoader CPU generator",
                "timing_scope": "Group inference seconds include checkpoint loading and small audit/repeat probes; exclude dataset loading, metrics, serialization, plotting, SHA-256 tree audit",
                "groups": []}
    write_json(manifest_path, manifest)
    try:
        tests = run_smoke(config, device, inventory, old_manifest, datasets, output)
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        with (config_path(config["checkpoint_root"]) / "raw_results.csv").open(newline="", encoding="utf-8") as stream:
            old_rows = list(csv.DictReader(stream))
        rows, comparisons = [], []
        new_inference_seconds, resumed = 0.0, 0
        for seed in config["seeds"]:
            for method in config["methods"]:
                stacks = {}
                for name, dataset in datasets.items():
                    metadata = group_metadata(seed, method, name, order[name], inventory, config)
                    path = cache_path(output, seed, method, name)
                    cached = read_group(path, signature, metadata, config)
                    if cached is None:
                        print(f"INFER seed={seed} {method} {name}: all {len(dataset)} images, M_max=5", flush=True)
                        stack, validation = infer_group(seed, method, name, dataset, inventory, old_manifest, config, device)
                        receipt = save_group(path, stack, metadata, validation, signature)
                        new_inference_seconds += validation["inference_seconds"]
                    else:
                        stack, receipt = cached
                        resumed += 1
                        print(f"RESUME verified seed={seed} {method} {name}", flush=True)
                    stacks[name] = stack
                    manifest["groups"].append({"seed": seed, "method": method, "dataset": name,
                                               "probabilities": str(path.relative_to(output)),
                                               "sha256": receipt["sha256"],
                                               "validation": receipt["validation"]})
                    write_json(manifest_path, manifest)
                for m in config["ensemble_sizes"]:
                    entropies = {}
                    for name, stack in stacks.items():
                        h = predictive_entropy(stack[:m])
                        entropies[name] = h
                        atomic_npy(output / "entropy" / f"seed_{seed}_{method}_{name}_M{m}.npy", h.numpy())
                    h_id, h_ood = entropies["mnist"], entropies["notmnist"]
                    rows.append({"run_seed": seed, "method": method, "M": m,
                                 "n_id": len(h_id), "n_ood": len(h_ood),
                                 "mean_id_entropy": mean_id_entropy(h_id), "mean_ood_entropy": mean_ood_entropy(h_ood),
                                 "entropy_gap": entropy_gap(h_id, h_ood), "ood_auroc": ood_auroc(h_id, h_ood)})
                    comparisons.append(compare_previous(stacks["mnist"], datasets["mnist"].tensors[1],
                                                        seed, method, m, config, old_rows))
                write_csv(output / "results_per_seed.csv", rows)
                write_csv(output / "classification_consistency.csv", comparisons)
        summary = make_summary(rows, config)
        if len(rows) != 36 or len(summary) != 12 or len(manifest["groups"]) != 24:
            raise AssertionError("Incomplete experiment grid")
        write_csv(output / "results_summary.csv", summary)
        full_checks = [{"name": f"full/{group['seed']}/{group['method']}/{group['dataset']}",
                        "passed": True, "validation": group["validation"]} for group in manifest["groups"]]
        full_checks.append({"name": "figure2a_deterministic_consistency", "passed": True,
                            "deterministic_comparisons": 27,
                            "all_36_within_tolerance": all(row["within_tolerance"] for row in comparisons),
                            "mc_dropout_comparisons": [row for row in comparisons if row["method"] == "mc_dropout"]})
        tests.update(full_evaluation_checks=full_checks, full_evaluation_passed=len(full_checks),
                     results_per_seed_rows=36, results_summary_rows=12, successful=True)
        write_json(output / "inference_test_results.json", tests)
        plot_settings = make_plots(output, config)
        manifest.update(status="verifying_integrity", result_rows=36, summary_rows=12,
                        total_inference_seconds=sum(group["validation"]["inference_seconds"] for group in manifest["groups"]),
                        new_inference_seconds_this_invocation=new_inference_seconds, resumed_groups=resumed,
                        smoke_seconds=tests["smoke_seconds"], plot_settings=plot_settings,
                        classification_comparisons_within_tolerance=sum(row["within_tolerance"] for row in comparisons))
        write_json(manifest_path, manifest)
        print("Checking SHA-256 of both complete protected experiment trees...", flush=True)
        subprocess.run([sys.executable, "-B", str(HERE / "verify_existing.py")], cwd=HERE.parent, check=True)
        integrity = json.loads((HERE / "outputs/protected_integrity.json").read_text(encoding="utf-8"))
        if not integrity["unchanged"]:
            raise AssertionError("Protected experiment files changed")
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        invocation_seconds=time.perf_counter() - started, protected_integrity=integrity,
                        artifact_sha256={str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*"))
                                         if path.is_file() and (path.suffix in (".csv", ".npy", ".png", ".pdf")
                                                                or path.name == "inference_test_results.json")})
        write_json(manifest_path, manifest)
        print(f"COMPLETE: 36 rows, 12 summaries; inference {manifest['total_inference_seconds']:.3f}s; {environment['gpu'] or device}", flush=True)
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=repr(exc), invocation_seconds=time.perf_counter() - started)
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
