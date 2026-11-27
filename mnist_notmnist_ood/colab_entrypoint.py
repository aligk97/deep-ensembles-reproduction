"""Colab storage/provenance driver reusing the unchanged OOD scientific functions."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import csv
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import torch

def never_train(*args, **kwargs):
    raise RuntimeError("OOD inference never permits optimizer steps or new model training")
torch.optim.Adam.step = never_train

from data import HERE, digest, write_json
from inference import (load_config, setup_device, checkpoint_inventory, load_datasets, infer_group,
                       cache_path, read_group, save_group, atomic_npy)
from metrics import predictive_entropy, mean_id_entropy, mean_ood_entropy, entropy_gap, ood_auroc
from run_experiment import json_hash, dataset_metadata, group_metadata, compare_previous, make_summary, write_csv
from test_inference import run_smoke
from plots import make_plots
from verify_outputs import verify as verify_outputs
from colab_source import inspect_result, verify_source_unchanged
from colab_data import load_reference_datasets

DRIVE_OOD = Path("/content/drive/MyDrive/DeepEnsembles_MNIST_NotMNIST_OOD")


def evaluate(output, source_info):
    output = Path(output).resolve()
    if not output.is_relative_to(DRIVE_OOD):
        raise ValueError("OOD results must stay in their separate persistent Drive directory")
    reference = json.loads((HERE / "colab_reference.json").read_text(encoding="utf-8"))
    config = load_config()
    if config != reference["ood_config"]:
        raise ValueError("The completed local OOD scientific configuration must remain unchanged")
    selected = json.loads(Path(source_info).read_text(encoding="utf-8"))
    source = inspect_result(selected["result_root"], selected["attestation"], reference)
    inventory, classification_manifest = checkpoint_inventory(config)
    if inventory != source["inventory"]:
        raise ValueError("Checkpoint bridge does not point to the explicitly verified Colab source")
    device = setup_device(config, "cuda")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": importlib.metadata.version("numpy"), "sklearn": importlib.metadata.version("scikit-learn"),
                   "platform": platform.platform(), "device": str(device), "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0), "tf32": False, "amp": False, "deterministic_algorithms": True}
    datasets = load_reference_datasets(config, reference, output)
    order = dataset_metadata(datasets, output)
    for name, item in order.items():
        if item["tensor_sha256"] != reference["dataset_tensor_sha256"][name]:
            raise ValueError(f"{name} preprocessing differs from the validated local dataset")
    sources = {name: digest(HERE / name) for name in reference["ood_scientific_source_sha256"]}
    if sources != reference["ood_scientific_source_sha256"]:
        raise ValueError("OOD scientific source code changed since the validated local run")
    drivers = {name: digest(HERE / name) for name in ("colab_source.py", "colab_runtime.py", "colab_entrypoint.py", "colab_data.py")}
    signature = json_hash({"config": config, "environment": environment, "sources": sources,
                           "colab_driver_sources": drivers,
                           "classifier_manifest_sha256": source["manifest_sha256"],
                           "classifier_audit_sha256": source["checkpoint_audit_sha256"],
                           "classifier_reference_sha256": source["figure2a_raw_sha256"],
                           "checkpoint_source_kind": source["attestation"]["kind"], "checkpoints": inventory,
                           "datasets": {name: item["order_sha256"] for name, item in order.items()}})
    manifest_path = output / "experiment_manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if old and old["inference_signature"] != signature:
        raise ValueError("OOD RUN_ID contains a different checkpoint/data/runtime signature; choose a new OOD_RUN_ID")
    manifest = {"status": "smoke_testing", "config": config, "environment": environment,
                "inference_signature": signature, "source_sha256": sources, "checkpoints": inventory,
                "colab_driver_source_sha256": drivers, "dataset_info_sha256": digest(output / "dataset_info.json"),
                "checkpoint_source": source, "original_training_manifest_sha256": source["manifest_sha256"],
                "dataset_order": {name: {k: v for k, v in item.items() if k not in ("sample_ids", "labels")}
                                  for name, item in order.items()},
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "first_started_utc": old["first_started_utc"] if old else datetime.now(timezone.utc).isoformat(),
                "new_training": False, "real_colab_inference_completed": False, "groups": [],
                "rng_policy": "Original member_seed + fixed dataset offset; seed once per stream; no per-forward reseeding",
                "resume_unit": "Verified seed/method/dataset probability cache",
                "timing_scope": "Checkpoint loading, small audit probes and inference; no dataset preparation/plotting"}
    write_json(manifest_path, manifest)
    try:
        tests = run_smoke(config, device, inventory, classification_manifest, datasets, output)
        tests["checkpoint_source"] = "verified_real_colab_drive_or_zip"
        tests["scope"] = "Verified real Colab final checkpoints; fixed 37 MNIST / 41 NotMNIST examples"
        write_json(output / "inference_test_results.json", tests)
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        with (Path(source["result_root"]) / "raw_results.csv").open(newline="", encoding="utf-8") as stream:
            previous_rows = list(csv.DictReader(stream))
        rows, comparisons = [], []
        resumed, new_seconds = 0, 0.0
        for seed in config["seeds"]:
            for method in config["methods"]:
                stacks = {}
                for name, dataset in datasets.items():
                    metadata = group_metadata(seed, method, name, order[name], inventory, config)
                    path = cache_path(output, seed, method, name)
                    restored = read_group(path, signature, metadata, config)
                    if restored is None:
                        print(f"COLAB INFER seed={seed} {method} {name}: {len(dataset)} images, M_max=5", flush=True)
                        stack, validation = infer_group(seed, method, name, dataset, inventory,
                                                        classification_manifest, config, device)
                        receipt = save_group(path, stack, metadata, validation, signature)
                        new_seconds += validation["inference_seconds"]
                    else:
                        stack, receipt = restored
                        resumed += 1
                        print(f"VERIFIED CACHE seed={seed} {method} {name}", flush=True)
                    stacks[name] = stack
                    manifest["groups"].append({"seed": seed, "method": method, "dataset": name,
                                               "probabilities": str(path.relative_to(output)), "sha256": receipt["sha256"],
                                               "validation": receipt["validation"]})
                    write_json(manifest_path, manifest)
                for m in config["ensemble_sizes"]:
                    entropies = {}
                    for name, stack in stacks.items():
                        entropies[name] = predictive_entropy(stack[:m])
                        atomic_npy(output / "entropy" / f"seed_{seed}_{method}_{name}_M{m}.npy", entropies[name].numpy())
                    hi, ho = entropies["mnist"], entropies["notmnist"]
                    rows.append({"run_seed": seed, "method": method, "M": m, "n_id": len(hi), "n_ood": len(ho),
                                 "mean_id_entropy": mean_id_entropy(hi), "mean_ood_entropy": mean_ood_entropy(ho),
                                 "entropy_gap": entropy_gap(hi, ho), "ood_auroc": ood_auroc(hi, ho)})
                    comparisons.append(compare_previous(stacks["mnist"], datasets["mnist"].tensors[1],
                                                        seed, method, m, config, previous_rows))
                write_csv(output / "results_per_seed.csv", rows)
                write_csv(output / "classification_consistency.csv", comparisons)
        summary = make_summary(rows, config)
        if len(rows) != 36 or len(summary) != 12 or len(manifest["groups"]) != 24:
            raise AssertionError("Incomplete Colab OOD grid")
        write_csv(output / "results_summary.csv", summary)
        full_checks = [{"name": f"full/{g['seed']}/{g['method']}/{g['dataset']}", "passed": True,
                        "validation": g["validation"]} for g in manifest["groups"]]
        full_checks.append({"name": "colab_figure2a_reference_consistency", "passed": True,
                            "deterministic_comparisons": 27,
                            "all_36_within_tolerance": all(row["within_tolerance"] for row in comparisons),
                            "mc_dropout_comparisons": [row for row in comparisons if row["method"] == "mc_dropout"]})
        tests.update(full_evaluation_checks=full_checks, full_evaluation_passed=len(full_checks),
                     results_per_seed_rows=36, results_summary_rows=12)
        write_json(output / "inference_test_results.json", tests)
        plot_settings = make_plots(output, config)
        verify_source_unchanged(source)
        manifest.update(status="complete", real_colab_inference_completed=True,
                        result_rows=36, summary_rows=12, smoke_seconds=tests["smoke_seconds"],
                        total_inference_seconds=sum(g["validation"]["inference_seconds"] for g in manifest["groups"]),
                        new_inference_seconds_this_invocation=new_seconds, resumed_groups=resumed,
                        classification_comparisons_within_tolerance=sum(row["within_tolerance"] for row in comparisons),
                        checkpoint_source_unchanged=True, plot_settings=plot_settings,
                        completed_utc=datetime.now(timezone.utc).isoformat(), invocation_seconds=time.perf_counter()-started,
                        artifact_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                                         if p.is_file() and (p.suffix in (".csv", ".npy", ".png", ".pdf")
                                                             or p.name == "inference_test_results.json")})
        write_json(manifest_path, manifest)
        verification = verify_outputs(output)
        manifest["independent_output_verification"] = {"successful": verification["successful"],
                                                       "passed": verification["passed"], "new_model_forward_passes": 0}
        write_json(manifest_path, manifest)
        print("REAL COLAB OOD COMPLETE: 36 seed rows / 12 summaries; no new training", flush=True)
    except BaseException as exc:
        manifest.update(status="interrupted_or_failed", error=repr(exc), real_colab_inference_completed=False,
                        invocation_seconds=time.perf_counter()-started)
        write_json(manifest_path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-info", type=Path, required=True)
    args = parser.parse_args()
    try:
        real_colab = importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        real_colab = False
    if not real_colab or os.environ.get("OOD_COLAB_VALIDATE_ONLY") == "1":
        raise RuntimeError("Actual checkpoint inference is disabled during local notebook preparation")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU runtime")
    evaluate(args.output, args.source_info)


if __name__ == "__main__":
    main()
