"""Real local-checkpoint smoke tests, run automatically before full evaluation."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import Subset, SequentialSampler

if __package__:
    from .inference import (HERE, load_config, setup_device, checkpoint_inventory, load_datasets,
                            infer_group, load_model, original_api, loader_for, save_group, read_group)
    from .data import write_json
else:
    from inference import (HERE, load_config, setup_device, checkpoint_inventory, load_datasets,
                           infer_group, load_model, original_api, loader_for, save_group, read_group)
    from data import write_json


def run_smoke(config, device, inventory, old_manifest, datasets, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"successful": False, "scope": "Real final local checkpoints; fixed 37 MNIST / 41 NotMNIST examples",
              "device": str(device), "unique_checkpoints": len(inventory), "checks": []}
    started = time.perf_counter()
    probe_config = {**config, "batch_size": 16}
    probes = {name: Subset(dataset, range(37 if name == "mnist" else 41)) for name, dataset in datasets.items()}
    try:
        first_mc, first_ensemble = {}, None
        for seed in config["seeds"]:
            for method in config["methods"]:
                for name, dataset in probes.items():
                    stack, validation = infer_group(seed, method, name, dataset, inventory, old_manifest,
                                                     probe_config, device)
                    report["checks"].append({"name": f"seed_{seed}/{method}/{name}", "passed": True,
                                             "validation": validation})
                    if seed == 42 and method == "mc_dropout":
                        first_mc[name] = stack
                    if seed == 42 and method == "ensemble" and name == "mnist":
                        first_ensemble = stack
        # Replay a whole dropout stream, not resetting before each forward pass.
        for name, expected in first_mc.items():
            actual, _ = infer_group(42, "mc_dropout", name, probes[name], inventory, old_manifest,
                                     probe_config, device)
            if not torch.equal(expected, actual):
                raise AssertionError("MC stream is not reproducible from its recorded initial seed")
            report["checks"].append({"name": f"mc_rng_replay/{name}", "passed": True})
        loader = loader_for(probes["mnist"], probe_config)
        sizes = [len(x) for x, _ in loader]
        if (not isinstance(loader.sampler, SequentialSampler) or loader.drop_last
                or sizes != [16, 16, 5]):
            raise AssertionError("Loader must preserve order and the partial batch")
        report["checks"].append({"name": "sequential_loader_partial_batch", "passed": True, "batch_sizes": sizes})
        model = load_model(inventory[0], old_manifest, device)
        broken_state = dict(model.state_dict())
        broken_state.pop(next(iter(broken_state)))
        try:
            model.load_state_dict(broken_state, strict=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Strict loading did not reject a missing key")
        del model
        report["checks"].append({"name": "strict_load_rejects_missing_key", "passed": True})
        metadata = {"n_examples": 37, "labels": [int(probes["mnist"][i][1]) for i in range(37)],
                    "scope": "cache smoke test"}
        with tempfile.TemporaryDirectory(dir=output) as directory:
            path = Path(directory) / "probe.npz"
            save_group(path, first_ensemble, metadata, {}, "smoke_signature")
            restored, _ = read_group(path, "smoke_signature", metadata, probe_config)
            if not torch.equal(restored, first_ensemble):
                raise AssertionError("Resume round-trip changed probabilities")
            with path.open("ab") as stream:
                stream.write(b"tamper")
            try:
                read_group(path, "smoke_signature", metadata, probe_config)
            except ValueError:
                pass
            else:
                raise AssertionError("Modified cached data was silently accepted")
        report["checks"].append({"name": "atomic_cache_roundtrip_and_tamper_detection", "passed": True})
        report.update(successful=True, passed=len(report["checks"]), failed=0)
    except BaseException as exc:
        report.update(error=repr(exc), failed=1, passed=len(report["checks"]))
        raise
    finally:
        report["smoke_seconds"] = time.perf_counter() - started
        write_json(output / "inference_test_results.json", report)
    print(f"INFERENCE SMOKE PASSED: {report['passed']} checks; all 48 final checkpoints", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/final.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    device = setup_device(config, args.device)
    inventory, manifest = checkpoint_inventory(config)
    datasets = load_datasets(config)
    run_smoke(config, device, inventory, manifest, datasets, HERE / "outputs")


if __name__ == "__main__":
    main()
