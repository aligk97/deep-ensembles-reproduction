"""Strict loading and reproducible read-only inference on existing local models."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import hashlib
import importlib.util
import json
import runpy
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

# Also protect the old source tree when callers omit python -B.
sys.dont_write_bytecode = True

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

if __package__:
    from .data import HERE, digest, make_loader, write_json, load_mnist_test, NotMNISTSmall
    from .metrics import predictive_entropy
else:
    from data import HERE, digest, make_loader, write_json, load_mnist_test, NotMNISTSmall
    from metrics import predictive_entropy

METHOD_NAMES = {"ensemble": "Ensemble", "random": "Ensemble + R",
                "adversarial": "Ensemble + AT", "mc_dropout": "MC Dropout 0.1"}
ORIGINAL = HERE.parent / "mnist_classification"
_ORIGINAL_API = None


def original_api():
    """Reuse original MLP, evaluation_mode, loader, seeding and scoring code.

    Temporarily resolve engine.py's absolute 'model' import, without inserting
    the old directory on sys.path or colliding with this stage's data/metrics.
    No training function is invoked and no old file is written.
    """
    global _ORIGINAL_API
    if _ORIGINAL_API is None:
        spec = importlib.util.spec_from_file_location("_ood_original_mnist_model", ORIGINAL / "model.py")
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        previous = sys.modules.get("model")
        try:
            sys.modules["model"] = model
            engine = runpy.run_path(str(ORIGINAL / "engine.py"))
        finally:
            if previous is None:
                sys.modules.pop("model", None)
            else:
                sys.modules["model"] = previous
        scoring = runpy.run_path(str(ORIGINAL / "metrics.py"))
        _ORIGINAL_API = SimpleNamespace(MLP=model.MLP, evaluation_mode=model.evaluation_mode,
                                       load_checkpoint=engine["load_checkpoint"],
                                       seed_all=engine["seed_all"], member_seed=engine["member_seed"],
                                       predict=engine["predict"],
                                       classification_metrics=scoring["classification_metrics"])
    return _ORIGINAL_API


def config_path(value):
    path = Path(value)
    return (HERE / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path=None):
    config = json.loads(Path(path or HERE / "configs/final.json").read_text(encoding="utf-8"))
    if (config["seeds"] != [42, 123, 2026]
            or config["methods"] != list(METHOD_NAMES)
            or config["ensemble_sizes"] != [1, 3, 5] or config["dropout"] != 0.1
            or config["batch_size"] <= 0 or config["num_workers"] != 0
            or not config["deterministic"]):
        raise ValueError("Configuration must preserve the authorized final experiment")
    return config


def setup_device(config, choice="auto"):
    device = torch.device("cuda" if choice == "auto" and torch.cuda.is_available()
                          else "cpu" if choice == "auto" else choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def checkpoint_inventory(config):
    root = config_path(config["checkpoint_root"])
    # Hard guard against accidentally evaluating smoke / unfinished runs.
    if root != (ORIGINAL / "outputs/local").resolve():
        raise ValueError("Only the completed mnist_classification/outputs/local run is allowed")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["trained_models"] != 48:
        raise ValueError("Expected a complete 48-model local run")
    for name, expected in manifest["source_sha256"].items():
        if digest(ORIGINAL / name) != expected:
            raise ValueError(f"Original source changed since training: {name}")
    inventory = []
    for seed in config["seeds"]:
        for method in config["methods"]:
            for member in range(1 if method == "mc_dropout" else 5):
                path = root / "checkpoints" / f"seed_{seed}" / method / f"member_{member:02d}.pt"
                inventory.append({"run_seed": seed, "method": method, "member": member,
                                  "path": str(path), "sha256": digest(path)})
    actual = {p.resolve() for p in (root / "checkpoints").rglob("*.pt")}
    if actual != {Path(row["path"]) for row in inventory} or len(actual) != 48:
        raise ValueError("Unexpected checkpoint set")
    return inventory, manifest


def load_model(record, manifest, device):
    api = original_api()
    state = api.load_checkpoint(record["path"])
    if (state["signature"] != manifest["signature"]
            or state["config"] != manifest["config"] or state["config"]["name"] != "local"
            or state["epoch"] != 10 or state["config"]["train_limit"] != 0
            or state["member_seed"] != api.member_seed(record["run_seed"], record["member"], record["method"])):
        raise ValueError(f"Checkpoint provenance mismatch: {record['path']}")
    last = state["history"][-1]
    if (last["run_seed"], last["method"], last["member"], last["epoch"]) != (
            record["run_seed"], record["method"], record["member"], 10):
        raise ValueError("Checkpoint history does not identify the requested final model")
    mc = record["method"] == "mc_dropout"
    model = api.MLP(0.1 if mc else 0.0).to(device)
    model.load_state_dict(state["model"], strict=True)
    api.evaluation_mode(model, mc_dropout=mc)
    dropouts = [layer for layer in model.modules() if isinstance(layer, nn.Dropout)]
    if mc and (len(dropouts) != 3 or any(layer.p != 0.1 or not layer.training for layer in dropouts)):
        raise AssertionError("Expected three active p=0.1 Dropout layers")
    if not mc and (model.training or dropouts):
        raise AssertionError("Ensemble must be in evaluation mode without dropout")
    if any(layer.training for layer in model.modules() if isinstance(layer, nn.modules.batchnorm._BatchNorm)):
        raise AssertionError("BatchNorm must stay in evaluation mode")
    return model


def buffer_snapshot(model):
    # Includes running_mean, running_var AND num_batches_tracked.
    return {name: value.detach().clone() for name, value in model.named_buffers()}


def assert_buffers_unchanged(model, before):
    if any(not torch.equal(before[name], value) for name, value in model.named_buffers()):
        raise AssertionError("BatchNorm buffers changed during inference")


def loader_for(dataset, config):
    loader = make_loader(dataset, config["batch_size"], config["num_workers"])
    # DataLoader creates a base_seed even with workers=0; isolate that CPU RNG
    # from the dropout stream (important for CPU parity with the old predictor).
    loader.generator = torch.Generator().manual_seed(config["loader_rng_seed"])
    return loader


@torch.inference_mode()
def predict_pass(model, dataset, config, device):
    chunks = []
    seen = 0
    for x, _ in loader_for(dataset, config):
        logits = model(x.to(device))
        if logits.shape != (len(x), 10):
            raise AssertionError("Unexpected model output shape")
        chunks.append(logits.softmax(dim=-1).cpu())
        seen += len(x)
    if seen != len(dataset):
        raise AssertionError("DataLoader did not cover the complete dataset")
    return torch.cat(chunks)


def mc_stream_seed(seed, dataset_name, config):
    if dataset_name not in ("mnist", "notmnist"):
        raise ValueError(dataset_name)
    return (original_api().member_seed(seed, 0, "mc_dropout") + config["mc_rng_offset"]
            + (config["notmnist_rng_offset"] if dataset_name == "notmnist" else 0))


def rng_digest(device):
    state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
    return hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()


def validate_stack(stack, n, config):
    if stack.shape != (5, n, 10) or stack.dtype != torch.float32:
        raise AssertionError(f"Expected float32 [5,{n},10], got {stack.shape}, {stack.dtype}")
    if not torch.isfinite(stack).all() or (stack < 0).any() or (stack > 1).any():
        raise AssertionError("Invalid class probabilities")
    error = float((stack.double().sum(-1) - 1).abs().max())
    if error > config["probability_sum_atol"]:
        raise AssertionError(f"Class probabilities do not sum to 1: {error}")
    bounds = {}
    for m in config["ensemble_sizes"]:
        h = predictive_entropy(stack[:m])
        bounds[str(m)] = {"shape": list(h.shape), "min_nat": float(h.min()), "max_nat": float(h.max())}
    return {"shape": list(stack.shape), "dtype": "float32", "nan_count": 0, "inf_count": 0,
            "max_probability_sum_error": error, "entropy_by_M": bounds}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer_group(seed, method, dataset_name, dataset, inventory, manifest, config, device):
    records = [row for row in inventory if row["run_seed"] == seed and row["method"] == method]
    samples, checks = [], []
    sync(device)
    started = time.perf_counter()
    rng_states = []
    for record in records:
        model = load_model(record, manifest, device)
        before = buffer_snapshot(model)
        if method == "mc_dropout":
            # Seed ONCE at stream start, never inside the pass or batch loop.
            original_api().seed_all(mc_stream_seed(seed, dataset_name, config))
            for _ in range(5):
                rng_states.append(rng_digest(device))
                samples.append(predict_pass(model, dataset, config, device))
        else:
            probe = Subset(dataset, range(min(17, len(dataset))))
            first = predict_pass(model, probe, config, device)
            second = predict_pass(model, probe, config, device)
            if not torch.equal(first, second):
                raise AssertionError("Deterministic ensemble produced different repeated predictions")
            probe_x = torch.stack([probe[index][0] for index in range(len(probe))]).to(device)
            reference = original_api().predict(model, probe_x, config["batch_size"], False)
            if not torch.equal(first, reference):
                raise AssertionError("DataLoader predictor differs from original engine.predict")
            samples.append(predict_pass(model, dataset, config, device))
        assert_buffers_unchanged(model, before)
        checks.append({"checkpoint": record["path"], "strict_loaded": True, "bn_buffers_unchanged": True,
                       "deterministic_repeat_equal": True if method != "mc_dropout" else None,
                       "original_predictor_equal": True if method != "mc_dropout" else None})
        del model
    sync(device)
    inference_seconds = time.perf_counter() - started
    stack = torch.stack(samples)
    validation = validate_stack(stack, len(dataset), config)
    if method == "mc_dropout":
        unequal = [not torch.equal(stack[0], stack[index]) for index in range(1, 5)]
        if not all(unequal) or len(set(rng_states)) != 5:
            raise AssertionError("MC Dropout passes did not advance RNG / produce different predictions")
        validation.update(mc_passes_differ=True, mc_rng_state_sha256_before_pass=rng_states,
                          mc_stream_seed=mc_stream_seed(seed, dataset_name, config))
    validation.update(models=checks, inference_seconds=inference_seconds)
    return stack, validation


def load_datasets(config):
    # No download/extract function is called in this experiment.
    mnist = load_mnist_test(config_path(config["mnist_root"]))
    notmnist = NotMNISTSmall(config_path(config["notmnist_root"]))
    if len(mnist) != 10000 or len(notmnist) != 18724:
        raise AssertionError("Expected every prepared test example")
    return {"mnist": mnist, "notmnist": notmnist}


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def atomic_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def cache_path(output, seed, method, dataset_name):
    return Path(output) / "probabilities" / f"seed_{seed}_{method}_{dataset_name}.npz"


def save_group(path, stack, metadata, validation, signature):
    atomic_npz(path, probabilities=stack.numpy(), metadata_json=np.array(json.dumps(metadata)),
               labels=np.asarray(metadata["labels"], dtype=np.int64),
               example_indices=np.arange(stack.shape[1], dtype=np.int64))
    receipt = {"signature": signature, "sha256": digest(path), "metadata": metadata,
               "validation": validation, "status": "complete"}
    # Receipt is the completion marker, committed only after the full atomic NPZ.
    write_json(path.with_suffix(".json"), receipt)
    return receipt


def read_group(path, signature, metadata, config):
    receipt_path = path.with_suffix(".json")
    if not path.exists() or not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["signature"] != signature or receipt["metadata"] != metadata:
        raise ValueError(f"Cache belongs to a different experiment: {path}")
    if receipt["status"] != "complete" or digest(path) != receipt["sha256"]:
        raise ValueError(f"Cache content failed integrity verification: {path}")
    with np.load(path, allow_pickle=False) as stored:
        if (json.loads(str(stored["metadata_json"])) != metadata
                or not np.array_equal(stored["example_indices"], np.arange(metadata["n_examples"]))
                or not np.array_equal(stored["labels"], metadata["labels"])):
            raise ValueError("Cached example ordering/provenance mismatch")
        stack = torch.from_numpy(stored["probabilities"].copy())
    validate_stack(stack, metadata["n_examples"], config)
    return stack, receipt
