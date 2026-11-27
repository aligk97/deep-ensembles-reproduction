"""Checkpoint validation/resume orchestration; original training math is untouched."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from engine import load_checkpoint, member_seed, atomic_json
from model import MLP

CORE = ("data.py", "model.py", "metrics.py", "engine.py", "run_experiment.py")
METHODS = ("ensemble", "random", "adversarial", "mc_dropout")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference(root):
    root = Path(root)
    reference = json.loads((root / "final_reference.json").read_text(encoding="utf-8"))
    config = json.loads((root / "configs/local.json").read_text(encoding="utf-8"))
    if config != reference["config"]:
        raise ValueError("Only the completed local experiment configuration is allowed")
    if (config["seeds"] != [42, 123, 2026] or config["ensemble_sizes"] != [1, 2, 3, 4, 5]
            or config["epochs"] != 10 or config["validation_size"] != 5000
            or config["split_seed"] != 4509 or config["train_limit"] or config["test_limit"]):
        raise ValueError("Unexpected scientific experiment scope")
    sources = {name: sha256(root / name) for name in CORE}
    if sources != reference["source_sha256"]:
        raise ValueError("Original scientific sources differ from completed local training")
    signature = hashlib.sha256(json.dumps({"config": config, "sources": sources}, sort_keys=True).encode()).hexdigest()
    if signature != reference["signature"]:
        raise ValueError("Training signature differs from the completed local experiment")
    return config, signature


def checkpoint_records(root, config):
    records = []
    for seed in config["seeds"]:
        for method in METHODS:
            for member in range(1 if method == "mc_dropout" else 5):
                path = Path(root) / "checkpoints" / f"seed_{seed}" / method / f"member_{member:02d}.pt"
                records.append((seed, method, member, path))
    return records


def validate_checkpoint(path, config, signature, method, seed, member, require_complete=False):
    """Read only; check provenance, strict architecture, optimizer, history and RNG.

    Local files contain Python/NumPy RNG objects, so the original trusted-local
    weights_only=False loader is intentionally reused. No optimizer step runs.
    """
    state = load_checkpoint(path, "cpu")
    if (state["signature"] != signature or state["config"] != config
            or state["member_seed"] != member_seed(seed, member, method)):
        raise ValueError(f"Checkpoint signature/config/member mismatch: {path}")
    epoch = state["epoch"]
    if not isinstance(epoch, int) or not 1 <= epoch <= config["epochs"]:
        raise ValueError("Checkpoint epoch is outside the authorized budget")
    history = state["history"]
    if len(history) != epoch:
        raise ValueError("Checkpoint history length does not match epoch")
    for index, row in enumerate(history, 1):
        if (row["epoch"], row["run_seed"], row["method"], row["member"], row["member_seed"]) != (
                index, seed, method, member, member_seed(seed, member, method)):
            raise ValueError("Checkpoint history belongs to another model or misses epochs")
        if not all(np.isfinite(row[key]) for key in ("objective_loss", "clean_loss", "perturbed_loss",
                                                     "train_accuracy", "validation_loss", "validation_accuracy", "epoch_seconds")):
            raise ValueError("Non-finite checkpoint history")
    model = MLP(config["dropout"] if method == "mc_dropout" else 0.0)
    model.load_state_dict(state["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in state["model"].values()):
        raise ValueError("Non-finite model parameters or BN buffers")
    # Loading optimizer state validates param groups without updating weights.
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    optimizer.load_state_dict(state["optimizer"])
    expected_steps = epoch * ((60000 - config["validation_size"] + config["batch_size"] - 1) // config["batch_size"])
    if len(optimizer.state) != len(list(model.parameters())):
        raise ValueError("Checkpoint misses trained optimizer parameter states")
    for parameter, entry in optimizer.state.items():
        if (float(entry["step"]) != expected_steps or entry["exp_avg"].shape != parameter.shape
                or entry["exp_avg_sq"].shape != parameter.shape or (entry["exp_avg_sq"] < 0).any()):
            raise ValueError("Optimizer step count/moment shapes do not match the saved epoch")
    for group in optimizer.param_groups:
        if (group["lr"] != config["learning_rate"] or tuple(group["betas"]) != (.9, .999)
                or group["eps"] != 1e-8 or group["weight_decay"] != 0):
            raise ValueError("Optimizer hyperparameters changed")
    for entry in state["optimizer"]["state"].values():
        if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all() for value in entry.values()):
            raise ValueError("Non-finite optimizer state")
    rng = state["rng"]
    torch.Generator().set_state(rng["torch"].cpu())
    torch.Generator().set_state(rng["shuffle"].cpu())
    np.random.RandomState().set_state(rng["numpy"])
    random.Random().setstate(rng["python"])
    if any(value.dtype != torch.uint8 or value.ndim != 1 for value in rng["cuda"]):
        raise ValueError("Invalid CUDA RNG state")
    if require_complete and epoch != config["epochs"]:
        raise ValueError("Checkpoint is not complete")
    return state


def make_train_member(original, allowed_root, allow_training, events):
    """Skip validated complete models; resume original code only for partial ones."""
    allowed_root = Path(allowed_root).resolve()

    def guarded(config, signature, output, method, seed, member, tx, ty, vx, vy, epsilon, stop_after_epoch=None):
        import pandas as pd
        output = Path(output).resolve()
        if output != allowed_root:
            raise ValueError("Checkpoint output escaped the authorized Colab result directory")
        if stop_after_epoch is not None:
            raise ValueError("The final Colab entrypoint never changes the epoch budget")
        path = output / "checkpoints" / f"seed_{seed}" / method / f"member_{member:02d}.pt"
        previous_epoch = 0
        if path.exists():
            state = validate_checkpoint(path, config, signature, method, seed, member)
            previous_epoch = state["epoch"]
            if previous_epoch == config["epochs"]:
                before = sha256(path)
                history = output / "histories" / f"seed_{seed}_{method}_{member:02d}.csv"
                history.parent.mkdir(parents=True, exist_ok=True)
                temporary = history.with_suffix(".csv.tmp")
                pd.DataFrame(state["history"]).to_csv(temporary, index=False)
                temporary.replace(history)
                if sha256(path) != before:
                    raise AssertionError("Completed checkpoint changed while being verified")
                events.append({"seed": seed, "method": method, "member": member,
                               "action": "verified_complete_skipped", "epoch": previous_epoch, "sha256": before})
                atomic_json(output / "colab_resume_audit.json", events)
                print(f"seed={seed} {method} member={member+1} epoch={previous_epoch}/10 VERIFIED COMPLETE - no training", flush=True)
                return path
        if not allow_training:
            raise RuntimeError("Training is prohibited in local compatibility validation")
        print(f"seed={seed} {method} member={member+1} START epoch={previous_epoch+1}/10", flush=True)
        result = original(config, signature, output, method, seed, member, tx, ty, vx, vy, epsilon)
        validate_checkpoint(result, config, signature, method, seed, member, require_complete=True)
        events.append({"seed": seed, "method": method, "member": member,
                       "action": "resumed_partial" if previous_epoch else "trained_new",
                       "resumed_after_epoch": previous_epoch, "epoch": config["epochs"], "sha256": sha256(result)})
        atomic_json(output / "colab_resume_audit.json", events)
        return result

    return guarded
