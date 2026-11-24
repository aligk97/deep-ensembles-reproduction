"""Independent training, epoch-boundary exact resume, and probability prediction."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model import MLP, evaluation_mode

METHODS = ("ensemble", "random", "adversarial", "mc_dropout")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def member_seed(run_seed, member, method):
    # Paired initializations across the three ensemble training treatments.
    return run_seed * 10000 + (9001 if method == "mc_dropout" else member + 1)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def save_checkpoint(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path, device="cpu"):
    # Local checkpoints contain optimizer and RNG states. Never load untrusted .pt files.
    return torch.load(path, map_location=device, weights_only=False)


def rng_state(generator):
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(), "shuffle": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    generator.set_state(state["shuffle"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def perturb(x, epsilon, method, gradient=None):
    if method == "adversarial":
        direction = gradient.detach().sign()
    elif method == "random":
        direction = torch.randint(0, 2, x.shape, device=x.device).mul(2).sub(1)
    else:
        raise ValueError(method)
    return (x.detach() + epsilon * direction).clamp(0, 1).detach()


def training_step(model, optimizer, x, y, epsilon, method):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x = x.detach().requires_grad_(method == "adversarial")
    logits = model(x)
    clean_loss = F.cross_entropy(logits, y)
    objective = clean_loss
    perturbed_loss = torch.zeros((), device=x.device)
    if method in ("random", "adversarial"):
        gradient = (torch.autograd.grad(clean_loss, x, retain_graph=True)[0]
                    if method == "adversarial" else None)
        xp = perturb(x, epsilon, method, gradient)
        perturbed_loss = F.cross_entropy(model(xp), y)
        # Algorithm 1 uses a sum, not a half-weighted mean.
        objective = clean_loss + perturbed_loss
    if not torch.isfinite(objective):
        raise FloatingPointError("Non-finite training objective")
    objective.backward()
    optimizer.step()
    return torch.stack([objective.detach(), clean_loss.detach(), perturbed_loss.detach(),
                        (logits.argmax(1) == y).float().mean()])


@torch.inference_mode()
def predict(model, x, batch_size=1000, mc_dropout=False):
    evaluation_mode(model, mc_dropout)
    return torch.cat([model(batch).softmax(1).cpu() for batch in x.split(batch_size)])


@torch.inference_mode()
def validation(model, x, y, batch_size):
    evaluation_mode(model)
    loss, correct = 0.0, 0
    for xb, yb in zip(x.split(batch_size), y.split(batch_size)):
        logits = model(xb)
        loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return loss / len(y), correct / len(y)


def train_member(config, signature, output, method, run_seed, member,
                 train_x, train_y, valid_x, valid_y, epsilon, stop_after_epoch=None):
    seed = member_seed(run_seed, member, method)
    seed_all(seed)
    device = train_x.device
    model = MLP(config["dropout"] if method == "mc_dropout" else 0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    shuffle = torch.Generator().manual_seed(seed + 100000000)
    folder = output / "checkpoints" / f"seed_{run_seed}" / method
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"member_{member:02d}.pt"
    history_path = output / "histories" / f"seed_{run_seed}_{method}_{member:02d}.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    start, history = 0, []
    if path.exists():
        state = load_checkpoint(path)
        if state["signature"] != signature or state["member_seed"] != seed:
            raise ValueError(f"Checkpoint configuration/source mismatch: {path}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, history = state["epoch"], state["history"]
        restore_rng(state["rng"], shuffle)
    import pandas as pd
    if start >= config["epochs"]:
        pd.DataFrame(history).to_csv(history_path, index=False)
        return path
    end = min(config["epochs"], stop_after_epoch or config["epochs"])
    for epoch in range(start, end):
        tic = time.perf_counter()
        indices = list(torch.randperm(len(train_y), generator=shuffle).split(config["batch_size"]))
        # BatchNorm needs at least two examples. Merge any singleton final batch.
        if len(indices[-1]) == 1:
            indices[-2] = torch.cat([indices[-2], indices[-1]])
            indices.pop()
        totals = torch.zeros(4, device=device)
        for idx in indices:
            idx = idx.to(device)
            totals += training_step(model, optimizer, train_x[idx], train_y[idx], epsilon, method) * len(idx)
        objective, clean_loss, pert_loss, accuracy = (totals / len(train_y)).tolist()
        vl, va = validation(model, valid_x, valid_y, config["eval_batch_size"])
        elapsed = time.perf_counter() - tic
        history.append({"run_seed": run_seed, "member_seed": seed, "method": method,
                        "member": member, "epoch": epoch + 1, "objective_loss": objective,
                        "clean_loss": clean_loss, "perturbed_loss": pert_loss,
                        "train_accuracy": accuracy, "validation_loss": vl,
                        "validation_accuracy": va, "epoch_seconds": elapsed})
        save_checkpoint(path, {"signature": signature, "member_seed": seed,
                               "epoch": epoch + 1, "model": model.state_dict(),
                               "optimizer": optimizer.state_dict(), "history": history,
                               "rng": rng_state(shuffle), "config": config})
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"seed={run_seed} {method} member={member + 1} epoch={epoch + 1}/{config['epochs']} "
              f"clean_CE={clean_loss:.4f} train_acc={accuracy:.4f} val_acc={va:.4f} "
              f"{elapsed:.2f}s", flush=True)
    return path
