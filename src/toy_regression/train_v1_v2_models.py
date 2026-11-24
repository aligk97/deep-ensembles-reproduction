import csv
from dataclasses import dataclass
from time import perf_counter

import torch
from torch.utils.data import DataLoader, TensorDataset

from .regression_losses import gaussian_nll, mse_loss
from .v1_v2_regression_networks import GaussianRegressor, MSERegressor
from .experiment_io import seed_all, write_json


@dataclass(frozen=True)
class Condition:
    name: str
    title: str
    gaussian: bool
    ensemble: bool = False
    adversarial: bool = False


CONDITIONS = (
    Condition("mse_ensemble_5", "MSE ensemble (5)", False, ensemble=True),
    Condition("nll_single", "Single NLL", True),
    Condition("nll_single_at", "Single NLL + AT", True, adversarial=True),
    Condition("nll_ensemble_5", "NLL ensemble (5)", True, ensemble=True),
)


def member_seeds(outer_seed, member_index):
    model_seed = outer_seed * 1000 + member_index
    return model_seed, model_seed + 1_000_000_000_000


def initialize_member(condition, config, model_seed):
    seed_all(model_seed)
    if condition.gaussian:
        return GaussianRegressor(config.hidden_dim, config.variance_floor)
    return MSERegressor(config.hidden_dim)


def fgsm_inputs(model, x, y, epsilon):
    source = x.detach().clone().requires_grad_(True)
    clean_nll = gaussian_nll(*model(source), y)
    gradient, = torch.autograd.grad(clean_nll, source)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Non-finite FGSM input gradient")
    return (source + epsilon * gradient.sign()).detach()


def training_losses(condition, model, x, y, epsilon):
    if not condition.gaussian:
        return mse_loss(model(x), y), None
    clean = gaussian_nll(*model(x), y)
    if condition.adversarial:
        adv = fgsm_inputs(model, x, y, epsilon)
        return clean, gaussian_nll(*model(adv), y)
    return clean, None


def train_member(condition, config, split, device, epsilon, seed, index, path, digest):
    model_seed, shuffle_seed = member_seeds(seed, index)
    model = initialize_member(condition, config, model_seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    parameters = tuple(model.parameters())
    generator = torch.Generator().manual_seed(shuffle_seed)
    loader = DataLoader(TensorDataset(split.x, split.y_noisy), batch_size=config.batch_size,
                        shuffle=config.shuffle, generator=generator)
    metadata = {
        "outer_seed": seed, "model_seed": model_seed, "shuffle_seed": shuffle_seed,
        "condition": condition.name, "member": index, "config_hash": digest,
        "architecture": model.__class__.__name__, "hidden_dim": config.hidden_dim,
        "variance_floor": config.variance_floor if condition.gaussian else None,
        "fgsm_epsilon": epsilon if condition.adversarial else None,
        "effective_batch_size": min(config.batch_size, config.train_size),
        "shuffle_changes_minibatch_composition": config.shuffle and config.batch_size < config.train_size,
        "train_data": "../data/train.npz",
    }
    write_json(path / "histories" / f"member_{index}_seeds.json", metadata)
    columns = ["epoch", "total_loss", "clean_loss", "adversarial_loss",
               "learning_rate", "duration_seconds", "finite"]
    with (path / "histories" / f"member_{index}.csv").open("w", encoding="utf-8", newline="") as stream:
        history = csv.DictWriter(stream, fieldnames=columns)
        history.writeheader()
        for epoch in range(1, config.max_epochs + 1):
            start = perf_counter()
            totals = torch.zeros(2, device=device)
            batch = 0
            try:
                for batch, (x, y) in enumerate(loader, 1):
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    clean, adv = training_losses(condition, model, x, y, epsilon)
                    loss = clean if adv is None else clean + adv
                    loss.backward()
                    if not torch.isfinite(torch.cat(
                        (loss.detach().reshape(1), *(p.grad.reshape(-1) for p in parameters))
                    )).all():
                        raise FloatingPointError("Non-finite training loss or gradient")
                    if config.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm,
                                                       error_if_nonfinite=True)
                    optimizer.step()
                    totals[0] += clean.detach() * len(x)
                    if adv is not None:
                        totals[1] += adv.detach() * len(x)
                if not torch.isfinite(torch.cat((totals, *(p.detach().flatten() for p in parameters)))).all():
                    raise FloatingPointError("Non-finite epoch loss or parameters")
            except (FloatingPointError, RuntimeError) as error:
                history.writerow({"epoch": epoch, "duration_seconds": perf_counter() - start,
                                  "learning_rate": optimizer.param_groups[0]["lr"], "finite": False})
                raise RuntimeError(f"{condition.name}, seed {seed}, member {index}, "
                                   f"epoch {epoch}, batch {batch}: {error}") from error
            clean_epoch, adv_epoch = (totals / len(split.x)).tolist()
            history.writerow({
                "epoch": epoch, "total_loss": clean_epoch + adv_epoch, "clean_loss": clean_epoch,
                "adversarial_loss": adv_epoch if condition.adversarial else None,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "duration_seconds": perf_counter() - start, "finite": True,
            })

    if config.save_checkpoints:
        torch.save({
            **metadata, "final_epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "shuffle_rng_state": generator.get_state(),
        }, path / "models" / f"member_{index}.pt")
    return model.eval()


def train_condition(condition, config, split, device, epsilon, seed, path, digest, log):
    (path / "histories").mkdir(parents=True)
    if config.save_checkpoints:
        (path / "models").mkdir()
    models = []
    count = config.ensemble_size if condition.ensemble else 1
    for index in range(count):
        log.info("Seed %s | %s | training member %s/%s", seed, condition.name, index + 1, count)
        models.append(train_member(condition, config, split, device, epsilon, seed, index, path, digest))
    return models
