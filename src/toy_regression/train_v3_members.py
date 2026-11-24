"""Train independent members in parallel with the project's Figure 1 networks and losses."""
import torch
from torch.func import functional_call, stack_module_state, vmap
from torch.utils.data import DataLoader

from .ensemble_predictions import predict, save_predictions
from .experiment_io import write_csv
from .generate_cubic_data import fgsm_epsilon
from .regression_losses import gaussian_nll, mse_loss
from .train_v1_v2_models import initialize_member, member_seeds


def atomic_save(value, path):
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def train_group(folder, arm, condition, config, splits, device):
    count = config.ensemble_size if condition.ensemble else 1
    members = [(seed, index) for seed in config.outer_seeds for index in range(count)]
    models = [initialize_member(condition, config, member_seeds(seed, i)[0])
              for seed, i in members]
    params, _ = stack_module_state(models)
    params = {k: torch.nn.Parameter(v.detach().to(device)) for k, v in params.items()}
    optimizer = torch.optim.Adam(params.values(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    generators = [torch.Generator().manual_seed(member_seeds(s, i)[1]) for s, i in members]
    # DataLoader consumes its base seed and RandomSampler consumes its permutations
    # exactly as in V1/V2. Fetch just indices so every member has its own minibatches.
    loaders = [DataLoader(range(config.train_size), batch_size=config.train_size,
                          shuffle=True, generator=g) for g in generators]
    x_all = torch.stack([splits[s]["train"].x for s, _ in members]).to(device)
    y_all = torch.stack([splits[s]["train"].y_noisy for s, _ in members]).to(device)
    eps = torch.tensor([fgsm_epsilon(splits[s]["train"].x, config.fgsm_epsilon_fraction)
                        for s, _ in members], device=device).reshape(-1, 1, 1)
    batch_rows = torch.arange(len(members), device=device)[:, None]
    model = models[0].to(device)

    def loss_one(weights, x, y):
        output = functional_call(model, weights, (x,))
        return gaussian_nll(*output, y) if condition.gaussian else mse_loss(output, y)

    losses = vmap(loss_one)
    history = torch.zeros(config.max_epochs, len(members), 2)
    resume_file = folder / (condition.name + "_resume.pt")
    start = 0
    if resume_file.exists():
        state = torch.load(resume_file, map_location="cpu", weights_only=True)
        if state["arm"] != arm or state["config"] != config.to_dict() or state["condition"] != condition.name:
            raise ValueError("Checkpoint configuration mismatch")
        with torch.no_grad():
            for name in params:
                params[name].copy_(state["params"][name])
        optimizer.load_state_dict(state["optimizer"])
        for g, rng in zip(generators, state["rng"], strict=True):
            g.set_state(rng)
        start = state["epoch"]
        history[:start] = state["history"]
    steps_per_epoch = (config.train_size + config.batch_size - 1) // config.batch_size
    for epoch in range(start + 1, config.max_epochs + 1):
        order = torch.stack([list(loader)[0] for loader in loaders]).to(device)
        totals = torch.zeros(len(members), 2, device=device)
        for indices in order.split(config.batch_size, dim=1):
            x, y = x_all[batch_rows, indices], y_all[batch_rows, indices]
            if condition.adversarial:
                x.requires_grad_(True)
            clean = losses(params, x, y)
            if condition.adversarial:
                grad, = torch.autograd.grad(clean.sum(), x, retain_graph=True)
                if arm.startswith("v4_") and not torch.isfinite(grad).all():
                    raise FloatingPointError(f"{arm}/{condition.name} nonfinite FGSM gradient at epoch {epoch}")
                perturbed = (x + eps * grad.sign()).detach()
                adv = losses(params, perturbed, y)
                totals[:, 1] += adv.detach() * indices.shape[1]
                objective = clean + adv
            else:
                objective = clean
            optimizer.zero_grad(set_to_none=True)
            objective.sum().backward()
            if arm.startswith("v4_"):
                finite = torch.isfinite(objective.detach())
                for p in params.values():
                    finite &= torch.isfinite(p.grad).flatten(1).all(1)
                if not finite.all():
                    failed = [members[j] for j in (~finite).nonzero().flatten().tolist()]
                    raise FloatingPointError(f"{arm}/{condition.name} epoch {epoch}, nonfinite loss/gradient {failed}")
            optimizer.step()
            totals[:, 0] += clean.detach() * indices.shape[1]
        history[epoch - 1] = totals.detach().cpu() / config.train_size
        if not torch.isfinite(history[epoch - 1]).all() or not all(
                torch.isfinite(p).all() for p in params.values()):
            raise FloatingPointError(f"{arm}/{condition.name} epoch {epoch}")
        if epoch % 100 == 0 or epoch == config.max_epochs:
            atomic_save(dict(arm=arm, condition=condition.name, config=config.to_dict(),
                             epoch=epoch, global_step=epoch * steps_per_epoch, params=params,
                             optimizer=optimizer.state_dict(), rng=[g.get_state() for g in generators],
                             history=history[:epoch].clone()), resume_file)

    for j, (seed, index) in enumerate(members):
        model_dir = folder / f"seed_{seed}" / condition.name
        for name in ("models", "histories"):
            (model_dir / name).mkdir(parents=True, exist_ok=True)
        network = models[j].cpu()
        network.load_state_dict({k: v[j].detach().cpu() for k, v in params.items()})
        individual = torch.optim.Adam(network.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        for p, stacked in zip(network.parameters(), params.values(), strict=True):
            individual.state[p] = {k: (v[j] if v.ndim else v).detach().cpu().clone()
                                   for k, v in optimizer.state[stacked].items()}
        model_seed, shuffle_seed = member_seeds(seed, index)
        atomic_save(dict(arm_id=arm, outer_seed=seed, condition=condition.name,
                         member=index, model_seed=model_seed, shuffle_seed=shuffle_seed,
                         learning_rate=config.learning_rate, weight_decay=config.weight_decay,
                         final_epoch=config.max_epochs, global_step=config.max_epochs * steps_per_epoch,
                         model_state_dict=network.state_dict(), optimizer_state_dict=individual.state_dict(),
                         shuffle_rng_state=generators[j].get_state()), model_dir / "models" / f"member_{index}.pt")
        clean, adv = history[:, j].numpy().T
        write_csv(model_dir / "histories" / f"member_{index}.csv", [
            dict(epoch=i + 1, clean_loss=float(c), adversarial_loss=float(a) if condition.adversarial else None,
                 total_loss=float(c + a), global_step=(i + 1) * steps_per_epoch)
            for i, (c, a) in enumerate(zip(clean, adv, strict=True))])
    for j, seed in enumerate(config.outer_seeds):
        trained = models[j * count:(j + 1) * count]
        for name in ("grid", "test_in", "test_ood"):
            split = splits[seed][name]
            prediction = predict(trained, split.x, condition.gaussian, "cpu")
            save_predictions(folder / f"seed_{seed}" / condition.name / f"predictions_{name}.npz",
                             prediction, split, condition, seed, True)

