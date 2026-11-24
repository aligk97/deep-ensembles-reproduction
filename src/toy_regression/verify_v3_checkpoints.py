"""Verify member training, saved datasets and predictions before computing V3 scores."""
import json
import math

import numpy as np
import pandas as pd
import torch
import yaml

from .compare_regional_performance import collect_region
from .compute_regional_metrics import REGIONS, distribution_for, evaluate_region
from .ensemble_predictions import predict
from .experiment_config import Config
from .load_saved_predictions import read_arrays, read_prediction, read_samples
from .run_v3_sweep import digest
from .train_v1_v2_models import CONDITIONS, initialize_member, member_seeds


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_member(folder, condition, seed, index, config, lr, steps):
    path = folder / condition.name
    member = torch.load(path / "models" / f"member_{index}.pt", weights_only=True, map_location="cpu")
    model_seed, shuffle_seed = member_seeds(seed, index)
    expected = dict(arm_id=folder.parent.name, condition=condition.name, outer_seed=seed, member=index,
                    model_seed=model_seed, shuffle_seed=shuffle_seed, final_epoch=config.max_epochs,
                    global_step=steps, learning_rate=lr, weight_decay=config.weight_decay)
    require(all(member[k] == v for k, v in expected.items()), "Checkpoint metadata mismatch")
    generator = torch.Generator()
    generator.set_state(member["shuffle_rng_state"])
    model = initialize_member(condition, config, model_seed)
    model.load_state_dict(member["model_state_dict"])
    require(all(p.isfinite().all() for p in model.parameters()), "Nonfinite weights")
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(member["optimizer_state_dict"])
    require(optimizer.param_groups[0]["lr"] == lr and
            optimizer.param_groups[0]["weight_decay"] == config.weight_decay, "Adam settings")
    for p in model.parameters():
        state = optimizer.state[p]
        require(state["step"].item() == steps and all(state[k].shape == p.shape and
                state[k].isfinite().all() for k in ("exp_avg", "exp_avg_sq")), "Invalid Adam state")
    history = pd.read_csv(path / "histories" / f"member_{index}.csv")
    epochs = np.arange(1, config.max_epochs + 1)
    require(np.array_equal(history.epoch, epochs) and
            np.array_equal(history.global_step, epochs * math.ceil(config.train_size / config.batch_size)),
            "Epoch/step history mismatch")
    require(np.isfinite(history[["total_loss", "clean_loss"]]).all().all(), "Nonfinite history")
    if condition.adversarial:
        require(np.isfinite(history.adversarial_loss).all(), "Nonfinite FGSM loss")
    return model.eval()


def score_predictions(folder, condition, models, seed, config, labels, *, overall=False):
    splits = []
    for name, count in (("grid", config.plot_points), ("test_in", config.n_test_in), ("test_ood", config.n_test_ood)):
        source = folder / "data" / f"{name}.npz"
        data = read_arrays(source)
        arrays = read_arrays(folder / condition.name / f"predictions_{name}.npz")
        for key in ("x", "y_true") if name == "grid" else ("x", "y_true", "y_noisy"):
            np.testing.assert_array_equal(data[key], arrays[key])
        require(data["x"].shape == (count, 1), f"Split shape {name}")
        samples = read_samples(source, data, name != "grid")
        current = predict(models, torch.tensor(data["x"]), condition.gaussian, "cpu")
        for key, value in vars(current).items():
            if value is not None:
                np.testing.assert_allclose(arrays[key], value.numpy(), rtol=1e-5, atol=1e-6)
        prediction = read_prediction(arrays, samples, condition, seed)
        if name != "grid":
            splits.append((samples, prediction))
    rows = []
    for region in [*REGIONS, *(["overall"] if overall else [])]:
        x, y, prediction = collect_region(splits, region)
        expected = config.n_test_in if region == "in_domain" else config.n_test_ood
        if region == "overall":
            expected = config.n_test_in + config.n_test_ood
        if region in ("left_ood", "right_ood"):
            expected //= 2
        require(len(x) == expected, f"Region sample count {region}")
        scores = evaluate_region(x, y, prediction, condition, config.variance_floor, region)
        require(all(math.isfinite(v) for v in scores.values()), "Nonfinite metric")
        for metric, value in scores.items():
            rows.append(dict(**labels, region=region, n_samples=len(x), metric=metric, value=value,
                             distribution=distribution_for(metric, condition), status="verified"))
    return rows


def verify_run(raw, *, selected_arm=None, overall=False):
    manifest = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    spec = yaml.safe_load((raw / "config_resolved.yaml").read_text(encoding="utf-8"))
    require(digest(raw / "config_resolved.yaml") == manifest["config_sha256"], "Matrix hash")
    require(not manifest["smoke"], "Smoke runs cannot enter the evaluation")
    rows, records, hashes, problems = [], [], [], []
    for arm, options in spec["arms"].items():
        if selected_arm is not None and arm != selected_arm:
            continue
        config_data = manifest["base"].copy()
        config_data.update({k: v for k, v in options.items() if k != "mse_learning_rate"})
        config_data["outer_seeds"] = tuple(spec["seeds"])
        config = Config(**config_data)
        counts = {}
        steps = config.max_epochs * math.ceil(config.train_size / config.batch_size)
        for seed in config.outer_seeds:
            folder = raw / "arms" / arm / f"seed_{seed}"
            try:
                for name in ("train", "test_in", "test_ood", "grid"):
                    saved = folder / "data" / f"{name}.npz"
                    h = digest(saved)
                    require(h == digest(raw / "data" / f"seed_{seed}" / saved.name), f"{name} differs")
                    hashes.append(dict(arm_id=arm, seed=seed, split=name, sha256=h))
            except Exception as e:
                problems.append(f"{arm}/{seed}: {e}")
                continue
            for condition in CONDITIONS:
                lr = config.learning_rate if condition.gaussian else options.get("mse_learning_rate", config.learning_rate)
                labels = dict(run_id=raw.name, source_run_id=raw.name, experiment_version=spec["experiment_version"],
                              arm_id=arm, seed=seed, model=condition.name, batch_size=config.batch_size,
                              effective_batch_size=min(20,config.batch_size), epochs=config.max_epochs,
                              learning_rate=lr, weight_decay=config.weight_decay, optimizer="Adam", optimizer_steps=steps)
                try:
                    models = [verify_member(folder, condition, seed, i, config, lr, steps)
                              for i in range(5 if condition.ensemble else 1)]
                    rows.extend(score_predictions(folder, condition, models, seed, config, labels, overall=overall))
                    counts[f"{seed}/{condition.name}"] = len(models)
                except Exception as e:
                    problems.append(f"{arm}/{seed}/{condition.name}: {e}")
        records.append(dict(arm_id=arm, verified_models=sum(counts.values()), models_by_seed_method=counts,
                            expected_epochs=config.max_epochs, expected_steps_per_member=steps))
        print("Verified", arm, sum(counts.values()), flush=True)
    return pd.DataFrame(rows), records, hashes, problems

