from statistics import NormalDist

import torch

from .regression_losses import gaussian_nll, mixture_nll
from .experiment_io import write_csv

COVERAGE_LEVELS = (50, 80, 90, 95)


def mixture_quantile(means, variances, probability):
    """Invert the equally weighted Gaussian-mixture CDF by vectorized bisection."""
    means, stds = means.double(), variances.double().sqrt()
    # The mixture quantile lies between the most extreme component quantiles.
    component = means + NormalDist().inv_cdf(probability) * stds
    low, high = component.amin(dim=0), component.amax(dim=0)
    for _ in range(40):
        mid = (low + high) * 0.5
        below = torch.special.ndtr((mid - means) / stds).mean(dim=0) < probability
        low = torch.where(below, mid, low)
        high = torch.where(below, high, mid)
    return (low + high) * 0.5


def coverage(target, low, high):
    return ((target >= low) & (target <= high)).double().mean().item()


def mean_if_available(tensor):
    return None if tensor is None else tensor.mean().item()


def evaluate_noisy(prediction, target, condition, floor):
    mean = prediction.mean.double()
    variance = prediction.total_variance.double()
    target = target.double()
    proxy = not condition.gaussian
    mixture = condition.gaussian and condition.ensemble
    kind = "gaussian"
    if proxy:
        variance = variance.clamp_min(floor)
        kind = "proxy_gaussian"
    elif mixture:
        kind = "mixture"
    results = {
        "rmse_noisy": (mean - target).square().mean().sqrt().item(),
        "mean_pred_variance": prediction.total_variance.double().mean().item(),
        "mean_aleatoric": mean_if_available(prediction.aleatoric),
        "mean_epistemic": mean_if_available(prediction.epistemic),
    }
    if proxy:
        results["near_zero_variance_count"] = (prediction.total_variance <= floor).sum().item()
    gaussian_score = gaussian_nll(mean, variance, target).item()
    if mixture:
        member_means = prediction.member_means.double()
        member_variances = prediction.member_variances.double()
        score = mixture_nll(member_means, member_variances, target).item()
        results["moment_matched_nll"] = gaussian_score
    else:
        score = gaussian_score
    results[f"{kind}_nll"] = results["predictive_nll"] = score

    for level in COVERAGE_LEVELS:
        lower_probability = (1 - level / 100) / 2
        radius = NormalDist().inv_cdf(1 - lower_probability) * variance.sqrt()
        gaussian_coverage = coverage(target, mean - radius, mean + radius)
        if mixture:
            low = mixture_quantile(member_means, member_variances, lower_probability)
            high = mixture_quantile(member_means, member_variances, 1 - lower_probability)
            results[f"moment_matched_coverage{level}"] = gaussian_coverage
            empirical = coverage(target, low, high)
        else:
            empirical = gaussian_coverage
        results[f"coverage{level}"] = empirical
    return kind, results


def evaluate_condition(condition, predictions, splits, config, seed):
    result = {
        "condition": condition.name, "outer_seed": seed, "smoke": config.smoke,
        "sample_counts": {name: len(split.x) for name, split in splits.items()},
        "variance_floor": config.variance_floor if not condition.gaussian else None,
        "metric_types": {},
        "metrics": {},
    }
    metrics = result["metrics"]
    for domain in ("in", "ood"):
        split = f"test_{domain}"
        kind, values = evaluate_noisy(predictions[split], splits[split].y_noisy, condition,
                                      config.variance_floor)
        for name, value in values.items():
            key = f"{name}_{domain}"
            metrics[key] = value
            if name.startswith("moment_matched"):
                result["metric_types"][key] = "moment_matched_gaussian"
            elif name.endswith("nll") or name.startswith("coverage"):
                result["metric_types"][key] = kind
    grid, prediction = splits["grid"], predictions["grid"]
    in_domain = (grid.x >= config.train_x_min) & (grid.x <= config.train_x_max)
    metrics["rmse_clean_grid"] = (prediction.mean.double() - grid.y_true).square().mean().sqrt().item()
    for region, mask in (("train_region", in_domain), ("outside_train_region", ~in_domain)):
        values = prediction.epistemic
        metrics[f"mean_epistemic_{region}"] = None if values is None else values[mask].mean().item()
    return result


def metric_rows(result):
    return [{
        "outer_seed": result["outer_seed"], "condition": result["condition"],
        "metric": metric, "metric_type": result["metric_types"].get(metric, "shared"),
        "value": value if value is not None else "N/A", "smoke": result["smoke"],
    } for metric, value in result["metrics"].items()]


def save_metrics_csv(path, results):
    write_csv(path, [row for result in results for row in metric_rows(result)])
