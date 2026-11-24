"""Metrics for the fixed Figure 1 evaluation support."""

from math import erf, sqrt
from statistics import NormalDist, mean, stdev

import torch

from .evaluate_predictions import coverage, mixture_quantile
from .regression_losses import gaussian_nll, mixture_nll

REGIONS = {
    "left_ood": "[-6,-4)",
    "in_domain": "[-4,4]",
    "right_ood": "(4,6]",
    "combined_ood": "[-6,-4) U (4,6]",
}
LEVELS = (("50", .50), ("80", .80), ("90", .90), ("95", .95),
          ("99_73", erf(3 / sqrt(2))))
Z95 = NormalDist().inv_cdf(.975)
NOISE_VARIANCE = 9.0

METRICS = {
    "rmse_noisy": "Root mean square of prediction mean minus saved noisy target",
    "rmse_true": "Root mean square of prediction mean minus x cubed at the same test inputs",
    "predictive_nll": "Full Gaussian or equally weighted Gaussian mixture negative log-likelihood",
    "moment_matched_nll": "Full Gaussian negative log-likelihood using ensemble moments",
    "aleatoric_variance": "Mean of learned component variances",
    "epistemic_variance": "Population variance of member means averaged over inputs",
    "total_predictive_variance": "Mean raw total predictive variance",
    "aleatoric_signed_error_from_9": "Mean in-domain aleatoric variance minus true noise variance 9",
    "aleatoric_abs_error_from_9": "Absolute deviation of the seed's mean in-domain aleatoric variance from 9",
}
for prefix in ("", "mixture_"):
    family = "Equal-tailed Gaussian mixture" if prefix else "Central Gaussian"
    for label, nominal in LEVELS:
        pct = f"{100 * nominal:.2f}".rstrip("0").rstrip(".")
        METRICS[f"{prefix}coverage_{label}"] = f"{family} {pct}% interval coverage of noisy targets (fraction)"
        METRICS[f"{prefix}true_function_containment_{label}"] = (
            f"{family} {pct}% interval containment of x cubed (fraction)")
        if label in ("95", "99_73"):
            METRICS[f"{prefix}interval_width_{label}"] = f"Mean {family} {pct}% interval width in target units"
    METRICS[f"{prefix}coverage_mae"] = (
        f"Mean absolute discrepancy from the five nominal {family} coverage levels (fraction)")


def region_masks(x):
    left = (x >= -6) & (x < -4)
    right = (x > 4) & (x <= 6)
    return {"left_ood": left, "in_domain": (x >= -4) & (x <= 4),
            "right_ood": right, "combined_ood": left | right}


def rmse(prediction, target):
    return (prediction - target).square().mean().sqrt().item()


def add_interval_metrics(result, target, truth, bounds, prefix=""):
    gaps = []
    for (label, nominal), (low, high) in zip(LEVELS, bounds, strict=True):
        measured = coverage(target, low, high)
        gaps.append(abs(measured - nominal))
        result[f"{prefix}coverage_{label}"] = measured
        result[f"{prefix}true_function_containment_{label}"] = coverage(truth, low, high)
        if label in ("95", "99_73"):
            result[f"{prefix}interval_width_{label}"] = (high - low).mean().item()
    result[f"{prefix}coverage_mae"] = mean(gaps)


def gaussian_bounds(mu, variance):
    std = variance.sqrt()
    for label, nominal in LEVELS:
        radius = (3.0 if label == "99_73" else NormalDist().inv_cdf((1 + nominal) / 2)) * std
        yield mu - radius, mu + radius


def mixture_bounds(means, variances):
    for _, nominal in LEVELS:
        tail = (1 - nominal) / 2
        yield (mixture_quantile(means, variances, tail),
               mixture_quantile(means, variances, 1 - tail))


def evaluate_region(x, target, prediction, condition, floor, region):
    truth = x.pow(3)
    mu = prediction["mean"]
    result = {"rmse_noisy": rmse(mu, target), "rmse_true": rmse(mu, truth)}
    for key, metric in (("aleatoric", "aleatoric_variance"), ("epistemic", "epistemic_variance"),
                        ("total_variance", "total_predictive_variance")):
        if key in prediction:
            result[metric] = prediction[key].mean().item()
    if region == "in_domain" and "aleatoric" in prediction:
        error = result["aleatoric_variance"] - NOISE_VARIANCE
        result.update(aleatoric_signed_error_from_9=error, aleatoric_abs_error_from_9=abs(error))
    if "total_variance" in prediction:
        var = prediction["total_variance"]
        var = var if condition.gaussian else var.clamp_min(floor)
        gaussian_score = gaussian_nll(mu, var, target).item()
        score_name = "moment_matched_nll" if condition.gaussian and condition.ensemble else "predictive_nll"
        result[score_name] = gaussian_score
        add_interval_metrics(result, target, truth, gaussian_bounds(mu, var))
    if condition.gaussian and condition.ensemble and "member_variances" in prediction:
        means, variances = prediction["member_means"], prediction["member_variances"]
        result["predictive_nll"] = mixture_nll(means, variances, target).item()
        add_interval_metrics(result, target, truth, mixture_bounds(means, variances), "mixture_")
    return result


def distribution_for(metric, condition):
    if metric.startswith("rmse") or "variance" in metric or "error_from_9" in metric:
        return "point_or_raw_moments"
    if not condition.gaussian:
        return "gaussian_proxy"
    if condition.ensemble:
        return "gaussian_mixture" if metric == "predictive_nll" or metric.startswith("mixture_") else (
            "moment_matched_gaussian")
    return "gaussian"


def sample_summary(values):
    return (mean(values), stdev(values) if len(values) > 1 else None)
