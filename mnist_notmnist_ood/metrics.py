"""Pure evaluation metrics for already computed classification probabilities.

References: Lakshminarayanan et al. (2017), Sections 2.4 and 3.5.
Input: softmax probabilities [M, N, K], never logits. Output entropy: [N],
float64, natural-log units (nat), on the input device. No model/data loading.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import torch

__all__ = ["predictive_entropy", "mean_id_entropy", "mean_ood_entropy",
           "entropy_gap", "ood_auroc", "summarize_seed_metrics"]


def _probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    if not isinstance(probabilities, torch.Tensor):
        raise TypeError("probabilities must be a torch.Tensor with shape [M, N, K]")
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have exactly three dimensions [M, N, K]")
    m, n, k = probabilities.shape
    if m < 1 or n < 1 or k < 2:
        raise ValueError("Require M >= 1, N >= 1, K >= 2")
    tolerances = {torch.float64: 1e-12, torch.float32: 5e-7,
                  torch.float16: 1e-3, torch.bfloat16: 1e-3}
    if probabilities.dtype not in tolerances or probabilities.layout != torch.strided:
        raise TypeError("Use a dense floating-point tensor (float16/bfloat16/float32/float64)")
    p = probabilities.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(p).all()):
        raise ValueError("probabilities contain NaN or Inf")
    if bool(((p < 0) | (p > 1)).any()):
        raise ValueError("Every probability must lie in [0, 1]")
    mass = p.sum(dim=-1, keepdim=True)
    if bool(((mass - 1).abs() > tolerances[probabilities.dtype]).any()):
        raise ValueError("Each [m, n, :] probability vector must sum to 1; logits are not accepted")
    # Correct only validated floating-point rounding; arbitrary weights are rejected.
    return p / mass


@torch.no_grad()
def predictive_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    r"""H[n] = -sum_k p_bar[n,k] * ln(p_bar[n,k]); p_bar = mean_m p.

    M denotes ensemble members or stochastic MC Dropout passes. All members
    must refer to the same N examples, in the same order and with identical
    class ordering. There is no logit averaging or mean of member entropies.

    xlogy implements 0*log(0)=0 without epsilon clipping, preserving exact zero
    entropy for one-hot vectors and contributions from arbitrarily small positive
    float64 probabilities. Accepted probability-mass roundoff is normalized in
    float64 before averaging (tolerance: 1e-12/5e-7/1e-3 for float64/32/16-bit).
    The final clamp removes only rounding outside the theoretical [0, ln(K)].
    """
    p = _probabilities(probabilities)
    averaged = p.mean(dim=0)
    averaged = averaged / averaged.sum(dim=-1, keepdim=True)
    entropy = -torch.special.xlogy(averaged, averaged).sum(dim=-1)
    upper = math.log(p.shape[-1])
    tolerance = 32 * torch.finfo(torch.float64).eps * p.shape[-1]
    if (not bool(torch.isfinite(entropy).all())
            or bool(((entropy < -tolerance) | (entropy > upper + tolerance)).any())):
        raise ArithmeticError("Entropy computation violated finite/theoretical bounds")
    return entropy.clamp(min=0.0, max=upper)


def _entropy_vector(values, name: str) -> np.ndarray:
    """Accept a real, finite, nonnegative, nonempty 1-D entropy vector."""
    if isinstance(values, torch.Tensor):
        if values.is_complex() or values.dtype == torch.bool:
            raise TypeError(f"{name} must be real-valued entropies")
        array = values.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        array = np.asarray(values)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{name} must be real-valued entropies")
        array = array.astype(np.float64, copy=False)
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a nonempty 1-D entropy vector [N]")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must contain finite, nonnegative entropies")
    return array


def mean_id_entropy(id_entropies) -> float:
    """Unweighted mean of per-example MNIST predictive entropies, in nat."""
    return float(_entropy_vector(id_entropies, "id_entropies").mean())


def mean_ood_entropy(ood_entropies) -> float:
    """Unweighted mean of per-example NotMNIST predictive entropies, in nat."""
    return float(_entropy_vector(ood_entropies, "ood_entropies").mean())


def entropy_gap(id_entropies, ood_entropies) -> float:
    """mean(OOD entropy) - mean(ID entropy), in nat; a negative gap is allowed."""
    return mean_ood_entropy(ood_entropies) - mean_id_entropy(id_entropies)


def ood_auroc(id_entropies, ood_entropies) -> float:
    """sklearn binary AUROC: MNIST=0, NotMNIST=1, higher entropy means more OOD.

    The two datasets may have different sizes. Scores are raw entropy values;
    no threshold tuning, balancing, normalization or score inversion is applied.
    AUROC is an additional project metric, not the Figure 3(a) histogram itself.
    """
    id_values = _entropy_vector(id_entropies, "id_entropies")
    ood_values = _entropy_vector(ood_entropies, "ood_entropies")
    from sklearn.metrics import roc_auc_score

    labels = np.concatenate((np.zeros(id_values.size, dtype=np.int64),
                             np.ones(ood_values.size, dtype=np.int64)))
    scores = np.concatenate((id_values, ood_values))
    return float(roc_auc_score(labels, scores))


def summarize_seed_metrics(seed_metrics: Mapping[int, Mapping[str, Real]]) -> dict:
    """Summarize scalar metrics for ONE method/M across independent run seeds.

    Input: {42: {'mean_id_entropy': ..., 'ood_auroc': ...}, 123: {...}, ...}.
    Each seed has the same metric names and one finite scalar per metric.
    At least two seeds are required: sample std (ddof=1) is undefined for one.
    Ensemble members/examples are never treated as independent seed repeats.
    """
    if not isinstance(seed_metrics, Mapping) or len(seed_metrics) < 2:
        raise ValueError("Provide metrics for at least two independent seeds")
    if any(not isinstance(seed, Integral) or isinstance(seed, bool) for seed in seed_metrics):
        raise TypeError("Seed keys must be integers")
    seeds = sorted(seed_metrics)
    rows = [seed_metrics[seed] for seed in seeds]
    if any(not isinstance(row, Mapping) or not row for row in rows):
        raise ValueError("Each seed must provide a nonempty metric mapping")
    names = set(rows[0])
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("Metric names must be nonempty strings")
    if any(set(row) != names for row in rows):
        raise ValueError("Every seed must have exactly the same metric names")
    summary = {}
    for name in sorted(names):
        values = [row[name] for row in rows]
        if any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) for value in values):
            raise TypeError(f"Metric {name} must have real scalar values")
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"Metric {name} contains NaN or Inf")
        with np.errstate(over="ignore", invalid="ignore"):
            mean, std = float(np.mean(array)), float(np.std(array, ddof=1))
        if not math.isfinite(mean) or not math.isfinite(std):
            raise ValueError(f"Metric {name} aggregation overflowed")
        summary[name] = {"mean": mean, "std": std}
    return {"n_seeds": len(seeds), "seeds": [int(seed) for seed in seeds],
            "ddof": 1, "metrics": summary}
