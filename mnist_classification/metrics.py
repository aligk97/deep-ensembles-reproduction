"""Metrics on predictive probabilities, never averaged logits."""
import torch
from torch.nn import functional as F


def classification_metrics(probabilities, labels):
    p = probabilities.double()
    if p.ndim != 2 or p.shape[1] != 10 or len(p) != len(labels):
        raise ValueError("Expected [N,10] probabilities and [N] labels")
    if not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid probabilities")
    if not torch.allclose(p.sum(1), torch.ones(len(p), device=p.device, dtype=p.dtype), atol=1e-5):
        raise ValueError("Probabilities must sum to one")
    accuracy = (p.argmax(1) == labels).double().mean().item()
    nll = -p[torch.arange(len(p), device=p.device), labels].clamp_min(1e-12).log().mean().item()
    # Mean over BOTH examples and K=10 classes implements the paper's 1/K.
    brier = (p - F.one_hot(labels, 10).double()).square().mean().item()
    return {"accuracy": accuracy, "classification_error": 1 - accuracy,
            "nll": nll, "brier": brier}
