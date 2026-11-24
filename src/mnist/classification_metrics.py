"""Sample-weighted single-model metrics for Figure 2(a)."""

import torch
from torch.nn import functional as F


def score_totals(log_probs, labels):
    """Additive correct count, negative log-likelihood and class-normalized Brier."""
    correct = (log_probs.argmax(dim=1) == labels).sum()
    nll = F.nll_loss(log_probs, labels, reduction="sum")
    brier = (log_probs.exp() - F.one_hot(labels, num_classes=10)).square().mean(dim=1).sum()
    return torch.stack((correct, nll, brier))


def mean_scores(totals, count):
    if not count or not totals.isfinite().all():
        raise ValueError("Expected nonempty predictions with finite metrics")
    correct, nll_sum, brier_sum = totals.tolist()
    return dict(test_samples=count, correct=int(correct), accuracy=correct / count,
                classification_error=100 * (count - correct) / count,
                nll=nll_sum / count, brier_score=brier_sum / count)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=loader.pin_memory)
        labels = labels.to(device, non_blocking=loader.pin_memory)
        logits = model(images).double()
        if logits.shape != (len(labels), 10):
            raise ValueError("Expected ten logits per MNIST image")
        log_probs = logits.log_softmax(dim=1)
        totals += score_totals(log_probs, labels)
        count += len(labels)
    return mean_scores(totals, count)
