"""Probability averaging, prefix scores and outer-seed uncertainty for MNIST."""

import math
from statistics import mean, stdev

import torch

from .classification_metrics import mean_scores, score_totals

METRICS = ("accuracy", "classification_error", "nll", "brier_score")


@torch.inference_mode()
def predict_log_probs(model, loader, device):
    model.eval()
    predictions, targets = [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=loader.pin_memory))
        predictions.append(logits.double().log_softmax(dim=1).cpu())
        targets.append(labels)
    return torch.cat(predictions), torch.cat(targets)


def prefix_scores(predictions, sizes):
    """Stream ordered members as (log_softmax, labels); retain one CPU accumulator."""
    member_scores, ensemble_scores = [], {}
    log_sum, targets = None, None
    for size, (log_probs, labels) in enumerate(predictions, start=1):
        if log_probs.shape != (len(labels), 10) or not torch.isfinite(log_probs).all():
            raise ValueError("Expected ten finite log probabilities for each MNIST image")
        normalizer = log_probs.logsumexp(dim=1)
        torch.testing.assert_close(normalizer, torch.zeros_like(normalizer), atol=1e-8, rtol=0)
        if log_sum is None:
            targets, log_sum = labels, log_probs
        else:
            if not torch.equal(labels, targets):
                raise ValueError("Inconsistent evaluation sample order between members")
            log_sum = torch.logaddexp(log_sum, log_probs)
        member_scores.append(mean_scores(score_totals(log_probs, labels), len(labels)))
        if size in sizes:
            # log((p1 + ... + pM) / M), including true-class probabilities near underflow.
            mixture = log_sum - math.log(size)
            ensemble_scores[size] = mean_scores(score_totals(mixture, labels), len(labels))
    if set(ensemble_scores) != set(sizes) or len(member_scores) != max(sizes):
        raise ValueError("Missing or extra ensemble members")
    return member_scores, ensemble_scores


def summarize(rows, seeds, sizes):
    summary = []
    for size in sizes:
        group = [row for row in rows if row["ensemble_size"] == size]
        if sorted(row["experiment_seed"] for row in group) != sorted(seeds):
            raise ValueError("Expected exactly one result for each outer seed at each size")
        result = dict(ensemble_size=size, repetitions=len(group))
        for metric in METRICS:
            values = [row[metric] for row in group]
            result[metric + "_mean"] = mean(values)
            result[metric + "_std"] = stdev(values)
        summary.append(result)
    return summary


def plot_summary(summary, directory, smoke):
    from matplotlib.figure import Figure

    for metric, label in (("classification_error", "Classification error (%)"),
                          ("nll", "Negative log-likelihood"), ("brier_score", "Brier score (1/K)")):
        figure = Figure(figsize=(6.4, 4), constrained_layout=True)
        ax = figure.subplots()
        sizes = [row["ensemble_size"] for row in summary]
        ax.errorbar(sizes, [r[metric + "_mean"] for r in summary],
                    yerr=[r[metric + "_std"] for r in summary], marker="o", capsize=4)
        title = "CUDA smoke test" if smoke else "MNIST standard deep ensemble"
        ax.set(xlabel="Number of members (M)", ylabel=label, xticks=sizes,
               title=f"{title}\nMean ± sample SD across outer seeds")
        ax.grid(axis="y", alpha=0.25)
        for suffix in ("png", "pdf"):
            figure.savefig(directory / f"{metric}.{suffix}", dpi=180)
