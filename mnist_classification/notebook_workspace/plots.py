"""Publishable PNG (300 dpi) and vector PDF figures from actual saved results."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import read_idx

LABELS = {"ensemble": "Ensemble", "random": "Ensemble + R",
          "adversarial": "Ensemble + AT", "mc_dropout": "MC Dropout", "single": "Single model"}
COLORS = {"ensemble": "#d6504f", "random": "#23a6a2", "adversarial": "#5454c4",
          "mc_dropout": "#65a830", "single": "#666666"}


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(output, data_dir):
    output, data_dir = Path(output), Path(data_dir)
    folder = output / "figures"
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    summary = pd.read_csv(output / "summary.csv")
    expected = {(method, size) for method in ("ensemble", "random", "adversarial", "mc_dropout")
                for size in config["ensemble_sizes"]} | {("single", 1)}
    observed = set(zip(summary.method, summary.M))
    if observed != expected or not (summary.n_seeds == len(config["seeds"])).all():
        raise ValueError("Comparison plots require every configured method/M and all run seeds")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for ax, metric, title in zip(axes, ("classification_error", "nll", "brier"),
                                 ("Classification error (%)", "NLL (nats)", "Brier score (1/K)")):
        scale = 100 if metric == "classification_error" else 1
        for method in ("ensemble", "random", "adversarial", "mc_dropout"):
            group = summary[summary.method == method].sort_values("M")
            ax.errorbar(group.M, group[f"{metric}_mean"] * scale,
                        yerr=group[f"{metric}_std"] * scale, marker="o", markersize=4,
                        capsize=3, label=LABELS[method], color=COLORS[method], linewidth=1.5)
        baseline = summary[summary.method == "single"]
        ax.axhline(baseline[f"{metric}_mean"].iloc[0] * scale, linestyle="--", color=COLORS["single"],
                   linewidth=1, label="Single model mean")
        ax.set(title=title, xlabel="M (networks / MC samples)", xticks=config["ensemble_sizes"])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    scope = "SMOKE TEST - NOT A REPRODUCTION RESULT" if config["train_limit"] else "MNIST reproduction"
    fig.suptitle(f"{scope} | {config['epochs']} epochs | mean ± sample SD, n={len(config['seeds'])} seeds")
    save(fig, folder, "figure2a_comparison")

    history = pd.concat([pd.read_csv(p) for p in (output / "histories").glob("*.csv")], ignore_index=True)
    history.to_csv(output / "training_history.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for method in ("ensemble", "random", "adversarial", "mc_dropout"):
        # Average members WITHIN each run before averaging independent run seeds.
        h = history[history.method == method].groupby(["run_seed", "epoch"])[
            ["clean_loss", "train_accuracy", "validation_accuracy"]].mean().reset_index()
        for ax, key, title in zip(axes, ("clean_loss", "train_accuracy", "validation_accuracy"),
                                  ("Clean training cross entropy", "Training accuracy", "Validation accuracy")):
            g = h.groupby("epoch")[key].agg(["mean", "std"]).reset_index()
            ax.plot(g.epoch, g["mean"], color=COLORS[method], label=LABELS[method])
            ax.fill_between(g.epoch, g["mean"] - g["std"], g["mean"] + g["std"],
                            color=COLORS[method], alpha=.12)
            ax.set(title=title, xlabel="Epoch")
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig, folder, "training_curves")

    x = read_idx(data_dir / "t10k-images-idx3-ubyte.gz", True).numpy().reshape(-1, 28, 28)
    y = read_idx(data_dir / "t10k-labels-idx1-ubyte.gz", False).numpy()
    fig, axes = plt.subplots(2, 10, figsize=(12, 3.3), layout="constrained")
    for k in range(10):
        for row, idx in enumerate(np.flatnonzero(y == k)[:2]):
            axes[row, k].imshow(x[idx], cmap="gray", vmin=0, vmax=255)
            axes[row, k].set_title(f"Label {k}")
            axes[row, k].axis("off")
    fig.suptitle("MNIST test examples (visualization only)")
    save(fig, folder, "mnist_examples")

    seed = config["seeds"][0]
    with np.load(output / "probabilities" / f"seed_{seed}_adversarial.npz") as archive:
        p = archive["probabilities"].mean(axis=0)
        labels = archive["labels"]
    confidence = p.max(1)
    predicted = p.argmax(1)
    # Stable sorting gives reproducible ties at confidence==1.
    ordered = np.argsort(confidence, kind="stable")
    low, high = ordered[:8], ordered[-8:][::-1]
    for name, indices, title in (("low_confidence", low, "Lowest confidence"),
                                  ("high_confidence", high, "Highest confidence")):
        fig, axes = plt.subplots(2, 4, figsize=(10, 5.5), layout="constrained")
        for ax, idx in zip(axes.flat, indices):
            ax.imshow(x[idx], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"#{idx}: true={labels[idx]}, pred={predicted[idx]}\np={confidence[idx]:.4f}", fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{title} | Ensemble + AT, seed={seed}, M={max(config['ensemble_sizes'])}")
        save(fig, folder, name)
    selected = np.concatenate([high[:3], low[:3]])
    fig, axes = plt.subplots(6, 2, figsize=(10, 14), gridspec_kw={"width_ratios": [1, 3]}, layout="constrained")
    export = []
    for row, idx in enumerate(selected):
        axes[row, 0].imshow(x[idx], cmap="gray", vmin=0, vmax=255)
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"Test #{idx}: true {labels[idx]}")
        axes[row, 1].bar(range(10), p[idx], color=COLORS["adversarial"])
        axes[row, 1].set(xticks=range(10), ylim=(0, 1.03), ylabel="Probability",
                         title=f"Prediction {predicted[idx]}, confidence {confidence[idx]:.4f}")
        export.append({"test_index": int(idx), "true_label": int(labels[idx]),
                       "prediction": int(predicted[idx]), "confidence": float(confidence[idx]),
                       **{f"p_{k}": float(p[idx, k]) for k in range(10)}})
    fig.suptitle(f"Actual class probabilities | Ensemble + AT, seed={seed}, M={max(config['ensemble_sizes'])}")
    save(fig, folder, "prediction_probabilities")
    pd.DataFrame(export).to_csv(output / "example_predictions.csv", index=False)
    print(f"Saved PNG/PDF figures in {folder}", flush=True)
