"""Figure 3(a)-style entropy densities and explicitly additional metric plots."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .inference import METHOD_NAMES
else:
    from inference import METHOD_NAMES

COLORS = {1: "#0072B2", 3: "#D55E00", 5: "#009E73"}
STYLES = {1: "-", 3: "--", 5: "-."}


def save_figure(figure, folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / f"{stem}.png", dpi=300, facecolor="white", bbox_inches="tight")
    figure.savefig(folder / f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(figure)


def make_plots(output, config):
    output = Path(output)
    folder = output / "figures"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
    edges = np.linspace(0, math.log(10), config["histogram_bins"] + 1)
    histograms = {}
    for seed in config["seeds"]:
        for method in config["methods"]:
            for name in ("mnist", "notmnist"):
                for m in config["ensemble_sizes"]:
                    path = output / "entropy" / f"seed_{seed}_{method}_{name}_M{m}.npy"
                    h = np.load(path, allow_pickle=False)
                    if not np.isfinite(h).all() or h.min() < 0 or h.max() > math.log(10):
                        raise ValueError(f"Invalid entropy values in {path}")
                    density, _ = np.histogram(h, bins=edges, density=True)
                    if not np.isclose(np.sum(density * np.diff(edges)), 1, atol=1e-12):
                        raise AssertionError("Histogram density does not integrate to one")
                    histograms[seed, method, name, m] = density

    def panel_figure(selected_seeds, stem, subtitle):
        figure, axes = plt.subplots(2, 4, figsize=(15.8, 7.2), sharex=True, sharey="row")
        row_max = [0, 0]
        for column, method in enumerate(config["methods"]):
            axes[0, column].set_title(METHOD_NAMES[method], fontweight="bold", pad=12)
            for row, name in enumerate(("mnist", "notmnist")):
                axis = axes[row, column]
                for m in config["ensemble_sizes"]:
                    density = np.mean([histograms[seed, method, name, m] for seed in selected_seeds], axis=0)
                    axis.stairs(density, edges, label=f"M = {m}", color=COLORS[m],
                                linestyle=STYLES[m], linewidth=1.7)
                    row_max[row] = max(row_max[row], density.max())
                axis.set_xlim(0, math.log(10))
                axis.set_xticks([0, .5, 1, 1.5, 2, math.log(10)], ["0", "0.5", "1", "1.5", "2", "ln(10)"])
                axis.grid(axis="y", alpha=.18)
                if row == 1:
                    axis.set_xlabel("Predictive entropy (nat)")
                if column == 0:
                    axis.set_ylabel(f"{'MNIST (ID)' if row == 0 else 'NotMNIST (OOD)'}\nProbability density")
        for row in range(2):
            for axis in axes[row]:
                axis.set_ylim(0, row_max[row] * 1.08)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .945), ncol=3, frameon=False)
        figure.suptitle("MNIST / NotMNIST - predictive entropy", fontsize=16, y=.998)
        figure.text(.5, .018, subtitle + " | 50 shared bins; y scales shared within each row",
                    ha="center", fontsize=9, color="#444444")
        figure.subplots_adjust(top=.84, bottom=.12, left=.065, right=.985, hspace=.23, wspace=.14)
        save_figure(figure, folder, stem)

    panel_figure(config["seeds"], "figure3a", "Mean of 3 independently normalized seed histograms")
    for seed in config["seeds"]:
        panel_figure([seed], f"figure3a_seed{seed}", f"Run seed {seed}; all test images")

    with (output / "results_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for metric, stem, ylabel in (("ood_auroc", "ood_auroc", "Entropy-based OOD AUROC"),
                                 ("entropy_gap", "entropy_gap", "Mean OOD entropy - mean ID entropy (nat)")):
        figure, axis = plt.subplots(figsize=(8, 5))
        for index, method in enumerate(config["methods"]):
            selected = [next(row for row in rows if row["method"] == method and int(row["M"]) == m)
                        for m in config["ensemble_sizes"]]
            # Tiny horizontal offsets separate overlapping error bars without changing M labels.
            x = np.asarray(config["ensemble_sizes"]) + (index - 1.5) * .035
            axis.errorbar(x, [float(row[f"{metric}_mean"]) for row in selected],
                          yerr=[float(row[f"{metric}_std"]) for row in selected],
                          marker=("o", "s", "^", "D")[index], capsize=4, linewidth=1.6,
                          label=METHOD_NAMES[method])
        axis.set_xticks(config["ensemble_sizes"])
        axis.set_xlabel("M (ensemble members / MC Dropout passes)")
        axis.set_ylabel(ylabel)
        axis.set_title("Additional analysis - mean ± sample SD across 3 seeds", fontsize=11)
        axis.grid(alpha=.2)
        if metric == "ood_auroc":
            axis.set_ylim(0, 1.02)
        else:
            axis.axhline(0, color="black", alpha=.35, linewidth=.8)
        axis.legend(frameon=False, fontsize=9, loc="best")
        figure.tight_layout()
        save_figure(figure, folder, stem)
    return {"histogram_edges_nat": edges.tolist(), "density_integral": 1.0,
            "aggregation": "Arithmetic mean of per-seed density histograms, not raw counts",
            "y_axis": "Common limits within each ID/OOD row; rows may differ", "png_dpi": 300}
