r"""CMP4509 Toy Regression, Step 08: explain out-of-distribution overconfidence.

Windows PowerShell, from toy_regression/:
    .\.venv\Scripts\python.exe .\08_diagnose_uncertainty.py

NO TRAINING. Reads previously saved pointwise predictions from Step 07.
All NEW results go to outputs/08_uncertainty_diagnostics/{figures,metrics}.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "outputs" / "07_seed_experiments"
OUTPUT_DIR = ROOT / "outputs" / "08_uncertainty_diagnostics"
METHODS = ("mse_ensemble", "gaussian_nll", "gaussian_at", "deep_ensemble")
LABELS = {
    "mse_ensemble": "MSE Ensemble (5)",
    "gaussian_nll": "Gaussian NLL (1)",
    "gaussian_at": "Gaussian NLL + AT",
    "deep_ensemble": "Deep Ensemble (5)",
}
REGIONS = {
    "left_ood": lambda x: x < -4,
    "in_distribution": lambda x: (x >= -4) & (x <= 4),
    "right_ood": lambda x: x > 4,
}
Z95 = 1.959963984540054
VAR_FLOOR_NORM = 1e-6  # as in Step 07 GaussianRegressor.forward()
NOISE_STD_TRUE = 3.0


def load_csv(file_path):
    with file_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {file_path}")
    return rows


def save_csv(file_path, rows):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def to_array(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def get_inputs():
    config_path = INPUT_DIR / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}\n"
            "Do not use the aggregate.zip alone. Run this script inside the "
            "toy_regression folder that contains the full Step 07 outputs."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "gaussian_target_normalization" not in config:
        raise KeyError("No normalization info in experiment_config.json")
    y_scale = float(config["gaussian_target_normalization"]["scale"])
    run_seeds = [int(s) for s in config["run_seeds"]]
    if y_scale <= 0:
        raise ValueError("Invalid normalization scale")
    predictions = {}
    reference_x = None
    for seed in run_seeds:
        for method in METHODS:
            path = INPUT_DIR / f"seed_{seed}" / method / "metrics" / "predictions.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing pointwise predictions:\n{path}\n"
                    "Keep Step 07's seed_<number> folders; aggregate.zip only has summary tables."
                )
            rows = load_csv(path)
            data = {k: to_array(rows, k) for k in (
                "x", "true_y", "mean_prediction", "aleatoric_variance",
                "epistemic_variance", "total_variance", "std_used",
                "lower_95", "upper_95"
            )}
            if reference_x is None:
                reference_x = data["x"]
            elif len(reference_x) != len(data["x"]) or not np.allclose(reference_x, data["x"]):
                raise ValueError(f"Mismatched x positions: seed={seed}, method={method}")
            if not all(np.isfinite(v).all() for v in data.values()):
                raise ValueError(f"Nonfinite values in {path}")
            predictions[(seed, method)] = data
    return config, run_seeds, y_scale, reference_x, predictions


def build_summary(run_seeds, x, predictions, sigma_floor):
    summaries = []
    for seed in run_seeds:
        for method in METHODS:
            p = predictions[(seed, method)]
            for region, region_mask in REGIONS.items():
                mask = region_mask(x)
                err = np.abs(p["mean_prediction"][mask] - p["true_y"][mask])
                sd = p["std_used"][mask]
                target_inside = ((p["true_y"][mask] >= p["lower_95"][mask]) &
                                 (p["true_y"][mask] <= p["upper_95"][mask]))
                frac_at_floor = ""
                # Only single Gaussian model total standard deviation has a well-defined
                # strict bound = y_scale*sqrt(normalized_variance_floor).
                if method in ("gaussian_nll", "gaussian_at"):
                    frac_at_floor = float(np.mean(sd <= sigma_floor * 1.01))
                summaries.append({
                    "seed": seed,
                    "method": method,
                    "region": region,
                    "n_x": int(mask.sum()),
                    "mean_absolute_error_vs_clean": float(err.mean()),
                    "rmse_vs_clean": float(np.sqrt(np.mean(err ** 2))),
                    "mean_predicted_std": float(sd.mean()),
                    "mean_95_width": float((2 * Z95 * sd).mean()),
                    "clean_curve_coverage_95": float(target_inside.mean()),
                    "mean_aleatoric_std": float(np.sqrt(np.maximum(p["aleatoric_variance"][mask], 0)).mean()),
                    "mean_epistemic_std": float(np.sqrt(np.maximum(p["epistemic_variance"][mask], 0)).mean()),
                    "fraction_sigma_at_normalized_floor_1pct_tolerance": frac_at_floor,
                })
    return summaries


def plot_left_comparison(path, seed, x, predictions):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), sharex=True, sharey=True)
    for ax, method in zip(axes.flat, METHODS):
        p = predictions[(seed, method)]
        m = (x >= -6) & (x <= -3.55)
        xx, mu, sd = x[m], p["mean_prediction"][m], p["std_used"][m]
        ax.plot(xx, xx ** 3, c="royalblue", linewidth=2, label="Ground truth x³")
        ax.plot(xx, mu, c="darkorange", linewidth=2, label="Predicted mean")
        ax.fill_between(xx, mu-Z95*sd, mu+Z95*sd, color="gray", alpha=.3,
                        label="95% interval (±1.96σ)")
        ax.axvline(-4, color="gray", ls="--", lw=1)
        ax.set_title(LABELS[method])
        ax.grid(alpha=.2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(f"Left OOD: why low uncertainty can coexist with large error | repeat {seed}")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(.5, -.002), frameon=False)
    fig.tight_layout(rect=(0,.045,1,.96))
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_std(path, run_seeds, x, predictions, sigma_floor):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))
    for ax, method in zip(axes.flat, METHODS):
        errors = np.stack([
            np.abs(predictions[(s,method)]["mean_prediction"]-x**3) for s in run_seeds
        ])
        sigmas = np.stack([predictions[(s,method)]["std_used"] for s in run_seeds])
        for row in sigmas:
            ax.plot(x, row, color="teal", alpha=.12, lw=.7)
        for row in errors:
            ax.plot(x, np.maximum(row,1e-5), color="darkorange", alpha=.1, lw=.7)
        ax.plot(x, errors.mean(axis=0), color="darkorange", lw=2,
                label="Absolute error |mean − x³| (avg seeds)")
        ax.plot(x, sigmas.mean(axis=0), color="teal", lw=2,
                label="Predicted σ (avg seeds)")
        ax.axhline(NOISE_STD_TRUE, color="royalblue", lw=1.3, ls=":",
                   label="True noise std = 3")
        if method in ("gaussian_nll", "gaussian_at"):
            ax.axhline(sigma_floor, color="gray", lw=1, ls="--",
                       label="Normalized variance floor")
        ax.axvline(-4, color="gray", lw=.8, ls="--")
        ax.axvline(4, color="gray", lw=.8, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(0.008, 500)
        ax.set_title(LABELS[method])
        ax.set_xlabel("x")
        ax.set_ylabel("Original y units (log scale)")
        ax.grid(alpha=.18, which="both")
    handles, labels = axes.flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(.5, -.002), frameon=False, fontsize=9)
    fig.suptitle(f"Prediction error vs reported uncertainty ({len(run_seeds)} independent repeat(s))")
    fig.tight_layout(rect=(0,.08,1,.96))
    fig.savefig(path, dpi=165, bbox_inches="tight")
    plt.close(fig)


def plot_gaussian_left_floor(path, run_seeds, x, predictions, sigma_floor):
    fig, ax = plt.subplots(figsize=(9, 5.3))
    for method, linestyle, color in (
        ("gaussian_nll", "-", "darkorange"),
        ("gaussian_at", "-", "forestgreen"),
        ("deep_ensemble", "-", "purple")
    ):
        all_s = np.stack([predictions[(seed,method)]["std_used"] for seed in run_seeds])
        ax.plot(x, all_s.mean(0), ls=linestyle, c=color, lw=2, label=LABELS[method])
    ax.axhline(sigma_floor, ls="--", c="black", lw=1.3,
               label=f"Gaussian std floor ≈ {sigma_floor:.4f}")
    ax.axhline(NOISE_STD_TRUE, ls=":", c="royalblue", lw=1.3,
               label="True noise std = 3")
    ax.axvspan(-6,-4,color="gray",alpha=.09,label="Left OOD")
    ax.axvline(-4,ls="--",c="gray",lw=1)
    ax.axvline(4,ls="--",c="gray",lw=1)
    ax.set(xlim=(-6,6),xlabel="x",ylabel="Predicted standard deviation (log scale)",
           title="Where does predicted variance approach its numerical floor?")
    ax.set_yscale("log")
    ax.grid(alpha=.2, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=165)
    plt.close(fig)


def main():
    config, seeds, y_scale, x, preds = get_inputs()
    if len(seeds) < 3:
        print("WARNING: fewer than three repeats; diagnostic only, not report-ready.")
    sigma_floor = y_scale * math.sqrt(VAR_FLOOR_NORM)
    plot_dir = OUTPUT_DIR / "figures"
    metrics_dir = OUTPUT_DIR / "metrics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(seeds, x, preds, sigma_floor)
    save_csv(metrics_dir / "diagnostic_by_seed.csv", summary)
    plot_left_comparison(plot_dir / "left_ood_intervals.png", seeds[0], x, preds)
    plot_error_vs_std(plot_dir / "error_vs_predicted_sigma.png", seeds, x, preds, sigma_floor)
    plot_gaussian_left_floor(plot_dir / "predicted_sigma_and_floor.png", seeds, x, preds, sigma_floor)
    sel = [r for r in summary if r["method"]=="gaussian_nll" and r["region"]=="left_ood"]
    avg_s = float(np.mean([r["mean_predicted_std"] for r in sel]))
    frac = float(np.mean([r["fraction_sigma_at_normalized_floor_1pct_tolerance"] for r in sel]))
    avg_mae = float(np.mean([r["mean_absolute_error_vs_clean"] for r in sel]))
    print("\n=== Step 08 — left OOD diagnosis ===")
    print("Run seeds:", seeds)
    print(f"Training target scale: {y_scale:.6f}")
    print(f"Theoretical minimum predicted Gaussian std: {sigma_floor:.6f}")
    print(f"Observed left-OOD Gaussian NLL std (avg runs): {avg_s:.6f}")
    print(f"Share within 1% of min std: {frac:.1%}")
    print(f"Left-OOD mean absolute error vs noiseless x^3: {avg_mae:.3f}")
    print(f"Saved 3 figures + diagnostic table in: {OUTPUT_DIR}")
    print("NOTE: clean_curve_coverage_95 is NOT coverage on noisy observations.")
    print("Use Step 07 aggregate coverage95 for noisy-test coverage in the report.")
    # No model fitting, no edits to 01..07 folders.


if __name__ == "__main__":
    main()
