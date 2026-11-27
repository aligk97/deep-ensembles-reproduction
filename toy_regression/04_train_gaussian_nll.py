r"""CMP4509 | Toy Regression | Step 4: single Gaussian NLL network.

Windows PowerShell, inside the existing toy_regression folder:
    .\.venv\Scripts\python.exe .\04_train_gaussian_nll.py

Reads the SAME 20 samples from outputs/toy_training_data.csv.
Saves all files into the EXISTING outputs/ directory (does not create extra folders):
    gaussian_nll_single_model.pt
    gaussian_nll_single_prediction.png
    gaussian_nll_single_loss.png
    gaussian_nll_single_predictions.csv
    gaussian_nll_single_training_log.csv
    gaussian_nll_single_summary.csv
    gaussian_nll_vs_mse_ensemble_5.png  (if the baseline CSV exists)

Paper: Lakshminarayanan et al. (2017), Eq. (1), Sec. 2.2.1, Fig. 1 panel 2.
The network predicts mean and a positive INPUT-DEPENDENT Gaussian variance.
This is one network, WITHOUT adversarial training and WITHOUT an ensemble.

OUR declared simplifications compared with the paper:
  - input range [-4,4], 20 data points (range is our choice)
  - 1 hidden layer with 50 ReLU neurons (paper refers to another architecture)
  - Adam(lr=0.001), 3000 full-batch epochs, seed=123
  - normalize y using training-set mean and std for stable NLL optimization.
    Reported predictions, variances, RMSE and NLL are converted BACK to original units.

WARNING: An NLL loss lower than zero can be valid for continuous densities.
The logged NLL includes the Gaussian constant and uses ORIGINAL y units.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA_FILE = OUT / "toy_training_data.csv"
BASELINE_CSV = OUT / "mse_ensemble_5_predictions.csv"

MODEL_SEED = 123
HIDDEN_UNITS = 50
EPOCHS = 3000
LEARNING_RATE = 0.001
MIN_NORMALIZED_VARIANCE = 1e-6
X_TRAIN_MIN, X_TRAIN_MAX = -4.0, 4.0


class GaussianRegressor(nn.Module):
    """1 input -> 50 ReLU neurons -> 2 raw outputs: mu, variance parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_outputs = self.network(x)
        mean = raw_outputs[:, :1]
        # Paper footnote: softplus + a small positive floor ensures sigma^2 > 0.
        variance = F.softplus(raw_outputs[:, 1:2]) + MIN_NORMALIZED_VARIANCE
        return mean, variance


def read_training_data(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            "First run 01_generate_data.py from your toy_regression folder."
        )
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or any("x" not in r or "y" not in r for r in rows):
        raise ValueError("Expected 20 CSV rows with columns 'x' and 'y'.")
    x = torch.tensor([[float(r["x"])] for r in rows], dtype=torch.float32)
    y = torch.tensor([[float(r["y"])] for r in rows], dtype=torch.float32)
    return x, y


def gaussian_nll_without_constant(
    mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean of Eq. (1): 0.5*log(var) + 0.5*(target-mu)^2/var."""
    return (0.5 * (variance.log() + (target - mean).square() / variance)).mean()


def add_shared_context(ax: plt.Axes, train_x, train_y, plot_x, ground_truth) -> None:
    ax.plot(
        plot_x, ground_truth, color="royalblue", lw=2.2, label="Ground truth: y = x^3"
    )
    ax.scatter(
        train_x,
        train_y,
        s=34,
        color="crimson",
        zorder=4,
        label="20 noisy training samples",
    )
    ax.axvline(X_TRAIN_MIN, color="gray", ls="--", alpha=0.55)
    ax.axvline(X_TRAIN_MAX, color="gray", ls="--", alpha=0.55)
    ax.set(xlabel="x", ylabel="y", xlim=(-6, 6))
    ax.grid(alpha=0.2)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = OUT / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x_cpu, y_cpu = read_training_data(DATA_FILE)

    # Normalize targets ONLY for training stability. No held-out/test data is used.
    y_mean = float(y_cpu.mean().item())
    y_scale = max(float(y_cpu.std(unbiased=False).item()), 1.0)
    y_norm_cpu = (y_cpu - y_mean) / y_scale
    full_nll_offset = math.log(y_scale) + 0.5 * math.log(2.0 * math.pi)

    torch.manual_seed(MODEL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = x_cpu.to(device)
    y_train_norm = y_norm_cpu.to(device)
    model = GaussianRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {len(x_cpu)} training examples (same CSV as previous steps)")
    print(f"Network: 1 -> {HIDDEN_UNITS} (ReLU) -> 2 (mean + variance)")
    print(f"Epochs: {EPOCHS}; learning rate: {LEARNING_RATE}; seed: {MODEL_SEED}")
    print(f"y normalization: mean={y_mean:.4f}, std={y_scale:.4f} (train only)")

    history: list[tuple[int, float]] = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        mean_norm, variance_norm = model(x_train)
        loss = gaussian_nll_without_constant(mean_norm, variance_norm, y_train_norm)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss is not finite at epoch {epoch}.")
        loss.backward()
        optimizer.step()

        # Measure AFTER updating the weights so that the last log entry
        # agrees with the metrics computed from the saved final model.
        with torch.no_grad():
            updated_mu, updated_var = model(x_train)
            updated_loss = gaussian_nll_without_constant(
                updated_mu, updated_var, y_train_norm
            )
        # Full Gaussian NLL of original (not standardized) target y.
        nll_original = float(updated_loss.detach().cpu()) + full_nll_offset
        history.append((epoch, nll_original))
        if epoch == 1 or epoch % 500 == 0:
            print(f"Epoch {epoch:4d}/{EPOCHS} | train Gaussian NLL: {nll_original:.4f}")

    model.eval()
    with torch.no_grad():
        mean_train_norm, variance_train_norm = model(x_train)
        train_mean = mean_train_norm * y_scale + y_mean
        train_variance = variance_train_norm * (y_scale**2)
        train_rmse = float(
            torch.sqrt(((train_mean - y_cpu.to(device)) ** 2).mean()).item()
        )
        train_nll = float(
            (
                0.5
                * (
                    torch.log(2.0 * math.pi * train_variance)
                    + (y_cpu.to(device) - train_mean).square() / train_variance
                )
            )
            .mean()
            .item()
        )

        x_plot = torch.linspace(-6.0, 6.0, 400, device=device).unsqueeze(1)
        mu_plot_norm, var_plot_norm = model(x_plot)
        mu_plot = (mu_plot_norm * y_scale + y_mean).cpu().squeeze(-1).numpy()
        var_plot = (var_plot_norm * y_scale**2).cpu().squeeze(-1).numpy()

    x_np = x_plot.cpu().squeeze(-1).numpy()
    true_np = x_np**3
    std_np = np.sqrt(var_plot)
    lower_np = mu_plot - 3.0 * std_np
    upper_np = mu_plot + 3.0 * std_np
    train_x_np = x_cpu.squeeze(-1).numpy()
    train_y_np = y_cpu.squeeze(-1).numpy()

    checkpoint = OUT / "gaussian_nll_single_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_seed": MODEL_SEED,
            "hidden_units": HIDDEN_UNITS,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "min_normalized_variance": MIN_NORMALIZED_VARIANCE,
            "y_mean": y_mean,
            "y_scale": y_scale,
            "train_rmse": train_rmse,
            "train_gaussian_nll_original_units": train_nll,
        },
        checkpoint,
    )

    log_path = OUT / "gaussian_nll_single_training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "training_nll_original_y_units"])
        writer.writerows(history)

    pred_path = OUT / "gaussian_nll_single_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x",
                "ground_truth",
                "mean_prediction",
                "predicted_variance",
                "predicted_std",
                "lower_3std",
                "upper_3std",
            ]
        )
        for i in range(len(x_np)):
            writer.writerow(
                [
                    float(x_np[i]),
                    float(true_np[i]),
                    float(mu_plot[i]),
                    float(var_plot[i]),
                    float(std_np[i]),
                    float(lower_np[i]),
                    float(upper_np[i]),
                ]
            )

    summary_path = OUT / "gaussian_nll_single_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "seed",
                "training_rmse",
                "training_nll_original_y_units",
                "y_mean",
                "y_scale",
            ]
        )
        writer.writerow(
            ["gaussian_nll_single", MODEL_SEED, train_rmse, train_nll, y_mean, y_scale]
        )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    add_shared_context(ax, train_x_np, train_y_np, x_np, true_np)
    ax.fill_between(
        x_np,
        lower_np,
        upper_np,
        color="gray",
        alpha=0.22,
        label="Gaussian mean +/- 3 predicted std",
    )
    ax.plot(x_np, lower_np, color="gray", lw=1.1)
    ax.plot(x_np, upper_np, color="gray", lw=1.1)
    ax.plot(
        x_np,
        mu_plot,
        color="darkorange",
        lw=2.2,
        label="Single Gaussian NLL network mean",
    )
    ax.set_title("Toy Regression | Single Gaussian NLL network - Figure 1 panel 2")
    ax.legend(loc="upper left", fontsize=9)
    fig_path = save_figure(fig, "gaussian_nll_single_prediction.png")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot([a for a, _ in history], [b for _, b in history], color="darkorange")
    ax.set(
        xlabel="Epoch",
        ylabel="Gaussian NLL (original y units)",
        title="Training loss - single Gaussian NLL network",
    )
    ax.grid(alpha=0.2)
    loss_path = save_figure(fig, "gaussian_nll_single_loss.png")

    compare_path = None
    if BASELINE_CSV.is_file():
        with BASELINE_CSV.open("r", newline="", encoding="utf-8") as handle:
            baseline = list(csv.DictReader(handle))
        required = {"x", "mean_prediction", "lower_3std", "upper_3std"}
        if baseline and required.issubset(baseline[0]):
            bx = np.array([float(row["x"]) for row in baseline])
            bm = np.array([float(row["mean_prediction"]) for row in baseline])
            bl = np.array([float(row["lower_3std"]) for row in baseline])
            bu = np.array([float(row["upper_3std"]) for row in baseline])
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharex=True, sharey=True)
            for ax in axes:
                add_shared_context(ax, train_x_np, train_y_np, x_np, true_np)
            axes[0].fill_between(
                bx, bl, bu, color="gray", alpha=0.23, label="Empirical mean +/- 3 std"
            )
            axes[0].plot(
                bx, bm, color="darkorange", lw=2, label="MSE Ensemble (5) mean"
            )
            axes[0].set_title("Figure 1 panel 1: MSE Ensemble (5)")
            axes[1].fill_between(
                x_np,
                lower_np,
                upper_np,
                color="gray",
                alpha=0.23,
                label="Gaussian mean +/- 3 std",
            )
            axes[1].plot(
                x_np, mu_plot, color="darkorange", lw=2, label="Single NLL model mean"
            )
            axes[1].set_title("Figure 1 panel 2: Single Gaussian NLL")
            for ax in axes:
                ax.legend(loc="upper left", fontsize=8)
            compare_path = save_figure(fig, "gaussian_nll_vs_mse_ensemble_5.png")
        else:
            print("Baseline CSV has unexpected columns, skipping comparison figure.")
    else:
        print("Previous MSE Ensemble CSV not found; comparison image is skipped.")

    print("\n=== RESULTS (20 TRAINING POINTS; NOT HELD-OUT TEST SCORES) ===")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Train Gaussian NLL (original y units): {train_nll:.4f}")
    print("\nx       true y      pred mean      pred std      in mean +/- 3 std?")
    with torch.no_grad():
        checks = torch.tensor([[-6.0], [-4.0], [0.0], [4.0], [6.0]], device=device)
        check_mu_n, check_var_n = model(checks)
        check_mu = (check_mu_n * y_scale + y_mean).cpu().flatten().numpy()
        check_std = (check_var_n.sqrt() * y_scale).cpu().flatten().numpy()
    for xval, m, s in zip([-6.0, -4.0, 0.0, 4.0, 6.0], check_mu, check_std):
        true = xval**3
        inside = abs(float(m) - true) <= 3.0 * float(s)
        print(
            f"{xval:+.0f}   {true:+9.2f}    {m:+10.2f}      {s:9.2f}          {inside}"
        )

    print("\nSaved files:")
    for path in [checkpoint, fig_path, loss_path, pred_path, log_path, summary_path]:
        print(f"  {path}")
    if compare_path is not None:
        print(f"  {compare_path}")
    print(
        "\nIMPORTANT: NLL is NOT MSE; their numeric loss values cannot be directly compared."
    )
    print(
        "The predicted std is a learned noise estimate, NOT a guarantee about extrapolation error."
    )


if __name__ == "__main__":
    main()
