r"""CMP4509 | Toy Regression | Step 6: Deep Ensemble of 5 Gaussian NLL networks.

Windows PowerShell (in Desktop/deep-ensembles/toy_regression):
    .\.venv\Scripts\python.exe .\06_train_deep_ensemble.py

Reads the EXISTING outputs/toy_training_data.csv (same 20 points as steps 1-5).
Creates NO new directories beyond the already-used outputs/ folder.

Outputs (all inside outputs/):
    gaussian_nll_ensemble_5_model_01.pt ... model_05.pt
    gaussian_nll_ensemble_5_prediction.png
    gaussian_nll_ensemble_5_members.png
    gaussian_nll_ensemble_5_uncertainty_decomposition.png
    gaussian_nll_ensemble_5_loss.png
    gaussian_nll_ensemble_5_predictions.csv
    gaussian_nll_ensemble_5_training_log.csv
    gaussian_nll_ensemble_5_summary.csv
    gaussian_nll_ensemble_5_diagnostics.csv
    toy_figure1_all_4_panels.png (if prior step CSVs are present)

Scientific correspondence: Lakshminarayanan et al. (2017), Eq. (1), Sec. 2.4,
Figure 1 (fourth panel). M=5 independently trained probabilistic regressors;
equal-weight Gaussian mixture moment matching:
  mu_star = mean_m(mu_m)
  var_star = mean_m(var_m) + mean_m((mu_m - mu_star)**2).
  This run does NOT use adversarial training; Step 5 examined AT separately.

Our simplifications, not asserted to be the exact paper hyperparameters:
  20 fixed noisy points from same CSV; x generated from [-4,4];
  1-50-ReLU-2 network, Adam(lr=0.001), 3000 FULL-BATCH epochs;
  seeds 123..127; normalize y with train-set stats for NLL stability.
  Reported predictions and NLL are in ORIGINAL y units.

IMPORTANT: 5 different MODEL seeds inside one ensemble are NOT 5 independent
experimental repeats. Do 3 separately seeded ensemble runs later for mean +/- sd.
The +/-3 sigma bands are model-implied intervals, not verified coverage claims.
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

NUM_MODELS = 5
SEEDS = (123, 124, 125, 126, 127)
HIDDEN_UNITS = 50
EPOCHS = 3000
LEARNING_RATE = 0.001
MIN_NORMALIZED_VARIANCE = 1e-6
TRAIN_X_MIN, TRAIN_X_MAX = -4.0, 4.0
PLOT_MIN, PLOT_MAX, GRID_SIZE = -6.0, 6.0, 400

# Pre-existing predictions from Steps 3, 4 (stable LR), and 5.
PRIOR_PANELS = [
    ("mse_ensemble_5_predictions.csv", "1. MSE Ensemble (5)"),
    ("gaussian_nll_single_lr0001_predictions.csv", "2. Single Gaussian NLL"),
    ("gaussian_nll_at_single_predictions.csv", "3. Single NLL + AT"),
]


class GaussianRegressor(nn.Module):
    """Exactly the same probabilistic architecture as Steps 4 and 5."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.network(x)
        mu = outputs[:, :1]
        variance = F.softplus(outputs[:, 1:2]) + MIN_NORMALIZED_VARIANCE
        return mu, variance


def gaussian_nll_no_constant(mu: torch.Tensor, variance: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
    """Eq. (1) in normalized y units (Gaussian constant omitted)."""
    return (0.5 * (variance.log() + (target - mu).square() / variance)).mean()


def gaussian_nll_original(mu: torch.Tensor, variance: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
    """Full Gaussian NLL in original y units; can be compared with Step 4/5."""
    return (0.5 * (torch.log(2 * math.pi * variance)
                   + (target - mu).square() / variance)).mean()


def load_training_data() -> tuple[torch.Tensor, torch.Tensor]:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Missing {DATA_FILE}. Run 01_generate_data.py first.")
    with DATA_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 20 or any("x" not in row or "y" not in row for row in rows):
        raise ValueError("Expected 20 data rows and CSV columns x,y.")
    x = torch.tensor([[float(r["x"])] for r in rows], dtype=torch.float32)
    y = torch.tensor([[float(r["y"])] for r in rows], dtype=torch.float32)
    return x, y


def load_previous_predictions(name: str) -> dict[str, np.ndarray] | None:
    file = OUT / name
    if not file.exists():
        print(f"Previous panel unavailable: {name}")
        return None
    with file.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    columns = ["x", "mean_prediction", "lower_3std", "upper_3std"]
    if len(rows) != GRID_SIZE or any(c not in rows[0] for c in columns):
        raise ValueError(f"Unexpected columns/grid in {file.name}")
    result = {col: np.array([float(r[col]) for r in rows]) for col in columns}
    return result


def save_fig(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def context(ax: plt.Axes, x_plot: np.ndarray, y_true: np.ndarray,
            x_train: np.ndarray, y_train: np.ndarray) -> None:
    ax.plot(x_plot, y_true, color="royalblue", lw=2.2, label="Ground truth: y = x^3")
    ax.scatter(x_train, y_train, color="crimson", s=33, zorder=5,
               label="20 noisy training samples")
    ax.axvline(TRAIN_X_MIN, color="gray", ls="--", alpha=0.56)
    ax.axvline(TRAIN_X_MAX, color="gray", ls="--", alpha=0.56)
    ax.set(xlim=(PLOT_MIN, PLOT_MAX), xlabel="x", ylabel="y")
    ax.grid(alpha=0.2)


def uncertainty_plot(ax: plt.Axes, x_plot: np.ndarray, mu: np.ndarray,
                     low: np.ndarray, high: np.ndarray, label: str) -> None:
    ax.fill_between(x_plot, low, high, color="gray", alpha=0.23,
                    label="Mean +/- 3 predicted std")
    ax.plot(x_plot, low, color="gray", alpha=0.7, lw=1.0)
    ax.plot(x_plot, high, color="gray", alpha=0.7, lw=1.0)
    ax.plot(x_plot, mu, color="darkorange", lw=2.2, label=label)


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)
    x_cpu, y_cpu = load_training_data()
    y_mean = float(y_cpu.mean())
    y_scale = max(float(y_cpu.std(unbiased=False)), 1.0)
    y_normalized_cpu = (y_cpu - y_mean) / y_scale
    nll_offset = math.log(y_scale) + 0.5 * math.log(2 * math.pi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = x_cpu.to(device)
    y_norm = y_normalized_cpu.to(device)
    y_train = y_cpu.to(device)
    x_grid = torch.linspace(PLOT_MIN, PLOT_MAX, GRID_SIZE, device=device).unsqueeze(1)

    print("=== Deep Ensemble: 5 Gaussian NLL networks, NO adversarial training ===")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training points: {len(x_cpu)} (same CSV as earlier steps)")
    print(f"Architecture: 1 -> {HIDDEN_UNITS} ReLU -> 2 (mu, variance)")
    print(f"Seeds: {SEEDS} | Epochs/member: {EPOCHS} | Adam lr={LEARNING_RATE}")
    print(f"Training y mean={y_mean:.4f} / std={y_scale:.4f}")

    member_means = []
    member_variances = []
    member_train_means = []
    member_train_variances = []
    training_histories = []
    member_summaries = []

    for idx, seed in enumerate(SEEDS, start=1):
        torch.manual_seed(seed)
        # Build on CPU, then transfer; matches model initialization in earlier steps.
        model = GaussianRegressor().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        history = []
        model.train()
        print(f"\nTraining member {idx}/{NUM_MODELS} | seed={seed}")
        for epoch in range(1, EPOCHS + 1):
            optimizer.zero_grad(set_to_none=True)
            mu_norm, var_norm = model(x_train)
            loss = gaussian_nll_no_constant(mu_norm, var_norm, y_norm)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss not finite: member={idx}, epoch={epoch}")
            loss.backward()
            optimizer.step()

            # Full Gaussian NLL after weight update, in original units.
            with torch.no_grad():
                updated_mu, updated_var = model(x_train)
                nll_epoch = float(gaussian_nll_no_constant(
                    updated_mu, updated_var, y_norm).item()) + nll_offset
            history.append(nll_epoch)
            if epoch == 1 or epoch % 500 == 0:
                print(f"  Epoch {epoch:4d}/{EPOCHS} | clean train NLL: {nll_epoch:.4f}")

        model.eval()
        with torch.no_grad():
            train_mu_norm, train_var_norm = model(x_train)
            train_mu = train_mu_norm * y_scale + y_mean
            train_var = train_var_norm * (y_scale ** 2)
            rmse = float(torch.sqrt((train_mu - y_train).square().mean()).item())
            nll = float(gaussian_nll_original(train_mu, train_var, y_train).item())
            grid_mu_norm, grid_var_norm = model(x_grid)
            grid_mu = (grid_mu_norm * y_scale + y_mean).cpu().squeeze(-1)
            grid_var = (grid_var_norm * (y_scale ** 2)).cpu().squeeze(-1)

        checkpoint = OUT / f"gaussian_nll_ensemble_5_model_{idx:02d}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "seed": seed, "model_number": idx,
            "hidden_units": HIDDEN_UNITS, "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "min_normalized_variance": MIN_NORMALIZED_VARIANCE,
            "y_mean": y_mean, "y_scale": y_scale,
            "train_rmse": rmse, "train_gaussian_nll_original_units": nll,
            "adversarial_training": False,
        }, checkpoint)
        member_means.append(grid_mu)
        member_variances.append(grid_var)
        member_train_means.append(train_mu.squeeze(-1).detach().cpu())
        member_train_variances.append(train_var.squeeze(-1).detach().cpu())
        training_histories.append(history)
        member_summaries.append((idx, seed, rmse, nll))
        print(f"  Final member train RMSE={rmse:.4f} | Gaussian NLL={nll:.4f}")

    # Shape: 5 models x 400 grid points; y units already restored.
    mus = torch.stack(member_means, dim=0)
    vars_ = torch.stack(member_variances, dim=0)
    mu_star = mus.mean(dim=0)
    aleatoric = vars_.mean(dim=0)
    epistemic = mus.var(dim=0, correction=0)  # equal mixture weights: divide by M
    total_variance = aleatoric + epistemic
    std_total = torch.sqrt(total_variance)
    lo = mu_star - 3 * std_total
    hi = mu_star + 3 * std_total

    # Training RMSE/NLL of moment-matched ensemble Gaussian, not held-out metrics.
    tr_mus = torch.stack(member_train_means, dim=0)
    tr_vars = torch.stack(member_train_variances, dim=0)
    tr_mu_star = tr_mus.mean(dim=0)
    tr_total_var = tr_vars.mean(dim=0) + tr_mus.var(dim=0, correction=0)
    tr_target = y_cpu.squeeze(-1)
    ensemble_rmse = float((tr_mu_star - tr_target).square().mean().sqrt().item())
    ensemble_nll = float(gaussian_nll_original(
        tr_mu_star, tr_total_var, tr_target).item())

    x_np = x_grid.detach().cpu().numpy().ravel()
    ground_np = x_np**3
    train_x_np = x_cpu.numpy().ravel()
    train_y_np = y_cpu.numpy().ravel()
    mean_np = mu_star.numpy()
    var_np = total_variance.numpy()
    sd_np = std_total.numpy()
    low_np, high_np = lo.numpy(), hi.numpy()
    aleat_np, epis_np = aleatoric.numpy(), epistemic.numpy()
    mus_np, vars_np = mus.numpy(), vars_.numpy()

    # Save all machine-readable model results for reproducible comparisons.
    with (OUT / "gaussian_nll_ensemble_5_training_log.csv").open(
        "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + [f"model_{i}_clean_nll_original_units" for i in range(1, 6)])
        for epoch in range(EPOCHS):
            w.writerow([epoch + 1] + [history[epoch] for history in training_histories])

    with (OUT / "gaussian_nll_ensemble_5_predictions.csv").open(
        "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        fields = ["x", "ground_truth", "mean_prediction", "predicted_variance",
                  "predicted_std", "lower_3std", "upper_3std", "aleatoric_variance",
                  "epistemic_variance", "aleatoric_std", "epistemic_std"]
        fields += [f"model_{i}_mean" for i in range(1, 6)]
        fields += [f"model_{i}_variance" for i in range(1, 6)]
        w.writerow(fields)
        for j in range(len(x_np)):
            row = [x_np[j], ground_np[j], mean_np[j], var_np[j], sd_np[j],
                   low_np[j], high_np[j], aleat_np[j], epis_np[j],
                   math.sqrt(float(aleat_np[j])), math.sqrt(float(epis_np[j]))]
            row += [mus_np[i, j] for i in range(NUM_MODELS)]
            row += [vars_np[i, j] for i in range(NUM_MODELS)]
            w.writerow([float(v) for v in row])

    with (OUT / "gaussian_nll_ensemble_5_summary.csv").open(
        "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "model_number", "seed", "train_rmse", "train_nll_original_y_units"])
        for idx, seed, rmse, nll in member_summaries:
            w.writerow(["member", idx, seed, rmse, nll])
        w.writerow(["ensemble_moment_matched", "all", "123-127", ensemble_rmse, ensemble_nll])

    # Explicit numerical checks at exact x positions (NOT rounded CSV-grid matches).
    checks = [-6.0, -4.0, 0.0, 4.0, 6.0]
    with (OUT / "gaussian_nll_ensemble_5_diagnostics.csv").open(
        "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "ground_truth", "ensemble_mean", "total_std",
                    "aleatoric_std", "epistemic_std", "lower_3std", "upper_3std",
                    "ground_truth_inside_3std"])
        print("\n   x      true    mean     std(total)   std(aleat)   std(epistemic)  in 3sigma?")
        with torch.no_grad():
            for xv in checks:
                xx = torch.tensor([[xv]], dtype=torch.float32, device=device)
                mm, vv = [], []
                # We do not keep live models to reduce memory; reconstruct from saved checkpoints.
                for idx in range(1, NUM_MODELS + 1):
                    ckpt = torch.load(OUT / f"gaussian_nll_ensemble_5_model_{idx:02d}.pt",
                                      map_location=device, weights_only=True)
                    current = GaussianRegressor().to(device)
                    current.load_state_dict(ckpt["model_state_dict"])
                    current.eval()
                    mn, vn = current(xx)
                    mm.append(float((mn * y_scale + y_mean).item()))
                    vv.append(float((vn * y_scale**2).item()))
                mean_v = float(np.mean(mm))
                var_a = float(np.mean(vv))
                var_e = float(np.mean((np.asarray(mm) - mean_v)**2))
                sd_v = math.sqrt(var_a + var_e)
                lo_v, hi_v = mean_v - 3*sd_v, mean_v + 3*sd_v
                inside = lo_v <= xv**3 <= hi_v
                w.writerow([xv, xv**3, mean_v, sd_v, math.sqrt(var_a),
                            math.sqrt(var_e), lo_v, hi_v, inside])
                print(f"{xv:+5.1f} {xv**3:+9.1f} {mean_v:+9.1f} {sd_v:10.2f}"
                      f" {math.sqrt(var_a):12.2f} {math.sqrt(var_e):15.2f}   {inside}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    context(ax, x_np, ground_np, train_x_np, train_y_np)
    uncertainty_plot(ax, x_np, mean_np, low_np, high_np, "Deep Ensemble mean (5 NLL models)")
    ax.set_title("Toy Regression | Deep Ensemble (5 Gaussian NLL networks)")
    ax.legend(loc="upper left", fontsize=8.5)
    save_fig(fig, "gaussian_nll_ensemble_5_prediction.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    context(ax, x_np, ground_np, train_x_np, train_y_np)
    for idx in range(NUM_MODELS):
        ax.plot(x_np, mus_np[idx], lw=1.4, alpha=0.82, label=f"Model {idx+1} (seed {SEEDS[idx]})")
    ax.plot(x_np, mean_np, c="black", lw=2.6, label="Ensemble mean")
    ax.set_title("Toy Regression | Mean predictions from 5 Gaussian NLL networks")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    save_fig(fig, "gaussian_nll_ensemble_5_members.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx, history in enumerate(training_histories):
        ax.plot(range(1, EPOCHS+1), history, lw=1.2, label=f"Model {idx+1} (seed {SEEDS[idx]})")
    ax.set(xlabel="Epoch", ylabel="Clean training Gaussian NLL (original y units)",
           title="Training Gaussian NLL: 5 independent ensemble members")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    save_fig(fig, "gaussian_nll_ensemble_5_loss.png")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x_np, np.sqrt(aleat_np), lw=2.1, label="Aleatoric std (mean predicted variance)")
    ax.plot(x_np, np.sqrt(epis_np), lw=2.1, label="Epistemic proxy std (member disagreement)")
    ax.plot(x_np, sd_np, lw=2.4, label="Total predictive std")
    ax.axvline(TRAIN_X_MIN, color="gray", ls="--", alpha=.5)
    ax.axvline(TRAIN_X_MAX, color="gray", ls="--", alpha=.5)
    ax.set(xlabel="x", ylabel="Standard deviation (original y units)",
           xlim=(PLOT_MIN, PLOT_MAX), title="Deep Ensemble uncertainty decomposition")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    save_fig(fig, "gaussian_nll_ensemble_5_uncertainty_decomposition.png")

    prior_data = [load_previous_predictions(filename) for filename, _ in PRIOR_PANELS]
    if all(item is not None for item in prior_data):
        fig, axs = plt.subplots(2, 2, figsize=(15.5, 10), sharex=True, sharey=True)
        panels = [(prior_data[i], PRIOR_PANELS[i][1]) for i in range(3)] + [
            ({"x":x_np, "mean_prediction":mean_np, "lower_3std":low_np,
              "upper_3std":high_np}, "4. Deep Ensemble (5 NLL models)")]
        for ax, (data, title) in zip(axs.ravel(), panels):
            context(ax, x_np, ground_np, train_x_np, train_y_np)
            uncertainty_plot(ax, data["x"], data["mean_prediction"],
                             data["lower_3std"], data["upper_3std"], "Predicted mean")
            ax.set_title(title)
            ax.legend(loc="upper left", fontsize=7.7)
            ax.set_ylim(-270, 320)  # same visible axes across all 4 panels
        fig.suptitle("Reproducing Figure 1: Toy Regression | Our simplified PyTorch setup", fontsize=15)
        save_fig(fig, "toy_figure1_all_4_panels.png")
        print("\nSaved 4-panel comparison: outputs/toy_figure1_all_4_panels.png")
    else:
        print("\nFour-panel plot skipped because prior prediction CSV(s) are missing.")
        print("Individual Deep Ensemble plots and all CSV/model files were saved.")

    print("\n=== TRAIN metrics on the SAME 20 samples (NOT independent test) ===")
    print(f"Deep Ensemble train RMSE: {ensemble_rmse:.4f}")
    print(f"Deep Ensemble moment-matched Gaussian train NLL: {ensemble_nll:.4f}")
    print(f"Saved results into: {OUT}")
    print("NOTE: 5 member seeds != 3 independent experimental repetitions.")
    print("NOTE: No AT in this panel. AT was examined in the previous step.")


if __name__ == "__main__":
    main()
