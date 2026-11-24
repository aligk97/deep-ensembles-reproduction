r"""CMP4509 | Toy Regression | Step 5: Gaussian NLL + adversarial training.

Windows PowerShell, from the EXISTING toy_regression directory:
    .\.venv\Scripts\python.exe .\05_train_gaussian_nll_at.py

Reads the SAME 20 examples from outputs/toy_training_data.csv.
Writes into the EXISTING outputs/ folder (no new folders):
    gaussian_nll_at_single_model.pt
    gaussian_nll_at_single_prediction.png
    gaussian_nll_at_single_loss.png
    gaussian_nll_at_single_predictions.csv
    gaussian_nll_at_single_training_log.csv
    gaussian_nll_at_single_summary.csv
    gaussian_nll_at_vs_no_at.png          (if Step 4 baseline CSV is present)
    toy_figure1_panels_1_to_3.png        (if Steps 3 and 4 CSVs are present)

Paper: Lakshminarayanan, Pritzel & Blundell (NeurIPS 2017),
Sec. 2.2.1 Eq. 1 (Gaussian NLL), Sec. 2.3 (FGSM), Algorithm 1 (clean + AT),
and Sec. 3.2 Figure 1, panel 3 (single NLL + AT network).

Paper-faithful AT details:
    x_adv = x + epsilon * sign(d loss(theta,x,y) / d x)
    epsilon = 0.01 * (max(x_train) - min(x_train))
    optimize GaussianNLL(clean x, y) + GaussianNLL(adversarial x, y)
    adversarial labels are unchanged. Each step generates new x_adv from
    CURRENT model; NOT using a fixed dataset of adversarial samples.

Our simplifications / ablation control:
    - Same x, y data and same 1-50-ReLU-2 architecture as Step 4;
    - Adam lr=0.001, 3000 full-batch epochs, initialization seed=123;
    - y normalized with TRAIN data stats for stability;
    - variances via softplus + 1e-6 (in normalized y coordinates);
    - no clipping of x_adv, consistent with the paper's un-clipped formula.

NLL histories are on original y units, include Gaussian constant; for AT
we log the clean NLL separately from the combined optimization objective.
The combined training loss is the SUM of two NLL terms, not comparable in
magnitude to the clean-only loss from Step 4.
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
DATA_CSV = OUT / "toy_training_data.csv"
NLL_BASELINE_PRED_CSV = OUT / "gaussian_nll_single_lr0001_predictions.csv"
NLL_BASELINE_LOG_CSV = OUT / "gaussian_nll_single_lr0001_training_log.csv"
MSE_ENSEMBLE_CSV = OUT / "mse_ensemble_5_predictions.csv"

SEED = 123
HIDDEN_UNITS = 50
EPOCHS = 3000
LEARNING_RATE = 0.001
MIN_NORMALIZED_VARIANCE = 1e-6
FGSM_EPS_FRACTION = 0.01  # 1% of the observed training input range
X_DISPLAY_MIN, X_DISPLAY_MAX = -6.0, 6.0
X_TRAIN_DESIGN_MIN, X_TRAIN_DESIGN_MAX = -4.0, 4.0


class GaussianRegressor(nn.Module):
    """Same architecture as Step 4: one x -> 50 ReLU -> (mu, var)."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(x)
        mu = raw[:, :1]
        var = F.softplus(raw[:, 1:2]) + MIN_NORMALIZED_VARIANCE
        return mu, var


def read_training_data(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(
            f"Data not found: {path}\nFirst run 01_generate_data.py."
        )
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or not all("x" in r and "y" in r for r in rows):
        raise ValueError("Expected exactly 20 rows and x, y columns.")
    x = torch.tensor([[float(r["x"])] for r in rows], dtype=torch.float32)
    y = torch.tensor([[float(r["y"])] for r in rows], dtype=torch.float32)
    return x, y


def nll_no_constant(
    mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    # Gaussian NLL, without its additive 0.5*log(2*pi) constant.
    return (0.5 * (var.log() + (target - mean).square() / var)).mean()


def get_fgsm_adversarial_inputs(
    model: GaussianRegressor,
    x: torch.Tensor,
    y_norm: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FGSM perturbation is re-computed from the CURRENT model per epoch."""
    x_attack = x.detach().clone().requires_grad_(True)
    mu_attack, var_attack = model(x_attack)
    attack_loss = nll_no_constant(mu_attack, var_attack, y_norm)

    # Only differentiate w.r.t. input x, not to perform the weight update here.
    dx = torch.autograd.grad(attack_loss, x_attack)[0]
    x_adv = (x + epsilon * dx.sign()).detach()
    return x_adv, dx.detach()


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_context(ax: plt.Axes, x_data, y_data, x_grid, y_ground_truth) -> None:
    ax.plot(x_grid, y_ground_truth, c="royalblue", lw=2.3,
            label="Ground truth: y = x^3")
    ax.scatter(x_data, y_data, color="crimson", s=34, zorder=5,
               label="20 noisy training samples")
    ax.axvline(X_TRAIN_DESIGN_MIN, ls="--", color="gray", alpha=0.55)
    ax.axvline(X_TRAIN_DESIGN_MAX, ls="--", color="gray", alpha=0.55)
    ax.set(xlim=(X_DISPLAY_MIN, X_DISPLAY_MAX), xlabel="x", ylabel="y")
    ax.grid(alpha=0.2)


def load_prediction_csv(path: Path):
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    needed = ("x", "mean_prediction", "lower_3std", "upper_3std")
    if not rows or not all(key in rows[0] for key in needed):
        raise ValueError(f"Unexpected prediction columns in {path.name}")
    return {key: np.asarray([float(row[key]) for row in rows]) for key in needed}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x_cpu, y_cpu = read_training_data(DATA_CSV)
    y_mean = float(y_cpu.mean())
    y_scale = max(float(y_cpu.std(unbiased=False)), 1.0)
    y_norm_cpu = (y_cpu - y_mean) / y_scale
    nll_offset = math.log(y_scale) + 0.5 * math.log(2 * math.pi)

    # Paper: epsilon is one percent of the observed range for this input.
    epsilon = FGSM_EPS_FRACTION * float((x_cpu.max() - x_cpu.min()).item())
    if not epsilon > 0:
        raise ValueError("All x inputs are identical: epsilon cannot be set.")

    # Same initial weights, optimizer, epochs and data as stable Step 4 model.
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = x_cpu.to(device)
    y_train = y_cpu.to(device)
    y_norm = y_norm_cpu.to(device)
    model = GaussianRegressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Step 5: Gaussian NLL + FGSM adversarial training (1 network)")
    print(f"Dataset: {len(x_cpu)} training points (same CSV)")
    print(f"Network: 1 -> {HIDDEN_UNITS} (ReLU) -> 2 (mean + variance)")
    print(f"Epochs: {EPOCHS} | Adam lr: {LEARNING_RATE} | seed: {SEED}")
    print(f"FGSM epsilon = 0.01 * (max x - min x) = {epsilon:.6f}")
    print("Training objective: NLL(clean) + NLL(FGSM), each in normalized units")
    print("No clipping of adversarial x; same target y.")

    history: list[tuple[int, float, float, float]] = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        # Step A: choose the direction that increases the CURRENT NLL.
        x_adv, input_grad = get_fgsm_adversarial_inputs(model, x_train, y_norm, epsilon)
        if not torch.isfinite(x_adv).all():
            raise FloatingPointError(f"Non-finite adversarial input at epoch {epoch}")
        if not torch.isfinite(input_grad).all():
            raise FloatingPointError(f"Non-finite input gradient at epoch {epoch}")

        # Step B: update the SAME model using clean + adversarial examples.
        optimizer.zero_grad(set_to_none=True)
        mu_clean, var_clean = model(x_train)
        mu_adv, var_adv = model(x_adv)
        loss_clean = nll_no_constant(mu_clean, var_clean, y_norm)
        loss_adv = nll_no_constant(mu_adv, var_adv, y_norm)
        combined = loss_clean + loss_adv
        if not torch.isfinite(combined):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        combined.backward()
        optimizer.step()

        # Report the UPDATED model's clean NLL for apples-to-apples comparison.
        with torch.no_grad():
            mu_after, var_after = model(x_train)
            clean_nll = float(nll_no_constant(mu_after, var_after, y_norm)) + nll_offset
        history.append((epoch, clean_nll, float(loss_clean.detach()), float(loss_adv.detach())))
        if epoch == 1 or epoch % 500 == 0:
            print(f"Epoch {epoch:4d}/{EPOCHS} | clean training NLL: {clean_nll:.4f}")

    model.eval()
    with torch.no_grad():
        train_mu_norm, train_var_norm = model(x_train)
        train_mu = train_mu_norm * y_scale + y_mean
        train_var = train_var_norm * (y_scale ** 2)
        train_rmse = torch.mean((train_mu - y_train).square()).sqrt().item()
        train_nll = (
            0.5 * (
                torch.log(2 * math.pi * train_var)
                + (y_train - train_mu).square() / train_var
            )
        ).mean().item()

        x_grid = torch.linspace(X_DISPLAY_MIN, X_DISPLAY_MAX, 400, device=device).unsqueeze(1)
        mu_norm_grid, var_norm_grid = model(x_grid)
        mu_grid = (mu_norm_grid * y_scale + y_mean).cpu().numpy().ravel()
        var_grid = (var_norm_grid * (y_scale ** 2)).cpu().numpy().ravel()

    x_np = x_grid.cpu().numpy().ravel()
    true_np = x_np ** 3
    std_np = np.sqrt(var_grid)
    lower_np = mu_grid - 3 * std_np
    upper_np = mu_grid + 3 * std_np
    x_train_np = x_cpu.numpy().ravel()
    y_train_np = y_cpu.numpy().ravel()

    saved: list[Path] = []
    ckpt = OUT / "gaussian_nll_at_single_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "seed": SEED,
        "hidden_units": HIDDEN_UNITS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "fgsm_epsilon_fraction": FGSM_EPS_FRACTION,
        "fgsm_epsilon_absolute": epsilon,
        "training_objective": "NLL(clean) + NLL(FGSM)",
        "y_mean": y_mean,
        "y_scale": y_scale,
        "train_rmse": train_rmse,
        "train_nll_original_units": train_nll,
    }, ckpt)
    saved.append(ckpt)

    log = OUT / "gaussian_nll_at_single_training_log.csv"
    with log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "clean_nll_original_y_units", "clean_loss_normalized_no_const", "adversarial_loss_normalized_no_const"])
        writer.writerows(history)
    saved.append(log)

    preds = OUT / "gaussian_nll_at_single_predictions.csv"
    with preds.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "ground_truth", "mean_prediction", "predicted_variance", "predicted_std", "lower_3std", "upper_3std"])
        for row in zip(x_np, true_np, mu_grid, var_grid, std_np, lower_np, upper_np):
            writer.writerow([float(v) for v in row])
    saved.append(preds)

    summ = OUT / "gaussian_nll_at_single_summary.csv"
    with summ.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seed", "epsilon_fraction", "epsilon_absolute", "learning_rate", "training_rmse", "training_nll_original_y_units"])
        writer.writerow(["gaussian_nll_at_single", SEED, FGSM_EPS_FRACTION, epsilon, LEARNING_RATE, train_rmse, train_nll])
    saved.append(summ)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    draw_context(ax, x_train_np, y_train_np, x_np, true_np)
    ax.fill_between(x_np, lower_np, upper_np, color="gray", alpha=0.23,
                    label="Gaussian mean +/- 3 predicted std")
    ax.plot(x_np, lower_np, color="gray", lw=1.0)
    ax.plot(x_np, upper_np, color="gray", lw=1.0)
    ax.plot(x_np, mu_grid, color="darkorange", lw=2.2,
            label="Single Gaussian NLL + AT model")
    ax.set_title("Toy Regression | Gaussian NLL + FGSM AT - Figure 1 panel 3")
    ax.legend(loc="upper left", fontsize=9)
    saved.append(save_fig(fig, "gaussian_nll_at_single_prediction.png"))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot([r[0] for r in history], [r[1] for r in history], c="darkorange")
    ax.set(xlabel="Epoch", ylabel="Clean Gaussian NLL (original y units)",
           title="Training loss - single Gaussian NLL + adversarial training")
    ax.grid(alpha=0.2)
    saved.append(save_fig(fig, "gaussian_nll_at_single_loss.png"))

    baseline = load_prediction_csv(NLL_BASELINE_PRED_CSV)
    if baseline is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharex=True, sharey=True)
        for ax in axes:
            draw_context(ax, x_train_np, y_train_np, x_np, true_np)
        axes[0].fill_between(baseline["x"], baseline["lower_3std"], baseline["upper_3std"],
                             color="gray", alpha=0.23, label="Mean +/- 3 std")
        axes[0].plot(baseline["x"], baseline["mean_prediction"],
                     color="darkorange", lw=2, label="Single NLL mean")
        axes[0].set_title("Panel 2: Gaussian NLL, no AT")
        axes[1].fill_between(x_np, lower_np, upper_np, color="gray", alpha=0.23,
                             label="Mean +/- 3 std")
        axes[1].plot(x_np, mu_grid, color="darkorange", lw=2, label="NLL + AT mean")
        axes[1].set_title("Panel 3: Gaussian NLL + AT")
        for ax in axes:
            ax.legend(loc="upper left", fontsize=8)
        saved.append(save_fig(fig, "gaussian_nll_at_vs_no_at.png"))
    else:
        print("Step 4 prediction CSV not found; skipping 2-panel comparison image.")

    mse = load_prediction_csv(MSE_ENSEMBLE_CSV)
    if mse is not None and baseline is not None:
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), sharex=True, sharey=True)
        for ax in axes:
            draw_context(ax, x_train_np, y_train_np, x_np, true_np)
        scenarios = [
            (mse, "Panel 1: MSE Ensemble (5)", "MSE Ensemble mean"),
            (baseline, "Panel 2: Single Gaussian NLL", "Single Gaussian NLL mean"),
            ({"x": x_np, "mean_prediction": mu_grid, "lower_3std": lower_np, "upper_3std": upper_np},
             "Panel 3: Single Gaussian NLL + AT", "NLL + AT mean"),
        ]
        for ax, (data, title, label) in zip(axes, scenarios):
            ax.fill_between(data["x"], data["lower_3std"], data["upper_3std"],
                            color="gray", alpha=0.24, label="Mean +/- 3 std")
            ax.plot(data["x"], data["mean_prediction"], c="darkorange", lw=2, label=label)
            ax.set_title(title)
            ax.legend(loc="upper left", fontsize=7.5)
        saved.append(save_fig(fig, "toy_figure1_panels_1_to_3.png"))
    else:
        print("MSE Ensemble and/or Step 4 prediction CSV missing; skipping 3-panel figure.")

    if NLL_BASELINE_LOG_CSV.exists():
        with NLL_BASELINE_LOG_CSV.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and "training_nll_original_y_units" in rows[0]:
            fig, ax = plt.subplots(figsize=(9, 4.6))
            ax.plot([int(r["epoch"]) for r in rows],
                    [float(r["training_nll_original_y_units"]) for r in rows],
                    c="royalblue", label="Gaussian NLL (no AT)")
            ax.plot([r[0] for r in history], [r[1] for r in history],
                    c="darkorange", label="Gaussian NLL + AT (clean NLL)")
            ax.set(xlabel="Epoch", ylabel="Clean NLL (original y units)",
                   title="Clean training NLL: Gaussian NLL vs Gaussian NLL + AT")
            ax.grid(alpha=0.2)
            ax.legend()
            saved.append(save_fig(fig, "gaussian_nll_at_vs_no_at_loss.png"))

    print("\n=== TRAIN METRICS (same 20 points, NOT a test set) ===")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Train Gaussian NLL (original units): {train_nll:.4f}")
    print("\nx      true y     pred mean    pred std    true in +/-3sigma?")
    checks = torch.tensor([[-6.], [-4.], [0.], [4.], [6.]], device=device)
    with torch.no_grad():
        mu_c, var_c = model(checks)
        mu_c = (mu_c * y_scale + y_mean).cpu().numpy().ravel()
        sd_c = (var_c.sqrt() * y_scale).cpu().numpy().ravel()
    for xv, mu, sd in zip([-6., -4., 0., 4., 6.], mu_c, sd_c):
        inside = abs(xv ** 3 - float(mu)) <= 3 * float(sd)
        print(f"{xv:+.0f}  {xv**3:+9.2f}  {mu:+10.2f}  {sd:10.2f}   {inside}")

    print("\nSaved files:")
    for p in saved:
        print(f"  {p}")
    print("\nNote: AT helps local smoothness, but does not guarantee correct uncertainty outside train support.")
    print("Three independent ensemble-level repetitions will be done later for reporting mean +/- std.")


if __name__ == "__main__":
    main()
