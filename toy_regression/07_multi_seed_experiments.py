r"""CMP4509 Toy Regression — Step 07: three independent repeats of Figure 1.

Run on Windows PowerShell (inside Desktop/deep-ensembles/toy_regression):
    .\.venv\Scripts\python.exe .\07_multi_seed_experiments.py

Reuses the existing 20 samples in outputs/toy_training_data.csv and NEVER
moves/deletes earlier files. All NEW output goes to outputs/07_seed_experiments/.

Default: 3 independent run seeds (42,123,2026), 4 methods/run:
  mse_ensemble  : 5 nets, 1-50-ReLU-1, Adam lr=.01, MSE, 3000 epochs
  gaussian_nll  : 1 net,  1-50-ReLU-2, Adam lr=.001, Gaussian NLL
  gaussian_at   : 1 net,  same as Gaussian NLL, plus FGSM (1% observed x-range)
  deep_ensemble : 5 nets, same as Gaussian NLL, no AT

Within each run, a model seed is run_seed*100 + member_index (0..4).
Those five member seeds do not count as five independent repetitions.

Shared held-out SYNTHETIC evaluation: 400 x positions in [-6,6], each with
20 independent noisy y observations generated once from x**3 + N(0, 3**2).
The same observations are used to evaluate every model/seed. We report:
  * RMSE_clean: prediction vs noiseless x**3 (not test RMSE on noisy labels)
  * NLL_moment: Gaussian NLL from mean and moment-matched total variance
  * coverage95 / width95: against independent NOISY observations, +/-1.96sd
  * NLL_exact_mixture: optional, for the probabilistic deep ensemble.
MSE ensemble NLL is only a Gaussian proxy from empirical member variance:
it was NOT trained probabilistically; its variance has a numeric floor 1e-6.

These are OUR scaled-down training/evaluation choices; not exact paper
hyperparameters. Reporting 3-run mean +/- sample standard deviation is
consistent with the CMP4509 project announcement.

Optional 1-seed short smoke test without touching production results:
    .\.venv\Scripts\python.exe .\07_multi_seed_experiments.py --epochs 10 --run-seeds 42 --output-dir outputs/07_smoke_test

If a session is interrupted, completed model checkpoints are reused on
relaunch provided the run configuration and source data are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

# Needed BEFORE initialising CUDA for deterministic CuBLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "outputs" / "toy_training_data.csv"
DEFAULT_OUT = ROOT / "outputs" / "07_seed_experiments"
RUN_SEEDS = (42, 123, 2026)
NUM_ENSEMBLE_MEMBERS = 5
EPOCHS_DEFAULT = 3000
HIDDEN = 50
MSE_LR = 0.01
NLL_LR = 0.001
VAR_NORM_FLOOR = 1e-6
VAR_Y_FLOOR = 1e-6  # Only for stable evaluation of empirical (MSE) variance.
AT_EPS_FRACTION = 0.01
TEST_SEED = 20261122
N_TEST_X = 400
N_NOISE_REPLICATES = 20
NOISE_STD = 3.0
Z95 = 1.959963984540054
METHODS = ("mse_ensemble", "gaussian_nll", "gaussian_at", "deep_ensemble")
METHOD_LABELS = {
    "mse_ensemble": "MSE Ensemble (5)",
    "gaussian_nll": "Single Gaussian NLL",
    "gaussian_at": "Gaussian NLL + AT",
    "deep_ensemble": "Deep Ensemble (5)",
}
RUN_VERSION = 1


class MSERegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    def forward(self, x):
        return self.network(x)


class GaussianRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 2))

    def forward(self, x):
        raw = self.network(x)
        mean = raw[:, :1]
        variance = F.softplus(raw[:, 1:2]) + VAR_NORM_FLOOR
        return mean, variance


def nll_loss(mean, var, target):
    """Eq (1) of Lakshminarayanan et al.; additive constant omitted in training."""
    return (0.5 * (var.log() + (target - mean).square() / var)).mean()


def read_train(path):
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV missing: {path}\nRun 01_generate_data.py first.")
    with path.open(newline="", encoding="utf-8-sig") as f:
        data = list(csv.DictReader(f))
    if len(data) != 20 or any("x" not in r or "y" not in r for r in data):
        raise ValueError("Expected 20 rows with columns 'x' and 'y' in existing CSV.")
    x = torch.tensor([[float(r["x"])] for r in data], dtype=torch.float32)
    y = torch.tensor([[float(r["y"])] for r in data], dtype=torch.float32)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Training CSV contains non-finite values.")
    return x, y, hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def mkdirs(method_root):
    paths = {k: method_root / k for k in ("figures", "models", "metrics")}
    for d in paths.values():
        d.mkdir(parents=True, exist_ok=True)
    return paths


def model_seed_for(run_seed, member_idx):
    return run_seed * 100 + member_idx


def train_or_restore(method, member_idx, run_seed, x_train, y_train, y_norm,
                     y_mean, y_scale, epsilon, data_hash, device, method_paths,
                     epochs, log_every):
    probabilistic = method != "mse_ensemble"
    has_at = method == "gaussian_at"
    lr = MSE_LR if not probabilistic else NLL_LR
    member_seed = model_seed_for(run_seed, member_idx)
    meta = {
        "version": RUN_VERSION, "method": method, "member_seed": member_seed,
        "member_index": member_idx, "run_seed": run_seed, "epochs": epochs,
        "lr": lr, "epsilon": epsilon if has_at else 0.0,
        "data_sha256": data_hash, "hidden_units": HIDDEN,
        "y_mean": y_mean, "y_scale": y_scale,
    }
    path = method_paths["models"] / f"member_{member_idx + 1:02d}.pt"
    history_path = method_paths["metrics"] / f"member_{member_idx + 1:02d}_loss.csv"
    model = (GaussianRegressor() if probabilistic else MSERegressor())
    model.to(device)

    if path.exists() and history_path.exists():
        try:
            stored = torch.load(path, map_location="cpu", weights_only=True)
            if stored.get("meta") == meta:
                model.load_state_dict(stored["model_state_dict"])
                model.eval()
                print(f"    Model {member_idx+1}: checkpoint yuklendi (yeniden egitilmedi).", flush=True)
                return model
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            pass

    # Model construction happens on CPU for reproducible weight initialization.
    torch.manual_seed(member_seed)
    model = (GaussianRegressor() if probabilistic else MSERegressor()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        if has_at:
            x_gradient_input = x_train.detach().clone().requires_grad_(True)
            attack_mu, attack_var = model(x_gradient_input)
            attack_loss = nll_loss(attack_mu, attack_var, y_norm)
            grad_x = torch.autograd.grad(attack_loss, x_gradient_input)[0]
            x_adv = (x_train + epsilon * grad_x.sign()).detach()

        optimizer.zero_grad(set_to_none=True)
        if method == "mse_ensemble":
            loss = F.mse_loss(model(x_train), y_train)
        else:
            mu, var = model(x_train)
            loss = nll_loss(mu, var, y_norm)
            if has_at:
                adv_mu, adv_var = model(x_adv)
                loss = loss + nll_loss(adv_mu, adv_var, y_norm)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            # Report CLEAN training loss (for AT: not combined objective).
            with torch.no_grad():
                if probabilistic:
                    mu_after, var_after = model(x_train)
                    clean_loss = float(nll_loss(mu_after, var_after, y_norm))
                    clean_loss += math.log(y_scale) + 0.5 * math.log(2 * math.pi)
                else:
                    clean_loss = float(F.mse_loss(model(x_train), y_train))
            if not math.isfinite(clean_loss):
                raise FloatingPointError(f"Non-finite loss: {method}, seed={member_seed}, epoch={epoch}")
            history.append({"epoch": epoch, "train_clean_loss": clean_loss})
            if epoch == 1 or epoch % 500 == 0 or epoch == epochs:
                print(f"    Model {member_idx+1} | epoch {epoch:4d}/{epochs} | clean loss {clean_loss:.5f}", flush=True)

    model.eval()
    write_csv(history_path, history, ["epoch", "train_clean_loss"])
    # Save checkpoint LAST. A missing history file means not yet fully completed.
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, path)
    return model


def predict_method(method, trained_models, x_grid, y_mean, y_scale):
    means = []
    vars_y = []
    with torch.no_grad():
        for model in trained_models:
            if method == "mse_ensemble":
                means.append(model(x_grid).squeeze(-1).double().cpu().numpy())
            else:
                mu_norm, var_norm = model(x_grid)
                means.append((mu_norm.squeeze(-1).double().cpu().numpy() * y_scale) + y_mean)
                vars_y.append(var_norm.squeeze(-1).double().cpu().numpy() * (y_scale ** 2))
    member_mu = np.stack(means, axis=0)   # [M, G]
    pred_mu = np.mean(member_mu, axis=0)
    var_epistemic = np.mean((member_mu - pred_mu) ** 2, axis=0)
    if method == "mse_ensemble":
        member_var = None
        var_aleatoric = np.zeros_like(pred_mu)
    else:
        member_var = np.stack(vars_y, axis=0)
        var_aleatoric = np.mean(member_var, axis=0)
    var_total = var_aleatoric + var_epistemic
    # For MSE, clamp zero variance only to avoid dividing by zero in proxy NLL.
    var_used = np.maximum(var_total, VAR_Y_FLOOR)
    return {
        "mean": pred_mu, "var_total": var_total, "var_used": var_used,
        "var_aleatoric": var_aleatoric, "var_epistemic": var_epistemic,
        "member_mu": member_mu, "member_var": member_var,
    }


def evaluate(method, prediction, x, noisy_obs, regions):
    pred_mu = prediction["mean"]
    var_used = prediction["var_used"]
    std_used = np.sqrt(var_used)
    out = []
    for region, mask in regions.items():
        target_clean = x[mask] ** 3
        target_noisy = noisy_obs[mask]  # [G_region, repeat_count]
        mu = pred_mu[mask, None]
        v = var_used[mask, None]
        residual = target_noisy - mu
        row = {
            "region": region,
            "rmse_clean": float(np.sqrt(np.mean((pred_mu[mask] - target_clean) ** 2))),
            "rmse_noisy": float(np.sqrt(np.mean(residual ** 2))),
            "nll_moment": float(np.mean(0.5 * (np.log(2 * math.pi * v) + residual ** 2 / v))),
            "coverage95": float(np.mean(np.abs(residual) <= Z95 * np.sqrt(v))),
            "width95": float(np.mean(2 * Z95 * std_used[mask])),
            "avg_aleatoric_std": float(np.mean(np.sqrt(prediction["var_aleatoric"][mask]))),
            "avg_epistemic_std": float(np.mean(np.sqrt(prediction["var_epistemic"][mask]))),
            "n_test_x": int(mask.sum()),
            "n_test_observations": int(target_noisy.size),
            "nll_exact_mixture": "",
        }
        if method == "deep_ensemble":
            # Uniform mixture p(y|x) = (1/5) sum_m N(y;mu_m,var_m).
            m_mu = prediction["member_mu"][:, mask, None]
            m_var = np.maximum(prediction["member_var"][:, mask, None], VAR_Y_FLOOR)
            per_component_lp = -0.5 * (
                np.log(2 * math.pi * m_var) + (target_noisy[None, :, :] - m_mu) ** 2 / m_var
            )
            mx = np.max(per_component_lp, axis=0)
            log_mix = mx + np.log(np.mean(np.exp(per_component_lp - mx), axis=0))
            row["nll_exact_mixture"] = float(-np.mean(log_mix))
        out.append(row)
    return out


def plot_method_prediction(path, method, prediction, x_grid, x_train, y_train):
    fig, ax = plt.subplots(figsize=(9, 5))
    draw_prediction(ax, method, prediction, x_grid, x_train, y_train)
    ax.set_title(METHOD_LABELS[method] + " — 3-sigma band")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def draw_prediction(ax, method, p, x, x_train, y_train):
    mu = p["mean"]
    s = np.sqrt(p["var_used"])
    ax.plot(x, x**3, c="royalblue", lw=1.9, label="True y=x^3")
    ax.fill_between(x, mu-3*s, mu+3*s, color="gray", alpha=0.25, label="Predicted +/-3 sd")
    ax.plot(x, mu, color="darkorange", lw=1.8, label="Mean prediction")
    ax.scatter(x_train, y_train, c="crimson", s=16, zorder=4, label="20 train points")
    ax.axvline(-4, ls="--", c="gray", alpha=.4)
    ax.axvline(4, ls="--", c="gray", alpha=.4)
    ax.set(xlim=(-6, 6), ylim=(-290,290), xlabel="x", ylabel="y")
    ax.grid(alpha=.15)


def save_four_panels(path, predictions, x_grid, x_train, y_train, seed):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, method in zip(axes.flat, METHODS):
        draw_prediction(ax, method, predictions[method], x_grid, x_train, y_train)
        ax.set_title(METHOD_LABELS[method])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Figure 1 reproduction | repeat seed {seed}", fontsize=14)
    fig.tight_layout(rect=(0, .04, 1, .97))
    fig.savefig(path, dpi=175)
    plt.close(fig)


def aggregate_results(raw, run_seeds, output_dir):
    cols = ["rmse_clean", "rmse_noisy", "nll_moment", "coverage95", "width95",
            "avg_aleatoric_std", "avg_epistemic_std", "nll_exact_mixture"]
    summary = []
    for method in METHODS:
        for region in ("left_ood", "in_distribution", "right_ood", "all"):
            subset = [r for r in raw if r["method"] == method and r["region"] == region]
            if len(subset) != len(run_seeds):
                raise AssertionError(f"Missing result for {method} {region}")
            row = {"method": method, "region": region, "n_runs": len(subset)}
            for col in cols:
                values = np.asarray([float(r[col]) for r in subset if r[col] != ""], dtype=np.float64)
                row[f"{col}_mean"] = float(np.mean(values)) if values.size else ""
                row[f"{col}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else ""
            summary.append(row)
    summary_cols = ["method", "region", "n_runs"] + [f"{k}_{suffix}" for k in cols for suffix in ("mean", "sd")]
    write_csv(output_dir / "aggregate" / "metrics" / "summary_mean_sd.csv", summary, summary_cols)
    return summary


def plot_summary(path, summary, n_runs):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    stats = [
        ("rmse_clean", "RMSE against noiseless y=x^3 (lower is better)"),
        ("nll_moment", "Gaussian moment NLL on noisy observations (lower is better)"),
        ("coverage95", "Observed 95% coverage on noisy observations"),
        ("width95", "Mean 95% interval width (narrower if calibrated)"),
    ]
    display_regions = ("left_ood", "in_distribution", "right_ood")
    palette = ["#577590", "#43aa8b", "#f8961e"]
    for ax, (column, title) in zip(axes.flat, stats):
        positions = np.arange(len(METHODS))
        width = .24
        for j, region in enumerate(display_regions):
            vals, sd = [], []
            for method in METHODS:
                row = next(r for r in summary if r["region"] == region and r["method"] == method)
                vals.append(float(row[f"{column}_mean"]))
                s = row[f"{column}_sd"]
                sd.append(float(s) if s != "" else 0.)
            x_pos = positions + (j - 1) * width
            ax.bar(x_pos, vals, width=width, label=region, color=palette[j], alpha=.85)
            ax.errorbar(x_pos, vals, yerr=sd, fmt="none", ecolor="black", capsize=2, lw=.75)
        if column == "nll_moment":
            # NLL may span several orders of magnitude after variance collapse.
            # Symmetric-log scale preserves finite error bars even if mean-sd < 0.
            ax.set_yscale("symlog", linthresh=10)
            ax.set_ylabel("symmetric log scale", fontsize=8)
        if column == "coverage95":
            ax.axhline(.95, ls="--", color="black", linewidth=1, alpha=.65)
        ax.set_xticks(positions)
        ax.set_xticklabels(["MSE 5", "NLL 1", "NLL+AT", "Deep 5"])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ["Left OOD [-6,-4)", "Train range [-4,4]", "Right OOD (4,6]"],
               loc="lower center", bbox_to_anchor=(.5, .060), ncol=3, frameon=False)
    fig.suptitle(f"{n_runs} independent repeat(s) | bars=mean, error bars=sample std", fontsize=14)
    fig.text(.5, .018, "MSE ensemble NLL is an empirical-Gaussian proxy (not a learned likelihood).",
             ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=(0, .14, 1, .96))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="CMP4509 Figure 1 multi-seed experiments")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(RUN_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.run_seeds)) != len(args.run_seeds):
        parser.error("epochs must be positive and run-seeds must be unique")
    return args


def main():
    args = parse_args()
    # Controlled CuBLAS reproducibility (small CPU-built networks).
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)  # Small 20-point full batches run well with one CPU thread.
    x_cpu, y_cpu, dataset_hash = read_train(INPUT_CSV)
    y_mean = float(y_cpu.mean())
    y_scale = max(float(y_cpu.std(unbiased=False)), 1.0)
    y_norm_cpu = (y_cpu - y_mean) / y_scale
    epsilon = AT_EPS_FRACTION * float(x_cpu.max() - x_cpu.min())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(-6, 6, N_TEST_X, dtype=np.float64)
    eval_rng = np.random.default_rng(TEST_SEED)
    y_eval = x_grid[:, None] ** 3 + eval_rng.normal(0, NOISE_STD, (N_TEST_X, N_NOISE_REPLICATES))
    regions = {
        "left_ood": x_grid < -4,
        "in_distribution": (x_grid >= -4) & (x_grid <= 4),
        "right_ood": x_grid > 4,
        "all": np.ones(N_TEST_X, dtype=bool),
    }
    run_info = {
        "run_seeds": list(args.run_seeds), "num_members_per_ensemble": NUM_ENSEMBLE_MEMBERS,
        "member_seed_rule": "run_seed * 100 + member_index (starting at 0)",
        "epochs_per_model": args.epochs, "data_file": str(INPUT_CSV),
        "data_sha256": dataset_hash, "number_train": len(x_cpu),
        "mse_lr": MSE_LR, "gaussian_lr": NLL_LR,
        "gaussian_target_normalization": {"mean": y_mean, "scale": y_scale},
        "fgsm_epsilon": epsilon, "fgsm_fraction_of_observed_x_range": AT_EPS_FRACTION,
        "eval_seed": TEST_SEED, "eval_x_points": N_TEST_X,
        "eval_noise_repeats_per_x": N_NOISE_REPLICATES,
        "eval_noise_sd": NOISE_STD, "metric_z95": Z95,
        "variance_floor_for_evaluation_y2": VAR_Y_FLOOR,
        "source_paper": "Lakshminarayanan et al. 2017 arXiv:1612.01474 Figure 1",
        "simplifications": "50-ReLU, full-batch, fixed original 20-point dataset; user-chosen x range",
        "torch_version": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    (out / "experiment_config.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print("=== CMP4509 Toy Regression | Step 07: Multi-Seed ===", flush=True)
    print(f"Device: {device} " + (f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)
    print(f"Run seeds: {args.run_seeds}; epochs/model: {args.epochs}; models/run: 12", flush=True)
    print(f"Data: 20 FIXED points | FGSM epsilon={epsilon:.6f} | output={out}", flush=True)
    if len(args.run_seeds) < 3:
        print("UYARI: Bu bir kisa test; hoca icin en az 3 bagimsiz tekrar gerekir.", flush=True)

    x_train = x_cpu.to(device)
    y_train = y_cpu.to(device)
    y_norm = y_norm_cpu.to(device)
    x_grid_dev = torch.tensor(x_grid, dtype=torch.float32, device=device).unsqueeze(1)
    all_rows = []
    all_columns = ["run_seed", "method", "region", "rmse_clean", "rmse_noisy", "nll_moment",
                   "coverage95", "width95", "avg_aleatoric_std", "avg_epistemic_std",
                   "n_test_x", "n_test_observations", "nll_exact_mixture"]
    for run_index, seed in enumerate(args.run_seeds, 1):
        print(f"\n======== RUN {run_index}/{len(args.run_seeds)} seed={seed} ========", flush=True)
        run_root = out / f"seed_{seed}"
        pred_by_method = {}
        for method in METHODS:
            print(f"\n  --- {METHOD_LABELS[method]} ---", flush=True)
            paths = mkdirs(run_root / method)
            count = NUM_ENSEMBLE_MEMBERS if "ensemble" in method else 1
            models = []
            for member_idx in range(count):
                model = train_or_restore(
                    method, member_idx, seed, x_train, y_train, y_norm,
                    y_mean, y_scale, epsilon, dataset_hash, device, paths,
                    args.epochs, log_every=50,
                )
                models.append(model)
            pred = predict_method(method, models, x_grid_dev, y_mean, y_scale)
            pred_by_method[method] = pred
            rows = evaluate(method, pred, x_grid, y_eval, regions)
            for row in rows:
                row.update({"run_seed": seed, "method": method})
            write_csv(paths["metrics"] / "region_metrics.csv", rows, all_columns)
            all_rows.extend(rows)
            save_rows = []
            for i, xx in enumerate(x_grid):
                sigma = math.sqrt(float(pred["var_used"][i]))
                save_rows.append({
                    "x": float(xx), "true_y": float(xx**3),
                    "mean_prediction": float(pred["mean"][i]),
                    "aleatoric_variance": float(pred["var_aleatoric"][i]),
                    "epistemic_variance": float(pred["var_epistemic"][i]),
                    "total_variance": float(pred["var_total"][i]),
                    "std_used": sigma,
                    "lower_3sd": float(pred["mean"][i] - 3*sigma),
                    "upper_3sd": float(pred["mean"][i] + 3*sigma),
                    "lower_95": float(pred["mean"][i] - Z95*sigma),
                    "upper_95": float(pred["mean"][i] + Z95*sigma),
                })
            write_csv(paths["metrics"] / "predictions.csv", save_rows, list(save_rows[0]))
            plot_method_prediction(paths["figures"] / "prediction.png", method, pred,
                                   x_grid, x_cpu.numpy().ravel(), y_cpu.numpy().ravel())
            row_train = next(r for r in rows if r["region"] == "in_distribution")
            print(f"  In-range: clean RMSE={row_train['rmse_clean']:.3f}  "
                  f"NLL={row_train['nll_moment']:.3f}  "
                  f"95% cover={row_train['coverage95']:.1%}", flush=True)
            del models
        run_fig_dir = run_root / "figures"
        run_fig_dir.mkdir(exist_ok=True)
        save_four_panels(run_fig_dir / "figure1_four_panels.png", pred_by_method,
                         x_grid, x_cpu.numpy().ravel(), y_cpu.numpy().ravel(), seed)
        print(f"  Grafik kaydedildi: {run_fig_dir / 'figure1_four_panels.png'}", flush=True)

    aggregate_metrics = out / "aggregate" / "metrics"
    aggregate_figures = out / "aggregate" / "figures"
    aggregate_metrics.mkdir(parents=True, exist_ok=True)
    aggregate_figures.mkdir(parents=True, exist_ok=True)
    write_csv(aggregate_metrics / "metrics_by_seed.csv", all_rows, all_columns)
    summary = aggregate_results(all_rows, args.run_seeds, out)
    plot_summary(aggregate_figures / "summary_metrics.png", summary, len(args.run_seeds))
    print("\n====== SEED OZET (IN DISTRIBUTION) ======", flush=True)
    print(f"{'Method':<19} {'RMSE_clean mean+/-sd':<25} {'Coverage95 mean+/-sd':<24} {'Width95 mean+/-sd':<22}", flush=True)
    for method in METHODS:
        r = next(s for s in summary if s["method"] == method and s["region"] == "in_distribution")
        def fmt(name):
            sd = r[f"{name}_sd"]
            return f"{r[f'{name}_mean']:.3f} +/- {sd:.3f}" if sd != "" else f"{r[f'{name}_mean']:.3f}"
        print(f"{method:<19} {fmt('rmse_clean'):<25} {fmt('coverage95'):<24} {fmt('width95'):<22}", flush=True)
    print(f"\nTAMAMLANDI. Sonuclar: {out}", flush=True)
    print(f"Ozet CSV : {aggregate_metrics / 'summary_mean_sd.csv'}", flush=True)
    print(f"Ozet PNG : {aggregate_figures / 'summary_metrics.png'}", flush=True)
    print("NOT: mean+/-sd, run seeds uzerindendir; model uyelerinden degil.", flush=True)


if __name__ == "__main__":
    main()
