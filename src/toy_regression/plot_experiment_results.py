import csv

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from .evaluate_predictions import COVERAGE_LEVELS
from .train_v1_v2_models import CONDITIONS

COLORS = ("#777777", "#276da3", "#b25435", "#258473")


def figure(size):
    fig = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=300)
    fig.savefig(directory / f"{name}.pdf")
    fig.clear()


def train_preview(train, grid):
    fig = figure((7, 4))
    ax = fig.subplots()
    ax.plot(grid.x, grid.y_true, color="#2364aa", label="True function")
    ax.scatter(train.x, train.y_noisy, color="#c43c35", label="Noisy training points (20)")
    ax.set(xlabel="x", ylabel="y")
    ax.legend()
    return fig


def draw_figure1(axes, seed_dir, config, seed):
    with np.load(seed_dir / "data" / "train.npz") as archive:
        train = dict(archive)
    with np.load(seed_dir / "data" / "grid.npz") as archive:
        x, truth = archive["x"].ravel(), archive["y_true"].ravel()
    for ax, condition in zip(axes, CONDITIONS):
        with np.load(seed_dir / condition.name / "predictions_grid.npz") as archive:
            mean = archive["mean"].ravel()
            radius = config.plot_sigma_multiplier * archive["std"].ravel()
        ax.fill_between(x, mean - radius, mean + radius, color="#989898", alpha=0.35,
                        label="±3σ predictive band")
        ax.plot(x, mean, color="#555555", label="Predictive mean")
        ax.plot(x, truth, color="#2364aa", lw=1.6, label="True function")
        ax.scatter(train["x"], train["y_noisy"], color="#c43c35", s=11, zorder=4,
                   label="Noisy observations")
        for boundary in (config.train_x_min, config.train_x_max):
            ax.axvline(boundary, color="#cccccc", linestyle=":", zorder=0)
        ax.set(title=condition.title, xlabel="x", xlim=(config.plot_min, config.plot_max))
    axes[0].set_ylabel(f"y · seed {seed}")
    axes[-1].legend(loc="upper left", fontsize=8)


def plot_seed(seed_dir, config, seed, splits, predictions, results):
    directory = seed_dir / "figures"
    directory.mkdir()
    qualifier = " · smoke" if config.smoke else ""

    fig = figure((14, 3.7))
    draw_figure1(fig.subplots(1, 4, sharex=True, sharey=True), seed_dir, config, seed)
    fig.suptitle(f"Figure 1 recreation · seed {seed}{qualifier}")
    save(fig, directory, "figure1_four_panels")

    fig = figure((12, 7))
    for ax, condition in zip(fig.subplots(2, 2).flat, CONDITIONS):
        for path in sorted((seed_dir / condition.name / "histories").glob("member_*.csv")):
            with path.open(encoding="utf-8", newline="") as stream:
                history = list(csv.DictReader(stream))
            epochs = [int(row["epoch"]) for row in history]
            line, = ax.plot(epochs, [float(row["total_loss"]) for row in history],
                            label=f"Member {int(path.stem.split('_')[-1]) + 1}")
            if condition.adversarial:
                for column, label, style in (("clean_loss", "Clean", "--"),
                                             ("adversarial_loss", "Adversarial", ":")):
                    ax.plot(epochs, [float(row[column]) for row in history],
                            color=line.get_color(), ls=style, label=label)
        ax.set(title=condition.title, xlabel="Epoch", ylabel="Training loss", yscale="symlog")
        ax.legend(fontsize=8)
    fig.suptitle(f"Training losses · seed {seed}{qualifier}")
    save(fig, directory, "loss_curves")

    grid = splits["grid"]
    prediction = predictions["nll_ensemble_5"]
    fig = figure((8, 4.5))
    ax = fig.subplots()
    ax.plot(grid.x, prediction.total_variance, label="Total")
    ax.plot(grid.x, prediction.aleatoric, label="Aleatoric")
    ax.plot(grid.x, prediction.epistemic, label="Epistemic")
    for boundary in (config.train_x_min, config.train_x_max):
        ax.axvline(boundary, color="#999999", ls=":")
    ax.set(title=f"NLL ensemble · seed {seed}{qualifier}", xlabel="x", ylabel="Predictive variance")
    ax.legend()
    save(fig, directory, "uncertainty_decomposition")

    fig = figure((10, 4.5))
    for ax, domain in zip(fig.subplots(1, 2, sharey=True), ("in", "ood")):
        ax.plot([0, 100], [0, 100], color="#aaaaaa", ls=":", label="Ideal")
        for color, condition, result in zip(COLORS, CONDITIONS, results):
            label = condition.title + (" (proxy)" if not condition.gaussian else
                                       " (mixture)" if condition.ensemble else "")
            ax.plot(COVERAGE_LEVELS,
                    [100 * result["metrics"][f"coverage{level}_{domain}"] for level in COVERAGE_LEVELS],
                    marker="o", color=color, label=label)
        ax.set(title="In-domain" if domain == "in" else "OOD", xlabel="Nominal coverage (%)",
               xlim=(0, 100), ylim=(0, 100))
    fig.axes[0].set_ylabel("Empirical coverage (%)")
    fig.axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Noisy test coverage · seed {seed}{qualifier}")
    save(fig, directory, "coverage_comparison")

    fig = figure((12, 4.5))
    for ax, condition in zip(fig.subplots(1, 2, sharex=True, sharey=True), (CONDITIONS[0], CONDITIONS[3])):
        ax.plot(grid.x, grid.y_true, color="#2364aa", ls="--", label="True function")
        for index, mean in enumerate(predictions[condition.name].member_means, 1):
            ax.plot(grid.x, mean, label=f"Member {index}", alpha=0.8)
        ax.set(title=condition.title, xlabel="x")
        ax.legend(fontsize=8)
    fig.axes[0].set_ylabel("y")
    fig.suptitle(f"Member predictions · seed {seed}{qualifier}")
    save(fig, directory, "predictions_by_member")


def plot_aggregate(directory, run_dir, config, stats):
    fig = figure((14, 3.5 * len(config.outer_seeds)))
    axes = fig.subplots(len(config.outer_seeds), 4, squeeze=False, sharex=True, sharey=True)
    for row, seed in zip(axes, config.outer_seeds):
        draw_figure1(row, run_dir / f"seed_{seed}", config, seed)
    fig.suptitle("Figure 1 recreation by outer seed" + (" · smoke" if config.smoke else ""))
    save(fig, directory, "figure1_seed_comparison")

    fig = figure((12, 8))
    cards = [
        ("rmse_noisy", "Noisy test RMSE", CONDITIONS),
        ("coverage95", "95% coverage · Gaussian predictions", CONDITIONS[1:]),
        ("predictive_nll", "Predictive NLL · Gaussian densities and mixture", CONDITIONS[1:]),
        ("proxy_gaussian_nll", "Gaussian-proxy NLL · MSE disagreement", CONDITIONS[:1]),
    ]
    for ax, (metric, title, conditions) in zip(fig.subplots(2, 2).flat, cards):
        positions = np.arange(len(conditions))
        for offset, domain, label in ((-0.13, "in", "In-domain"), (0.13, "ood", "OOD")):
            entries = [stats[c.name][f"{metric}_{domain}"] for c in conditions]
            ax.errorbar(positions + offset, [entry["mean"] for entry in entries],
                        yerr=[entry["sample_std"] for entry in entries]
                        if len(config.outer_seeds) > 1 else None,
                        fmt="o", capsize=4, label=label)
        if metric == "coverage95":
            ax.axhline(0.95, color="#aaaaaa", ls=":")
        ax.set_title(title)
        ax.set_xticks(positions, [c.title for c in conditions], fontsize=8)
        ax.legend(fontsize=8)
    n = len(config.outer_seeds)
    fig.suptitle(f"Across outer seeds (n={n})" + (" · smoke" if config.smoke else "") +
                 (" · mean ± sample SD" if n > 1 else " · mean only"))
    save(fig, directory, "metrics_comparison")
