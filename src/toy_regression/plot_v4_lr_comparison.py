"""Paired V2-A/V4 figures using saved predictions and three-seed summaries."""
import numpy as np

from .plot_experiment_results import figure, save
from .train_v1_v2_models import CONDITIONS

REGION_NAMES = {"left_ood": "Sol OOD", "in_domain": "Eğitim aralığı", "right_ood": "Sağ OOD",
                "combined_ood": "Birleşik OOD"}
COLORS = ("#296385", "#a34024")


def render_v4(baseline, arm, table, destination, lr):
    for seed in (42, 123, 2026):
        fig = figure((14, 7))
        axes = fig.subplots(2, 4, sharex=True, sharey=True)
        for row, (source, label) in enumerate(((baseline, "V2-A · LR 0.1"), (arm, f"V4 · LR {lr}"))):
            folder = source / f"seed_{seed}"
            with np.load(folder / "data/train.npz") as data:
                x_obs, y_obs = data["x"].ravel(), data["y_noisy"].ravel()
            for ax, method in zip(axes[row], CONDITIONS):
                with np.load(folder / method.name / "predictions_grid.npz") as data:
                    x, mean, std, truth = (data[k].ravel() for k in ("x", "mean", "std", "y_true"))
                ax.fill_between(x, mean-3*std, mean+3*std, color="#dedede", label="±3σ")
                ax.fill_between(x, mean-1.959964*std, mean+1.959964*std, color="#a8bdc8", label="%95")
                ax.plot(x, truth, color="#373737", linestyle="--", label="x³")
                ax.plot(x, mean, color=COLORS[row], label="Tahmin")
                ax.scatter(x_obs, y_obs, s=8, color="#222222", label="Eğitim")
                for bound in (-4, 4):
                    ax.axvline(bound, color="#666666", linestyle=":", linewidth=.7)
                ax.set(xlim=(-6, 6), xlabel="x")
                if row == 0:
                    ax.set_title(method.title)
            axes[row, 0].set_ylabel(label + "\ny")
        axes[0,-1].legend(fontsize=7)
        fig.suptitle(f"Şekil 1 · seed {seed} · dikey çizgiler eğitim desteğinin sınırları")
        save(fig, destination, f"figure1_v2a_vs_v4_seed_{seed}")
    groups = [
        ("rmse_nll", [("rmse_true", "RMSE (x³)"), ("predictive_nll", "NLL")]),
        ("coverage_width", [("coverage_95", "%95 kapsama"), ("interval_width_95", "%95 genişlik"),
                            ("coverage_99_73", "±3σ kapsama"), ("interval_width_99_73", "±3σ genişlik")]),
    ]
    x = np.arange(len(REGION_NAMES))
    for name, metrics in groups:
        fig = figure((14, 2.8*len(metrics)))
        axes = fig.subplots(len(metrics), 4, squeeze=False)
        for row, (metric, label) in enumerate(metrics):
            for ax, method in zip(axes[row], CONDITIONS):
                data = table[(table.model==method.name) & (table.metric==metric)].set_index("region").loc[list(REGION_NAMES)]
                for column, tag, offset, color in (("v2a", "V2-A", -.09, COLORS[0]), ("v4", "V4", .09, COLORS[1])):
                    ax.errorbar(x+offset, data[column+"_mean"], yerr=data[column+"_std"], fmt="o",
                                color=color, ms=3, capsize=2, label=tag)
                if metric.startswith("coverage"):
                    ax.axhline(.95 if metric=="coverage_95" else .9973, linestyle=":", color="#555555")
                ax.set_xticks(x, REGION_NAMES.values(), rotation=20, ha="right", fontsize=7)
                ax.set_ylabel(label)
                if row==0:
                    ax.set_title(method.title)
        axes[0,-1].legend(fontsize=7)
        fig.suptitle("V2-A ve V4 · üç seed ortalaması ± örneklem SS")
        save(fig, destination, name)
    fig = figure((13,4))
    for ax, metric, label in zip(fig.subplots(1,3), ("aleatoric_variance","epistemic_variance","total_predictive_variance"),
                                 ("Aleatorik","Epistemik","Toplam")):
        data = table[(table.model=="nll_ensemble_5") & (table.metric==metric)].set_index("region").loc[list(REGION_NAMES)]
        for col, tag, offset, color in (("v2a","V2-A",-.08,COLORS[0]),("v4","V4",.08,COLORS[1])):
            ax.errorbar(x+offset, data[col+"_mean"], yerr=data[col+"_std"], fmt="o", capsize=2, label=tag, color=color)
        ax.set_xticks(x, REGION_NAMES.values(), rotation=20, ha="right")
        ax.set_title(label)
    fig.axes[-1].legend()
    fig.suptitle("NLL ensemble · tahmin varyansı · üç seed ortalaması ± SS")
    save(fig,destination,"variance_decomposition")

