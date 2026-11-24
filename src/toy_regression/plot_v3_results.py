"""V3 Figure 1 panels and paired arm/region diagnostics from verified outputs."""
import numpy as np
import pandas as pd

from .plot_experiment_results import figure, save
from .train_v1_v2_models import CONDITIONS


def render(raw, out, scores):
    arm_ids = list(scores.arm_id.unique())
    regional = scores[scores.model.eq("nll_ensemble_5")]
    for arm in arm_ids:
        arm_dir = raw / "arms" / arm
        destination = out / arm
        destination.mkdir(parents=True, exist_ok=True)
        for seed in sorted(scores[scores.arm_id.eq(arm)].seed.unique()):
            folder = arm_dir / f"seed_{seed}"
            with np.load(folder / "data/train.npz") as obs:
                xs, ys = obs["x"].ravel(), obs["y_noisy"].ravel()
            fig = figure((14, 4))
            axes = fig.subplots(1, 4, sharex=True, sharey=True)
            for ax, cond in zip(axes, CONDITIONS):
                with np.load(folder / cond.name / "predictions_grid.npz") as pred:
                    x, y, mean, std = (pred[k].ravel() for k in ("x", "y_true", "mean", "std"))
                ax.fill_between(x, mean-3*std, mean+3*std, color="#dce9f2", label="±3σ")
                ax.fill_between(x, mean-1.959964*std, mean+1.959964*std, color="#a3c4db", label="%95 (moment)")
                ax.plot(x, y, color="#963c36", lw=1.3, label="x³")
                ax.plot(x, mean, color="#18354d", lw=1.5, label="Tahmin")
                ax.scatter(xs, ys, color="#171717", s=7, label="Eğitim verisi")
                for bound in (-4,4):
                    ax.axvline(bound, color="#777777", ls=":", lw=0.8)
                ax.set(xlim=(-6,6), xlabel="x", title=cond.title)
            axes[0].set_ylabel("y")
            axes[-1].legend(fontsize=7, loc="upper left")
            fig.suptitle(f"{arm} · seed {seed} · kesikli çizgilerin dışı OOD")
            save(fig, destination, f"figure1_seed_{seed}")
        fig = figure((11,6))
        for ax, cond in zip(fig.subplots(2,2).flat, CONDITIONS):
            for seed in sorted(scores[scores.arm_id.eq(arm)].seed.unique()):
                files = sorted((arm_dir / f"seed_{seed}" / cond.name / "histories").glob("member_*.csv"))
                histories = [pd.read_csv(f).total_loss for f in files]
                ax.plot(np.arange(1,len(histories[0])+1), np.mean(histories,axis=0), label=str(seed), lw=.8)
            ax.set(xlabel="Epoch", ylabel="Ortalama eğitim kaybı", title=cond.title, yscale="symlog")
            ax.legend(fontsize=7)
        fig.suptitle(arm)
        save(fig, destination, "learning_curves")
    regions = ("left_ood","in_domain","right_ood")
    short = [arm.split("_")[1].upper() for arm in arm_ids]
    for metric in ("rmse_true","interval_width_95","coverage_95","predictive_nll"):
        stats = regional[regional.metric.eq(metric)]
        fig = figure((12,4))
        for ax, region in zip(fig.subplots(1,3), regions):
            table = stats[stats.region.eq(region)].groupby("arm_id").value.agg(["mean","std"]).reindex(arm_ids)
            ax.errorbar(np.arange(len(arm_ids)), table["mean"], yerr=table["std"], fmt="o", ms=3, capsize=2)
            ax.set(xticks=np.arange(len(arm_ids)),xticklabels=short, title=region, xlabel="V3 konfigürasyonu")
            if metric=="coverage_95":
                ax.axhline(.95,ls=":",color="#963c36")
        fig.suptitle(f"NLL ensemble · {metric} · 3 seed ortalaması ± örneklem SS")
        save(fig, out, "arm_comparison_"+metric)
    fig = figure((12,4))
    levels = ["50","80","90","95","99_73"]
    x = [0.5,0.8,0.9,0.95,0.9973]
    for ax,region in zip(fig.subplots(1,3),regions):
        for arm, label in zip(arm_ids,short):
            sub = regional[(regional.region==region)&(regional.arm_id==arm)].groupby("metric").value.mean()
            ax.plot(x,[sub["coverage_"+l] for l in levels],lw=.85,label=label)
        ax.plot([0,1],[0,1],ls=":",color="black")
        ax.set(xlabel="Nominal kapsama",ylabel="Ölçülen kapsama",title=region,ylim=(-.02,1.02))
    fig.axes[-1].legend(ncol=2, fontsize=6)
    fig.suptitle("V3 NLL ensemble · moment-Gaussian aralıklarının kalibrasyonu")
    save(fig,out,"calibration")

