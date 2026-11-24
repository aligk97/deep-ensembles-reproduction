"""Matplotlib figures for the regional analysis."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .compute_regional_metrics import NOISE_VARIANCE, REGIONS, Z95
from .load_saved_predictions import SEEDS
from .train_v1_v2_models import CONDITIONS

DEEP = "nll_ensemble_5"
PRIMARY = ("left_ood", "in_domain", "right_ood")
TITLES = dict(zip(REGIONS, ("Sol OOD", "Eğitim aralığı", "Sağ OOD", "Birleşik OOD")))
V2_COLORS = {"V2-A": "#27689c", "V2-C": "#bf5625"}


def save(figure, output, name):
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def error(row, factor=1):
    value = row["sample_std_ddof1"]
    return factor * value if value is not None else None


def empty(ax, message="Kayıt yok"):
    ax.text(.5, .5, message, ha="center", va="center", color="#586474", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def regional_v2(output, index):
    metrics = (("rmse_true", "RMSE (x³)", 1), ("rmse_noisy", "RMSE (gürültülü)", 1),
               ("predictive_nll", "NLL", 1), ("coverage_95", "%95 kapsama (%)", 100),
               ("interval_width_95", "%95 aralık genişliği", 1))
    figure, axes = plt.subplots(len(metrics), len(CONDITIONS), figsize=(15, 12), layout="constrained")
    figure.suptitle("V2-A ve V2-C · Bölgesel karşılaştırma\nNoktalar seed ortalaması, çubuklar örneklem SS",
                   fontsize=16)
    for row_index, (metric, ylabel, factor) in enumerate(metrics):
        for col_index, condition in enumerate(CONDITIONS):
            ax = axes[row_index, col_index]
            available = False
            for offset, (label, color) in zip((-.10, .10), V2_COLORS.items()):
                for i, region in enumerate(PRIMARY):
                    row = index.get((label, condition.name, region, metric))
                    if row is not None:
                        ax.errorbar(i + offset, row["mean"] * factor, yerr=error(row, factor),
                                    color=color, fmt="o", markersize=4, capsize=3)
                        available = True
            if available:
                ax.set_xticks(range(3), ("Sol", "İç", "Sağ"))
                ax.grid(axis="y", alpha=.2)
                if metric == "coverage_95":
                    ax.axhline(95, color="#56606a", linestyle="--", linewidth=.8)
            else:
                empty(ax, "V2-A / V2-C kaydı yok")
            if col_index == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title(condition.title)
    handles = [Line2D([], [], color=color, marker="o", linestyle="none", label=label)
               for label, color in V2_COLORS.items()]
    figure.legend(handles=handles, loc="outside lower center", ncol=2)
    save(figure, output, "01_v2a_vs_v2c_regional")


def plot_grids(output, grids, multiplier, title, stem):
    configs_and_seeds = [(label, seed) for label in V2_COLORS for seed in SEEDS]
    figure, axes = plt.subplots(len(configs_and_seeds), len(CONDITIONS), figsize=(15, 17),
                              sharex=True, sharey=True, layout="constrained")
    figure.suptitle(f"V2-A ve V2-C · Şekil 1 · {title}\nKayıtlı gürültüsüz çizim grid’i", fontsize=16)
    for grid_row, (label, seed) in zip(axes, configs_and_seeds):
        for col, (ax, condition) in enumerate(zip(grid_row, CONDITIONS)):
            for boundary in (-4, 4):
                ax.axvline(boundary, linestyle="--", color="#556471", linewidth=.8)
            available = grids.get((label, seed, condition.name))
            if available is not None:
                samples, prediction = available
                order = samples.x.flatten().argsort()
                x = samples.x[order].flatten()
                mean = prediction["mean"][order].flatten()
                color = V2_COLORS[label]
                if "total_variance" in prediction:
                    radius = multiplier * prediction["total_variance"][order].flatten().sqrt()
                    ax.fill_between(x, mean - radius, mean + radius, color=color, alpha=.18, linewidth=0)
                else:
                    ax.text(.04, .92, "Varyans kaydı yok", transform=ax.transAxes, fontsize=7)
                ax.plot(x, mean, color=color, linewidth=1.2)
                ax.plot(x, x.pow(3), color="#293440", linewidth=.95, linestyle=":")
            else:
                ax.text(.5, .5, "Grid tahmini yok", ha="center", va="center", color="#586474",
                        transform=ax.transAxes)
            ax.set_xlim(-6, 6)
            if col == 0:
                ax.set_ylabel(f"{label}\nseed {seed}")
    for ax, condition in zip(axes[0], CONDITIONS):
        ax.set_title(condition.title)
    for ax in axes[-1]:
        ax.set_xlabel("x")
    figure.legend(handles=[
        Line2D([], [], color=V2_COLORS["V2-A"], label="V2-A tahmin ortalaması"),
        Line2D([], [], color=V2_COLORS["V2-C"], label="V2-C tahmin ortalaması"),
        Line2D([], [], color="#293440", linestyle=":", label="x³"),
    ], loc="outside lower center", ncol=3)
    save(figure, output, stem)


def coverage_width(output, index, labels):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    figure.suptitle("Deep Ensemble · %95 kapsama ve aralık genişliği\nMoment eşlemeli Gaussian", fontsize=15)
    colors = plt.get_cmap("tab20")
    oracle_width = 2 * Z95 * np.sqrt(NOISE_VARIANCE)
    present = set()
    has_outlier = False
    for ax, region in zip(axes.flat, REGIONS):
        data = []
        for i, label in enumerate(labels):
            width = index.get((label, DEEP, region, "interval_width_95"))
            measured = index.get((label, DEEP, region, "coverage_95"))
            if width is not None and measured is not None:
                data.append((i, label, width, measured))
        ax.set_title(TITLES[region])
        if not data:
            empty(ax, "Deep Ensemble verisi yok")
            continue
        q1, q3 = np.quantile([width["mean"] for _, _, width, _ in data], [.25, .75])
        upper_fence = q3 + 1.5 * (q3 - q1) if len(data) >= 4 else float("inf")
        for i, label, width, measured in data:
            present.add(i)
            outlier = width["mean"] > upper_fence
            has_outlier |= outlier
            ax.errorbar(width["mean"], 100 * measured["mean"], xerr=error(width), yerr=error(measured, 100),
                        fmt="D" if outlier else "o", color=colors(i % 20), markersize=7 if outlier else 5,
                        markeredgecolor="#293440" if outlier else colors(i % 20), capsize=2)
            if outlier:
                ax.annotate(label, (width["mean"], 100 * measured["mean"]),
                            xytext=(5, -12), textcoords="offset points", fontsize=8)
        ax.axhline(95, color="#52616e", linestyle="--", linewidth=1, label="Nominal %95")
        ax.axvline(oracle_width, color="#52616e", linestyle=":", linewidth=1, label="x³ ve σ=3 referansı")
        ax.set_xlabel("Ortalama aralık genişliği")
        ax.set_ylabel("Gürültülü hedef kapsaması (%)")
        ax.grid(alpha=.18)
    handles = [Line2D([], [], color=colors(i % 20), marker="o", linestyle="none", label=labels[i])
               for i in sorted(present)]
    if present:
        handles.extend([Line2D([], [], color="#52616e", linestyle=style, label=label) for style, label in (
            ("--", "Nominal %95"), (":", "x³ ve σ=3 referansı"))])
    if has_outlier:
        handles.append(Line2D([], [], color="#293440", marker="D", linestyle="none", label="Q3 + 1.5 × IQR üstü"))
    if handles:
        figure.legend(handles=handles, loc="outside lower center", ncol=min(5, len(handles)))
    save(figure, output, "03_coverage_vs_interval_width")


def compare_ood(output, index, labels):
    figure, axes = plt.subplots(3, len(CONDITIONS), figsize=(15, 10 + max(0, len(labels) - 5) * .45),
                              layout="constrained")
    figure.suptitle("Sol OOD ve sağ OOD · seed ortalaması ve örneklem SS", fontsize=16)
    for rows, (metric, xlabel, factor) in zip(axes, (
        ("rmse_true", "RMSE (x³)", 1), ("rmse_noisy", "RMSE (gürültülü)", 1),
        ("coverage_95", "%95 kapsama (%)", 100),
    )):
        for col, (condition, ax) in enumerate(zip(CONDITIONS, rows)):
            ax.set_title(condition.title)
            entries = [(i, index.get((label, condition.name, region, metric)), offset, color)
                       for i, label in enumerate(labels) for region, offset, color in (
                           ("left_ood", -.13, "#27689c"), ("right_ood", .13, "#bf5625"))]
            if not any(row for _, row, _, _ in entries):
                empty(ax)
                continue
            for i, row, offset, color in entries:
                if row is not None:
                    ax.errorbar(row["mean"] * factor, i + offset, xerr=error(row, factor),
                                color=color, fmt="o", markersize=4, capsize=2)
            ax.set_ylim(len(labels) - .5, -.5)
            ax.set_yticks(range(len(labels)), labels if col == 0 else [""] * len(labels))
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", alpha=.2)
            if metric == "coverage_95":
                ax.axvline(95, color="#56606a", linestyle="--", linewidth=.8)
    figure.legend(handles=[Line2D([], [], color=color, marker="o", linestyle="none", label=label)
                           for color, label in (("#27689c", "Sol OOD"), ("#bf5625", "Sağ OOD"))],
                  loc="outside lower center", ncol=2)
    save(figure, output, "04_left_vs_right_ood")


def noise_variance(output, index, labels):
    figure, ax = plt.subplots(figsize=(10, max(4, .42 * len(labels) + 2)), layout="constrained")
    ax.set_title("Eğitim aralığında aleatoric varyans · seed ortalaması ve örneklem SS")
    ax.axvline(NOISE_VARIANCE, color="#293440", linestyle="--", label="Gerçek gürültü varyansı = 9")
    present = False
    for condition, offset, color in zip(CONDITIONS[1:], (-.2, 0, .2), ("#26735d", "#bf5625", "#27689c")):
        label = condition.title
        for i, config in enumerate(labels):
            row = index.get((config, condition.name, "in_domain", "aleatoric_variance"))
            if row is not None:
                present = True
                ax.errorbar(row["mean"], i + offset, xerr=error(row), color=color, fmt="o",
                            markersize=5, capsize=3, label=label)
                label = "_nolegend_"
    if present:
        ax.set_ylim(len(labels) - .5, -.5)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Ortalama aleatoric varyans")
        ax.grid(axis="x", alpha=.2)
    else:
        empty(ax, "Öğrenilmiş varyans kaydı yok")
    ax.legend(loc="best")
    save(figure, output, "05_noise_variance")


def make_figures(output, summary, grids):
    index = {(row["configuration"], row["model"], row["region"], row["metric"]): row for row in summary}
    labels = sorted({row["configuration"] for row in summary})
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9}):
        regional_v2(output, index)
        plot_grids(output, grids, Z95, "%95 Gaussian aralığı", "02a_figure1_gaussian_95")
        plot_grids(output, grids, 3, "±3σ Gaussian aralığı", "02b_figure1_gaussian_3sigma")
        coverage_width(output, index, labels)
        compare_ood(output, index, labels)
        noise_variance(output, index, labels)
