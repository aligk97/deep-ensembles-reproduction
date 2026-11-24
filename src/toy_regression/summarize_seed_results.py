from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
from statistics import mean, stdev

from .experiment_config import load_config
from .evaluate_predictions import metric_rows, save_metrics_csv
from .plot_experiment_results import plot_aggregate
from .train_v1_v2_models import CONDITIONS
from .experiment_io import write_csv, write_json


CAPTION = (
    "Lakshminarayanan ve ark. (2017), Şekil 1'in bu çalışmadaki yeniden çizimi. "
    "Mavi çizgi gerçek x³, kırmızı noktalar her satırdaki ortak 20 gürültülü eğitim gözlemidir. "
    "Koyu gri çizgi tahmin ortalamasını, açık gri bant ortalama ±3σ’yı gösterir. "
    "MSE panelinde σ, üye tahminlerinin ampirik popülasyon standart sapmasıdır. "
    "NLL ensemble panelinde σ, Gaussian mixture’ın momentlerinden gelir. "
    "Her seed ayrı bir eğitim veri kümesidir. Paneller ortak y ekseni kullanır."
)
TABLE_METRICS = ("rmse_noisy_in", "rmse_noisy_ood", "predictive_nll_in", "predictive_nll_ood",
                 "coverage95_in", "coverage95_ood")


def summarize(results):
    groups = defaultdict(dict)
    condition_seeds = defaultdict(set)
    for result in results:
        condition_seeds[result["condition"]].add(result["outer_seed"])
        for row in metric_rows(result):
            key = row["condition"], row["metric"], row["metric_type"]
            if row["outer_seed"] in groups[key]:
                raise ValueError(f"Duplicate outer seed for {key}")
            groups[key][row["outer_seed"]] = row["value"]
    stats = defaultdict(dict)
    rows = []
    for (condition, metric, metric_type), by_seed in groups.items():
        if by_seed.keys() != condition_seeds[condition]:
            raise ValueError(f"Missing seed or inconsistent metric type for {condition}/{metric}")
        values = list(by_seed.values())
        available = [value for value in values if value != "N/A"]
        if available and len(available) != len(values):
            raise ValueError(f"Inconsistent availability across seeds for {condition}/{metric}")
        entry = {
            "metric_type": metric_type,
            "mean": mean(available) if available else None,
            "sample_std": stdev(available) if len(available) > 1 else None,
            "n": len(available),
        }
        stats[condition][metric] = entry
        rows.append({"condition": condition, "metric": metric, **entry})
    return dict(stats), rows


def display_value(entry, *, scale=1, tex=False):
    if entry["mean"] is None:
        return "N/A"
    center = f"{entry['mean'] * scale:.3f}"
    spread = entry["sample_std"]
    if spread is None:
        return center
    if tex:
        return r"$%s \pm %.3f$" % (center, spread * scale)
    return f"{center} ± {spread * scale:.3f}"


def write_table(directory, stats, config):
    n = len(config.outer_seeds)
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Method & RMSE in & RMSE OOD & NLL in & NLL OOD & 95\% in & 95\% OOD \\",
             r"\hline"]
    wide_rows = []
    for condition in CONDITIONS:
        entries = stats[condition.name]
        title = condition.title + (" (proxy scores)" if not condition.gaussian else "")
        cells = [display_value(entries[metric], scale=100 if metric.startswith("coverage") else 1,
                               tex=True) for metric in TABLE_METRICS]
        lines.append(" & ".join([title, *cells]) + r" \\")
        wide_rows.append({
            "condition": condition.name, "n_outer_seeds": n, "smoke": config.smoke,
            "predictive_metric_type": entries["predictive_nll_in"]["metric_type"],
            **{metric: display_value(entries[metric], scale=100 if metric.startswith("coverage") else 1)
               for metric in TABLE_METRICS},
        })
    mode = "Smoke run. " if config.smoke else ""
    count = "Means and sample standard deviations" if n > 1 else "Single-seed values (SD unavailable)"
    lines += [r"\hline", r"\end{tabular}}",
              rf"\caption{{{mode}{count} ($n={n}$ outer seeds), on held-out noisy targets. "
              r"MSE uses a Gaussian proxy. The NLL ensemble uses the Gaussian mixture density "
              r"and mixture quantile intervals. Coverage is in percent.}",
              r"\label{tab:toy-regression}", r"\end{table}"]
    (directory / "report_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(directory / "report_table.csv", wide_rows)


def findings(directory, stats, results, config, command, provenance_link,
             figure_link="figures/figure1_seed_comparison.png"):
    n = len(config.outer_seeds)
    mse, single, at, ensemble = [c.name for c in CONDITIONS]

    def value(condition, metric):
        return stats[condition][metric]["mean"]

    def shown(condition, metric):
        return display_value(stats[condition][metric])

    def gap(condition):
        return abs(value(condition, "coverage95_in") - 0.95)

    variance_support = gap(single) < gap(mse)
    ensemble_support = value(ensemble, "predictive_nll_ood") < value(single, "predictive_nll_ood")
    if config.smoke or n < 3:
        verdict = "Belirsiz. Yorum için en az üç tam dış seed gerekir."
    elif variance_support and ensemble_support:
        verdict = "Desteklendi. İki yönsel ölçüt bu koşuda sağlandı."
    elif variance_support or ensemble_support:
        verdict = "Kısmen desteklendi. İki yönsel ölçütten biri bu koşuda sağlandı."
    else:
        verdict = "Desteklenmedi. İki yönsel ölçüt de bu koşuda sağlanamadı."

    lines = [
        "# Toy regression bulguları", "",
        f"**Keşifsel değerlendirme.** {verdict}", "",
        "## Amaç", "",
        "Yirmi gözlemli kübik regresyonda NLL ile varyans öğrenmenin ve bağımsız üyeleri "
        "birleştirmenin tahmin belirsizliğine etkisi incelendi.", "",
        "## Makaleye uygunluk ve sapmalar", "",
        f"Adam LR {config.learning_rate:g}, minibatch {config.batch_size}, {config.max_epochs} epoch, "
        f"preset {config.preset}. Makale §3.2 toy deneyine özel eğitim süresi vermez. "
        "Örnekleme aralığı HLA2015 §5.3’ten alındı. PyTorch başlatması, ortak eksenler, "
        "bağımsız gürültülü testler ve üç-seed protokolü proje seçimleridir. "
        f"[Ayrıntılı kaynak ve seçim tablosu]({provenance_link}).", "",
        "## Dört yöntem ve ölçümler", "",
        f"Dış seed sayısı **n={n}**, seed’ler " + ", ".join(map(str, config.outer_seeds)) + ". "
        f"{'Değerler ortalama ± örneklem standart sapmasıdır.' if n > 1 else 'Tek seed için standart sapma N/A.'}"
        + (" Smoke ölçümleri rapora dahil edilmez." if config.smoke else ""), "",
        "| Koşul | RMSE iç | RMSE OOD | NLL iç | NLL OOD | %95 iç (%) | %95 OOD (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        title = condition.title + (" · proxy" if not condition.gaussian else "")
        entries = [display_value(stats[condition.name][metric],
                                 scale=100 if metric.startswith("coverage") else 1) for metric in TABLE_METRICS]
        lines.append("| " + " | ".join([title, *entries]) + " |")
    lines += [
        "", "NLL ensemble skorları Gaussian mixture yoğunluğundan, %95 kapsam mixture "
        "CDF kuantillerinden gelir. Moment-matched Gaussian sonuçları ayrı kaydedilir. "
        "MSE satırında varyans tabanlı Gaussian proxy kullanılır.", "",
        "## Eğrilerde gözlenen davranış", "",
        "Temiz grid üzerindeki yardımcı RMSE değerleri " +
        ", ".join(f"{condition.title} {shown(condition.name, 'rmse_clean_grid')}"
                  for condition in CONDITIONS) + ". " +
        f"[Seed panelleri]({figure_link}) her veri kümesinin tahmin eğrisini ve ±3σ bandını gösterir.",
        "", "## NLL ve MSE", "",
        f"Nominal %95 iç kapsamdan mutlak fark MSE proxy için {100 * gap(mse):.2f}, "
        f"tek NLL için {100 * gap(single):.2f} yüzde puandı. "
        f"OOD test NLL’si tek Gaussian için {shown(single, 'predictive_nll_ood')}, "
        f"ensemble mixture için {shown(ensemble, 'predictive_nll_ood')}.", "",
        "## Aleatoric ve epistemic belirsizlik", "",
        f"Ensemble iç test aleatoric varyansı {shown(ensemble, 'mean_aleatoric_in')}. "
        f"Veri üretim varyansı {config.noise_std ** 2:g}. Ensemble grid üye anlaşmazlığı "
        f"eğitim aralığında {shown(ensemble, 'mean_epistemic_train_region')}, "
        f"aralığın dışında {shown(ensemble, 'mean_epistemic_outside_train_region')}. "
        "Tek modeller için epistemic bileşen N/A, MSE ensemble için aleatoric bileşen N/A.", "",
        "## AT etkisi", "",
        f"Tek NLL OOD skoru {shown(single, 'predictive_nll_ood')}, AT ile "
        f"{shown(at, 'predictive_nll_ood')}. İç test skorları sırasıyla "
        f"{shown(single, 'predictive_nll_in')} ve {shown(at, 'predictive_nll_in')}.", "",
        "## Seed duyarlılığı", "",
    ]
    for seed in config.outer_seeds:
        paired = {item["condition"]: item["metrics"] for item in results if item["outer_seed"] == seed}
        delta = paired[ensemble]["predictive_nll_ood"] - paired[single]["predictive_nll_ood"]
        lines.append(f"- Seed {seed}, OOD NLL farkı (ensemble eksi tek NLL) {delta:+.3f}.")
    lines += [
        "", "## Sınırlamalar ve sonuç kontrolü", "",
        "- Varyans öğrenme iddiası için tek NLL’nin iç %95 kapsam hatasını MSE proxy ile karşılaştırın. "
        "Nominal kapsamdan daha küçük fark, bu ölçüt için olumlu yöndür.",
        "- Ensemble etkisi için eşleştirilmiş OOD mixture ve tek Gaussian NLL farklarını inceleyin. "
        "Daha düşük NLL olumlu yöndür.",
        "- Kapsamı RMSE, tahmin varyansı ve her seed’in bandıyla birlikte yorumlayın. Geniş bantlar "
        "kapsamı yükseltebilir. Yönsel karar bir istatistiksel anlamlılık testi içermez.",
        "- Şekil 1 nitel bir karşılaştırmadır. Gürültülü test skorları, x eksenindeki OOD tanımı ve "
        "bu seed’lerden elde edilen değerler bu çalışmanın ek analizleridir. Üye anlaşmazlığı "
        "epistemic belirsizlik için bir yaklaşımdır.",
        "- Tamamlanmamış koşular toplu analize alınamaz. Sorun giderilirse yeni bir run ID altında "
        "bütün seed’lerle yeniden çalıştırın.", "", "## Yeniden üretim", "", "~~~text",
        command, "~~~", "",
    ]
    (directory / "findings_tr.md").write_text("\n".join(lines), encoding="utf-8")


def read_results(run_dir, config):
    results = []
    for seed in config.outer_seeds:
        for condition in CONDITIONS:
            path = run_dir / f"seed_{seed}" / condition.name / "metrics.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            if (result["outer_seed"], result["condition"], result["smoke"]) != (
                    seed, condition.name, config.smoke):
                raise ValueError(f"Unexpected seed, condition or smoke status in {path}")
            results.append(result)
    return results


def build_aggregate(run_dir, config, command, *, destination=None):
    results = read_results(run_dir, config)
    stats, rows = summarize(results)
    directory = destination if destination is not None else run_dir / "aggregate"
    directory.mkdir(parents=True)
    figures = directory / "figures"
    figures.mkdir()
    save_metrics_csv(directory / "metrics_by_seed.csv", results)
    write_csv(directory / "metrics_mean_std.csv",
              [{**row, "mean": "N/A" if row["mean"] is None else row["mean"],
                "sample_std": "N/A" if row["sample_std"] is None else row["sample_std"]}
               for row in rows])
    write_json(directory / "metrics_mean_std.json", {
        "outer_seeds": list(config.outer_seeds), "smoke": config.smoke,
        "n_outer_seeds": len(config.outer_seeds), "ddof": 1, "conditions": stats,
    })
    plot_aggregate(figures, run_dir, config, stats)
    write_table(directory, stats, config)
    caption = ("Smoke çalıştırması. " if config.smoke else "") + CAPTION
    (directory / "report_figure_caption.txt").write_text(caption + "\n", encoding="utf-8")
    try:
        provenance_link = Path(os.path.relpath(run_dir / "provenance.md", directory)).as_posix()
    except ValueError:
        provenance_link = (run_dir / "provenance.md").resolve().as_uri()
    findings(directory, stats, results, config, command, provenance_link.replace(" ", "%20"))
    return directory


def export_report(run_dir, destination, manifest, aggregate, command):
    config = load_config(run_dir / "config_resolved.yaml")
    if config.smoke or len(config.outer_seeds) < 3:
        raise ValueError("Report export requires a complete non-smoke run with at least 3 outer seeds")
    target = Path(destination).resolve() / run_dir.name
    target.mkdir(parents=True)
    for filename in ("report_table.tex", "report_table.csv", "report_figure_caption.txt",
                     "metrics_mean_std.csv"):
        shutil.copy2(aggregate / filename, target)
    for filename in ("figure1_seed_comparison.pdf", "metrics_comparison.pdf"):
        shutil.copy2(aggregate / "figures" / filename, target)
    shutil.copy2(run_dir / "provenance.md", target)
    stats = json.loads((aggregate / "metrics_mean_std.json").read_text(encoding="utf-8"))["conditions"]
    findings(target, stats, read_results(run_dir, config), config, command,
             "provenance.md", "figure1_seed_comparison.pdf")
    write_json(target / "source_run.json", manifest)
    return target


def load_completed(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError(f"Run status is {manifest['status']}; inspect {run_dir / 'logs' / 'run.log'} "
                         "and failure.json if present. Rerun all requested seeds under a new run ID.")
    return manifest
