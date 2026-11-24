"""Write the Turkish analysis narrative and a LaTeX table fragment."""

from pathlib import Path
from statistics import mean

from .plot_regional_comparisons import TITLES
from .compute_regional_metrics import NOISE_VARIANCE, REGIONS, Z95
from .load_saved_predictions import SEEDS
from .train_v1_v2_models import CONDITIONS

DEEP = "nll_ensemble_5"


def number(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def summary_value(row, factor=1):
    if row is None:
        return "—"
    value = number(factor * row["mean"], 2 if factor == 100 else 3)
    deviation = row["sample_std_ddof1"]
    return (f"{value} ± {number(factor * deviation, 2 if factor == 100 else 3)}" if deviation is not None
            else f"{value} (n=1)")


def write_latex(path, summary):
    keys = ("rmse_true", "rmse_noisy", "predictive_nll", "coverage_95", "interval_width_95")
    index = {(row["configuration"], row["model"], row["region"], row["metric"]): row for row in summary}
    region_labels = dict(zip(REGIONS, ("Left OOD", "In-domain", "Right OOD", "Combined OOD")))
    model_labels = dict(zip((item.name for item in CONDITIONS), ("MSE (5)", "NLL", "NLL + AT", "Deep Ens.")))

    def tex(value):
        escapes = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$",
                   "{": r"\{", "}": r"\}", "\\": r"\textbackslash{}", "~": r"\textasciitilde{}",
                   "^": r"\textasciicircum{}"}
        return "".join(escapes.get(char, char) for char in value)

    def cell(row, factor):
        if row is None:
            return "--"
        average = row["mean"] * factor
        std = row["sample_std_ddof1"]
        precision = ".3g" if abs(average) >= 1000 else ".2f"
        return (rf"${average:{precision}} \pm {std * factor:{precision}}$" if std is not None
                else rf"${average:{precision}}$ (1)")

    lines = [
        "% Requires booktabs (already loaded by the CMP4509 template).",
        "% Input this fragment in the body of the report.",
        r"\begingroup", r"\setlength{\tabcolsep}{3pt}", r"\scriptsize",
    ]
    for label in sorted({row["configuration"] for row in summary}):
        lines.extend([
            r"\begin{table}[htbp]", r"\centering",
            rf"\caption{{Regional toy regression metrics for {tex(label)}. Mean and sample SD across seeds.}}",
            r"\begin{tabular}{llrrrrr}", r"\toprule",
            r"Region & Model & RMSE (true) & RMSE (noisy) & NLL & Cov. 95\% & Width 95\% \\",
            r"\midrule",
        ])
        for region, region_title in region_labels.items():
            for model, title in model_labels.items():
                rows = [index.get((label, model, region, metric)) for metric in keys]
                if not any(rows):
                    continue
                cells = [cell(row, 100 if metric == "coverage_95" else 1)
                         for metric, row in zip(keys, rows)]
                if model == "mse_ensemble_5" and rows[2] is not None:
                    cells[2] += r"\textsuperscript{*}"
                lines.append(" & ".join((region_title, title, *cells)) + r" \\")
            if region != "combined_ood":
                lines.append(r"\addlinespace")
        lines.extend([
            r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
            r"\footnotesize * MSE uses a Gaussian proxy. Deep Ensemble NLL uses the Gaussian mixture. "
            r"Coverage (\%) and width use Gaussian intervals. (1) marks a single seed.", r"\end{table}",
        ])
    if not summary:
        lines.append(r"No measured regional data were available.")
    lines.append(r"\endgroup")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path, summary, v2, validation, manifest):
    model_names = {condition.name: condition.title for condition in CONDITIONS}
    index = {(row["configuration"], row["model"], row["region"], row["metric"]): row for row in summary}
    labels = [row["configuration"] for row in manifest["selected_runs"]]
    limitations = []
    expected_seeds = "|".join(map(str, SEEDS))

    def matched(a, b):
        return a["seeds"] == b["seeds"] == expected_seeds and a["test_sha256_per_seed"] == b["test_sha256_per_seed"]

    def eligible(region, metric, models=None):
        rows = [row for row in summary if row["region"] == region and row["metric"] == metric
                and (models is None or row["model"] in models)]
        reference = next((row for row in rows if row["seeds"] == expected_seeds), None)
        selected = [row for row in rows if reference is not None and matched(reference, row)]
        if len(selected) < len(rows):
            limitations.append(f"{TITLES[region]} · `{metric}` sıralamasına "
                               f"{len(rows) - len(selected)} kayıt eşleşen üç seed koşulunu karşılamadığı için girmedi.")
        return selected

    def winner(region, metric, models=None, highest=False):
        rows = eligible(region, metric, models)
        if not rows:
            return None
        return (max if highest else min)(rows, key=lambda row: row["mean"])

    def describe(row, factor=1):
        return f"{row['configuration']} / {row['model_label']} ({summary_value(row, factor)})" if row else "Karşılaştırılabilir ölçüm yok"

    lines = [
        "# CMP4509 · Toy Regression bölgesel değerlendirmesi", "",
        "## 1. Toy Regression’ın amacı", "",
        "20 eğitim gözlemiyle öğrenilen ortalama tahmini ve tahmin belirsizliğini ayrı ölçtük. "
        "Veri üretiminde koşullu ortalama x³, Gaussian gürültü varyansı 9 ve eğitim desteği [−4,4]. "
        "RMSE, kayıtlı test girdilerinde hem gürültülü hedefe hem de x³’e göre hesaplandı.", "",
        "## 2. V1, V2 ve V3 zaman çizelgesi", "",
        "| Konfigürasyon | Arşiv koşusu | Başlangıç (UTC) | Kayıt durumu |",
        "|---|---|---|---|",
    ]
    for run in manifest["selected_runs"]:
        meta = run["original_manifest"]
        lines.append(f"| {run['configuration']} | `{Path(run['path']).name}` | "
                     f"{meta.get('timestamp_utc', '—')} | {meta.get('status', 'belirtilmemiş')} |")
    for label in validation["missing_named_configurations"]:
        lines.append(f"| {label} | — | — | Ham koşu kaydı bulunamadı |")
    if validation["missing_v3_configuration_count"]:
        lines.append(f"| V3 | — | — | Beklenen 14 koşuldan {validation['detected_v3_configurations']} bulundu |")
    lines.extend([
        "", "## 3. Hangi parametreleri neden değiştirdik?", "",
        "| Konfigürasyon | Learning rate | Batch | Amaç veya kayıtlı ayar |", "|---|---:|---:|---|",
        "| V1 | 0,001 | 5 | Pratik Adam başlangıç ayarı. |",
        "| V2-A | 0,1 | 5 | V1’in batch boyutuyla learning rate duyarlılığını ölçmek. |",
        "| V2-B | 0,001 | 100 | V1’in learning rate’iyle tam batch duyarlılığını ölçmek. |",
        "| V2-C | 0,1 | 100 | Makalenin §3.1’de verdiği genel Adam ayarlarını birlikte sınamak. |",
    ])
    for run in manifest["selected_runs"]:
        if run["version"] == "V3":
            p = run["parameters"]
            lines.append(f"| {run['configuration']} | {p.get('learning_rate', '—')} | {p.get('batch_size', '—')} | "
                         f"{p.get('optimizer', '—')}, {p.get('max_epochs', '—')} epoch |")
    lines.extend([
        "", "20 eğitim gözleminde nominal batch 100, tüm gözlemleri tek optimizer adımında kullanır. "
        "V2-A ile V2-C aynı learning rate’i paylaşır. V3 tasarımı GitHub örneklerinden esinlenmiştir. "
        "Konfigürasyon bazındaki doğrulanmış parametreler ve komutlar manifestte yer alır.", "",
        "## 4. Sol OOD, eğitim aralığı ve sağ OOD sonuçları", "",
        "Bölgeler [−6,−4), [−4,4] ve (4,6]. Birleşik OOD, iki taraftaki örnekler havuzlandıktan sonra "
        "hesaplandı. Her satırdaki değerler seed ortalaması ± örneklem standart sapmasıdır (ddof=1). "
        "Kapsama yüzdelik, aralık genişliği hedefle aynı birimdedir.", "",
        "| Konfigürasyon · model | Bölge | Seed (n) | RMSE x³ | RMSE gürültülü | NLL | %95 kapsama | %95 genişlik |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    names = ("rmse_true", "rmse_noisy", "predictive_nll", "coverage_95", "interval_width_95")
    for label in labels:
        for model, title in model_names.items():
            for region in REGIONS:
                rows = [index.get((label, model, region, metric)) for metric in names]
                if rows[0] is None:
                    continue
                values = [summary_value(row, 100 if metric == "coverage_95" else 1)
                          for metric, row in zip(names, rows)]
                if model == "mse_ensemble_5" and rows[2] is not None:
                    values[2] += " (proxy)"
                lines.append(f"| {label} · {title} | {TITLES[region]} | {rows[0]['n_seeds']} | "
                             + " | ".join(values) + " |")
    lines.extend(["", "## 5. V2-A ile V2-C arasındaki gerçek sayısal fark", ""])
    pairs = [row for row in v2 if row["metric"] in (*names, "coverage_mae") and row["n_paired_seeds"]]
    if pairs:
        lines.extend([
            "| Model | Bölge | Metrik | Eşleşen seed | V2-A | V2-C | C − A ± SS |",
            "|---|---|---|---|---:|---:|---:|",
        ])
        for row in pairs:
            f = 100 if row["metric"].startswith("coverage") else 1
            delta = {"mean": row["paired_delta_c_minus_a"], "sample_std_ddof1": row["paired_delta_sample_std_ddof1"]}
            lines.append(f"| {row['model_label']} | {TITLES[row['region']]} | {row['metric']} | "
                         f"{row['paired_seeds'].replace('|', ', ')} | {number(row['paired_a_mean'] * f)} | "
                         f"{number(row['paired_c_mean'] * f)} | {summary_value(delta, f)} |")
        for region in ("left_ood", "right_ood"):
            items = [row for row in pairs if row["region"] == region and row["model"] == DEEP
                     and row["metric"] in ("rmse_true", "rmse_noisy", "predictive_nll", "coverage_mae")]
            for item in items:
                delta = item["paired_delta_c_minus_a"]
                verdict = "V2-C daha düşük" if delta < 0 else "V2-A daha düşük" if delta > 0 else "Eşit"
                lines.append(f"\n{TITLES[region]} Deep Ensemble `{item['metric']}` için **{verdict}** "
                             f"(C−A = {number(delta)}, n={item['n_paired_seeds']}).")
        right = next((row for row in pairs if row["region"] == "right_ood" and row["model"] == DEEP
                      and row["metric"] == "rmse_true"), None)
        if right:
            change = right["paired_delta_c_minus_a"]
            verdict = ("destekliyor" if change < 0 else "desteklemiyor")
            lines.append(f"\nSağ OOD’de Deep Ensemble’ın x³ RMSE’si V2-C’nin ortalama eğriye daha yakın "
                         f"göründüğü gözlemini {verdict} (C−A = {number(change)}, n={right['n_paired_seeds']}).")
        else:
            lines.append("\nSağ OOD’deki görsel izlenimi sınayacak eşleşen Deep Ensemble x³ RMSE’si yok.")
    else:
        lines.append("V2-A ve V2-C için eşleşen ham test tahmini bulunamadı. Sol ve sağ OOD üstünlüğü "
                     "ile V2-C’nin x>4 bölgesindeki görsel izlenimi bu kayıtlardan doğrulanamıyor.")

    lines.extend([
        "", "## 6. Kapsama ve aralık genişliği neden birlikte değerlendirilmelidir?", "",
        "Bir aralığın genişletilmesi, merkez tahmini aynı kalsa bile daha fazla hedefi içine alabilir. "
        "Kalibrasyonu %50, %80, %90, %95 ve yaklaşık %99,73 seviyelerindeki mutlak kapsama hatalarının "
        "ortalamasıyla ölçtük. %95 aralığı yaklaşık ±1,96σ, %99,73 aralığı ±3σ. x³ kapsaması "
        "`true_function_containment_*` olarak ayrıca kaydedildi. Gerçek ortalama ve σ=3 ile beklenen "
        f"%95 aralık genişliği {number(2 * Z95 * NOISE_VARIANCE**.5)} birimdir.", "",
        "| Bölge | En yüksek %95 kapsama | Genişlik | 5 seviye sapması (yüzde puan) | En düşük 5 seviye sapması |",
        "|---|---|---:|---:|---|",
    ])
    for region in REGIONS:
        high = winner(region, "coverage_95", highest=True)
        calibrated = winner(region, "coverage_mae")
        width = gap = None
        if high:
            group = (high["configuration"], high["model"], region)
            width = index.get((*group, "interval_width_95"))
            gap = index.get((*group, "coverage_mae"))
        if high is not None or calibrated is not None:
            lines.append(f"| {TITLES[region]} | {describe(high, 100)} | {summary_value(width)} | "
                         f"{summary_value(gap, 100)} | {describe(calibrated, 100)} |")
    for label in labels:
        inside = index.get((label, DEEP, "in_domain", "interval_width_95"))
        outside = index.get((label, DEEP, "combined_ood", "interval_width_95"))
        cover = index.get((label, DEEP, "combined_ood", "coverage_95"))
        inside_cover = index.get((label, DEEP, "in_domain", "coverage_95"))
        at_cover = index.get((label, "nll_single_at", "in_domain", "coverage_95"))
        at_width = index.get((label, "nll_single_at", "in_domain", "interval_width_95"))
        if inside and inside_cover and at_cover and at_width and matched(inside_cover, at_cover):
            lines.append(f"\n{label} eğitim aralığında Tek NLL + AT ve Deep Ensemble’ın %95 kapsaması "
                         f"sırasıyla %{number(100 * at_cover['mean'], 2)} ve %{number(100 * inside_cover['mean'], 2)}. "
                         f"Genişlikleri sırasıyla {number(at_width['mean'])} ve {number(inside['mean'])}. "
                         "Bu iki ölçüm, aralık genişliğinin ve tahmin merkezlerinin kapsamadaki payını tek başına belirleyemez.")
        if inside and outside and cover and inside["mean"] > 0:
            lines.append(f"\n{label} Deep Ensemble’da birleşik OOD %95 genişliği, eğitim aralığının "
                         f"{number(outside['mean'] / inside['mean'], 2)} katı. Birleşik OOD kapsaması "
                         f"{summary_value(cover, 100)}%. Nominal %95’ten mutlak fark "
                         f"{number(100 * abs(cover['mean'] - .95), 2)} yüzde puan.")
    lines.extend([
        "", "Deep Ensemble NLL, tam Gaussian mixture yoğunluğundan logsumexp ile hesaplandı. "
        "Ana kapsama ve genişlik tablosu Gaussian moment eşlemesini kullanır. Mixture CDF’sinin "
        "sayısal tersinden elde edilen eşit kuyruklu aralıklar, `mixture_*` metriklerindedir. "
        "MSE Ensemble için Gaussian proxy, üye ortalamalarının popülasyon varyansını kullanır ve "
        "aralık ile NLL hesabında arşivdeki varyans tabanını uygular (V1’de 10⁻⁶). Bu proxy, "
        "ölçüm gürültüsünü öğrenmediği için NLL sıralamasından çıkarıldı.", "",
        "## 7. Makaledeki Figure 1 ile karşılaştırma", "",
        "Figure 1, gerçek x³ eğrisini, gürültülü eğitim noktalarını ve ±3σ bantlarını gösterir. "
        "§3.2’deki iki iddiayı bu kayıtlarda ölçülebilir sonuçlarla inceledik. "
        "[Lakshminarayanan, Pritzel ve Blundell, §3.2 ve Figure 1](https://arxiv.org/pdf/1612.01474v3#page=6)", "",
    ])
    for label in labels:
        lines.append(f"**{label}**")
        for region in REGIONS:
            base = index.get((label, "mse_ensemble_5", region, "coverage_mae"))
            single = index.get((label, "nll_single", region, "coverage_mae"))
            if base and single and matched(base, single):
                verdict = ("destekliyor" if single["mean"] < base["mean"] else "desteklemiyor")
                lines.append(f"- {TITLES[region]} kapsama sapması, varyansı NLL ile öğrenmenin MSE proxy’sine "
                             f"göre belirsizliği iyileştirdiği iddiasını {verdict}. Tek NLL "
                             f"{summary_value(single, 100)}, MSE proxy {summary_value(base, 100)} yüzde puan.")
        for region in ("left_ood", "right_ood", "combined_ood"):
            for metric in ("predictive_nll", "rmse_true"):
                ensemble = index.get((label, DEEP, region, metric))
                single = index.get((label, "nll_single", region, metric))
                adversarial = index.get((label, "nll_single_at", region, metric))
                if ensemble and single and matched(ensemble, single):
                    verdict = ("destekliyor" if ensemble["mean"] < single["mean"] else "desteklemiyor")
                    extra = (f" Tek NLL + AT {summary_value(adversarial)}."
                             if adversarial and matched(ensemble, adversarial) else "")
                    lines.append(f"- {TITLES[region]} `{metric}`, ensemble’ın tek NLL modeline "
                                 f"göre OOD performansını iyileştirdiği iddiasını {verdict}. "
                                 f"Deep Ensemble {summary_value(ensemble)}, tek NLL {summary_value(single)}.{extra}")
        epistemic = [index.get((label, DEEP, region, "epistemic_variance"))
                     for region in ("in_domain", "left_ood", "right_ood")]
        if all(epistemic):
            values = [summary_value(row) for row in epistemic]
            lines.append(f"- Deep Ensemble epistemic varyansı iç, sol ve sağ bölgelerde sırasıyla "
                         f"{', '.join(values)}. Bu değerler belirsizliğin eğitim desteği dışında "
                         "hangi tarafa doğru arttığını gösteriyor.")
        lines.append("")
    lines.extend(["## 8. En güçlü ölçülen sonuçlar", ""])
    gaussian_models = tuple(item.name for item in CONDITIONS if item.gaussian)
    for region in REGIONS:
        accuracy = winner(region, "rmse_true")
        likelihood = winner(region, "predictive_nll", gaussian_models)
        calibrated = winner(region, "coverage_mae")
        lines.append(f"- **{TITLES[region]}** · En düşük x³ RMSE {describe(accuracy)}. "
                     f"En düşük NLL {describe(likelihood)}. En düşük kapsama sapması {describe(calibrated, 100)} "
                     "yüzde puan.")
    for models, description in ((gaussian_models, "Tüm Gaussian modeller"), ((DEEP,), "Deep Ensemble")):
        closest = winner("in_domain", "aleatoric_abs_error_from_9", models)
        if closest:
            row = index[closest["configuration"], closest["model"], "in_domain", "aleatoric_variance"]
            lines.append(f"- **{description}** · Varyans 9’a en yakın sonuç {describe(row)}. "
                         f"Seed başına mutlak sapma {summary_value(closest)}.")
    lines.extend([
        "", "## 9. Sınırlamalar", "",
        f"- Beklenen {validation['expected_configuration_count']} konfigürasyonun "
        f"{validation['detected_configuration_count']} tanesi kullanılabilir arşivde tespit edildi. "
        "Eksik koşuların yanı sıra varsa eksik seed, model, bölge ve dosya bilgileri `validation.json` içinde.",
        "- Her bölge için seed’ler eşit ağırlıklı. Seed sayısı üç ve sonuçlar betimsel. "
        "Örneklem standart sapmaları anlamlılık testi veya güven aralığı işlevi görmez.",
        "- Kaydedilmiş gürültülü test setleri değiştirilmedi. Her seed ve split için bulunan ilk kayıtlı "
        "test seti ortak referans oldu. Eşleşen girdilerde arşiv tahmini kullanıldı. Gerekirse checkpoint "
        "ile CPU inference, o da yoksa kayıtlı yerel test seti kullanılır. Test çifti kimlikleri CSV’de.",
        "- Şekil 1 panelleri varsa kayıtlı gürültüsüz çizim grid’ini kullanır. Tablolardaki kapsama "
        "ve RMSE ölçümleri kayıtlı test girdilerinden gelir.",
        f"- Çalıştırma ortamı `{manifest['environment']['os']}`, cihaz CPU. "
        "Her okunmuş giriş dosyasının SHA-256 değeri çalıştırma sonunda yeniden doğrulandı.",
    ])
    for limitation in dict.fromkeys(limitations):
        lines.append(f"- {limitation}")
    lines.extend(["", "## 10. Nihai bilimsel değerlendirme", ""])
    region = "combined_ood"
    accuracy = winner(region, "rmse_true")
    likelihood = winner(region, "predictive_nll", gaussian_models)
    calibration = winner(region, "coverage_mae")
    if accuracy and likelihood and calibration:
        lines.append(f"Birleşik OOD’de x³ eğrisine en yakın ortalama {describe(accuracy)}, olasılıksal "
                     f"tahminde en düşük NLL {describe(likelihood)} ve nominal kapsama seviyelerine en yakın "
                     f"sonuç {describe(calibration, 100)} yüzde puan sapmayla ölçüldü.")
    if validation["missing_named_configurations"] or validation["missing_v3_configuration_count"]:
        lines.append("V1, V2 ve V3 arasında nihai bir karşılaştırma için eksik V2 ve V3 koşularının "
                     "ham test tahminleri veya checkpoint’leri ile kayıtlı test setleri gerekiyor."
                     if not validation["detected_v3_configurations"] else
                     "Sürümler arası bulgular erişilebilen arşivlerle sınırlı.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
